from copy import deepcopy
# import gpytorch
import math
import torch
import tensorly
import numpy as np
import os
import sys
import random

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from tensorly.decomposition import tucker
from tensorly import tucker_to_tensor
from utils import *
from kernel import kernel_utils
from utils.data_utils import data_register

tensorly.set_backend('pytorch')


default_module_config = {
    'dataset' : {'name': 'Piosson_mfGent_v5',
                 'interp_data': False,
                 
                 # preprocess
                 'seed': None,
                 'train_start_index': 0,
                 'train_sample': 8, 
                 'eval_start_index': 0, 
                 'eval_sample':256,
                
                 'inputs_format': ['x[0]'],
                 'outputs_format': ['y[2]'],

                 'force_2d': False,
                 'x_sample_to_last_dim': False,
                 'y_sample_to_last_dim': True,
                 'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                 },

    'lr': {'kernel':0.01, 
           'optional_param':0.01, 
           'noise':0.01},
    'weight_decay': 1e-3,
    # kernel number as dim + 1
    'kernel': {
            'K1': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
            # 'K2': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
            # 'K3': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
            # 'K4': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
              },
    'auto_broadcast_kernel': True,
    # 'evaluate_method': ['mae', 'rmse', 'r2', 'gaussian_loss'],
    'evaluate_method': ['mae', 'rmse', 'r2'],
    'optimizer': 'adam',
    'exp_restrict': False,
    'input_normalize': True,
    'output_normalize': True,
    'noise_init' : 1.0,
    'grid_config': {'grid_size': [-1], 
                    'type': 'fixed', # learnable, fixed
                    'dimension_map': 'identity', # latent space: identity, learnable_identity, learnable_map
                    'auto_broadcast_grid_size': True,
                    'squeeze_to_01': False,
                    },
    'cuda': False,
    'BayeSTA': False,
}

def _last_dim_to_fist(_tensor):
    _dim = [i for i in range(_tensor.ndim)]
    _dim.insert(0, _dim.pop())
    return _tensor.permute(*_dim)

def _first_dim_to_last(_tensor):
    _dim = [i+1 for i in range(_tensor.ndim-1)]
    _dim.append(0)
    return _tensor.permute(*_dim)


class HOGP_MODULE(torch.nn.Module):
    # def __init__(self, grid_params_list, kernel_list, target_list, normalize=True, restrict_method= 'exp') -> None:
    def __init__(self, module_config, data=None) -> None:
        super().__init__()
        # default_module_config.update(module_config)
        _final_config = smart_update(default_module_config, module_config)
        self.module_config = deepcopy(_final_config)
        module_config = deepcopy(_final_config)

        # param check
        assert module_config['optimizer'] in ['adam'], 'now optimizer only support adam, but get {}'.format(module_config['optimizer'])

        # load_data
        if data == None:
            # load_data
            data_register.data_regist(self, module_config['dataset'], module_config['cuda'])
        else:
            mlgp_log.i("Data is given, directly using input data. Notice data should be given as list, order is [input_x, output_y, eval_input_x, eval_input_y]")
            self.module_config['dataset'] = 'custom asigned, tracking source failed'
            self.inputs_tr = data[0]
            self.outputs_tr = data[1]
            self.inputs_eval = data[2]
            self.outputs_eval = data[3]

        self._grid_setup(module_config['grid_config'])

        # TODO if param allow more than single kernel, optimize code here
        self.vector_dims = len(self.grid)
        self.param_dims = 1

        # X - normalize
        if module_config['input_normalize'] is True:
            self.X_normalizer = Normalizer(self.inputs_tr[0])
            self.inputs_tr[0] = self.X_normalizer.normalize(self.inputs_tr[0])
        else:
            self.X_normalizer = None
        self.latent_input = []
        if self.module_config['BayeSTA']:
            self.latent_num = 3
            for i in range(self.latent_num):
                self.latent_input.append(torch.nn.Parameter(torch.tensor(i).float()*400, requires_grad=True))

        # Y - normalize
        # TODO normalize according to dims
        if module_config['output_normalize'] is True:
            self.Y_normalizer = Normalizer(self.outputs_tr[0], dim=[i for i in range(len(self.outputs_tr[0].shape))])
            self.outputs_tr[0] = self.Y_normalizer.normalize(self.outputs_tr[0])
        else:
            self.Y_normalizer = None

        # init kernel
        if len(module_config['kernel']) != self.outputs_tr[0].dim():
            # broadcast kernel
            if len(module_config['kernel']) != 1:
                mlgp_log.e("kernel broadcast only valid on single kernel, but got {}".format(len(module_config)))
            else:
                for i in range(self.outputs_tr[0].dim()-1):
                    module_config['kernel']['K{}'.format(i+2)] = deepcopy(module_config['kernel'][list(module_config['kernel'].keys())[0]])
        kernel_utils.register_kernel(self, module_config['kernel'])

        # init noise
        if module_config['exp_restrict'] is True:
            self.noise = torch.nn.Parameter(torch.log(torch.tensor(module_config['noise_init'])))
        else:
            self.noise = torch.nn.Parameter(torch.tensor(module_config['noise_init']))

        # init optimizer
        self._optimizer_setup()


    def _grid_setup(self, grid_config):
        if grid_config['auto_broadcast_grid_size'] is True and len(grid_config['grid_size'])==1:
            grid_config['grid_size'] = grid_config['grid_size']* (self.outputs_tr[0].ndim - 1)
        self.grid = []
        for i,_value in enumerate(grid_config['grid_size']):
            if _value == -1:
                _value = self.outputs_tr[0].shape[i]
            if grid_config['type'] == 'fixed':
                self.grid.append(torch.tensor(range(_value)).reshape(-1,1).float())
            elif grid_config['type'] == 'learnable':
                self.grid.append(torch.nn.Parameter(torch.tensor(range(_value)).reshape(-1,1).float()))
            if grid_config['squeeze_to_01'] is True:
                self.grid[-1] = self.grid[-1]/(_value-1)
        
        # identity, learnable_identity, learnable_map
        self.mapping_vector = []
        for i,_value in enumerate(grid_config['grid_size']):
            if _value == -1:
                _value = self.outputs_tr[0].shape[i]
            if grid_config['dimension_map'] == 'identity':
                self.mapping_vector.append(torch.eye(_value))
            elif grid_config['dimension_map'] == 'learnable_identity':
                self.mapping_vector.append(torch.nn.Parameter(torch.eye(_value)))
            elif grid_config['dimension_map'] == 'learnable_map':
                # TODO add init function
                self.mapping_vector.append(torch.nn.Parameter(torch.randn(_value, self.outputs_tr[0].shape[i])))

        if self.module_config['cuda']:
            for _params in [self.grid, self.mapping_vector]:
                for i, _v in enumerate(_params):
                    _params[i] = _v.cuda()

    def _optimizer_setup(self):
        optional_params = []
        if self.module_config['grid_config']['type'] == 'learnable':
            for i in range(len(self.grid)):
                optional_params.append(self.grid[i])
        if self.module_config['grid_config']['dimension_map'] not in ['identity']:
            for i in range(len(self.mapping_vector)):
                optional_params.append(self.mapping_vector[i])
        
        kernel_learnable_param = []
        for _kernel in self.kernel_list:
            _kernel.get_param(kernel_learnable_param)

        # TODO support SGD?
        # module_config['lr'] = {'kernel':0.01, 'optional_param':0.01, 'noise':0.01}
        self.optimizer = torch.optim.Adam([{'params': optional_params, 'lr': self.module_config['lr']['optional_param']}, 
                                           {'params': [self.noise], 'lr': self.module_config['lr']['noise']},
                                           {'params': kernel_learnable_param + self.latent_input, 'lr': self.module_config['lr']['kernel']}],
                                           weight_decay = self.module_config['weight_decay']) # 改了lr从0.01 改成0.0001

    def compute_var(self):
        # init in first time
        if not hasattr(self, 'K'):
            self.K = []
        if not hasattr(self, 'K_eigen'):
            self.K_eigen = []
        
        # clear
        self.K.clear()
        self.K_eigen.clear()

        # update grid
        for i in range(len(self.grid)):
            _in = tensorly.tenalg.mode_dot(self.grid[i], self.mapping_vector[i], 0)
            self.K.append(self.kernel_list[i](_in, _in))
            self.K_eigen.append(eigen_pairs(self.K[i]))

        # update x
        _index = len(self.grid)
        # for i in range(len(self.inputs_tr)):
        if not self.module_config['BayeSTA']:
            self.K.append(self.kernel_list[_index](self.inputs_tr[0], self.inputs_tr[0]))
            self.K_eigen.append(eigen_pairs(self.K[-1]))
        else:
            latent_space = self._get_latent_space(self.inputs_tr[1])
            _in = torch.concat([self.inputs_tr[0], latent_space], dim=1)
            self.K.append(self.kernel_list[_index](_in, _in))
            self.K_eigen.append(eigen_pairs(self.K[-1]))

    def _get_latent_space(self, inputs):
        latent_space = torch.zeros_like(inputs)
        for i in range(self.latent_num):
            latent_space += self.latent_input[i]* (inputs==i)
        return latent_space

    '''
    def compute_Kstar(self, new_param):
        # use for predict decorator
        all_params = torch.cat([self.grid_params_list[-1], new_param], 0)
        if self.K != []:
            self.Kstar = [*self.K[:-1], self.kernel_list[-1](all_params, all_params)]
        return self.Kstar
    '''

    def update_product(self):
        # during training, due to the change of params in Kernel, recalculate K again.
        self.compute_var()

        # Kruskal operator
        # compute log(|S|) = sum over the logarithm of all the elements in A. O(nd) complexity.
        _init_value = torch.tensor([1.0],  device=list(self.parameters())[0].device).reshape(*[1 for i in self.K])
        lambda_list = [eigen.value.reshape(-1, 1) for eigen in self.K_eigen]
        A = tucker_to_tensor((_init_value, lambda_list))

        if self.module_config['exp_restrict'] is True:
            _noise = self.noise.exp()
        else:
            _noise = self.noise
        # A = A + _noise * tensorly.ones(A.shape)
        A = A + _noise.pow(-1)* tensorly.ones(A.shape,  device=list(self.parameters())[0].device)
        # TODO: add jitter limite here?
        
        # vec(z).T@ S.inverse @ vec(z) = b.T @ b,  b = S.pow(-1/2) @ vec(z)
        T_1 = tensorly.tenalg.multi_mode_dot(self.outputs_tr[0], [eigen.vector.T for eigen in self.K_eigen])
        T_2 = T_1 * A.pow(-1/2)
        T_3 = tensorly.tenalg.multi_mode_dot(T_2, [eigen.vector for eigen in self.K_eigen])
        b = tensorly.tensor_to_vec(T_3)

        # g = S.inverse@vec(z)
        g = tensorly.tenalg.multi_mode_dot(T_1 * A.pow(-1), [eigen.vector for eigen in self.K_eigen])
        # g_vec = tensorly.tensor_to_vec(g)

        self.b = b
        self.A = A
        self.g = g
        # self.g_vec = g_vec


    def predict(self, input_param):
        input_param = deepcopy(input_param)
        with torch.no_grad():
            if self.module_config['input_normalize'] is True:
                input_param[0] = self.X_normalizer.normalize(input_param[0])
                if self.module_config['BayeSTA']:
                    latent_space = self._get_latent_space(input_param[1])
                    _in = torch.concat([input_param[0], latent_space], dim=1)
                    input_param[0] = _in

            #! may needn't?
            # self.update_product()

            # /*** Get predict mean***/
            if len(input_param[0].shape) != len(self.inputs_tr[0].shape):
                input_param[0] = input_param[0].reshape(1, *input_param[0].shape)
            if self.module_config['BayeSTA']:
                latent_space = self._get_latent_space(self.inputs_tr[1])
                src_in = torch.concat([self.inputs_tr[0], latent_space], dim=1)
                K_star = self.kernel_list[-1](input_param[0], src_in)
            else:
                K_star = self.kernel_list[-1](input_param[0], self.inputs_tr[0])

            K_predict = self.K[:-1] + [K_star]

            '''
            predict_u = tensorly.tenalg.kronecker(K_predict)@self.g_vec #-> memory error
            so we use tensor.tenalg.multi_mode_dot instead
            '''
            predict_u = tensorly.tenalg.multi_mode_dot(self.g, K_predict)
            # predict_u = predict_u.reshape_as(self.outputs_eval[:,:,0])
            if self.module_config['output_normalize'] is True:
                predict_u = self.Y_normalizer.denormalize(predict_u)

            # /*** Get predict var***/
            # NOTE: now only work for the normal predict
            _init_value = torch.tensor([1.0], device=list(self.parameters())[0].device).reshape(*[1 for i in self.K])
            diag_K = tucker_to_tensor(( _init_value, [K.diag().reshape(-1,1) for K in self.K[:-1]]))
            diag_K = self.kernel_list[-1](input_param[0], input_param[0]).diag()* diag_K

            S = self.A * self.A.pow(-1/2)
            S_2 = S.pow(2)
            # S_product = tensorly.tenalg.multi_mode_dot(S_2, [eigen_vector_d1.pow(2), eigen_vector_d2.pow(2), (K_star@K_p.inverse()@eigen_vector_p).pow(2)])
            S_product = tensorly.tenalg.multi_mode_dot(S_2, [self.K_eigen[i].vector.pow(2) for i in range(len(self.K_eigen)-1)]+[(K_star@(self.K[-1]+JITTER*torch.eye(self.K[-1].shape[0], device=list(self.parameters())[0].device)).inverse()@self.K_eigen[-1].vector).pow(2)])
            M = diag_K + S_product
            if self.module_config['output_normalize'] is True:
                M = M * self.Y_normalizer.std ** 2

        return predict_u, M


    def train(self):
        self.update_product()
        #! loss = -1/2* torch.log(abs(self.A)).mean()
        nd = torch.prod(torch.tensor([value for value in self.A.shape]))
        loss = -1/2* nd * torch.log(torch.tensor(2 * math.pi, device=list(self.parameters())[0].device))
        loss += -1/2* torch.log(self.A).sum()
        loss += -1/2* self.b.t() @ self.b

        loss = -loss/nd
        # loss = -loss
        # print('loss:', loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if 'GP_DEBUG' in os.environ and os.environ['GP_DEBUG'] == 'True':
            print('self.noise:{}'.format(self.noise.data))


    def eval(self):
        print('---> start eval')
        predict_y, predict_var = self.predict(self.inputs_eval)
        self.predict_y = deepcopy(predict_y)
        self.predict_var = predict_var
        # result = performance_evaluator(predict_y, self.outputs_eval[0], self.module_config['evaluate_method'])
        predict_y = _last_dim_to_fist(predict_y)
        predict_var = _last_dim_to_fist(predict_var)
        target = _last_dim_to_fist(self.outputs_eval[0])
        result = high_level_evaluator([predict_y, predict_var], target, self.module_config['evaluate_method'])
        print(result)

        if self.module_config['BayeSTA'] is True:
            from utils.gp_output_decorator import posterior_output_decorator
            if not hasattr(self, 'random_mask'):
                # gen mask only once
                self.random_mask = []
                mask_number_per_sample = 0.1
                sample_size = self.outputs_eval[0][...,0].numel()
                for i in range(self.outputs_eval[0].shape[-1]):
                    if (i+1)%5 != 0:
                        self.random_mask.append([True]* self.outputs_eval[0].shape[0])
                    else:
                        self.random_mask.append([False]* self.outputs_eval[0].shape[0])
                    # For STA mask
                    # mask = [i for i in range(sample_size)]
                    # pop_item = np.array([i for i in range(sample_size)]).reshape(5,3)[2:,2].tolist()
                    # for _v in pop_item:
                    #     mask.remove(_v)
                    # self.random_mask.append(mask)

            pod = posterior_output_decorator(self, 'hogp', self.random_mask, lr=0.001)
            for i in range(100):
                pod.train()
                print('pod epoch {}/{}'.format(i+1, 100),end='\r')
            pod_result = high_level_evaluator([_last_dim_to_fist(pod.eval()), predict_var], target, self.module_config['evaluate_method'])
            print(pod_result, '<-- posterior output optimize')
            for _k, _v in pod_result.items():
                result[_k + '_STA'] = _v

        return result


if __name__ == '__main__':
    module_config = {}

    x = np.load('./data/sample/input.npy')
    y0 = np.load('./data/sample/output_fidelity_1.npy')
    y2 = np.load('./data/sample/output_fidelity_2.npy')

    x = torch.tensor(x).float()
    y0 = torch.tensor(y0).float()
    y2 = torch.tensor(y2).float()

    train_x = [x[:128,:],]      # permute for sample to last dim
    train_y = [y2[:128,...].permute(1,2,0)]
    
    eval_x = [x[128:,:]]
    eval_y = [y2[128:,...].permute(1,2,0)]
    source_shape = y0[128:,...].shape

    cigp = HOGP_MODULE(module_config, [train_x, train_y, eval_x, eval_y])
    for epoch in range(300):
        print('epoch {}/{}'.format(epoch+1, 300), end='\r')
        cigp.train()
    print('\n')
    cigp.eval()

    from visualize_tools.plot_field import plot_container
    data_list = [cigp.outputs_eval[0].numpy(), cigp.predict_y.numpy()]
    data_list.append(abs(data_list[0] - data_list[1]))
    label_list = ['groundtruth','predict', 'diff']
    pc = plot_container(data_list, label_list, sample_dim=2)
    pc.plot()