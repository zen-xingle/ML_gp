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
    'dataset' : {'name': 'poisson_v4_02',
                 'interp_data': False,
                 
                 # preprocess
                 'seed': None,
                 'train_start_index': 0, 
                 'train_sample': 8, 
                 'eval_start_index': 0, 
                 'eval_sample':256,

                 'inputs_format': ['x[0]', 'y[0]'],
                 'outputs_format': ['y[2]'],

                 'force_2d': True,
                 'x_sample_to_last_dim': False,
                 'y_sample_to_last_dim': False,
                 'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                 },

    'connection_method': 'res_mapping',
    'lr': {'kernel':0.1, 
           'optional_param':0.1, 
           'noise':0.1},
    'weight_decay': 1e-3,
    # kernel number as dim + 1
    'kernel': {
            'K1': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
            # 'K2': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
            # 'K3': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
              },
    'auto_broadcast_kernel': True,
    'evaluate_method': ['mae', 'rmse', 'r2'],
    'optimizer': 'adam',
    'exp_restrict': True,
    'input_normalize': True,
    'output_normalize': True,
    'noise_init' : 1.,
    'grid_config': {'grid_size': [-1], 
                    'type': 'fixed', # learnable, fixed
                    'dimension_map': 'identity', # latent space, identity, learnable_identity, learnable_map
                    'auto_broadcast_grid_size': True,
                    },
    'cuda': False,
}

def _last_dim_to_fist(_tensor):
    _dim = [i for i in range(_tensor.ndim)]
    _dim.insert(0, _dim.pop())
    return _tensor.permute(*_dim)

def _first_dim_to_last(_tensor):
    _dim = [i+1 for i in range(_tensor.ndim-1)]
    _dim.append(0)
    return _tensor.permute(*_dim)


class CIGAR(torch.nn.Module):
    # def __init__(self, grid_params_list, kernel_list, target_list, normalize=True, restrict_method= 'exp') -> None:
    def __init__(self, module_config, data=None) -> None:
        super().__init__()
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
        self._select_connection_kernel(module_config['connection_method'])

        # TODO if param allow more than single kernel, optimize code here
        self.vector_dims = len(self.grid)
        self.param_dims = 1

        # X - normalize
        if module_config['input_normalize'] is True:
            self.X_normalizer = Normalizer(self.inputs_tr[0])
            self.inputs_tr[0] = self.X_normalizer.normalize(self.inputs_tr[0])
        else:
            self.X_normalizer = None

        # Y - normalize
        # TODO normalize according to dims
        if module_config['output_normalize'] is True:
            self.Y_normalizer = Normalizer(self.outputs_tr[0], dim=[i for i in range(len(self.outputs_tr[0].shape))])
            self.outputs_tr[0] = self.Y_normalizer.normalize(self.outputs_tr[0])
            self.inputs_tr[1] = self.Y_normalizer.normalize(self.inputs_tr[1])
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

    def set_up_unknown_yl_var(self, yl_var_unknow, k_star_l, k_l_inv):
        self.yl_var_unknow = yl_var_unknow
        self.k_star_l = k_star_l
        self.k_l_inv = k_l_inv

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

    def _select_connection_kernel(self, type_name):
        from kernel.Multi_fidelity_connection import rho_connection, mapping_connection
        assert type_name in ['res_rho', 'res_mapping']
        if type_name in ['res_rho']:
            self.target_connection = rho_connection()
        elif type_name in ['res_mapping']:
            self.target_connection = mapping_connection(self.inputs_tr[1][0,...].shape, 
                                                        self.outputs_tr[0][0,...].shape,
                                                        sample_last_dim=False)

    def _optimizer_setup(self):
        optional_params = []
        if self.module_config['grid_config']['type'] == 'learnable':
            for i in range(len(self.grid)):
                optional_params.append(self.grid[i])
        if self.module_config['grid_config']['dimension_map'] not in ['identity']:
            for i in range(len(self.mapping_vector)):
                optional_params.append(self.mapping_vector[i])
        
        kernel_learnable_param = []
        # for _kernel in self.kernel_list:
        #     _kernel.get_param(kernel_learnable_param)
        _kernel = self.kernel_list[-1]
        _kernel.get_param(kernel_learnable_param)
        self.target_connection.get_param(kernel_learnable_param)

        # TODO support SGD?
        # module_config['lr'] = {'kernel':0.01, 'optional_param':0.01, 'noise':0.01}
        self.optimizer = torch.optim.Adam([{'params': optional_params, 'lr': self.module_config['lr']['optional_param']}, 
                                           {'params': [self.noise], 'lr': self.module_config['lr']['noise']},
                                           {'params': kernel_learnable_param , 'lr': self.module_config['lr']['kernel']}],
                                           weight_decay = self.module_config['weight_decay']) # 改了lr从0.01 改成0.0001

    def update_kernel_result(self):
        # init in first time
        if not hasattr(self, 'K'):
            self.K = []
        if not hasattr(self, 'K_eigen'):
            self.K_eigen = []
        
        # clear
        self.K.clear()
        self.K_eigen.clear()

        # update x
        self.K.append(self.kernel_list[0](self.inputs_tr[0], self.inputs_tr[0]))
        self.K_eigen.append(eigen_pairs(self.K[-1]))

    '''
    def compute_Kstar(self, new_param):
        # use for predict decorator
        all_params = torch.cat([self.grid_params_list[-1], new_param], 0)
        if self.K != []:
            self.Kstar = [*self.K[:-1], self.kernel_list[-1](all_params, all_params)]
        return self.Kstar
    '''

    def compute_loss(self):
        # during training, due to the change of params in Kernel, recalculate K again.
        self.update_kernel_result()
        train_num, ydim = self.outputs_tr[0].shape

        if self.module_config['exp_restrict'] is True:
            _noise = self.noise.exp()
        else:
            _noise = self.noise
        Sigma = self.K[0] + _noise.pow(-1)* torch.eye(train_num,  device=list(self.parameters())[0].device) + JITTER* torch.eye(train_num,  device=list(self.parameters())[0].device)
        
        if hasattr(self, 'yl_var_unknow'):
            Sigma += torch.diag_embed(self.yl_var_unknow)

        L = torch.linalg.cholesky(Sigma)
        _res = self.target_connection(self.inputs_tr[-1], self.outputs_tr[0])
        gamma,_ = torch.triangular_solve(_res, L, upper = False)

        nll =  0.5 * (gamma ** 2).sum() +  L.diag().log().sum() * ydim \
          + 0.5 * train_num * torch.log(2 * torch.tensor(PI)) * ydim
        return nll


    def predict(self, input_param):
        input_param = deepcopy(input_param)
        with torch.no_grad():
            if self.module_config['input_normalize'] is True:
                input_param[0] = self.X_normalizer.normalize(input_param[0])
            
            self.update_kernel_result() 

            if self.module_config['exp_restrict'] is True:
                _noise = self.noise.exp()
            else:
                _noise = self.noise
            Sigma = self.K[0] + _noise.pow(-1)* tensorly.ones(self.K[0].size(0),  device=list(self.parameters())[0].device) + JITTER
            L = torch.cholesky(Sigma)

            # /*** Get predict mean***/
            if len(input_param[0].shape) != len(self.inputs_tr[0].shape):
                input_param[0] = input_param[0].reshape(1, *input_param[0].shape)
            K_star = self.kernel_list[-1](self.inputs_tr[0], input_param[0])

            predict_u = K_star.T @ torch.cholesky_solve(self.outputs_tr[0], L)  # torch.linalg.cholesky()

            if self.module_config['output_normalize'] is True:
                base_yl = self.Y_normalizer.normalize(input_param[1])
                predict_u = self.target_connection.low_2_high(base_yl, predict_u)
                predict_u = self.Y_normalizer.denormalize(predict_u)
            else:
                predict_u = self.target_connection.low_2_high(input_param[1], predict_u)

            # /*** Get predict var***/
            LinvKx,_ = torch.triangular_solve(K_star, L, upper = False)
            pred_var = self.kernel_list[-1](input_param[0], input_param[0]).diag().view(-1,1) - (LinvKx.t() @ LinvKx).diag().view(-1, 1)
            pred_var += _noise.pow(-1)

            if hasattr(self, 'yl_var_unknow') and len(input_param)>2:
                k_r_inv = torch.inverse(self.K[0]) + _noise.pow(-1)* tensorly.ones(self.K[0].size(0),  device=list(self.parameters())[0].device) + JITTER
                gamma = K_star.T@k_r_inv - self.k_star_l.T@self.k_l_inv
                pred_uncertainty = gamma @ torch.diag_embed(self.yl_var_unknow) @ gamma.T
                pred_var = input_param[2] + pred_var + pred_uncertainty.diag().reshape(-1, 1)
                if self.module_config['output_normalize'] is True:
                    pred_var = pred_var* self.Y_normalizer.std

        return predict_u, pred_var

    def train(self):
        loss = self.compute_loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if 'GP_DEBUG' in os.environ and os.environ['GP_DEBUG'] == 'True':
            print('self.noise:{}'.format(self.noise.data))


    def eval(self):
        print('---> start eval')
        if hasattr(self, 'base_var'):
            predict_y, predict_var = self.predict(self.inputs_eval + [self.base_var])
        else:
            predict_y, predict_var = self.predict(self.inputs_eval)
        self.predict_y = deepcopy(predict_y)
        self.predict_var = deepcopy(predict_var)
        target = self.outputs_eval[0]
        result = high_level_evaluator([predict_y, self.predict_var.expand_as(predict_y)], target, self.module_config['evaluate_method'])
        print(result)
        return result


if __name__ == '__main__':
    module_config = {'noise_init' : 1.,}

    x = np.load('./data/sample/input.npy')
    y0 = np.load('./data/sample/output_fidelity_1.npy')
    y2 = np.load('./data/sample/output_fidelity_2.npy')

    x = torch.tensor(x).float()
    source_shape = y0[128:,...].shape
    y0 = torch.tensor(y0).float().reshape(y0.shape[0], -1)
    y2 = torch.tensor(y2).float().reshape(y2.shape[0], -1)

    train_x = [x[:128,:], y0[:128,...]]      # permute for sample to last dim
    train_y = [y2[:128,...]]
    
    eval_x = [x[128:,:], y0[128:,...]]
    eval_y = [y2[128:,...]]

    cigp = CIGAR(module_config, [train_x, train_y, eval_x, eval_y])
    for epoch in range(1000):
        print('epoch {}/{}'.format(epoch+1, 300), end='\r')
        cigp.train()
    print('\n')
    cigp.eval()

    from visualize_tools.plot_field import plot_container
    data_list = [cigp.outputs_eval[0].numpy().reshape(source_shape), cigp.predict_y.numpy().reshape(source_shape)]
    data_list.append(abs(data_list[0] - data_list[1]))
    label_list = ['groundtruth','predict', 'diff']
    pc = plot_container(data_list, label_list, sample_dim=0)
    pc.plot()
