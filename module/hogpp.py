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
from utils.eigen import eigen_pairs
from utils.normalizer import Normalizer
# from module.gp_output_decorator import posterior_output_decorator
from utils.performance_evaluator import performance_evaluator, high_level_evaluator
from scipy.io import loadmat

# optimize for main_controller

JITTER = 1e-6
EPS = 1e-10
PI = 3.1415

tensorly.set_backend('pytorch')

fidelity_map = {
    'low': 0,
    'medium': 1,
    'high': 2
}

mat_dataset_paths = {
                'poisson_v4_02': 'data/MultiFidelity_ReadyData/poisson_v4_02.mat',
                'burger_v4_02': 'data/MultiFidelity_ReadyData/burger_v4_02.mat',
                'Burget_mfGent_v5': 'data/MultiFidelity_ReadyData/Burget_mfGent_v5.mat',
                'Burget_mfGent_v5_02': 'data/MultiFidelity_ReadyData/Burget_mfGent_v5_02.mat',
                'Heat_mfGent_v5': 'data/MultiFidelity_ReadyData/Heat_mfGent_v5.mat',
                'Piosson_mfGent_v5': 'data/MultiFidelity_ReadyData/Piosson_mfGent_v5.mat',
                'Schroed2D_mfGent_v1': 'data/MultiFidelity_ReadyData/Schroed2D_mfGent_v1.mat',
                'TopOP_mfGent_v5': 'data/MultiFidelity_ReadyData/TopOP_mfGent_v5.mat',
                'DoublePendu_mfGent_v01': 'data/MultiFidelity_ReadyData/DoublePendu_mfGent_v01.mat',
            } # they got the same data format


default_module_config = {
    'dataset' : {'name': 'Piosson_mfGent_v5',
                 'fidelity': ['low'],
                 'type':'x_2_y',    # x_yl_2_yh, x_2_y
                 'connection_method': 'res_mapping',  # Only valid when x_yl_2_yh, identity, res_rho, res_mapping
                 'train_start_index': 0, 
                 'train_sample': 8, 
                 'eval_start_index': 0, 
                 'eval_sample':256,
                 'seed': 0},

    'lr': {'kernel':0.01, 
           'optional_param':0.01, 
           'noise':0.01},
    # kernel number as dim + 1
    'kernel': {
            'K1': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
            'K2': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
            'K3': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
              },
    'evaluate_method': ['mae', 'rmse', 'r2', 'gaussian_loss'],
    'optimizer': 'adam',
    'exp_restrict': False,
    'input_normalzie': True,
    'output_normalize': True,
    'noise_init' : 100.,
    'grid_config': {'grid_size': [-1, -1], 
                    'type': 'fixed', # learnable, fixed
                    'dimension_map': 'identity', # latent space: identity, learnable_identity, learnable_map
                    },
}

def _last_dim_to_fist(_tensor):
    _dim = [i for i in range(_tensor.ndim)]
    _dim.insert(0, _dim.pop())
    return _tensor.permute(*_dim)

def _first_dim_to_last(_tensor):
    _dim = [i+1 for i in range(_tensor.ndim-1)]
    _dim.append(0)
    return _tensor.permute(*_dim)


class HOGP_MODULE:
    # def __init__(self, grid_params_list, kernel_list, target_list, normalize=True, restrict_method= 'exp') -> None:
    def __init__(self, module_config) -> None:
        default_module_config.update(module_config)
        self.module_config = deepcopy(default_module_config)
        module_config = deepcopy(default_module_config)

        # param check
        assert module_config['optimizer'] in ['adam'], 'now optimizer only support adam, but get {}'.format(module_config['optimizer'])

        # load_data
        self._load_data(module_config['dataset'])
        self._grid_setup(module_config['grid_config'])

        # TODO if param allow more than single kernel, optimize code here
        self.vector_dims = len(self.grid)
        self.param_dims = 1

        # X - normalize
        if module_config['input_normalzie'] is True:
            self.X_normalizer = Normalizer(self.inputs_tr[0])
            self.inputs_tr[0] = self.X_normalizer.normalize(self.inputs_tr[0])
        else:
            self.X_normalizer = None

        # Y - normalize
        # TODO normalize according to dims
        if module_config['output_normalize'] is True:
            self.Y_normalizer = Normalizer(self.outputs_tr[0], dim=[i for i in range(len(self.outputs_tr[0].shape))])
            self.outputs_tr[0] = self.Y_normalizer.normalize(self.outputs_tr[0])
        else:
            self.Y_normalizer = None


        # init kernel
        self._init_kernel(module_config['kernel'])

        # init noise
        if module_config['exp_restrict'] is True:
            self.noise = torch.nn.Parameter(torch.log(torch.tensor(module_config['noise_init'])))
        else:
            self.noise = torch.nn.Parameter(torch.tensor(module_config['noise_init']))

        # init optimizer
        self._optimizer_setup()


    def _load_data(self, dataset_config):
        print('dataset_config name:', dataset_config['name'])
        if dataset_config['name'] in mat_dataset_paths:
            # they got the same data format
            data = loadmat(mat_dataset_paths[dataset_config['name']])

            if dataset_config['type'] == 'x_2_y':
                assert len(dataset_config['fidelity']) == 1, 'for x_2_y, fidelity length must be 1'
                _fidelity = fidelity_map[dataset_config['fidelity'][0]]
                # 0 for low, 1 for middle, 2 for high
                x_tr = torch.tensor(data['xtr'], dtype=torch.float32)
                y_tr = torch.tensor(data['Ytr'][0][_fidelity], dtype=torch.float32)
                x_eval = torch.tensor(data['xte'], dtype=torch.float32)
                y_eval = torch.tensor(data['Yte'][0][_fidelity], dtype=torch.float32)
                # shuffle
                if self.module_config['dataset']['seed'] is not None:
                    x_tr, y_tr = self._random_shuffle([[x_tr, 0], [y_tr, 0]])

                # gen vector, put num to the last dim
                _index = dataset_config['train_start_index']
                self.inputs_tr = []
                self.inputs_tr.append(torch.tensor(x_tr[_index:_index+dataset_config['train_sample'], ...]))
                self.outputs_tr = []
                self.outputs_tr.append(torch.tensor(y_tr[_index:_index+dataset_config['train_sample'], ...]))
                self.outputs_tr[-1] = _first_dim_to_last(self.outputs_tr[-1])

                _index = dataset_config['eval_start_index']
                self.inputs_eval = []
                self.inputs_eval.append(torch.tensor(x_eval[_index:_index+dataset_config['eval_sample'], ...]))
                self.outputs_eval = []
                self.outputs_eval.append(torch.tensor(y_eval[_index:_index+dataset_config['eval_sample'], ...]))
                self.outputs_eval[-1] = _first_dim_to_last(self.outputs_eval[-1])
        else:
            assert False
        

    def _grid_setup(self, grid_config):
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


    def _init_kernel(self, kernel_config):
        from kernel.kernel_generator import kernel_generator
        self.kernel_list = []
        for key, value in kernel_config.items():
            for _kernel_type, _kernel_params in value.items():
                # broadcast exp_restrict
                if not hasattr(_kernel_params, 'exp_restrict'):
                    _kernel_params['exp_restrict'] = self.module_config['exp_restrict']
                self.kernel_list.append(kernel_generator(_kernel_type, _kernel_params))


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
                                           {'params': kernel_learnable_param , 'lr': self.module_config['lr']['kernel']}]) # 改了lr从0.01 改成0.0001

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
        self.K.append(self.kernel_list[_index](self.inputs_tr[0], self.inputs_tr[0]))
        self.K_eigen.append(eigen_pairs(self.K[-1]))

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
        _init_value = torch.tensor([1.0]).reshape(*[1 for i in self.K])
        lambda_list = [eigen.value.reshape(-1, 1) for eigen in self.K_eigen]
        A = tucker_to_tensor((_init_value, lambda_list))

        if self.module_config['exp_restrict'] is True:
            _noise = self.noise.exp()
        else:
            _noise = self.noise
        # A = A + _noise * tensorly.ones(A.shape)
        A = A + _noise.pow(-1)* tensorly.ones(A.shape)
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
            if self.module_config['input_normalzie'] is True:
                input_param[0] = self.X_normalizer.normalize(input_param[0])
            
            #! may needn't?
            # self.update_product()

            # /*** Get predict mean***/
            if len(input_param[0].shape) != len(self.inputs_tr[0].shape):
                input_param[0] = input_param[0].reshape(1, *input_param[0].shape)
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
            '''
            '''
            # NOTE: now only work for the normal predict
            _init_value = torch.tensor([1.0]).reshape(*[1 for i in self.K])
            diag_K = tucker_to_tensor(( _init_value, [K.diag().reshape(-1,1) for K in self.K[:-1]]))

            # var则为0, 怎么利用该信息更新输入？
            S = self.A * self.A.pow(-1/2)
            S_2 = S.pow(2)
            # S_product = tensorly.tenalg.multi_mode_dot(S_2, [eigen_vector_d1.pow(2), eigen_vector_d2.pow(2), (K_star@K_p.inverse()@eigen_vector_p).pow(2)])
            S_product = tensorly.tenalg.multi_mode_dot(S_2, [self.K_eigen[i].vector.pow(2) for i in range(len(self.K_eigen)-1)]+[(K_star@self.K[-1].inverse()@self.K_eigen[-1].vector).pow(2)])
            M = diag_K + S_product
            # if self.module_config['output_normalize'] is True:
            #     M = M * self.Y_normalizer.std ** 2

        return predict_u, M

    '''
    def predict_postprior(self, input_param, target_y, target_mask, method):
        # method 0 - 
        # method 1 - direct gp

        if len(input_param.shape) != len(self.grid_params_list[-1].shape):
                input_param = input_param.reshape(1, *input_param.shape)
        
        if method == '0':
            # get predict_u for init
            predict_u, _ = self.predict(input_param)

            # method 0
            new_param = torch.cat([self.grid_params_list[-1],input_param], 0)
            new_k = [*self.K[:-1], self.kernel_list[-1](new_param, new_param)]
            pod = posterior_output_decorator(new_k, predict_u, target_y, target_mask, self.target_list)
            pod.noise = deepcopy(self.noise.data)
            pod_train_number = 100
            for i in range(pod_train_number):
                pod.train()
                print("posterior_output_decorator finish {}/{}".format(i+1, pod_train_number), end='\r')
            output = pod.eval()

        # method 1
        # /*** not implement yet***/
        # y_known = torch.stack(self.target_list, dim=1)
        # predict_direct_gp = K_star@(torch.inverse(self.K[0])@y_known).T

        return output
    '''

    '''
    def train_l2_loss(self, params, target):
        # no used anymore
        self.update_product()
        mean, var = self.predict(params)

        loss = (mean.reshape(target.shape)-target).pow(2).sum()/mean.size()[0]
        loss *= -1
        loss.backward()
        self.optimizer.step()
    '''

    def train(self):
        self.update_product()
        #! loss = -1/2* torch.log(abs(self.A)).mean()
        nd = torch.prod(torch.tensor([value for value in self.A.shape]))
        loss = -1/2* nd * torch.log(torch.tensor(2 * math.pi))
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


    def _random_shuffle(self, np_array_list):
        random.seed(self.module_config['dataset']['seed'])
        # check dim shape
        # TODO support -1 dim
        dim_lenth = []
        for _np_array in np_array_list:
            dim = _np_array[1]
            if dim < 0:
                dim = len(_np_array[0].shape) + dim
            dim_lenth.append(_np_array[0].shape[dim])

        assert len(set(dim_lenth)) == 1, "length of dim is not the same"

        shuffle_index = [i for i in range(dim_lenth[0])]
        random.shuffle(shuffle_index)

        output_array = []
        for _np_array in np_array_list:
            dim = _np_array[1]
            np_array = deepcopy(_np_array[0])
            if dim < 0:
                dim = len(_np_array[0].shape) + dim

            if dim==0:
                output_array.append(np_array[shuffle_index])
            else:
                switch_dim = [i for i in range(len(np_array.shape))]
                switch_dim[switch_dim.index(dim)] = 0
                switch_dim[0] = dim
                np_array = np.ascontiguousarray(np.transpose(np_array, switch_dim))
                np_array = np_array[shuffle_index]
                np_array = np.ascontiguousarray(np.transpose(np_array, switch_dim))
                output_array.append(np_array)
        return output_array


    def get_params_need_check(self):
        params_need_check = []
        for i in range(len(self.kernel_list)):
            params_need_check.extend(self.kernel_list[i].get_params_need_check())
        params_need_check.append(self.noise)
        
        if self.module_config['grid_config']['type'] == 'learnable':
            params_need_check.extend(self.grid)

        if self.module_config['grid_config']['dimension_map'] not in ['identity']:
            params_need_check.extend(self.mapping_vector)

        return params_need_check


    def save_state(self):
        state_dict = []
        for i, _kernel in enumerate(self.kernel_list):
            state_dict.extend(_kernel.get_param([]))

        state_dict.append(self.noise)

        if self.module_config['grid_config']['type'] == 'learnable':
            state_dict.extend(self.grid)

        if self.module_config['grid_config']['dimension_map'] not in ['identity']:
            state_dict.extend(self.mapping_vector)
        
        return state_dict

        # TODO save word

    def load_state(self, params_list):
        index = 0
        for i, _kernel in enumerate(self.kernel_list):
            _temp_list = _kernel.get_param([])
            _kernel.set_param(params_list[index: index + len(_temp_list)])
            index += len(_temp_list)
        
        with torch.no_grad():
            self.noise.copy_(params_list[index])
            index += 1

            if self.module_config['grid_config']['type'] == 'learnable':
                for i in range(len(self.grid)):
                    self.grid[i].copy_(params_list[index + i])
                index += len(self.grid)

            if self.module_config['grid_config']['dimension_map'] not in ['identity']:
                for i in range(len(self.mapping_vector)):
                    self.mapping_vector[i].copy_(params_list[index + i])
                index += len(self.mapping_vector)

    def eval(self):
        print('---> start eval')
        predict_y, predict_var = self.predict(self.inputs_eval)
        self.predict_y = deepcopy(predict_y)
        # result = performance_evaluator(predict_y, self.outputs_eval[0], self.module_config['evaluate_method'])
        predict_y = _last_dim_to_fist(predict_y)
        predict_var = _last_dim_to_fist(predict_var)
        target = _last_dim_to_fist(self.outputs_eval[0])
        result = high_level_evaluator([predict_y, predict_var], target, self.module_config['evaluate_method'])
        print(result)
        return result