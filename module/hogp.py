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


JITTER = 1e-6
EPS = 1e-10
PI = 3.1415

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
    # kernel number as dim + 1
    'kernel': {
            'K1': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
            # 'K2': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
            # 'K3': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
            # 'K4': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
              },
    'auto_broadcast_kernel': True,
    'evaluate_method': ['mae', 'rmse', 'r2', 'gaussian_loss'],
    'optimizer': 'adam',
    'exp_restrict': False,
    'input_normalzie': True,
    'output_normalize': True,
    'noise_init' : 0.00001,
    'grid_config': {'grid_size': [-1], 
                    'type': 'fixed', # learnable, fixed
                    'dimension_map': 'identity', # latent space: identity, learnable_identity, learnable_map
                    'auto_broadcast_grid_size': True,
                    'squeeze_to_01': False,
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
        # default_module_config.update(module_config)
        _final_config = smart_update(default_module_config, module_config)
        self.module_config = deepcopy(_final_config)
        module_config = deepcopy(_final_config)

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
        loaded = False
        for _loader in [SP_DataLoader, Standard_mat_DataLoader]:
            if dataset_config['name'] in _loader.dataset_available:
                self.data_loader = _loader(dataset_config['name'], dataset_config['interp_data'])
                _data = self.data_loader.get_data()
                loaded = True
                break
        if loaded is False:
            assert False, 'dataset {} not found in all loader'.format(dataset_config['name'])

        dp = Data_preprocess(dataset_config)
        self.inputs_tr, self.outputs_tr, self.inputs_eval, self.outputs_eval = dp.do_preprocess(_data, numpy_to_tensor=True)

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


    def _init_kernel(self, kernel_config):
        from kernel.kernel_generator import kernel_generator
        self.kernel_list = []
        if len(kernel_config) == 1 and self.module_config['auto_broadcast_kernel'] is True:
            print('auto broadcast kernel')
            kernel_need = len(self.inputs_tr[0].shape) - 1 +\
                          len(self.outputs_tr[0].shape) - 1
            for key, value in kernel_config.items():
                for _kernel_type, _kernel_params in value.items():
                    # get base _kernel_type, _kernel_params
                    pass
            for _ in range(kernel_need):
            # broadcast exp_restrict
                if not hasattr(_kernel_params, 'exp_restrict'):
                    _kernel_params['exp_restrict'] = self.module_config['exp_restrict']
                self.kernel_list.append(kernel_generator(_kernel_type, _kernel_params))
        else:
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