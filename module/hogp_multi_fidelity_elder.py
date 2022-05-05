from copy import deepcopy
# import gpytorch
import math
import torch
import tensorly
import numpy as np
import os
import sys

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('Hogp')])
sys.path.append(realpath)


from tensorly.decomposition import tucker
from tensorly import tucker_to_tensor
from utils.eigen import eigen_pairs
from utils.normalizer import Normalizer
from module.gp_output_decorator import posterior_output_decorator
from utils.performance_evaluator import performance_evaluator

from kernel.Multi_fidelity_connection import rho_connection, mapping_connection
# optimize for main_controller

JITTER = 1e-6
EPS = 1e-10
PI = 3.1415

tensorly.set_backend('pytorch')
import random

class HOGP_MODULE:
    # def __init__(self, grid_params_list, kernel_list, target_list, normalize=True, restrict_method= 'exp') -> None:
    def __init__(self, module_config) -> None:
        # module_config = {}
        module_config['lr'] = {'kernel':0.01, 'optional_param':0.01, 'noise':0.01}
        module_config['kernel'] = {
            'K1': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale':1.}},
            'K2': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale':1.}},
            'K3': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale':1.}},
        }
        # module_config['dataset'] = {'name': 'burger_v4_02', 'train_start_index': 0, 'train':512, 'eval_start_index': 512, 'eval':128} #
        module_config['evaluate_method'] = ['mae', 'rmse', 'r2'] #-
        module_config['optimizer'] = 'adam' 
        module_config['exp_restrict'] = False
        module_config['normal_input'] = True
        module_config['normal_output'] = True
        module_config['changeable_vector'] = False
        module_config['mapping'] = False
        module_config['noise_init'] = 0.0005
        module_config['target_type'] = {'type_name': 'res_mapping'}
        # assert type_name in ['res_rho', 'res_mapping']

        self.module_config = deepcopy(module_config)

        # param check
        assert module_config['optimizer'] in ['adam'], 'now optimizer only support adam, but get {}'.format(module_config['optimizer'])

        # load_data
        self._load_data(module_config['dataset'])
        self._select_connection_kernel(module_config['target_type']['type_name'])

        # TODO if param allow more than single kernel, optimize code here
        self.vector_dims = len(self.grid_params_list)-1
        self.param_dims = 1

        # X - normalize
        if module_config['normal_input'] is True:
            self.X_normalizer = Normalizer(self.grid_params_list[-1])
            self.grid_params_list[-1] = self.X_normalizer.normalize(self.grid_params_list[-1])
        else:
            self.X_normalizer = None

        # Y - normalize
        # Y = [yl, yh]
        if module_config['normal_output'] is True:
            self.Y_normalizer = Normalizer(self.target_list[1],dim=[i for i in range(len(self.target_list[1].shape))])
            self.target_list[1] = self.Y_normalizer.normalize(self.target_list[1])
            self.target_list[0] = self.Y_normalizer.normalize(self.target_list[0])
        else:
            self.Y_normalizer = None

        # init kernel
        self._init_kernel(module_config['kernel'])

        # init noise
        if module_config['exp_restrict'] is True:
            self.noise = torch.nn.Parameter(torch.log(module_config['noise_init']))
        else:
            self.noise = torch.nn.Parameter(torch.tensor(module_config['noise_init']))

        # init optimizer
        self._optimizer_setup()

    def _select_connection_kernel(self, type_name):
        assert type_name in ['res_rho', 'res_mapping']
        if type_name == 'identity':
            self.target_connection = None
        elif type_name in ['res_rho']:
            self.target_connection = rho_connection()
        elif type_name in ['res_mapping']:
            self.target_connection = mapping_connection(self.target_list[0][:,:,0].shape, 
                                                        self.target_list[1][:,:,0].shape,
                                                        self.module_config['mapping'])

    def _load_data(self, dataset_config):
        # TODO optimize code
        if dataset_config['name'] == 'TinMeltingFront':
            x = np.load('./data/datasets/np_format/TinMeltingFront/in_0.npy')
            y = np.load('./data/datasets/np_format/TinMeltingFront/out_0.npy')
        elif dataset_config['name'] == 'poisson_v4_02':
            x = np.load('./data/datasets/np_format/poisson_v4_02/x.npz')
            y = np.load('./data/datasets/np_format/poisson_v4_02/y.npz')
            yl = y['yl'].transpose(1,2,0).astype(np.float32)
            yh = y['yh'].transpose(1,2,0).astype(np.float32)
            x = x['x'].astype(np.float32)
        elif dataset_config['name'] == 'burger_v4_02':
            x = np.load('./data/datasets/np_format/burger_v4_02/x.npz')
            y = np.load('./data/datasets/np_format/burger_v4_02/y.npz')
            yl = y['yl'].astype(np.float32)
            yh = y['yh'].astype(np.float32)
            x = x['x'].astype(np.float32)
        else:
            assert False

        x,yl,yh = self._random_shuffle([[x,0], [yl,-1], [yh,-1]])

        # gen vector grid
        self.grid_params_list = []
        for dim_value in yh.shape[:-1]:
            vector = torch.tensor(np.array(range(1,dim_value+1))).reshape(-1,1).float()
            self.grid_params_list.append(vector)

        if 'train_start_index' in dataset_config:
            _index = dataset_config['train_start_index']
        else:
            _index = 0

        self.grid_params_list.append(torch.tensor(x[_index:_index+dataset_config['train'], :]))
        self.target_list = [torch.tensor(yl[:,:,_index:_index+dataset_config['train']]), torch.tensor(yh[:,:,:dataset_config['train']])]

        if 'eval_start_index' in dataset_config:
            _index = dataset_config['eval_start_index']
        else:
            _index = dataset_config['train'] + _index

        self.eval_grid_params_list =  torch.tensor(deepcopy(x[_index: _index+ dataset_config['eval'], :]))
        # self.eval_target_list =  torch.tensor(deepcopy(y[:,:, _index: _index+ dataset_config['eval']]))
        self.eval_target_list = [torch.tensor(yl[:,:,_index: _index+ dataset_config['eval']]), 
                                 torch.tensor(yh[:,:,_index: _index+ dataset_config['eval']])]

    def _init_kernel(self, kernel_config):
        from kernel.kernel_generator import kernel_generator
        self.kernel_list = []
        for key, value in kernel_config.items():
            for _kernel_type, _kernel_params in value.items():
                # broadcast exp_restrict
                if not hasattr(_kernel_params, 'exp_restrict'):
                    _kernel_params['exp_restrict'] = self.module_config['exp_restrict']
                self.kernel_list.append(kernel_generator(_kernel_type, _kernel_params))


    def _result_evaluate(self, result, target):
        from utils.performance_evaluator import performance_evaluator
        return performance_evaluator(result, target, self.module_config['evaluate_method'])


    def _optimizer_setup(self):
        kernel_learnable_param = []
        if self.module_config['mapping'] is True:
            mapping_list = [True for i in range(self.vector_dims)] + [False for i in range(self.param_dims)]
        elif self.module_config['mapping'] is False:
            mapping_list = [False for i in range(self.vector_dims)] + [False for i in range(self.param_dims)]
        else:
            assert False, "not support yet, please check"
        self.mapping_list = mapping_list
        
        learnable_mapping = []
        self.mapping_grid = []
        for i,_mapping_sate in enumerate(self.mapping_list):
            if _mapping_sate is True:
                # _grid = torch.ones([self.grid_params_list[i].shape[0], self.target_list.shape[i]])
                # _grid = _grid/self.grid_params_list[i].shape[0]     # normalize
                _grid = torch.diag_embed(tensorly.ones([100]))
                _grid.requires_grad = True
                self.mapping_grid.append(_grid)
                # kernel_learnable_param.append(_grid)
                learnable_mapping.append(_grid)
            else:
                self.mapping_grid.append(None)

        kernel_learnable_param = []
        for _kernel in self.kernel_list:
            _kernel.get_param(kernel_learnable_param)
        if self.target_connection is not None:
            self.target_connection.get_param(kernel_learnable_param)

        # TODO support SGD?
        # module_config['lr'] = {'kernel':0.01, 'optional_param':0.01, 'noise':0.01}
        self.optimizer = torch.optim.Adam([{'params': learnable_mapping, 'lr': self.module_config['lr']['optional_param']}, 
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

        # update
        for i in range(len(self.grid_params_list)):
            if self.mapping_list[i] is False:
                self.K.append(self.kernel_list[i](self.grid_params_list[i], self.grid_params_list[i]))
            elif self.mapping_list[i] is True:
                _in = tensorly.tenalg.mode_dot(self.grid_params_list[i], self.mapping_grid[i], 0)
                self.K.append(self.kernel_list[i](_in, _in))
            self.K_eigen.append(eigen_pairs(self.K[i]))

    def compute_Kstar(self, new_param):
        # use for predict decorator
        all_params = torch.cat([self.grid_params_list[-1], new_param], 0)
        if self.K != []:
            self.Kstar = [*self.K[:-1], self.kernel_list[-1](all_params, all_params)]
        return self.Kstar

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

        if self.target_connection == None:
            _target_learnable = self.target_list
        else:
            _target_learnable = self.target_connection(*self.target_list)
        
        # vec(z).T@ S.inverse @ vec(z) = b.T @ b,  b = S.pow(-1/2) @ vec(z)
        T_1 = tensorly.tenalg.multi_mode_dot(_target_learnable, [eigen.vector.T for eigen in self.K_eigen])
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


    def predict(self, input_param, base_yl):
        with torch.no_grad():
            if self.module_config['normal_input'] is True:
                input_param = self.X_normalizer.normalize(input_param)
            #! may needn't?
            # self.update_product()

            # /*** Get predict mean***/
            if len(input_param.shape) != len(self.grid_params_list[-1].shape):
                input_param = input_param.reshape(1, *input_param.shape)
            K_star = self.kernel_list[-1](input_param, self.grid_params_list[-1])
            # 对于self.K[:-1], 反映了y上各点的关系, 将 已知的y带进去, 随后进行迭代（怎么迭代）？
            K_predict = self.K[:-1] + [K_star]

            '''
            predict_u = tensorly.tenalg.kronecker(K_predict)@self.g_vec #-> memory error
            so we use tensor.tenalg.multi_mode_dot instead
            '''
            predict_u = tensorly.tenalg.multi_mode_dot(self.g, K_predict)
            predict_u = predict_u.reshape_as(self.target_list[1][:,:,0])

            if self.module_config['normal_output'] is True:
                base_yl = self.Y_normalizer.normalize(base_yl)
                predict_u = self.target_connection.low_2_high(base_yl, predict_u)
                predict_u = self.Y_normalizer.denormalize(predict_u)
            else:
                predict_u = self.target_connection.low_2_high(base_yl, predict_u)

            # /*** Get predict var***/
            # NOTE: now only work for the normal predict
            _init_value = torch.tensor([1.0]).reshape(*[1 for i in self.K])
            diag_K = tucker_to_tensor(( _init_value, [K.diag().reshape(-1,1) for K in self.K[:-1]]))

            # var则为0, 怎么利用该信息更新输入？
            S = self.A * self.A.pow(-1/2)
            S_2 = S.pow(2)
            # S_product = tensorly.tenalg.multi_mode_dot(S_2, [eigen_vector_d1.pow(2), eigen_vector_d2.pow(2), (K_star@K_p.inverse()@eigen_vector_p).pow(2)])
            S_product = tensorly.tenalg.multi_mode_dot(S_2, [self.K_eigen[i].vector.pow(2) for i in range(len(self.K_eigen)-1)]+[(K_star@self.K[-1].inverse()@self.K_eigen[-1].vector).pow(2)])
            M = diag_K + S_product
            if self.module_config['normal_output'] is True:
                M = M * self.Y_normalizer.std ** 2

        return predict_u, M

    def _random_shuffle(self, np_array_list):
        random.seed(1024)
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
        # if self.restrict_method == 'clamp':
        #     self._pos_as_clamp()
        #     with torch.no_grad():
        #         self.noise.copy_(self.noise.clamp_(min=0))

        if 'GP_DEBUG' in os.environ and os.environ['GP_DEBUG'] == 'True':
            print('self.noise:{}'.format(self.noise.data))

    # def _pos_as_clamp(self):
    #     for i,_kernel in enumerate(self.kernel_list):
    #         _kernel.clamp_to_positive()

    def get_params_need_check(self):
        params_need_check = []
        for i in range(len(self.kernel_list)):
            params_need_check.extend(self.kernel_list[i].get_params_need_check())
        params_need_check.append(self.noise)

        params_need_check.extend(self.target_connection.get_params_need_check())
        
        if hasattr(self, 'mapping_param'):
            params_need_check.append(self.mapping_param)
        return params_need_check


    def save_state(self):
        state_dict = []
        for i, _kernel in enumerate(self.kernel_list):
            state_dict.extend(_kernel.get_param([]))

        state_dict.append(self.noise)

        state_dict.extend(self.target_connection.get_params_need_check())

        if self.module_config['mapping'] is True:
            state_dict.append(self.mapping_list)
        
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
            self.target_connection.set_param(params_list[index:])

            if self.module_config['mapping'] is True:
                self.mapping_list.copy_(params_list[-1])

    def eval(self):
        print('---> start eval')
        predict_y = []
        for i in range(self.module_config['dataset']['eval']):
            predict_y.append(self.predict(self.eval_grid_params_list[i:i+1,:], self.eval_target_list[0][:,:,i:i+1])[0])
            print('    finish {}/{}'.format(i+1, self.module_config['dataset']['eval']), end='\r')

        predict_y = torch.stack(predict_y,-1)
        if np.isnan(predict_y.data.cpu().numpy()).any():
            return {'fail':'reach NaN'}

        result = performance_evaluator(predict_y, self.eval_target_list[1], self.module_config['evaluate_method'])
        print(result)
        return result