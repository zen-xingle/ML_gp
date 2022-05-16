# import gpytorch
import math
import torch
import tensorly
import numpy as np
import os
import sys
import random

from copy import deepcopy
from scipy.io import loadmat

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

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
                
                 'inputs_format': ['x[0]', 'y[0]'],
                 'outputs_format': ['y[2]'],

                 'force_2d': True,
                 'x_sample_to_last_dim': False,
                 'y_sample_to_last_dim': True,
                 'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                 },

    'lr': {'kernel':0.1, 
           'optional_param':0.1, 
           'noise':0.1},

    'kernel': {
            'K1': {'SE': {'exp_restrict':False, 'length_scale':1., 'scale': 1.}},
              },
    'evaluate_method': ['mae', 'rmse', 'r2'],
    'optimizer': 'adam',
    'exp_restrict': False,
    'input_normalize': True,
    'output_normalize': True,
    'noise_init' : 100.,
    'pca': {'type': 'listPCA', 
            'r': 0.99, } # listPCA, resPCA_mf,
}

pca_map = {'listPCA': listPCA, 'resPCA_mf': resPCA_mf}


class DC_CIGP_MODULE:
    # def __init__(self, grid_params_list, kernel_list, target_list, normalize=True, restrict_method= 'exp') -> None:
    def __init__(self, module_config) -> None:
        _final_config = smart_update(default_module_config, module_config)
        self.module_config = deepcopy(_final_config)
        module_config = deepcopy(_final_config)

        # param check
        assert module_config['optimizer'] in ['adam'], 'now optimizer only support adam, but get {}'.format(module_config['optimizer'])

        # load_data
        self._load_data(module_config['dataset'])

        # X - normalize
        if module_config['input_normalize'] is True:
            self.X_normalizer_0 = Normalizer(self.inputs_tr[0])
            self.inputs_tr[0] = self.X_normalizer_0.normalize(self.inputs_tr[0])
            self.X_normalizer_1 = Normalizer(self.inputs_tr[1])
            self.inputs_tr[1] = self.X_normalizer_1.normalize(self.inputs_tr[1])
        else:
            self.X_normalizer = None

        # Y - normalize
        if module_config['output_normalize'] is True:
            self.Y_normalizer = Normalizer(self.outputs_tr[0])
            self.outputs_tr[0] = self.Y_normalizer.normalize(self.outputs_tr[0])
        else:
            self.Y_normalizer = None

        # PCA for y
        if module_config['pca'] is not None:
            if module_config['pca']['type'] in pca_map:
                self.pca_model = pca_map[module_config['pca']['type']]([self.inputs_tr[1], self.outputs_tr[0]], module_config['pca']['r'])
                _temp_list = self.pca_model.project([self.inputs_tr[1], self.outputs_tr[0]])
                self.inputs_tr = [torch.cat([self.inputs_tr[0],_temp_list[0]], dim=1)]
                self.outputs_tr = [_temp_list[1]]
            else:
                assert False
        else:
            self.pca_model = None
            assert False, "DC need pca"

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


    def _select_connection_kernel(self, type_name):
        from kernel.Multi_fidelity_connection import rho_connection, mapping_connection
        assert type_name in ['res_standard', 'res_rho', 'res_mapping']
        if type_name == 'res_standard':
            self.target_connection = rho_connection(rho=1., trainable=False)
        elif type_name in ['res_rho']:
            self.target_connection = rho_connection(rho=1., trainable=True)
        elif type_name in ['res_mapping']:
            self.target_connection = mapping_connection(self.target_list[0][:,:,0].shape, 
                                                        self.target_list[1][:,:,0].shape,
                                                        self.module_config['mapping'])


    def _init_kernel(self, kernel_config):
        from kernel.kernel_generator import kernel_generator
        self.kernel_list = []
        for key, value in kernel_config.items():
            for _kernel_type, _kernel_params in value.items():
                # broadcast exp_restrict
                if 'exp_restrict' not in _kernel_params:
                    _kernel_params['exp_restrict'] = self.module_config['exp_restrict']
                self.kernel_list.append(kernel_generator(_kernel_type, _kernel_params))

    def get_params_need_check(self):
        params_need_check = []
        for i in range(len(self.kernel_list)):
            params_need_check.extend(self.kernel_list[i].get_params_need_check())
        params_need_check.append(self.noise)
        
        if hasattr(self, 'mapping_param'):
            params_need_check.append(self.mapping_param)
        return params_need_check


    def _optimizer_setup(self):
        optional_params = []

        kernel_learnable_param = []
        for _kernel in self.kernel_list:
            _kernel.get_param(kernel_learnable_param)

        # TODO support SGD?
        # module_config['lr'] = {'kernel':0.01, 'optional_param':0.01, 'noise':0.01}
        self.optimizer = torch.optim.Adam([{'params': optional_params, 'lr': self.module_config['lr']['optional_param']}, 
                                           {'params': [self.noise], 'lr': self.module_config['lr']['noise']},
                                           {'params': kernel_learnable_param , 'lr': self.module_config['lr']['kernel']}]) # 改了lr从0.01 改成0.0001
        
    def negative_log_likelihood(self):
        # inputs / outputs
        # x: [num, vector_dims]
        # y: [num, vector_dims]

        Sigma = self.kernel_list[0](self.inputs_tr[0], self.inputs_tr[0]) + JITTER * torch.eye(self.inputs_tr[0].size(0))
        if self.module_config['exp_restrict'] is True:
            _noise = self.noise.exp()
        else:
            _noise = self.noise
        Sigma = Sigma + _noise.pow(-1) * torch.eye(self.inputs_tr[0].size(0))

        L = torch.linalg.cholesky(Sigma)
        #option 1 (use this if torch supports)
        # Gamma,_ = torch.triangular_solve(self.Y, L, upper = False)
        #option 2

        gamma = L.inverse() @ self.outputs_tr[0]       # we can use this as an alternative because L is a lower triangular matrix.

        y_num, y_dimension = self.outputs_tr[0].shape
        nll =  0.5 * (gamma ** 2).sum() +  L.diag().log().sum() * y_dimension  \
            + 0.5 * y_num * torch.log(2 * torch.tensor(PI)) * y_dimension

        return nll


    def predict(self, input_param):
        # avoid changing the original input
        input_param = deepcopy(input_param)
    
        with torch.no_grad():
            if self.module_config['input_normalize'] is True:
                input_param[0] = self.X_normalizer_0.normalize(input_param[0])
                input_param[1] = self.X_normalizer_1.normalize(input_param[1])
            if self.pca_model is not None:
                _temp_record = self.pca_model.project([input_param[1], self.outputs_eval[0]])
                input_param[1] = _temp_record[0]
                # input_param[1] = self.pca_model.project([input_param[1], self.outputs_eval[0]])[0]
            input_param = [torch.cat(input_param, dim=1)]

            Sigma = self.kernel_list[0](self.inputs_tr[0], self.inputs_tr[0]) + JITTER * torch.eye(self.inputs_tr[0].size(0))
            if self.module_config['exp_restrict'] is True:
                _noise = self.noise.exp()
            else:
                _noise = self.noise
            Sigma = Sigma + _noise.pow(-1) * torch.eye(self.inputs_tr[0].size(0))

            kx = self.kernel_list[0](self.inputs_tr[0], input_param[0])
            L = torch.cholesky(Sigma)
            # LinvKx,_ = torch.triangular_solve(kx, L, upper = False)

            u = kx.t() @ torch.cholesky_solve(self.outputs_tr[0], L)

            if self.pca_model is not None:
                u = self.pca_model.recover([_temp_record[0],u])[1]

            if self.module_config['output_normalize'] is True:
                u = self.Y_normalizer.denormalize(u)
            # TODO var
            # if self.module_config['exp_restrict'] is True:
            #     var_diag = self.log_scale.exp() - (LinvKx**2).sum(dim = 0).view(-1,1)
            return u, None


    def train(self):
        self.optimizer.zero_grad()
        loss = self.negative_log_likelihood()
        loss.backward()
        self.optimizer.step()
        # print('loss_nll:', loss.item())


    def save_state(self):
        state_dict = []
        for i, _kernel in enumerate(self.kernel_list):
            state_dict.extend(_kernel.get_param([]))

        state_dict.append(self.noise)
        return state_dict


    def load_state(self, params_list):
        index = 0
        for i, _kernel in enumerate(self.kernel_list):
            _temp_list = _kernel.get_param([])
            _kernel.set_param(params_list[index: index + len(_temp_list)])
            index += len(_temp_list)
        
        with torch.no_grad():
            self.noise.copy_(params_list[index])
            index += 1

    def eval(self):
        print('---> start eval')
        predict_y, _var = self.predict(self.inputs_eval)
        result = performance_evaluator(predict_y, self.outputs_eval[0], self.module_config['evaluate_method'], sample_last_dim=False)
        self.predict_y = predict_y
        print(result)
        return result


if __name__ == '__main__':
    module_config = {}
    cigp = DC_CIGP_MODULE(module_config)
