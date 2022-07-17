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
from kernel import kernel_utils
from utils.data_utils import data_register

tensorly.set_backend('pytorch')


default_module_config = {
    'dataset' : {'name': 'poisson_v4_02',
                'interp_data': True,
                
                # preprocess
                'seed': 0,
                'train_start_index': 0,
                'train_sample': 8, 
                'eval_start_index': 0, 
                'eval_sample':256,
                
                'inputs_format': ['x[0]','y[0]'],
                'outputs_format': ['y[2]'],

                'force_2d': True,
                'x_sample_to_last_dim': False,
                'y_sample_to_last_dim': False,
                'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                },

    'lr': {'kernel':0.01, 
           'optional_param':0.01, 
           'noise':0.01},

    'kernel': {
            'K1': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
              },
    'evaluate_method': ['mae', 'rmse', 'r2'],
    'optimizer': 'adam',
    'exp_restrict': False,
    'input_normalize': True,
    'output_normalize': True,
    'noise_init' : 1.,
    'res_cigp': {'type_name': 'res_standard'}, # only available when x_yl_2_yh
    'cuda': False,
}


class CIGP_MODULE_Multi_Fidelity(torch.nn.Module):
    # def __init__(self, grid_params_list, kernel_list, target_list, normalize=True, restrict_method= 'exp') -> None:
    def __init__(self, module_config) -> None:
        super().__init__()
        _final_config = smart_update(default_module_config, module_config)
        self.module_config = deepcopy(_final_config)
        module_config = deepcopy(_final_config)

        # param check
        assert module_config['optimizer'] in ['adam'], 'now optimizer only support adam, but get {}'.format(module_config['optimizer'])

        # load_data
        data_register.data_regist(self, module_config['dataset'], module_config['cuda'])
        self._select_connection_kernel(module_config['res_cigp']['type_name'])

        # X - normalize
        if module_config['input_normalize'] is True:
            self.X_normalizer = Normalizer(self.inputs_tr[0])
            self.inputs_tr[0] = self.X_normalizer.normalize(self.inputs_tr[0])
        else:
            self.X_normalizer = None

        # Y - normalize
        if module_config['output_normalize'] is True:
            self.Y_normalizer = Normalizer(self.outputs_tr[0])
            self.outputs_tr[0] = self.Y_normalizer.normalize(self.outputs_tr[0])
            if self.module_config['res_cigp'] is not None:
                self.inputs_tr[1] = self.Y_normalizer.normalize(self.inputs_tr[1])
        else:
            self.Y_normalizer = None

        # init kernel
        kernel_utils.register_kernel(self, module_config['kernel'])

        # init noise
        if module_config['exp_restrict'] is True:
            self.noise = torch.nn.Parameter(torch.log(torch.tensor(module_config['noise_init'])))
        else:
            self.noise = torch.nn.Parameter(torch.tensor(module_config['noise_init']))

        # init optimizer
        self._optimizer_setup()


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


    def _optimizer_setup(self):
        optional_params = []
        if self.module_config['res_cigp'] is not None:
            optional_params = self.target_connection.get_param(optional_params)

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

        Sigma = self.kernel_list[0](self.inputs_tr[0], self.inputs_tr[0]) + JITTER * torch.eye(self.inputs_tr[0].size(0),  device=list(self.parameters())[0].device)
        if self.module_config['exp_restrict'] is True:
            _noise = self.noise.exp()
        else:
            _noise = self.noise
        Sigma = Sigma + _noise.pow(-1) * torch.eye(self.inputs_tr[0].size(0),  device=list(self.parameters())[0].device)

        L = torch.linalg.cholesky(Sigma)
        #option 1 (use this if torch supports)
        # Gamma,_ = torch.triangular_solve(self.Y, L, upper = False)
        #option 2
        if self.module_config['res_cigp'] is not None:
            gamma = L.inverse() @ self.target_connection(self.inputs_tr[1], self.outputs_tr[0])
        else:
            gamma = L.inverse() @ self.outputs_tr[0]       # we can use this as an alternative because L is a lower triangular matrix.

        y_num, y_dimension = self.outputs_tr[0].shape
        nll =  0.5 * (gamma ** 2).sum() +  L.diag().log().sum() * y_dimension  \
            + 0.5 * y_num * torch.log(2 * torch.tensor(PI,  device=list(self.parameters())[0].device)) * y_dimension

        return nll


    def predict(self, input_param):
        # avoid changing the original input
        input_param = deepcopy(input_param)
    
        with torch.no_grad():
            if self.module_config['input_normalize'] is True:
                input_param[0] = self.X_normalizer.normalize(input_param[0])

            Sigma = self.kernel_list[0](self.inputs_tr[0], self.inputs_tr[0]) + JITTER * torch.eye(self.inputs_tr[0].size(0),  device=list(self.parameters())[0].device)
            if self.module_config['exp_restrict'] is True:
                _noise = self.noise.exp()
            else:
                _noise = self.noise
            Sigma = Sigma + _noise.pow(-1) * torch.eye(self.inputs_tr[0].size(0),  device=list(self.parameters())[0].device)

            kx = self.kernel_list[0](self.inputs_tr[0], input_param[0])
            L = torch.cholesky(Sigma)
            # LinvKx,_ = torch.triangular_solve(kx, L, upper = False)

            if self.module_config['res_cigp'] is not None:
                u = kx.t() @ torch.cholesky_solve(self.target_connection(self.inputs_tr[1], self.outputs_tr[0]), L)
                if self.module_config['output_normalize'] is True:
                    input_param[1] = self.Y_normalizer.normalize(input_param[1])
                    u = self.target_connection.low_2_high(input_param[1], u)
                else:
                    u = self.target_connection.low_2_high(input_param[1], u)
            else:
                u = kx.t() @ torch.cholesky_solve(self.outputs_tr[0], L)

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


    def eval(self):
        print('---> start eval')
        predict_y, _var = self.predict(self.inputs_eval)
        result = performance_evaluator(predict_y, self.outputs_eval[0], self.module_config['evaluate_method'], sample_last_dim=False)
        self.predict_y = predict_y
        print(result)
        return result


if __name__ == '__main__':
    module_config = {}
    cigp = CIGP_MODULE_Multi_Fidelity(module_config)
    for i in range(1000):
        cigp.train()
