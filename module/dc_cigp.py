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

    'lr': {'kernel':0.01, 
           'optional_param':0.01, 
           'noise':0.01},
    'weight_decay': 1e-3,

    'kernel': {
            'K1': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
              },
    'evaluate_method': ['mae', 'rmse', 'r2'],
    'optimizer': 'adam',
    'exp_restrict': False,
    'input_normalize': True,
    'output_normalize': True,
    'noise_init' : 1.,
    'pca': {'type': 'listPCA', 
            'r': 0.99, }, # listPCA, resPCA_mf,
    'cuda': False,
}

pca_map = {'listPCA': listPCA, 'resPCA_mf': resPCA_mf}


class DC_CIGP_MODULE(torch.nn.Module):
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

        # X - normalize
        if module_config['input_normalize'] is True:
            # self.X_normalizer_0 = Normalizer(self.inputs_tr[0])
            self.X_normalizer_0 = Normalizer(self.inputs_tr[0],  dim=[i for i in range(len(self.inputs_tr[0].shape))])
            self.inputs_tr[0] = self.X_normalizer_0.normalize(self.inputs_tr[0])

            # self.X_normalizer_1 = Normalizer(self.inputs_tr[1])
            self.X_normalizer_1 = Normalizer(self.inputs_tr[0],  dim=[i for i in range(len(self.inputs_tr[0].shape))])
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
        kernel_utils.register_kernel(self, module_config['kernel'])

        # init noise
        if module_config['exp_restrict'] is True:
            self.noise = torch.nn.Parameter(torch.log(torch.tensor(module_config['noise_init'])))
        else:
            self.noise = torch.nn.Parameter(torch.tensor(module_config['noise_init']))

        # init optimizer
        self._optimizer_setup()

    def _optimizer_setup(self):
        optional_params = []

        kernel_learnable_param = []
        for _kernel in self.kernel_list:
            _kernel.get_param(kernel_learnable_param)

        # TODO support SGD?
        # module_config['lr'] = {'kernel':0.01, 'optional_param':0.01, 'noise':0.01}
        self.optimizer = torch.optim.Adam([{'params': optional_params, 'lr': self.module_config['lr']['optional_param']}, 
                                           {'params': [self.noise], 'lr': self.module_config['lr']['noise']},
                                           {'params': kernel_learnable_param , 'lr': self.module_config['lr']['kernel']}],
                                           weight_decay = self.module_config['weight_decay']) # 改了lr从0.01 改成0.0001
        
    def negative_log_likelihood(self):
        # inputs / outputs
        # x: [num, vector_dims]
        # y: [num, vector_dims]

        Sigma = self.kernel_list[0](self.inputs_tr[0], self.inputs_tr[0]) + JITTER * torch.eye(self.inputs_tr[0].size(0), device=list(self.parameters())[0].device)
        if self.module_config['exp_restrict'] is True:
            _noise = self.noise.exp()
        else:
            _noise = self.noise
        Sigma = Sigma + _noise.pow(-1) * torch.eye(self.inputs_tr[0].size(0), device=list(self.parameters())[0].device)

        L = torch.linalg.cholesky(Sigma)
        #option 1 (use this if torch supports)
        # Gamma,_ = torch.triangular_solve(self.Y, L, upper = False)
        #option 2

        gamma = L.inverse() @ self.outputs_tr[0]       # we can use this as an alternative because L is a lower triangular matrix.

        y_num, y_dimension = self.outputs_tr[0].shape
        nll =  0.5 * (gamma ** 2).sum() +  L.diag().log().sum() * y_dimension  \
            + 0.5 * y_num * torch.log(2 * torch.tensor(PI, device=list(self.parameters())[0].device)) * y_dimension

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

            Sigma = self.kernel_list[0](self.inputs_tr[0], self.inputs_tr[0]) + JITTER * torch.eye(self.inputs_tr[0].size(0), device=list(self.parameters())[0].device)
            if self.module_config['exp_restrict'] is True:
                _noise = self.noise.exp()
            else:
                _noise = self.noise
            Sigma = Sigma + _noise.pow(-1) * torch.eye(self.inputs_tr[0].size(0), device=list(self.parameters())[0].device)

            kx = self.kernel_list[0](self.inputs_tr[0], input_param[0])
            L = torch.linalg.cholesky(Sigma)
            LinvKx,_ = torch.triangular_solve(kx, L, upper = False)

            var_diag = self.kernel_list[0](input_param[0], input_param[0]).diag().view(-1, 1) - (LinvKx**2).sum(dim = 0).view(-1, 1)
            if self.module_config['exp_restrict'] is True:
                var_diag = var_diag + self.noise.exp().pow(-1)
            else:
                var_diag = var_diag + self.noise.pow(-1)
            
            u = kx.t() @ torch.cholesky_solve(self.outputs_tr[0], L)

            if self.pca_model is not None:
                var_diag = var_diag.expand_as(u)
                var_diag = torch.sqrt(var_diag)
                var_diag = self.pca_model.recover([_temp_record[0],var_diag])[1]
                var_diag = torch.pow(var_diag, 2)
                u = self.pca_model.recover([_temp_record[0],u])[1]

            if self.module_config['output_normalize'] is True:
                u = self.Y_normalizer.denormalize(u)
                var_diag = var_diag * self.Y_normalizer.std**2

            return u, var_diag


    def train(self):
        self.optimizer.zero_grad()
        loss = self.negative_log_likelihood()
        loss.backward()
        self.optimizer.step()
        # print('loss_nll:', loss.item())


    def eval(self):
        print('---> start eval')
        predict_y, predict_var = self.predict(self.inputs_eval)
        result = high_level_evaluator([predict_y, predict_var], self.outputs_eval[0], self.module_config['evaluate_method'], sample_last_dim=False)
        self.predict_y = predict_y
        self.predict_var = predict_var
        print(result)
        return result


if __name__ == '__main__':
    module_config = {}
    cigp = DC_CIGP_MODULE(module_config)
