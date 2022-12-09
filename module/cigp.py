# import gpytorch
import math
import torch
import tensorly
import numpy as np
import os
import sys
import random

from copy import deepcopy

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from utils import *
from kernel import kernel_utils
from utils.data_utils import data_register
from utils.mlgp_hook import set_function_as_module_to_catch_error

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
    'cuda': False,
}


class CIGP_MODULE(torch.nn.Module):
    # def __init__(self, grid_params_list, kernel_list, target_list, normalize=True, restrict_method= 'exp') -> None:
    def __init__(self, module_config, data=None) -> None:
        super().__init__()
        _final_config = smart_update(default_module_config, module_config)
        self.module_config = deepcopy(_final_config)
        module_config = deepcopy(_final_config)

        # param check
        assert module_config['optimizer'] in ['adam'], 'now optimizer only support adam, but get {}'.format(module_config['optimizer'])

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


        # X - normalize
        if module_config['input_normalize'] is True:
            # self.X_normalizer = Normalizer(self.inputs_tr[0])
            self.X_normalizer = Normalizer(self.inputs_tr[0],  dim=[i for i in range(len(self.inputs_tr[0].shape))])
            self.inputs_tr[0] = self.X_normalizer.normalize(self.inputs_tr[0])
        else:
            self.X_normalizer = None

        # Y - normalize
        if module_config['output_normalize'] is True:
            # self.Y_normalizer = Normalizer(self.outputs_tr[0], dim=[0,1])
            self.Y_normalizer = Normalizer(self.outputs_tr[0], dim=[i for i in range(len(self.outputs_tr[0].shape))])
            self.outputs_tr[0] = self.Y_normalizer.normalize(self.outputs_tr[0])
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

        # cholesky module as module
        self.cholesky_am = set_function_as_module_to_catch_error(torch.linalg.cholesky)


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

        # L = torch.linalg.cholesky(Sigma)
        L = self.cholesky_am(Sigma)
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
                input_param[0] = self.X_normalizer.normalize(input_param[0])

            Sigma = self.kernel_list[0](self.inputs_tr[0], self.inputs_tr[0]) + JITTER * torch.eye(self.inputs_tr[0].size(0), device=list(self.parameters())[0].device)
            if self.module_config['exp_restrict'] is True:
                _noise = self.noise.exp()
            else:
                _noise = self.noise
            Sigma = Sigma + _noise.pow(-1) * torch.eye(self.inputs_tr[0].size(0), device=list(self.parameters())[0].device)

            kx = self.kernel_list[0](self.inputs_tr[0], input_param[0])
            L = torch.linalg.cholesky(Sigma)
            LinvKx,_ = torch.triangular_solve(kx, L, upper = False)

            u = kx.t() @ torch.cholesky_solve(self.outputs_tr[0], L)

            if self.module_config['output_normalize'] is True:
                u = self.Y_normalizer.denormalize(u)

            var_diag = self.kernel_list[0](input_param[0], input_param[0]).diag().view(-1, 1) - (LinvKx**2).sum(dim = 0).view(-1, 1)
            if self.module_config['exp_restrict'] is True:
                var_diag = var_diag + self.noise.exp().pow(-1)
            else:
                var_diag = var_diag + self.noise.pow(-1)
            
            if self.module_config['output_normalize'] is True:
                var_diag = var_diag * (self.Y_normalizer.std**2)
            var_diag = var_diag.expand_as(u)
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

        if hasattr(self, 'base_cigp') and False:
            predict_var = None
            from torch.distributions import Normal
            sample_time = 100
            data_number = self.base_cigp.inputs_eval[0].shape[0]
            _dim_len = self.base_cigp.inputs_eval[0].shape[1]
            _base_mean, _base_var = self.base_cigp.predict([self.inputs_eval[0][:, :_dim_len]])
            for i in range(sample_time):
                # sample
                # sampler = Normal(_base_mean[i:i+1,:], torch.clip(_base_var[i:i+1,:].sqrt(), 1e-6))
                sampler = Normal(_base_mean, torch.clip(_base_var.sqrt(), 1e-6))
                
                # sample_input = torch.cat([sampler.sample() for i in range(sample_time)], dim=0)
                sample_input = sampler.sample()
                base_input = self.inputs_eval[0][:, :_dim_len]

                # real_input
                sample_input = torch.cat([base_input, sample_input], dim=1)
                _, sample_var = self.predict([sample_input])

                # replace var
                # sample_var = sample_var.mean(0, keepdim=True)
                if predict_var is None:
                    predict_var = sample_var.clone().reshape(1, *sample_var.shape)
                else:
                    predict_var = torch.cat([predict_var, sample_var.clone().reshape(1, *sample_var.shape)], dim=0)
            predict_var = predict_var.mean(0)

        result = high_level_evaluator([predict_y, predict_var], self.outputs_eval[0], self.module_config['evaluate_method'], sample_last_dim=False)
        self.predict_y = predict_y
        self.predict_var = predict_var
        print(result)
        return result


if __name__ == '__main__':
    module_config = {}

    x = np.load('./data/sample/input.npy')
    y = np.load('./data/sample/output_fidelity_2.npy')
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()

    train_x = [x[:128,:]]
    train_y = [y[:128,...].reshape(128, -1)]
    eval_x = [x[128:,:]]
    eval_y = [y[128:,...].reshape(128, -1)]
    source_shape = y[128:,...].shape

    cigp = CIGP_MODULE(module_config, [train_x, train_y, eval_x, eval_y])
    for epoch in range(300):
        print('epoch {}/{}'.format(epoch+1, 300), end='\r')
        cigp.train()
    print('\n')
    cigp.eval()

    from result_visualize.plot_field import plot_container
    data_list = [cigp.outputs_eval[0].numpy(), cigp.predict_y.numpy()]
    data_list.append(abs(data_list[0] - data_list[1]))
    data_list = [_d.reshape(source_shape) for _d in data_list]
    label_list = ['groundtruth','predict', 'diff']
    pc = plot_container(data_list, label_list, sample_dim=0)
    pc.plot()


