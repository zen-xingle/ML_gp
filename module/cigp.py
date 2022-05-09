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

from utils.normalizer import Normalizer
from utils.performance_evaluator import performance_evaluator

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

# other name: test_v1, 
mat_dataset_paths = {
                'poisson_v4_02': 'data/MultiFidelity_ReadyData/poisson_v4_02.mat',
                'burger_v4_02': 'data/MultiFidelity_ReadyData/burger_v4_02.mat',
                'Burget_mfGent_v5': 'data/MultiFidelity_ReadyData/Burget_mfGent_v5.mat',
                'Burget_mfGent_v5_02': 'data/MultiFidelity_ReadyData/Burget_mfGent_v5_02.mat',
                'Heat_mfGent_v5': 'data/MultiFidelity_ReadyData/Heat_mfGent_v5.mat',
                'Piosson_mfGent_v5': 'data/MultiFidelity_ReadyData/Piosson_mfGent_v5.mat',
                'Schroed2D_mfGent_v1': 'data/MultiFidelity_ReadyData/Schroed2D_mfGent_v1.mat',
                'TopOP_mfGent_v5': 'data/MultiFidelity_ReadyData/TopOP_mfGent_v5.mat',
            } # they got the same data format

default_module_config = {
    'dataset' : {'name': 'Burget_mfGent_v5',
                 'fidelity': ['low'],
                 'type':'x_2_y',    # x_yl_2_yh, x_2_y
                 'train_start_index': 0, 
                 'train_sample': 32, 
                 'eval_start_index': 0, 
                 'eval_sample':256},

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
    'noise_init' : 1.,
    'res_cigp': None,
}


class CIGP_MODULE:
    # def __init__(self, grid_params_list, kernel_list, target_list, normalize=True, restrict_method= 'exp') -> None:
    def __init__(self, module_config) -> None:
        default_module_config.update(module_config)
        module_config = default_module_config
        self.module_config = deepcopy(module_config)

        # param check
        assert module_config['optimizer'] in ['adam'], 'now optimizer only support adam, but get {}'.format(module_config['optimizer'])

        # load_data
        self._load_data(module_config['dataset'])
        if self.module_config['res_cigp'] is not None:
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
        if dataset_config['name'] == 'test_v1':
            _sample = 100
            x_eval = torch.linspace(0, 6, _sample).view(-1, 1)
            y_eval = torch.sin(x_eval) + 10
            x_tr = torch.rand(32, 1) * 6
            y_tr = torch.sin(x_tr) + torch.rand(32, 1) * 0.5 + 10
        elif dataset_config['name'] in mat_dataset_paths:
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
            elif dataset_config['type'] == 'x_yl_2_yh':
                assert len(dataset_config['fidelity']) == 2, 'for x_yl_2_yh, fidelity length must be 2'
                _first_fidelity = fidelity_map[dataset_config['fidelity'][0]]
                _second_fidelity = fidelity_map[dataset_config['fidelity'][1]]
                x_tr_0 = torch.tensor(data['xtr'], dtype=torch.float32)
                x_tr_1 = torch.tensor(data['Ytr'][0][_first_fidelity], dtype=torch.float32)
                x_tr = torch.cat([x_tr_0.reshape(x_tr_0.shape[0], -1), x_tr_1.reshape(x_tr_1.shape[0], -1)], dim=1)
                y_tr = torch.tensor(data['Ytr'][0][_second_fidelity], dtype=torch.float32)
                x_eval_0 = torch.tensor(data['xte'], dtype=torch.float32)
                x_eval_1 = torch.tensor(data['Yte'][0][_first_fidelity], dtype=torch.float32)
                x_eval = torch.cat([x_eval_0.reshape(x_eval_0.shape[0], -1), x_eval_1.reshape(x_eval_1.shape[0], -1)], dim=1)
                y_eval = torch.tensor(data['Yte'][0][_second_fidelity], dtype=torch.float32)
        else:
            assert False

        # shuffle
        if self.module_config['dataset']['seed'] is not None:
            x_tr, y_tr = self._random_shuffle([[x_tr, 0], [y_tr, 0]])
        
        # vectorize, reshape to 2D
        _temp_list = [x_tr, y_tr, x_eval, y_eval]
        for i,_value in enumerate(_temp_list):
            _temp_list[i] = _value.reshape(_value.shape[0], -1)
        x_tr = _temp_list[0]
        y_tr = _temp_list[1]
        x_eval = _temp_list[2]
        y_eval = _temp_list[3]
        
        _index = dataset_config['train_start_index']
        self.inputs_tr = []
        self.inputs_tr.append(torch.tensor(x_tr[_index:_index+dataset_config['train_sample'], :]))
        self.outputs_tr = []
        self.outputs_tr.append(torch.tensor(y_tr[_index:_index+dataset_config['train_sample'], :]))

        _index = dataset_config['eval_start_index']
        self.inputs_eval = []
        self.inputs_eval.append(torch.tensor(x_eval[_index:_index+dataset_config['eval_sample'], :]))
        self.outputs_eval = []
        self.outputs_eval.append(torch.tensor(y_eval[_index:_index+dataset_config['eval_sample'], :]))


    def _select_connection_kernel(self, type_name):
        from kernel.Multi_fidelity_connection import rho_connection, mapping_connection
        assert type_name in ['res_rho', 'res_mapping']
        if type_name == 'identity':
            self.target_connection = None
        elif type_name in ['res_rho']:
            self.target_connection = rho_connection()
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
                if not hasattr(_kernel_params, 'exp_restrict'):
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
        if self.module_config['res_cigp'] is not None:
            optional_params.append(self.res)
            assert False, 'res_cigp is not supported yet'

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
        if self.module_config['res_cigp'] is not None:
            gamma = L.inverse() @ self.target_connection(self.inputs_tr[1], self.outputs_tr[0])
        else:
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
                input_param[0] = self.X_normalizer.normalize(input_param[0])

            Sigma = self.kernel_list[0](self.inputs_tr[0], self.inputs_tr[0]) + JITTER * torch.eye(self.inputs_tr[0].size(0))
            if self.module_config['exp_restrict'] is True:
                _noise = self.noise.exp()
            else:
                _noise = self.noise
            Sigma = Sigma + _noise.pow(-1) * torch.eye(self.inputs_tr[0].size(0))

            kx = self.kernel_list[0](self.inputs_tr[0], input_param[0])
            L = torch.cholesky(Sigma)
            # LinvKx,_ = torch.triangular_solve(kx, L, upper = False)

            if self.module_config['res_cigp'] is not None:
                u = kx.t() @ torch.cholesky_solve(self.target_connection(self.inputs_tr[1], self.outputs_tr[0]))
                if self.module_config['output_normalize'] is True:
                    input_param[1] = self.Y_normalizer.normalize(input_param[1])
                    u = self.target_connection(input_param[1], u)
                else:
                    u = self.target_connection(input_param[1], u)
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


    def save_state(self):
        state_dict = []
        for i, _kernel in enumerate(self.kernel_list):
            state_dict.extend(_kernel.get_param([]))

        state_dict.append(self.noise)
        if self.module_config['res_cigp'] is True:
            # TODO
            assert False, "res_cigp is not supported"        
        return state_dict


    def load_state(self, params_list):
        index = 0
        for i, _kernel in enumerate(self.kernel_list):
            _temp_list = _kernel.get_param([])
            _kernel.set_param(params_list[index: index + len(_temp_list)])
            index += len(_temp_list)
        
        with torch.no_grad():
            self.noise.copy_(params_list[index])
            if self.module_config['res_cigp'] is True:
                # TODO
                assert False, "res_cigp is not supported" 

    def eval(self):
        print('---> start eval')
        predict_y, _var = self.predict(self.inputs_eval)
        result = performance_evaluator(predict_y, self.outputs_eval[0], self.module_config['evaluate_method'], sample_last_dim=False)
        self.predict_y = predict_y
        print(result)
        return result


if __name__ == '__main__':
    module_config = {}
    cigp = CIGP_MODULE(module_config)
