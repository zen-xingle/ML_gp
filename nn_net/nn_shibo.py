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


from nn_net.BaseNet import AdaptiveBaseNet
from scipy.io import loadmat
from utils import *


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
            } # they got the same data format


default_module_config = {
    # 'dataset' : {'name': 'Piosson_mfGent_v5',
    #              'fidelity': ['low', 'high'],
    #              'type':'x_yl_2_yh',    # x_yl_2_yh, x_2_y
    #              'train_start_index': 0, 
    #              'train_sample': [32, 8], 
    #              'eval_start_index': 0, 
    #              'eval_sample': [128, 128],
    #              'seed': 0,
    #              'interp_data': False},
    'dataset': {'name': 'poisson_v4_02',
                'interp_data': True,

                'seed': None,
                'train_start_index': 0, 
                'train_sample': 32, 
                'eval_start_index': 0,
                'eval_sample': 128,

                'inputs_format': ['x[0]'],
                'outputs_format': ['y[0]','y[-1]'],

                'force_2d': True,
                'x_sample_to_last_dim': False,
                'y_sample_to_last_dim': False,
                'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                },
    'second_fidelity_sample': 8,
    'lr': {'opt_param': 0.01},
    'evaluate_method': ['mae', 'rmse', 'r2'],
    'optimizer': 'adam',
    'exp_restrict': False,
    'input_normalize': True,
    'output_normalize': True,
    
    # according to original inplement
    # h_w, h_d determine laten dim
    # net_param
    'nn_param': {
        'hlayers_w': [40, 40],
        'hlayers_d': [2, 2],
        'base_dim': [32, 32], # ?
        'activation': 'relu', # ['tanh','relu','sigmoid']
    },
    # 'input_dim': auto
    # 'output_dim': auto
    'reg_strength': 1e-3,
}


class DeepMFnet:
    # def __init__(self, grid_params_list, kernel_list, target_list, normalize=True, restrict_method= 'exp') -> None:
    def __init__(self, module_config) -> None:
        default_module_config.update(module_config)
        self.module_config = deepcopy(default_module_config)
        module_config = deepcopy(default_module_config)

        # param check
        assert module_config['optimizer'] in ['adam'], 'now optimizer only support adam, but get {}'.format(module_config['optimizer'])

        # load_data
        self._load_data(module_config['dataset'])
        self.init_model_params()

        # X - normalize
        if module_config['input_normalize'] is True:
            self.X_normalizer_list = []
            for i in range(self.module_config['nn_param']['M']):
                self.X_normalizer_list.append(Normalizer(self.inputs_tr[i]))
                self.inputs_tr[i] = self.X_normalizer_list[i].normalize(self.inputs_tr[i])
        else:
            self.X_normalizer_list = None

        # Y - normalize
        # TODO normalize according to dims
        if module_config['output_normalize'] is True:
            self.Y_normalizer_list = []
            for i in range(self.module_config['nn_param']['M']):
                self.Y_normalizer_list.append(Normalizer(self.outputs_tr[i]))
                self.outputs_tr[i] = self.Y_normalizer_list[i].normalize(self.outputs_tr[i])
        else:
            self.Y_normalizer_list = None

        # init optimizer
        self._optimizer_setup()

    '''
    def _load_data_elder(self, dataset_config):
        print('dataset_config name:', dataset_config['name'])
        if dataset_config['name'] in mat_dataset_paths:
            # they got the same data format
            data = loadmat(mat_dataset_paths[dataset_config['name']])
            
            if dataset_config['type'] == 'x_yl_2_yh':
                assert len(dataset_config['fidelity']) == 2, 'for x_yl_2_yh, fidelity length must be 2'
                _first_fidelity = fidelity_map[dataset_config['fidelity'][0]]
                _second_fidelity = fidelity_map[dataset_config['fidelity'][1]]
                x_tr_0 = torch.tensor(data['xtr'], dtype=torch.float32)
                x_eval_0 = torch.tensor(data['xte'], dtype=torch.float32)

                if dataset_config['interp_data'] is True:
                    x_tr_1 = torch.tensor(data['Ytr_interp'][0][_first_fidelity], dtype=torch.float32)
                    y_tr = torch.tensor(data['Ytr_interp'][0][_second_fidelity], dtype=torch.float32)
                    x_eval_1 = torch.tensor(data['Yte_interp'][0][_first_fidelity], dtype=torch.float32)
                    y_eval = torch.tensor(data['Yte_interp'][0][_second_fidelity], dtype=torch.float32)
                else:
                    x_tr_1 = torch.tensor(data['Ytr'][0][_first_fidelity], dtype=torch.float32)
                    y_tr = torch.tensor(data['Ytr'][0][_second_fidelity], dtype=torch.float32)
                    x_eval_1 = torch.tensor(data['Yte'][0][_first_fidelity], dtype=torch.float32)
                    y_eval = torch.tensor(data['Yte'][0][_second_fidelity], dtype=torch.float32)
            else:
                assert False, NotImplemented
                
            # shuffle
            if self.module_config['dataset']['seed'] is not None:
                x_tr_0, x_tr_1, y_tr = self._random_shuffle([[x_tr_0, 0], [x_tr_1, 0], [y_tr, 0]])
            
            # vectorize, reshape to 2D
            _temp_list = [x_tr_0, x_tr_1, y_tr, x_eval_0, x_eval_1, y_eval]
            for i,_value in enumerate(_temp_list):
                _temp_list[i] = _value.reshape(_value.shape[0], -1)
            x_tr_0 = _temp_list[0]
            x_tr_1 = _temp_list[1]
            y_tr = _temp_list[2]
            x_eval_0 = _temp_list[3]
            x_eval_1 = _temp_list[4]
            y_eval = _temp_list[5]
            
            _index = dataset_config['train_start_index']
            self.inputs_tr = []
            self.inputs_tr.append(x_tr_0[_index:_index+dataset_config['train_sample'][0], :])
            self.inputs_tr.append(x_tr_0[_index:_index+dataset_config['train_sample'][1], :])
            self.outputs_tr = []
            self.outputs_tr.append(x_tr_1[_index:_index+dataset_config['train_sample'][0], :])
            self.outputs_tr.append(y_tr[_index:_index+dataset_config['train_sample'][1], :])

            _index = dataset_config['eval_start_index']
            self.inputs_eval = []
            self.inputs_eval.append(x_eval_0[_index:_index+dataset_config['eval_sample'][0], :])
            self.inputs_eval.append(x_eval_0[_index:_index+dataset_config['eval_sample'][1], :])
            self.outputs_eval = []
            self.outputs_eval.append(x_eval_1[_index:_index+dataset_config['eval_sample'][0], :])
            self.outputs_eval.append(y_eval[_index:_index+dataset_config['eval_sample'][1], :])
        else:
            assert False
    '''

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
        _inputs_tr, _outputs_tr, _inputs_eval, _outputs_eval = dp.do_preprocess(_data, numpy_to_tensor=True)
        self.inputs_tr = [_inputs_tr[0], _inputs_tr[0][0:self.module_config['second_fidelity_sample'],...]]
        self.outputs_tr = [_outputs_tr[0], _outputs_tr[1][0:self.module_config['second_fidelity_sample'],...]]
        self.inputs_eval = [_inputs_eval[0], _inputs_eval[0]]
        self.outputs_eval = [_outputs_eval[0], _outputs_eval[1]]

    def _optimizer_setup(self):
        opt_params = []
        lr = self.module_config['lr']['opt_param']
        for i in range(self.module_config['nn_param']['M']):
            for nn_param_name, nn_param in self.nn_list[i].parameters().items():
                opt_params.append({'params':nn_param, 'lr':lr})
            opt_params.append({'params': self.log_tau_list[i], 'lr':lr})
        self.optimizer = torch.optim.Adam(opt_params) # 改了lr从0.01 改成0.0001


    def init_model_params(self):
        nn_param = self.module_config['nn_param']
        #check lenth
        # TODO
        # _lenth_list = [len(self.inputs_tr)]
        # for _key, _value in nn_param.items():
        #     _lenth_list.append(len(_value))
        # assert len(set(_lenth_list)) == 1, 'nn_param length not equal'
        self.module_config['nn_param']['M'] = len(self.inputs_tr)
        self.M = len(self.inputs_tr)

        self.nn_list = []
        self.log_tau_list = []
        for i in range(self.M):
            _layers = []
            if i == 0:
                _layers.append(self.inputs_tr[i].shape[1]) #input dim
            else:
                _layers.append(self.inputs_tr[i].shape[1] + nn_param['base_dim'][i-1])
            _layers.extend([nn_param['hlayers_w'][i]]* nn_param['hlayers_d'][i] ) #hidden dim
            _layers.append(nn_param['base_dim'][i]) #transition dim
            _layers.append(self.outputs_tr[i].shape[1])
            self.nn_list.append(AdaptiveBaseNet(_layers, nn_param['activation'], 'cpu', torch.float))
            self.log_tau_list.append(torch.tensor(0.0, device='cpu', requires_grad=True, dtype=torch.float))
    

    def forward(self, x, m, sample=False):
        Y_m, base_m = self.nn_list[0].forward(x, sample)
        # propagate to the other fidelity levels
        for i in range(1,m+1):
            X_concat = torch.cat((base_m, x), dim=1)
            # print(X_concat.shape)
            Y_m, base_m = self.nn_list[i].forward(X_concat, sample)
        return Y_m, base_m


    def eval_llh(self, x, y, m):
        llh_samples_list = []
        pred_sample, _ = self.forward(x, m, sample=True)
        log_prob_sample = torch.sum(-0.5*torch.square(torch.exp(self.log_tau_list[m]))*torch.square(pred_sample-y) +\
                                self.log_tau_list[m] - 0.5*np.log(2*np.pi))
        llh_samples_list.append(log_prob_sample)
        return sum(llh_samples_list)

    def batch_eval_llh(self):
        llh_list = []
        for m in range(self.M):
            llh_m = self.eval_llh(self.inputs_tr[m], self.outputs_tr[m], m)
            llh_list.append(llh_m)
        return sum(llh_list)

    def batch_eval_kld(self):
        kld_list = []
        for m in range(self.M):
            kld_list.append(self.nn_list[m]._eval_kld())
        return sum(kld_list)

    def batch_eval_reg(self):
        reg_list = []
        for m in range(self.M):
            reg_list.append(self.nn_list[m]._eval_reg())
        return sum(reg_list)


    def predict(self, m, x):
        if self.module_config['input_normalize'] is True:
            x = self.X_normalizer_list[m].normalize(x)
        y_predict, _ = self.forward(x, m)
        if self.module_config['output_normalize'] is True:
            y_predict = self.Y_normalizer_list[m].denormalize(y_predict)
        return y_predict


    def train(self):
        self.optimizer.zero_grad()
        loss = -self.batch_eval_llh()
        # loss = self.batch_eval_kld()
        # loss = self.module_config['reg_strength']*self.batch_eval_reg()
        loss.backward(retain_graph=True)
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
        return params_need_check


    def save_state(self):
        state_dict = []
        return state_dict


    def load_state(self, params_list):
        return

    def eval(self):
        print('---> start eval')
        predict_y = self.predict(self.M-1, self.inputs_eval[1])
        # self.predict_y = predict_y
        result = high_level_evaluator([predict_y], self.outputs_eval[1], self.module_config['evaluate_method'])
        print(result)
        return result


if __name__ == '__main__':
    model = DeepMFnet({})
    model.train()
