import random
import torch
import numpy as np
from copy import deepcopy

fidelity_map = {
    'low': 0,
    'medium': 1,
    'high': 2
}

# fidelity order should be low -> high
preprocess_default_config_dict = {
    'random_shuffle_seed': None,

    'train_start_index': 0, 
    'train_sample': 8, 
    'eval_start_index': 0, 
    'eval_sample':256,
    
    'inputs_format': ['x[0]'],
    'outputs_format': ['y[0]', 'y[2]'],

    'force_2d': False,
    'x_sample_to_last_dim': False,
    'y_sample_to_last_dim': False,
}


def _flatten_inputs(inputs):
    _len_list = [len(_l) for _l in inputs]
    _temp_array_list = []
    [_temp_array_list.extend(deepcopy(_l)) for _l in inputs]
    return _temp_array_list, _len_list

def _reformat_inputs(array_list, _len_list):
    _index = 0
    outputs = []
    for _len in _len_list:
        outputs.append(array_list[_index:_index+_len])
        _index += _len
    return outputs

def _last_dim_to_fist(_tensor):
    _dim = [i for i in range(_tensor.ndim)]
    _dim.insert(0, _dim.pop())
    if isinstance(_tensor, torch.Tensor):
        return _tensor.permute(*_dim)
    elif isinstance(_tensor, np.ndarray):
        return _tensor.transpose(*_dim)
    else:
        assert False, '_tensor should be torch.Tensor or np.ndarray'

def _first_dim_to_last(_tensor):
    _dim = [i+1 for i in range(_tensor.ndim-1)]
    _dim.append(0)
    if isinstance(_tensor, torch.Tensor):
        return _tensor.permute(*_dim)
    elif isinstance(_tensor, np.ndarray):
        return _tensor.transpose(*_dim)
    else:
        assert False, '_tensor should be torch.Tensor or np.ndarray'


class Data_preprocess(object):
    # --------------------------------------------------
    # input data format:
    #           [x_train, y_train, x_eval, y_eval]
    # output data format:
    #           [x_train, y_train, x_eval, y_eval]
    # --------------------------------------------------
    def __init__(self, config_dict):
        default_config = deepcopy(preprocess_default_config_dict)
        default_config.update(config_dict)
        self.config_dict = default_config

    def do_preprocess(self, inputs):
        if self.config_dict['random_shuffle_seed'] is not None:
            out = self._random_shuffle(inputs)
        else:
            out = inputs

        out = self._get_sample(out)
        if self.config_dict['force_2d'] is True:
            out = self._force_2d(out)

        if self.config_dict['x_sample_to_last_dim'] is True:
            out[0] = [_first_dim_to_last(_array) for _array in out[0]]
            out[2] = [_first_dim_to_last(_array) for _array in out[2]]
        if self.config_dict['y_sample_to_last_dim'] is True:
            out[1] = [_last_dim_to_fist(_array) for _array in out[1]]
            out[3] = [_last_dim_to_fist(_array) for _array in out[3]]

        out = self._get_want_format(out)
        return out

    def _force_2d(self, inputs):
        _temp_array_list,_len_list = _flatten_inputs(inputs)
        _temp_array_list = [_array.reshape(_array.shape[0], -1) for _array in _temp_array_list]
        outputs = _reformat_inputs(_temp_array_list, _len_list)
        return outputs

    def _get_want_format(self, inputs):
        outputs = []
        x = deepcopy(inputs[0])
        y = deepcopy(inputs[1])
        
        _temp_list = []
        for _cmd in self.config_dict['inputs_format']:
            _temp_list.append(eval(_cmd))
        outputs.append(_temp_list) # inputs_tr
        
        _temp_list = []
        for _cmd in self.config_dict['outputs_format']:
            _temp_list.append(eval(_cmd))
        outputs.append(_temp_list) # outputs_tr

        x = deepcopy(inputs[2])
        y = deepcopy(inputs[3])
        _temp_list = []
        for _cmd in self.config_dict['inputs_format']:
            _temp_list.append(eval(_cmd))
        outputs.append(_temp_list) # inputs_eval
        
        _temp_list = []
        for _cmd in self.config_dict['outputs_format']:
            _temp_list.append(eval(_cmd))
        outputs.append(_temp_list) # outputs_eval
        return outputs

    def _get_sample(self, inputs):
        outputs = []
        for i in range(2):
            _a = self.config_dict['train_start_index']
            _b = self.config_dict['train_sample']
            _temp_list = [_array[_a:_a+_b,...] for _array in inputs[i]]
            outputs.append(_temp_list)

        for i in range(2,4):
            _a = self.config_dict['eval_start_index']
            _b = self.config_dict['eval_sample']
            _temp_list = [_array[_a:_a+_b,...] for _array in inputs[i]]
            outputs.append(_temp_list)
        return outputs

    def _random_shuffle(self, inputs):
        _temp_array_list,_len_list = _flatten_inputs(inputs)
        _temp_array_list = self._random_shuffle_array_list(_temp_array_list)
        outputs = _reformat_inputs(_temp_array_list, _len_list)
        return outputs

    def _random_shuffle_array_list(self, np_array_list):
        # [array_0, array_1, ...]
        random.seed(self.config_dict['random_shuffle_seed'])
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


if __name__ == '__main__':
    from data_loader import Standard_mat_DataLoader
    Stand_data = Standard_mat_DataLoader('poisson_v4_02')
    data = Stand_data.get_data()

    data_preprocess = Data_preprocess({})
    data = data_preprocess.do_preprocess(data)