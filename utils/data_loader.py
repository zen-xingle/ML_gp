import numpy as np
from scipy.io import loadmat
import torch

# fisrt for train, second for test
# int for exact number, point for percentage 
default_seperate = [0.6, 0.4]

def np_list_to_tensor_list(np_list):
    return [torch.from_numpy(np_list[i]).float() for i in range(len(np_list))]

def dict_pattern(path, seperate, function, interp_available):
    return {'path': path, 'seperate': seperate, 'function': function, 'interp_available': interp_available}

def _concat_on_new_last_dim(arrays):
    # TODO support multi in, now support 2
    in1 = arrays[0]
    in2 = arrays[1]
    assert in1.shape == in2.shape
    in1 = in1.reshape(*in1.shape, 1)
    in2 = in2.reshape(*in2.shape, 1)
    return np.concatenate([in1, in2], axis=-1)

def _force_2d(arrays):
    return [_array.reshape(_array.shape[0], -1) for _array in arrays]

class SP_DataLoader(object):
    dataset_available = ['FlowMix3D_MF', 'MolecularDynamic_MF', 'plasmonic2_MF', 'SOFC_MF']
    def __init__(self, dataset_name, force_2d=False) -> None:
        # self.dataset_info = {
        #     'FlowMix3D_MF': 
        #         dict_pattern('data\MF_data\FlowMix3D_MF.mat', default_seperate, self._FlowMix3D_MF, True),
        #     'MolecularDynamic_MF': 
        #         dict_pattern('data\MF_data\MolecularDynamic_MF.mat', default_seperate, self._MolecularDynamic_MF, True),
        #     'plasmonic2_MF': 
        #         dict_pattern('data\MF_data\plasmonic2_MF.mat', default_seperate, self._plasmonic2_MF, True),
        #     'SOFC_MF': 
        #         dict_pattern('data\MF_data\SOFC_MF.mat', default_seperate, self._SOFC_MF, True),
        #     }

        self.dataset_info = {
            'FlowMix3D_MF': 
                dict_pattern('data/MultiFidelity_ReadyData/FlowMix3D_MF.mat', default_seperate, self._FlowMix3D_MF, True),
            'MolecularDynamic_MF': 
                dict_pattern('data/MultiFidelity_ReadyData/MolecularDynamic_MF.mat', default_seperate, self._MolecularDynamic_MF, True),
            'plasmonic2_MF': 
                dict_pattern('data/MultiFidelity_ReadyData/plasmonic2_MF.mat', default_seperate, self._plasmonic2_MF, True),
            'SOFC_MF': 
                dict_pattern('data/MultiFidelity_ReadyData/SOFC_MF.mat', default_seperate, self._SOFC_MF, True),
            }

        if dataset_name not in self.dataset_info:
            assert False

        self.dataset_name = dataset_name
        self.force_2d = force_2d


    def get_data(self):
        outputs = self.dataset_info[self.dataset_name]['function']()
        if self.force_2d:
            outputs = [_force_2d(_out) for _out in outputs]
        return outputs

    def _seperate_for_real(self, length, seperate):
        if isinstance(seperate[0], int):
            return [seperate[0], seperate[1]]

        elif isinstance(seperate[0], float):
            assert seperate[0] + seperate[1] == 1, 'seperate sum should be 1'
            _tr = int(length * seperate[0])
            _te = int(length * seperate[1])
            while _tr + _te > length:
                _tr -= 1
                _te -= 1
            return _tr, _te

    def _general(self):
        _data = loadmat(self.dataset_info[self.dataset_name]['path'])
        _tr_sample, _te_sample = self._seperate_for_real(len(_data['X']), self.dataset_info[self.dataset_name]['seperate'])
        x_tr = [_data['X'][:_tr_sample, :]]
        x_te = [_data['X'][_tr_sample:_tr_sample + _te_sample, :]]
        y_tr = []
        y_te = []
        for i in range(len(_data['Y'][0])):
            y_tr.append(_data['Y'][0][i][:_tr_sample])
            y_te.append(_data['Y'][0][i][_tr_sample:_tr_sample + _te_sample])
        return x_tr, y_tr, x_te, y_te

    def _FlowMix3D_MF(self):
        return self._general()

    def _MolecularDynamic_MF(self):
        return self._general()

    def _plasmonic2_MF(self):
        return self._general()

    def _SOFC_MF(self):
        _data = loadmat(self.dataset_info[self.dataset_name]['path'])
        y = []
        for i in range(len(_data['Y1'][0])):
            y.append(_concat_on_new_last_dim([_data['Y1'][0][i], _data['Y2'][0][i]]))
        _tr_sample, _te_sample = self._seperate_for_real(len(_data['x']), self.dataset_info[self.dataset_name]['seperate'])
        x_tr = [_data['x'][:_tr_sample, :]]
        x_te = [_data['x'][_tr_sample:_tr_sample + _te_sample, :]]
        y_tr = []
        y_te = []
        for i in range(len(y)):
            y_tr.append(y[i][:_tr_sample])
            y_te.append(y[i][_tr_sample:_tr_sample + _te_sample])
        return x_tr, y_tr, x_te, y_te

    def _get_distribute(self):
        pass

if __name__ == '__main__':
    sd = SP_DataLoader('SOFC_MF', None)
    print(sd.get_data())