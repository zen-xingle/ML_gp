import numpy as np
from scipy.io import loadmat
import torch


def _smart_path(path):
    # TODO : add real path parse
    return path

# fisrt for train, second for test
# int for exact number, point for percentage 
default_slice_param = [0.6, 0.4]

def np_list_to_tensor_list(np_list):
    return [torch.from_numpy(np_list[i]).float() for i in range(len(np_list))]

def dict_pattern(path, function, interp_available):
    return {'path': path, 'function': function, 'interp_available': interp_available}

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

def _get_format_slice_data(length, slice):
    if isinstance(slice[0], int):
        return slice[0], slice[1]

    elif isinstance(slice[0], float):
        assert slice[0] + slice[1] == 1, 'slice sum should be 1'
        _tr = int(length * slice[0])
        _te = int(length * slice[1])
        while _tr + _te > length:
            _tr -= 1
            _te -= 1
        return _tr, _te


class SP_DataLoader(object):
    dataset_available = ['FlowMix3D_MF',
                         'MolecularDynamic_MF', 
                         'plasmonic2_MF', 
                         'SOFC_MF',
                         ]
    def __init__(self, dataset_name, need_interp=False) -> None:
        self.dataset_info = {
            'FlowMix3D_MF': 
                dict_pattern('data\MF_data\FlowMix3D_MF.mat', self._general, False),
            'MolecularDynamic_MF': 
                dict_pattern('data\MF_data\MolecularDynamic_MF.mat', self._general, False),
            'plasmonic2_MF': 
                dict_pattern('data\MF_data\plasmonic2_MF.mat', self._general, False),
            'SOFC_MF': 
                dict_pattern('data\MF_data\SOFC_MF.mat', self._SOFC_MF, False),
            }
        if dataset_name not in self.dataset_info:
            assert False
        if need_interp and self.dataset_info[self.dataset_name]['interp_available'] is False:
            assert False
        self.dataset_name = dataset_name
        self.need_interp = need_interp

    def get_data(self):
        outputs = self.dataset_info[self.dataset_name]['function']()
        return outputs

    def _general(self):
        _data = loadmat(_smart_path(self.dataset_info[self.dataset_name]['path']))
        x = [_data['X']]
        y = []
        for i in range(len(_data['Y'][0])):
            y.append(_data['Y'][0][i])
        return x, y, None, None

    # def _FlowMix3D_MF(self):
    #     return self._general()

    # def _MolecularDynamic_MF(self):
    #     return self._general()

    # def _plasmonic2_MF(self):
    #     return self._general()

    def _SOFC_MF(self):
        _data = loadmat(_smart_path(self.dataset_info[self.dataset_name]['path']))
        x = [_data['X']]
        y = []
        for i in range(len(_data['Y1'][0])):
            y.append(_concat_on_new_last_dim([_data['Y1'][0][i], _data['Y2'][0][i]]))
        return x, y, None, None

    def _get_distribute(self):
        pass


class Standard_mat_DataLoader(object):
    dataset_available = ['poisson_v4_02',
                        'burger_v4_02',
                        'Burget_mfGent_v5',
                        'Burget_mfGent_v5_02',
                        'Heat_mfGent_v5',
                        'Piosson_mfGent_v5',
                        'Schroed2D_mfGent_v1',
                        'TopOP_mfGent_v5',]
    def __init__(self, dataset_name, force_2d=False) -> None:
        self.dataset_info = {
            'poisson_v4_02': dict_pattern( 'data/MultiFidelity_ReadyData/poisson_v4_02.mat', self._general, True),
            'burger_v4_02': dict_pattern( 'data/MultiFidelity_ReadyData/burger_v4_02.mat', self._general, True),
            'Burget_mfGent_v5': dict_pattern( 'data/MultiFidelity_ReadyData/Burget_mfGent_v5.mat', self._general, True),
            'Burget_mfGent_v5_02': dict_pattern( 'data/MultiFidelity_ReadyData/Burget_mfGent_v5_02.mat', self._general, True),
            'Heat_mfGent_v5': dict_pattern( 'data/MultiFidelity_ReadyData/Heat_mfGent_v5.mat', self._general, True),
            'Piosson_mfGent_v5': dict_pattern( 'data/MultiFidelity_ReadyData/Piosson_mfGent_v5.mat', self._general, True),
            'Schroed2D_mfGent_v1': dict_pattern( 'data/MultiFidelity_ReadyData/Schroed2D_mfGent_v1.mat', self._general, True),
            'TopOP_mfGent_v5': dict_pattern( 'data/MultiFidelity_ReadyData/TopOP_mfGent_v5.mat', self._general, True),
        }
        if dataset_name not in self.dataset_info:
            assert False
        self.dataset_name = dataset_name
        self.force_2d = force_2d

    def _general(self):
        _data = loadmat(_smart_path(self.dataset_info[self.dataset_name]['path']))
        x_tr = [_data['xtr']]
        x_te = [_data['xte']]
        y_tr = []
        for i in range(len(_data['Ytr'][0])):
            y_tr.append(_data['Ytr'][0][i])
        y_te = []
        for i in range(len(_data['Yte'][0])):
            y_te.append(_data['Yte'][0][i])
        return x_tr, y_tr, x_te, y_te

    def get_data(self):
        outputs = self.dataset_info[self.dataset_name]['function']()
        return outputs






if __name__ == '__main__':
    # sp_data = SP_DataLoader('SOFC_MF', None)
    # print(sp_data.get_data())

    # stand_data = Standard_mat_DataLoader('poisson_v4_02')
    # print(stand_data.get_data())

    pass