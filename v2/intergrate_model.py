from re import M
import torch

from utils import *
from utils.data_utils import data_register
import numpy as np

dataset = {'x': np.load('x.npy'), 'yl': np.load('yl.npy'), 'yh': np.load('yh.npy')},

defualt_config = {
    'dataset' : {'name': 'poisson_v4_02',
                 'interp_data': False,
                 
                 # preprocess
                 'seed': None,
                 'train_set': '0:256',
                 'eval_set': '256:512',
                #  'train_set': '0:0.5',
                #  'eval_set':  '0.5:1',

                 'symbol_asign':
                        { 'x'  : 'inputs_list[0]',
                          'yl' : 'outputs_list[0]',
                          'yh' : 'outputs_list[1]',
                        },

                 'force_2d': True,
                #  'x_sample_to_last_dim': False,
                #  'y_sample_to_last_dim': False,
                #  'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                 },

    'gp_0': {
            'inputs': 'x',
            'outputs': 'yl',
            'data_preprocess':{
                "proc_imp": fo,
                'recall_for_output': True,
                "inputs_normalize": True,
                "outputs_normalize": True},

            'gp_model': {
                'name': 'CIGP',
                'kernel': {
                    'K1': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
                    },
                'noise_init' : 1.,
                'exp_restrict': True},

                
            'training_set':{
                "ALL": "0:32"}
            },

    'gp_1': { 
            'inputs': ['x', 'yl'],
            'outputs': 'yh',
            'data_preprocess':{
                "proc_imp": fo,
                'recall_for_output': True,
                "inputs_normalize": True,
                "outputs_normalize": True},

            'gp_model': {
                'name:': 'CIGAR',
                'kernel': {
                    'K1': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
                    },
                'noise_init' : 1.,
                'exp_restrict': True},

            'training_set':{
                "x" : "16:48",
                "yl": "PARTIAL-SUBSET",
                "yh": "16:48"}
            },

    'optimizer':{
        'type': 'Adam',

        'lr_init': {'noise': 0.1, 'kernel': 0.1, 'others': 0.1},
        # 'lr_init': 0.1,
        
        'epoch': 1000,
        'lr_decay': {'step': 100, 'decay_value': 0.5},
        'lr_decay': {'step': '10x', 'decay_value': 0.5},
        'lr_decay': {'epoch-100': 0.5, 'epoch-200': 0.25},
        },

}



# inputs_normalize will setup at the begining of the training, then it will be used in the eval/predict process
# outputs normalize will setup at the begining of the training, then it will be used in the eval/predict process

# base_gp_model should got the input_list and output_list

# pre_process should got the input_list and output_list
# post_process should got the input_list and output_list

class Intergrate_gp_model:
    def __init__(self, config) -> None:
        self.config = config

        self.sub_block(inputs_normalize, pre_process, base_gp_model, post_process, outputs_normalize)


    def train(input_list, output_list):
        pass


    def predict(input_list):
        pass