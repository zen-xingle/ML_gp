import os
import sys
import torch

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from utils.main_controller import controller
from gp_module.cigp import CIGP_MODULE
from gp_module.DC_I import DC_I

interp_data= True
real_dataset = ['FlowMix3D_MF',
                'MolecularDynamic_MF', 
                'plasmonic2_MF', 
                'SOFC_MF',]

gen_dataset = ['poisson_v4_02',
                'burger_v4_02',
                'Burget_mfGent_v5',
                'Burget_mfGent_v5_02',
                'Heat_mfGent_v5',
                'Piosson_mfGent_v5',
                # 'Schroed2D_mfGent_v1',
                'TopOP_mfGent_v5',]

if __name__ == '__main__':
    for _dataset in ['poisson_v4_02']:
        for _seed in [None]:
            controller_config = {'max_epoch':1000, 
                                 'record_file_path': 'DC.txt'} # use defualt config
            module_config = {
                'dataset': {'name': _dataset,
                            'interp_data': interp_data,

                            'seed': _seed,
                            'train_start_index': 0, 
                            'train_sample': 64, 
                            'eval_start_index': 0,
                            'eval_sample': 128,

                            'inputs_format': ['x[0]'],
                            'outputs_format': ['y[0]'],

                            'force_2d': True,
                            'x_sample_to_last_dim': False,
                            'y_sample_to_last_dim': False,
                            'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                            },
                'lr': {'kernel':0.01, 
                'optional_param':0.01, 
                'noise':0.01},
                'cuda': True,
                'evaluate_method': ['mae', 'rmse', 'r2', 'gaussian_loss'],
                'noise_init' : 100.0,
            } # only change dataset config, others use default config
            ct = controller(CIGP_MODULE, controller_config, module_config)
            ct.start_train()


            # ================================================================
            # Training x,yl -> yh part
            _second_sample = 32
            second_module_config = {
                'dataset': {'name': _dataset,
                            'interp_data': interp_data,

                            'seed': _seed,
                            'train_start_index': 0, 
                            'train_sample': _second_sample, 
                            'eval_start_index': 0,
                            'eval_sample': 128,

                            'inputs_format': ['x[0]','y[0]'],
                            'outputs_format': ['y[1]'],

                            'force_2d': True,
                            'x_sample_to_last_dim': False,
                            'y_sample_to_last_dim': False,
                            'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                            },
                'pca': {'type': 'listPCA', 
                        'r': 0.99, }, # listPCA, resPCA_mf,
                'noise_init' : 100.,
                'cuda': True,
                'evaluate_method': ['mae', 'rmse', 'r2', 'gaussian_loss'],
            }
            second_ct = controller(DC_I, controller_config, second_module_config)

            # x_dim = ct.module.inputs_eval[0].shape[1]
            # torch.dist(second_ct.module.inputs_eval[0][:,0:x_dim], ct.module.inputs_eval[0])
            # # check inputs y
            # torch.dist(second_ct.module.inputs_eval[1], ct.module.outputs_eval[0])
            # # check predict y
            # torch.dist(second_ct.module.inputs_eval[1], ct.module.predict_y)
            # # second_ct.module.inputs_eval[1] = ct.module.predict_y
            second_ct.module.inputs_eval[1] = ct.module.predict_y

            second_ct.start_train()


            _third_sample = 16
            third_module_config = {
                'dataset': {'name': _dataset,
                            'interp_data': interp_data,

                            'seed': _seed,
                            'train_start_index': 0, 
                            'train_sample': _third_sample, 
                            'eval_start_index': 0,
                            'eval_sample': 128,

                            'inputs_format': ['x[0]','y[0]','y[1]'],
                            'outputs_format': ['y[2]'],

                            'force_2d': True,
                            'x_sample_to_last_dim': False,
                            'y_sample_to_last_dim': False,
                            'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                            },
                'pca': {'type': 'listPCA', 
                        'r': 0.99, }, # listPCA, resPCA_mf,
                'noise_init' : 100.,
                'cuda': True,
                'evaluate_method': ['mae', 'rmse', 'r2', 'gaussian_loss'],
            }
            
            third_ct = controller(DC_I, controller_config, third_module_config)
            # replace ground truth eval data with low fidelity predict
            third_ct.module.inputs_eval[2] = second_ct.module.predict_y

            second_ct.start_train()


    # second_ct.clear_record()