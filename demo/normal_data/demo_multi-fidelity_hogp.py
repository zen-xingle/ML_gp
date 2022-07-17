import os
import sys
import torch

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from utils.main_controller import controller
from module.hogp import HOGP_MODULE
from module.hogp_multi_fidelity import HOGP_MF_MODULE


interp_data = False

real_dataset = ['FlowMix3D_MF',
                'MolecularDynamic_MF', 
                'plasmonic2_MF', 
                'SOFC_MF',]

gen_dataset = ['poisson_v4_02',
                'burger_v4_02',
                'Burget_mfGent_v5',
                'Burget_mfGent_v5_02',
                # 'Heat_mfGent_v5',
                'Piosson_mfGent_v5',
                'Schroed2D_mfGent_v1',
                'TopOP_mfGent_v5',]

if __name__ == '__main__':
    # for _dataset in real_dataset + gen_dataset:
    for _dataset in ['SOFC_MF']:
        for _seed in [None,0,1,2,3,4]:
            controller_config = {
                'max_epoch': 100
            } # use defualt config
            
            module_config = {
                'dataset': {'name': _dataset,
                            'interp_data': interp_data,

                            # preprocess
                            'seed': _seed,
                            'train_start_index': 0,
                            'train_sample': 32, 
                            'eval_start_index': 0, 
                            'eval_sample':128,
                            
                            'inputs_format': ['x[0]'],
                            'outputs_format': ['y[0]'],

                            'force_2d': False,
                            'x_sample_to_last_dim': False,
                            'y_sample_to_last_dim': True,
                            'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                            },
                } # only change dataset config, others use default config
            
            ct = controller(HOGP_MODULE, controller_config, module_config)
            ct.start_train()

            for _sample in [4, 8, 16, 32]:
                mfct_module_config = {
                    'dataset': {'name': _dataset,
                                'interp_data': interp_data,

                                # preprocess
                                'seed': _seed,
                                'train_start_index': 0,
                                'train_sample': _sample, 
                                'eval_start_index': 0, 
                                'eval_sample':128,
                                
                                'inputs_format': ['x[0]', 'y[0]'],
                                'outputs_format': ['y[-1]'],

                                'force_2d': False,
                                'x_sample_to_last_dim': False,
                                'y_sample_to_last_dim': True,
                                'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                                },
                } # only change dataset config, others use default config

                mfct = controller(HOGP_MF_MODULE, controller_config, mfct_module_config)
                
                with torch.no_grad():
                    # use x->yl_predict for test x+yl -> yh
                    mfct.module.inputs_eval[1] = ct.module.predict_y
                    pass

                mfct.start_train()

    mfct.clear_record()