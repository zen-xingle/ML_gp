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
from module.GAR import GAR


interp_data = False

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
                # 'TopOP_mfGent_v5',
                ]

if __name__ == '__main__':
    for _dataset in ['TopOP_mfGent_v5']:
        for _seed in [4]:
            # try:
                controller_config = {
                    'max_epoch': 1000,
                    'record_file_path': 'GAR.txt'
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
                                
                                'inputs_format': ['(x[0] - x[0].mean()) / x[0].std()'],
                                'outputs_format': ['(y[0] - y[0].mean()) / y[0].std()'],

                                'force_2d': False,
                                'x_sample_to_last_dim': False,
                                'y_sample_to_last_dim': True,
                                'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                                },
                        'cuda': False,
                        'evaluate_method': ['mae', 'rmse', 'r2', 'gaussian_loss'],
                        'noise_init' : 10.,
                        'exp_restrict': False,
                        'lr': {'kernel':0.01, 
                                'optional_param':0.01, 
                                'noise':0.01},
                    } # only change dataset config, others use default config
                
                ct = controller(HOGP_MODULE, controller_config, module_config)
                ct.start_train()
            # except:
            #     print('fisrt module stop early')

            # try:
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
                                    
                                    'inputs_format': ['(x[0] - x[0].mean()) / x[0].std()', '(y[0] - y[0].mean()) / y[0].std()'],
                                    'outputs_format': ['(y[-1] - y[-1].mean()) / y[-1].std()'],

                                    'force_2d': False,
                                    'x_sample_to_last_dim': False,
                                    'y_sample_to_last_dim': True,
                                    'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                                    },
                        'cuda': False,
                        'evaluate_method': ['mae', 'rmse', 'r2', 'gaussian_loss'],
                        'noise_init' : 10.,
                        'exp_restrict': False,
                        'lr': {'kernel':0.01, 
                                'optional_param':0.01, 
                                'noise':0.01},
                    } # only change dataset config, others use default config

                    mfct = controller(GAR, controller_config, mfct_module_config)
                    
                    with torch.no_grad():
                        # use x->yl_predict for test x+yl -> yh
                        mfct.module.inputs_eval[1] = ct.module.predict_y
                        if hasattr(ct.module, 'predict_var'):
                            mfct.module.base_var = ct.module.predict_var
                        pass

                    mfct.start_train()
            # except:
            #     print('second module stop early')

    # mfct.clear_record()