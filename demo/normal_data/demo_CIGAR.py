import os
import sys
import torch

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
demo_name = realpath[-1].rstrip('.py')
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from utils.main_controller import controller
from module.ind_hogp import HOGP_MODULE
from module.CIGAR import CIGAR

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
                'TopOP_mfGent_v5',]

if __name__ == '__main__':
    # for _dataset in real_dataset + gen_dataset:
    for _dataset in ['poisson_v4_02','burger_v4_02','Heat_mfGent_v5',]:
        for _seed in [0, 1, 2, 3, 4]:
            first_fidelity_sample = 32
            controller_config = {
                'max_epoch': 1000,
                'record_file_path': 'CIGAR.txt'
            } # use defualt config

            ct_module_config = {
                'dataset': {'name': _dataset,
                            'interp_data': interp_data,

                            'seed': _seed,
                            'train_start_index': 0, 
                            'train_sample': first_fidelity_sample, 
                            'eval_start_index': 0,
                            'eval_sample': 128,

                            'inputs_format': ['x[0]'],
                            'outputs_format': ['y[0]'],

                            'force_2d': False,
                            'x_sample_to_last_dim': False,
                            'y_sample_to_last_dim': True,
                            'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                            },
                'cuda': True,
                'evaluate_method': ['mae', 'rmse', 'r2', 'gaussian_loss'],
                'noise_init' : 10.,
                } # only change dataset config, others use default config
            ct = controller(HOGP_MODULE, controller_config, ct_module_config, demo_name)
            ct.start_train()

            # second_fidelity_sample = 32
            for second_fidelity_sample in [4, 8, 16, 32]:
                subset = 0.5 * second_fidelity_sample
                mfct_module_config = {
                    'dataset': {'name': _dataset,
                                'interp_data': interp_data,

                                # preprocess
                                'seed': _seed,
                                'train_start_index': int(first_fidelity_sample - subset), 
                                'train_sample': second_fidelity_sample, 
                                'eval_start_index': 0, 
                                'eval_sample':128,
                                
                                'inputs_format': ['x[0]', 'y[0]'],
                                'outputs_format': ['y[-1]'],

                                'force_2d': False,
                                'x_sample_to_last_dim': False,
                                'y_sample_to_last_dim': True,
                                'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                                },
                    'cuda': True,
                    'evaluate_method': ['mae', 'rmse', 'r2', 'gaussian_loss'],
                    'noise_init' : 10.,
                } # only change dataset config, others use default config

                mfct = controller(CIGAR, controller_config, mfct_module_config, demo_name)
                
                with torch.no_grad():
                    # use x->yl_predict for test x+yl -> yh
                    mfct.module.inputs_eval[1] = ct.module.predict_y
                    if hasattr(ct.module, 'predict_var'):
                        mfct.module.base_var = ct.module.predict_var
                    pass

                mfct.start_train()

    # mfct.clear_record()