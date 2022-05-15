
import os
import sys
import torch

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from utils.main_controller import controller
from module.ind_hogp import HOGP_MODULE
from module.ind_hogp_multi_fidelity import HOGP_MF_MODULE

interp_data=False

if __name__ == '__main__':
    for _seed in [None, 0, 1, 2, 3, 4]:
        with open('record.txt', 'a') as _temp_file:
            _temp_file.write('-'*40 + '\n')
            _temp_file.write('\n')
            _temp_file.write('  Demo sGAR \n')
            _temp_file.write('  seed: {} \n'.format(_seed))
            _temp_file.write('  interp_data: {} \n'.format(interp_data))
            _temp_file.write('\n')
            _temp_file.write('-'*40 + '\n')
            _temp_file.write('-'*3 + '> Training x -> yl part\n')
            _temp_file.flush()

        controller_config = {
            'max_epoch': 1000
        } # use defualt config

        ct_module_config = {
            'dataset': {'name': 'poisson_v4_02',
                        'interp_data': interp_data,

                        'seed': _seed,
                        'train_start_index': 0, 
                        'train_sample': 64, 
                        'eval_start_index': 0,
                        'eval_sample': 128,

                        'inputs_format': ['x[0]'],
                        'outputs_format': ['y[0]'],

                        'force_2d': False,
                        'x_sample_to_last_dim': False,
                        'y_sample_to_last_dim': True,
                        'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                        },
        } # only change dataset config, others use default config
        ct = controller(HOGP_MODULE, controller_config, ct_module_config)
        ct.start_train()
        ct.smart_restore_state(-1)
        ct.rc_file.write('---> final result')
        ct.rc_file.flush()
        ct.start_eval({'eval state':'final'})
        ct.rc_file.write('---> end\n\n')
        ct.rc_file.flush()

        for _sample in [32, 64]:
            with open('record.txt', 'a') as _temp_file:
                _temp_file.write('\n'+ '-'*10 + '>\n')
                _temp_file.write('SGAR for {} samples\n'.format(_sample))
                _temp_file.write('-'*3 + '> Training x,yl -> yh part\n\n')
                _temp_file.flush()

            mfct_module_config = {
                'dataset': {'name': 'poisson_v4_02',
                            'interp_data': interp_data,

                            # preprocess
                            'seed': _seed,
                            'train_start_index': 0,
                            'train_sample': _sample, 
                            'eval_start_index': 0, 
                            'eval_sample':256,
                            
                            'inputs_format': ['x[0]', 'y[0]'],
                            'outputs_format': ['y[2]'],

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

            mfct.start_train()
            mfct.smart_restore_state(-1)
            mfct.rc_file.write('---> final result')
            mfct.rc_file.flush()
            mfct.start_eval({'eval state':'final',
                            'module_name': 'SGAR',
                            'cp_record_file': True})
            mfct.rc_file.write('---> end\n\n')
            mfct.rc_file.flush()

    mfct.clear_record()