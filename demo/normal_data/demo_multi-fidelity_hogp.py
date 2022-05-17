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

if __name__ == '__main__':
    for _seed in [0,1,2,3,4]:
        with open('record.txt', 'a') as _temp_file:
            _temp_file.write('-'*40 + '\n')
            _temp_file.write('\n')
            _temp_file.write('  Demo GAR \n')
            _temp_file.write('  seed: {} \n'.format(_seed))
            _temp_file.write('  interp_data: {} \n'.format(interp_data))
            _temp_file.write('\n')
            _temp_file.write('-'*40 + '\n')
            _temp_file.write('-'*3 + '> Training x -> yl part\n')
            _temp_file.flush()

        # ================================================================
        # Training x -> yl part

        controller_config = {
            'max_epoch': 1000
        } # use defualt config

        module_config = {
            'dataset': {'name': 'TopOP_mfGent_v5',
                        'interp_data': interp_data,

                        'seed': _seed,
                        'train_start_index': 0, 
                        'train_sample': 128, 
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
        
        ct = controller(HOGP_MODULE, controller_config, module_config)
        ct.start_train()
        ct.smart_restore_state(-1)
        ct.rc_file.write('---> final result')
        ct.rc_file.flush()
        ct.start_eval({'eval state':'final'})
        ct.rc_file.write('-'*10 + '> finish x-yl training\n\n')
        ct.rc_file.flush()

        # ================================================================
        # Training x,yl -> yh part

        for _sample in [4, 8, 16, 32, 64, 128]:
            with open('record.txt', 'a') as _temp_file:
                _temp_file.write('\n'+ '-'*10 + '>\n')
                _temp_file.write('GAR for {} samples\n'.format(_sample))
                _temp_file.write('-'*3 + '> Training x,yl -> yh part\n\n')
                _temp_file.flush()
            

            mfct_module_config = {
                'dataset': {'name': 'TopOP_mfGent_v5',
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
            mfct.smart_restore_state(-1)
            mfct.rc_file.write('---> final result\n')
            mfct.rc_file.flush()
            mfct.start_eval({'eval state':'final',
                            'module_name': 'GAR',
                            'cp_record_file': True})
            mfct.rc_file.write('---> end\n\n')
            mfct.rc_file.flush()

    mfct.clear_record()