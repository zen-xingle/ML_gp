import os
import sys
import torch

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from utils.main_controller import controller
from module.cigp import CIGP_MODULE

interp_data=True

if __name__ == '__main__':
    for _seed in [None, 0, 1, 2, 3, 4]:
        with open('record.txt', 'a') as _temp_file:
            _temp_file.write('-'*40 + '\n')
            _temp_file.write('\n')
            _temp_file.write('  Demo NAR cigp \n')
            _temp_file.write('  seed: {} \n'.format(_seed))
            _temp_file.write('  interp_data: {} \n'.format(interp_data))
            _temp_file.write('\n')
            _temp_file.write('-'*40 + '\n')
            _temp_file.write('-'*3 + '> Training x -> yl part\n')
            _temp_file.flush()

        # ================================================================
        # Training x -> yl part

        controller_config = {} # use defualt config
        module_config = {
            'dataset': {'name': 'Heat_mfGent_v5',
                        'fidelity': ['low'],
                        'type':'x_2_y',    # x_yl_2_yh, x_2_y
                        'train_start_index': 0, 
                        'train_sample': 32, 
                        'eval_start_index': 0, 
                        'eval_sample':128,
                        'seed': _seed,
                        'interp_data': interp_data},
        } # only change dataset config, others use default config
        ct = controller(CIGP_MODULE, controller_config, module_config)
        ct.start_train()
        ct.smart_restore_state(-1)
        ct.rc_file.write('---> final result\n')
        ct.rc_file.flush()
        ct.start_eval({'eval state':'final'})
        ct.rc_file.write('-'*10 + '> finish x-yl training\n\n')
        ct.rc_file.flush()


        # ================================================================
        # Training x,yl -> yh part

        for _sample in [4,8,16,32]:
            with open('record.txt', 'a') as _temp_file:
                _temp_file.write('\n'+ '-'*10 + '>\n')
                _temp_file.write('NAR for {} samples\n'.format(_sample))
                _temp_file.write('-'*3 + '> Training x,yl -> yh part\n\n')
                _temp_file.flush()

            second_controller_config = {
                'max_epoch': 300,
            }
            second_module_config = {
                'dataset': {'name': 'Burget_mfGent_v5_02',
                            'fidelity': ['low','high'],
                            'type':'x_yl_2_yh',    # x_yl_2_yh, x_2_y
                            'train_start_index': 0, 
                            'train_sample': _sample,
                            'eval_start_index': 0, 
                            'eval_sample':128,
                            'seed': _seed,
                            'interp_data': interp_data},
                'res_cigp': None,
            }
            second_ct = controller(CIGP_MODULE, controller_config, second_module_config)
            # replace ground truth eval data with low fidelity predict
            # check inputs x
            x_dim = ct.module.inputs_eval[0].shape[1]
            torch.dist(second_ct.module.inputs_eval[0][:,0:x_dim], ct.module.inputs_eval[0])
            # check inputs y
            torch.dist(second_ct.module.inputs_eval[0][:,x_dim:], ct.module.outputs_eval[0])
            # check predict y
            torch.dist(second_ct.module.inputs_eval[0][:,x_dim:], ct.module.predict_y)
            second_ct.module.inputs_eval[0] = torch.cat([ct.module.inputs_eval[0], ct.module.predict_y],dim=1)

            second_ct.start_train()
            second_ct.smart_restore_state(-1)
            second_ct.rc_file.write('---> final result\n')
            second_ct.rc_file.flush()
            second_ct.start_eval({'eval state':'final'})
            second_ct.rc_file.flush()