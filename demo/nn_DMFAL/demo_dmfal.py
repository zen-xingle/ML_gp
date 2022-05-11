import os
import sys

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from utils.main_controller import controller
from nn_net.nn_shibo import DeepMFnet


interp_data = False

if __name__ == '__main__':
    for _seed in [None, 0, 1]:
        with open('record.txt', 'a') as _temp_file:
            _temp_file.write('-'*40 + '\n')
            _temp_file.write('\n')
            _temp_file.write('  Demo DMFAL \n')
            _temp_file.write('  seed: {} \n'.format(_seed))
            _temp_file.write('  interp_data: {} \n'.format(interp_data))
            _temp_file.write('\n')
            _temp_file.write('-'*40 + '\n')
            _temp_file.flush()

        module_config = {
            'dataset' : {'name': 'TopOP_mfGent_v5',
                 'fidelity': ['low', 'high'],
                 'type':'x_yl_2_yh',    # x_yl_2_yh, x_2_y
                 'train_start_index': 0, 
                 'train_sample': [32, 8], 
                 'eval_start_index': 0, 
                 'eval_sample': [128, 128],
                 'seed': 0,
                 'interp_data': False},
        }
        ct = controller(DeepMFnet, {}, module_config)
        ct.start_train()
        ct.smart_restore_state(-1)
        ct.rc_file.write('---> final result\n')
        ct.rc_file.flush()
        ct.start_eval({'eval state':'final',
                       'module_name':'dmfal',
                       'cp_record_file': True})
        ct.rc_file.write('---> end\n\n')
        ct.rc_file.flush()

    ct.clear_record()