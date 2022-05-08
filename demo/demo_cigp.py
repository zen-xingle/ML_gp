import os
import sys

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from utils.main_controller import controller
from module.cigp import CIGP_MODULE


if __name__ == '__main__':
    with open('record.txt', 'a') as _temp_file:
        _temp_file.write('-'*40 + '\n')
        _temp_file.write('\n')
        _temp_file.write('  Demo cigp \n')
        _temp_file.write('\n')
        _temp_file.write('-'*40 + '\n')
        _temp_file.flush()

    module_config = {
        'dataset': {'name': 'TopOP_mfGent_v5',
                    'fidelity': ['low'],
                    'type':'x_2_y',    # x_yl_2_yh, x_2_y
                    'train_start_index': 0, 
                    'train_sample': 8, 
                    'eval_start_index': 0,
                    'eval_sample':128,
                    'seed': None},
    }
    ct = controller(CIGP_MODULE, {}, module_config)
    ct.start_train()
    ct.smart_restore_state(-1)
    ct.rc_file.write('---> final result\n')
    ct.rc_file.flush()
    ct.start_eval({'eval state':'final'})
    ct.rc_file.write('---> end\n\n')
    ct.rc_file.flush()