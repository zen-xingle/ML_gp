import os
import sys

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from utils.main_controller import controller
from module.hogp import HOGP_MODULE

interp_data = False

if __name__ == '__main__':
    for _seed in [None, 0, 1, 2, 3, 4]:
        with open('record.txt', 'a') as _temp_file:
            _temp_file.write('-'*40 + '\n')
            _temp_file.write('\n')
            _temp_file.write('  Demo hogp \n')
            _temp_file.write('  seed: {} \n'.format(_seed))
            _temp_file.write('  interp_data: {} \n'.format(interp_data))
            _temp_file.write('\n')
            _temp_file.write('-'*40 + '\n')
            _temp_file.flush()

        module_config = {
            'dataset': {
                        'name': 'MolecularDynamic_MF',
                        # 'name': 'NavierStock_mfGent_v1_02',
                        'interp_data': interp_data,

                        'seed': _seed,
                        'train_start_index': 0, 
                        'train_sample': 32, 
                        'eval_start_index': 0,
                        'eval_sample': 128,

                        'inputs_format': ['x[0]'],
                        'outputs_format': ['y[-1]'],

                        'force_2d': False,
                        'x_sample_to_last_dim': False,
                        'y_sample_to_last_dim': True,
                        'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                        },
            }

        ct = controller(HOGP_MODULE, {'max_epoch': 1000}, module_config)
        # print(ct.module.module_config['dataset']['seed'])
        ct.start_train()
        ct.smart_restore_state(-1)
        ct.rc_file.write('---> final result\n')
        ct.rc_file.flush()
        ct.start_eval({'eval state':'final',
                       'module_name':'hogp',
                       'cp_record_file': True})
        ct.rc_file.write('---> end\n\n')
        ct.rc_file.flush()

    ct.clear_record()