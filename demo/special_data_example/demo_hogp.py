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
    for _seed in [None]:
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
            'dataset': {'name': 'Schroed2D_mfGent_v1',
                        'fidelity': ['low'],
                        'type':'x_2_y',    # x_yl_2_yh, x_2_y
                        'train_start_index': 0, 
                        'train_sample': 64, 
                        'eval_start_index': 0,
                        'eval_sample': 128,
                        'seed': _seed,
                        'interp_data': interp_data,
                        },
            # kernel number = len(sample.shape) + 1
            'kernel': {
                        'K1': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
                        'K2': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
                        'K3': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
                        'K4': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
                        },
            # len(grid_size) = len(sample.shape)
            'grid_config': {'grid_size': [-1, -1, -1], 
                            'type': 'fixed', # learnable, fixed
                            'dimension_map': 'identity', # latent space: identity, learnable_identity, learnable_map
                            },
        }
        ct = controller(HOGP_MODULE, {}, module_config)
        ct.start_train()
        ct.smart_restore_state(-1)
        ct.rc_file.write('---> final result\n')
        ct.rc_file.flush()
        ct.start_eval({'eval state':'final'})
        ct.rc_file.write('---> end\n\n')
        ct.rc_file.flush()