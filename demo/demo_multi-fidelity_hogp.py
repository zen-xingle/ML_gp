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


if __name__ == '__main__':
    controller_config = {
        'max_epoch': 1000
    } # use defualt config

    ct_module_config = {
        'dataset': {'name': 'burger_v4_02',
                    'fidelity': ['low'],
                    'type':'x_2_y',    # x_yl_2_yh, x_2_y
                    'train_start_index': 0, 
                    'train_sample': 32, 
                    'eval_start_index': 0,
                    'eval_sample':128,
                    'seed': None},
    } # only change dataset config, others use default config
    ct = controller(HOGP_MODULE, controller_config, ct_module_config)
    ct.start_train()
    ct.smart_restore_state(-1)
    ct.rc_file.write('---> final result')
    ct.rc_file.flush()
    ct.start_eval({'eval state':'final'})
    ct.rc_file.write('---> end\n\n')
    ct.rc_file.flush()

    for _sample in [4,8,16,32,64]:

        mfct_module_config = {
            'dataset': {'name': 'burger_v4_02',
                        'fidelity': ['low','high'],
                        'type':'x_yl_2_yh',    # x_yl_2_yh, x_2_y
                        'connection_method': 'res_mapping',
                        'train_start_index': 0, 
                        'train_sample': _sample, 
                        'eval_start_index': 0,
                        'eval_sample':128,
                        'seed': None},
        } # only change dataset config, others use default config

        mfct = controller(HOGP_MF_MODULE, controller_config, mfct_module_config)
        
        with torch.no_grad():
            # use x->yl_predict for test x+yl -> yh
            mfct.module.inputs_eval[1] = ct.module.predict_y
            pass

        mfct.start_train()
        mfct.smart_restore_state(-1)
        mfct.rc_file.write('---> final result')
        mfct.rc_file.flush()
        mfct.start_eval({'eval state':'final'})
        mfct.rc_file.write('---> end\n\n')
        mfct.rc_file.flush()