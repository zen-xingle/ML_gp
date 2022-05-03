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


if __name__ == '__main__':
    controller_config = {} # use defualt config
    module_config = {
        'dataset': {'name': 'poisson_v4_02',
                    'fidelity': ['low'],
                    'type':'x_2_y',    # x_yl_2_yh, x_2_y
                    'train_start_index': 0, 
                    'train_sample': 64, 
                    'eval_start_index': 0, 
                    'eval_sample':128},
    } # only change dataset config, others use default config
    ct = controller(CIGP_MODULE, controller_config, module_config)
    ct.start_train()
    ct.rc_file.write('---> Finsh the first module\n\n')
    ct.rc_file.flush()

    second_controller_config = {
        'max_epoch': 3000,
    }
    second_module_config = {
        'dataset': {'name': 'poisson_v4_02',
                    'fidelity': ['low','high'],
                    'type':'x_yl_2_yh',    # x_yl_2_yh, x_2_y
                    'train_start_index': 0, 
                    'train_sample': 16,
                    'eval_start_index': 0, 
                    'eval_sample':128},
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