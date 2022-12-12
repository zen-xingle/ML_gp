
import os
import sys

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
demo_name = realpath[-1].rstrip('.py')
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from utils.main_controller import controller
# from module.hogp import HOGP_MODULE
from module.ind_hogp import HOGP_MODULE

interp_data = True

real_dataset = ['FlowMix3D_MF',
                'MolecularDynamic_MF', 
                'plasmonic2_MF', 
                'SOFC_MF',]

gen_dataset = ['poisson_v4_02',
                'burger_v4_02',
                'Burget_mfGent_v5',
                'Burget_mfGent_v5_02',
                # 'Heat_mfGent_v5',
                'Piosson_mfGent_v5',
                'Schroed2D_mfGent_v1',
                'TopOP_mfGent_v5',]

if __name__ == '__main__':
    # for _dataset in real_dataset + gen_dataset:
    for _dataset in ['SOFC_MF']:
        for _seed in [None, 0, 1, 2, 3, 4]:
            module_config = {
                'dataset': {
                            'name': _dataset,
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
                'cuda': True,
                'evaluate_method': ['mae', 'rmse', 'r2', 'gaussian_loss'],
            }
            ct = controller(HOGP_MODULE, {}, module_config, demo_name)
            ct.start_train()

    ct.clear_record()