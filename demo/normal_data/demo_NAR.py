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

real_dataset = ['FlowMix3D_MF',
                'MolecularDynamic_MF', 
                'plasmonic2_MF', 
                'SOFC_MF',]

gen_dataset = ['poisson_v4_02',
                'burger_v4_02',
                'Burget_mfGent_v5',
                'Burget_mfGent_v5_02',
                'Heat_mfGent_v5',
                'Piosson_mfGent_v5',
                'Schroed2D_mfGent_v1',
                'TopOP_mfGent_v5',]
interp_data=False

if __name__ == '__main__':
    for _dataset in ['TopOP_mfGent_v5',]:
        for _seed in [4]:

            controller_config = {'max_epoch': 1000,
                                 'record_file_path': 'NAR.txt'} # use defualt config
            module_config = {
                'dataset': {'name': _dataset,
                            'interp_data': interp_data,

                            'seed': _seed,
                            'train_start_index': 0, 
                            'train_sample': 16, 
                            'eval_start_index': 0,
                            'eval_sample': 128,

                            # 'inputs_format': ['x[0]'],
                            # 'outputs_format': ['y[0]'],
                            'inputs_format': ['(x[0] - x[0].mean()) / x[0].std()'],
                            'outputs_format': ['(y[0] - y[0].mean()) / y[0].std()'],

                            'force_2d': True,
                            'x_sample_to_last_dim': False,
                            'y_sample_to_last_dim': False,
                            'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                            },
                'cuda': False,
                'evaluate_method': ['mae', 'rmse', 'r2', 'gaussian_loss'],
                'noise_init' : 10.,
                'exp_restrict': True,
            } # only change dataset config, others use default config
            ct = controller(CIGP_MODULE, controller_config, module_config)
            ct.start_train()

            for _sample in [4,8,16,32]:
                second_module_config = {
                    'dataset': {'name': _dataset,
                                'interp_data': interp_data,

                                'seed': _seed,
                                'train_start_index': 0, 
                                'train_sample': _sample, 
                                'eval_start_index': 0,
                                'eval_sample': 128,

                                # 'inputs_format': ['np.concatenate([x[0],y[0]], axis=1)'],
                                # 'outputs_format': ['y[-1]'],

                                # 'inputs_format': ['np.concatenate([(x[0] - x[0].mean()) / x[0].std(),y[0]], axis=1)'],
                                # 'outputs_format': ['y[-1]'],

                                'inputs_format': ['np.concatenate([(x[0] - x[0].mean()) / x[0].std(), (y[0] - y[0].mean()) / y[0].std()], axis=1)'],
                                'outputs_format': ['(y[-1] - y[-1].mean()) / y[-1].std()'],

                                'force_2d': True,
                                'x_sample_to_last_dim': False,
                                'y_sample_to_last_dim': False,
                                'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                                },
                    'cuda': False,
                    'evaluate_method': ['mae', 'rmse', 'r2', 'gaussian_loss'],
                    'noise_init' : 10.0,
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
                if hasattr(ct.module, 'predict_var'):
                    second_ct.module.base_cigp = ct.module

                second_ct.start_train()

    # second_ct.clear_record()