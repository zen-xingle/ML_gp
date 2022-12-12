import os
import sys
import torch
from copy import deepcopy
realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
demo_name = realpath[-1].rstrip('.py')
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from utils.main_controller import controller
from module.cigp import CIGP_MODULE
from module.cigp_multi_fidelity import CIGP_MODULE_Multi_Fidelity

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

interp_data = True

def non_subset(first_module, second_module):
    from copy import deepcopy
    # torch.dist(first_module.inputs_tr[0], second_module.inputs_tr[0][:,0:first_module.inputs_tr[0].shape[1]])
    f_start_index = first_module.module_config['dataset']['train_start_index']
    f_sample = first_module.module_config['dataset']['train_sample']
    f_input = deepcopy(first_module.inputs_tr[0])
    if first_module.module_config['input_normalize'] is True:
        f_input = first_module.X_normalizer.denormalize(f_input)
    s_start_index = second_module.module_config['dataset']['train_start_index']
    s_sample = second_module.module_config['dataset']['train_sample']
    # assert s_sample == f_sample

    subset_number = max(f_start_index + f_sample - s_start_index, 0)
    subset_number = min(subset_number, s_sample)

    subset_start_index = s_start_index
    s_input = deepcopy(second_module.inputs_tr[0])
    if second_module.module_config['input_normalize'] is True:
        s_input = second_module.X_normalizer.denormalize(s_input)
    s_subset_input = s_input[:subset_number,...]
    # torch.dist(f_input[subset_start_index:,...], s_subset_input[:, :f_input.shape[-1]]) # -> 0
    # update non-subset
    non_subset_input = s_input[subset_number:, :f_input.shape[-1]]
    predict_u, _ = first_module.predict([non_subset_input])
    if second_module.module_config['output_normalize'] is True:
        predict_u = second_module.Y_normalizer.normalize(predict_u)
    new_input_0 = torch.cat([s_subset_input, non_subset_input], dim=0)
    new_input_1 = torch.cat([second_module.inputs_tr[1][:subset_number,:], predict_u], dim=0)
    second_module.inputs_tr[0] = deepcopy(new_input_0)
    second_module.inputs_tr[1] = deepcopy(new_input_1)
    if second_module.module_config['input_normalize'] is True:
        second_module.inputs_tr[0] = second_module.X_normalizer.normalize(second_module.inputs_tr[0])
    _, eval_var = first_module.predict([second_module.inputs_eval[0]])
    second_module.base_var = eval_var


if __name__ == '__main__':
    # for _dataset in real_dataset + gen_dataset:
    for _dataset in ['poisson_v4_02']:
        for _seed in [None, 0, 1, 2, 3, 4]:
            first_fidelity_sample = 32
            controller_config = {'max_epoch': 1000,
                                 'record_file_path': 'NonSubset_LAR.txt'} # use defualt config
            module_config = {
                'dataset': {'name': _dataset,
                            'interp_data': interp_data,

                            'seed': _seed,
                            'train_start_index': 0, 
                            'train_sample': first_fidelity_sample, 
                            'eval_start_index': 0,
                            'eval_sample': 128,

                            'inputs_format': ['x[0]'],
                            'outputs_format': ['y[0]'],

                            'force_2d': True,
                            'x_sample_to_last_dim': False,
                            'y_sample_to_last_dim': False,
                            'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                            },
                'cuda': True,
                'evaluate_method': ['mae', 'rmse', 'r2', 'gaussian_loss'],
                'noise_init': 10.0
            } # only change dataset config, others use default config
            ct = controller(CIGP_MODULE, controller_config, module_config, demo_name)
            ct.start_train()

            second_fidelity_sample = 32
            for subset in [1, 2, 4, 8, 16, 32]:
                second_controller_config = {
                    'max_epoch': 1000,
                }
                second_module_config = {
                    'dataset': {'name': _dataset,
                                'interp_data': interp_data,

                                'seed': _seed,
                                'train_start_index': int(first_fidelity_sample - subset), 
                                'train_sample': second_fidelity_sample, 
                                'eval_start_index': 0,
                                'eval_sample': 128,

                                'inputs_format': ['x[0]', 'y[0]'],
                                'outputs_format': ['y[-1]'],

                                'force_2d': True,
                                'x_sample_to_last_dim': False,
                                'y_sample_to_last_dim': False,
                                'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                                },
                    'res_cigp': {'type_name': 'res_rho'},
                    'lr': {'kernel':0.1, 
                            'optional_param':0.1, 
                            'noise':0.1},
                    'cuda': True,
                    'evaluate_method': ['mae', 'rmse', 'r2', 'gaussian_loss'],
                    'noise_init': 10.0
                }
                second_ct = controller(CIGP_MODULE_Multi_Fidelity, second_controller_config, second_module_config, demo_name)

                # replace ground truth eval data with low fidelity predict
                # check inputs x, this should be 0
                torch.dist(second_ct.module.inputs_eval[0], ct.module.inputs_eval[0])
                # check inputs yl, this should be 0
                torch.dist(second_ct.module.inputs_eval[1], ct.module.outputs_eval[0])
                # check predict yh, as lower as better.
                torch.dist(second_ct.module.inputs_eval[1], ct.module.predict_y)
                second_ct.module.inputs_eval[1] = deepcopy(ct.module.predict_y)
                non_subset(ct.module, second_ct.module)

                second_ct.start_train()


    # second_ct.clear_record()