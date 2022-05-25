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
from module.dc_cigp import DC_CIGP_MODULE

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
interp_data=True

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
    non_subset_input = deepcopy(s_input[subset_number:, :f_input.shape[-1]])
    if second_module.module_config['input_normalize'] is True:
        non_subset_input = second_module.X_normalizer_0.denormalize(non_subset_input)
    predict_u, _ = first_module.predict([non_subset_input])
    if second_module.module_config['output_normalize'] is True:
        predict_u = second_module.Y_normalizer.normalize(predict_u)
    if second_module.pca_model is not None:
        origin_y = second_module.pca_model.recover([deepcopy(s_input[subset_number:, f_input.shape[-1]:]), second_module.outputs_tr[0][subset_number:, :]])
        _temp_record = second_module.pca_model.project([predict_u, origin_y[1]])
    s_input[subset_number:, f_input.shape[-1]:] = _temp_record[0]
    second_module.inputs_tr[0] = s_input

if __name__ == '__main__':
    # for _dataset in real_dataset + gen_dataset:
    for _dataset in ['poisson_v4_02']:
        for _seed in [None, 0, 1, 2, 3, 4]:
            first_fidelity_sample = 32
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

            controller_config = {'max_epoch':1000} # use defualt config
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
            second_fidelity_sample = 32
            for subset in [1, 2, 4, 8, 16, 32]:
                with open('record.txt', 'a') as _temp_file:
                    _temp_file.write('\n'+ '-'*10 + '>\n')
                    _temp_file.write('NAR for {} subset samples\n'.format(subset))
                    _temp_file.write('-'*3 + '> Training x,yl -> yh part\n\n')
                    _temp_file.flush()

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

                                'inputs_format': ['x[0]','y[0]'],
                                'outputs_format': ['y[-1]'],

                                'force_2d': True,
                                'x_sample_to_last_dim': False,
                                'y_sample_to_last_dim': False,
                                'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                                },
                    'pca': {'type': 'listPCA', 
                            'r': 0.99, }, # listPCA, resPCA_mf,
                    'noise_init' : 1000.,
                }
                second_ct = controller(DC_CIGP_MODULE, controller_config, second_module_config)
                # replace ground truth eval data with low fidelity predict
                # check inputs x
                x_dim = ct.module.inputs_eval[0].shape[1]
                torch.dist(second_ct.module.inputs_eval[0][:,0:x_dim], ct.module.inputs_eval[0])
                # check inputs y
                torch.dist(second_ct.module.inputs_eval[1], ct.module.outputs_eval[0])
                # check predict y
                torch.dist(second_ct.module.inputs_eval[1], ct.module.predict_y)
                second_ct.module.inputs_eval[1] = ct.module.predict_y
                non_subset(ct.module, second_ct.module)

                second_ct.start_train()
                second_ct.smart_restore_state(-1)
                second_ct.rc_file.write('---> final result\n')
                second_ct.rc_file.flush()
                second_ct.start_eval({'eval state':'final',
                        'module_name':'DC_cigp',
                        'subset': str(subset),
                        'cp_record_file': True})
                second_ct.rc_file.write('---> end\n\n')
                second_ct.rc_file.flush()

    second_ct.clear_record()