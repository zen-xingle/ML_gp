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


interp_data = False

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
    new_input_1 = torch.cat([second_module.inputs_tr[1][...,:subset_number], predict_u], dim=-1)
    second_module.inputs_tr[0] = deepcopy(new_input_0)
    second_module.inputs_tr[1] = deepcopy(new_input_1)
    if second_module.module_config['input_normalize'] is True:
        second_module.inputs_tr[0] = second_module.X_normalizer.normalize(second_module.inputs_tr[0])


if __name__ == '__main__':
    # for _dataset in real_dataset + gen_dataset:
    for _dataset in ['poisson_v4_02']:
        for _seed in [None, 0, 1, 2, 3, 4]:
            first_fidelity_sample = 32
            with open('record.txt', 'a') as _temp_file:
                _temp_file.write('-'*40 + '\n')
                _temp_file.write('\n')
                _temp_file.write('  Demo GAR \n')
                _temp_file.write('  seed: {} \n'.format(_seed))
                _temp_file.write('  interp_data: {} \n'.format(interp_data))
                _temp_file.write('\n')
                _temp_file.write('-'*40 + '\n')
                _temp_file.write('-'*3 + '> Training x -> yl part\n')
                _temp_file.flush()

            # ================================================================
            # Training x -> yl part

            controller_config = {
                'max_epoch': 1000
            } # use defualt config

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

                            'force_2d': False,
                            'x_sample_to_last_dim': False,
                            'y_sample_to_last_dim': True,
                            'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                            },

                # 'lr': {'kernel':0.01, 
                #         'optional_param':0.01, 
                #         'noise':0.01},
                #     # kernel number as dim + 1
                #     'kernel': {
                #             'K1': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
                #             # 'K1': {'Local_Periodic': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
                #             # 'K2': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
                #             # 'K3': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
                #             },
                #     'exp_restrict': False,
                #     'input_normalize': True,
                #     'output_normalize': True,
                    'noise_init' : 10.,
                #     'grid_config': {'grid_size': [-1], 
                #                     'type': 'fixed', # learnable, fixed
                #                     'dimension_map': 'identity', # latent space: identity, learnable_identity, learnable_map
                #                     'auto_broadcast_grid_size': True,
                #                     'squeeze_to_01': False,
                #                     },
                
                } # only change dataset config, others use default config
            
            ct = controller(HOGP_MODULE, controller_config, module_config)
            ct.start_train()
            ct.smart_restore_state(-1)
            ct.rc_file.write('---> final result')
            ct.rc_file.flush()
            ct.start_eval({'eval state':'final'})
            ct.rc_file.write('-'*10 + '> finish x-yl training\n\n')
            ct.rc_file.flush()

            # ================================================================
            # Training x,yl -> yh part
            # exit()
            second_fidelity_sample = 32
            for subset in [1, 2, 4, 8, 16, 32]:
                with open('record.txt', 'a') as _temp_file:
                    _temp_file.write('\n'+ '-'*10 + '>\n')
                    _temp_file.write('GAR for {} subset samples\n'.format(subset))
                    _temp_file.write('-'*3 + '> Training x,yl -> yh part\n\n')
                    _temp_file.flush()

                mfct_module_config = {
                    'dataset': {'name': _dataset,
                                'interp_data': interp_data,

                                # preprocess
                                'seed': _seed,
                                'train_start_index': int(first_fidelity_sample - subset), 
                                'train_sample': second_fidelity_sample, 
                                'eval_start_index': 0, 
                                'eval_sample':128,
                                
                                'inputs_format': ['x[0]', 'y[0]'],
                                'outputs_format': ['y[-1]'],

                                'force_2d': False,
                                'x_sample_to_last_dim': False,
                                'y_sample_to_last_dim': True,
                                'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                                },

                    # 'lr': {'kernel':0.01, 
                    #     'optional_param':0.01, 
                    #     'noise':0.01},
                    # # kernel number as dim + 1
                    # 'kernel': {
                    #         'K1': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
                    #         # 'K1': {'Periodic': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
                    #         # 'K2': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
                    #         # 'K3': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
                    #         },
                    # 'exp_restrict': True,
                    # 'input_normalize': True,
                    # 'output_normalize': True,
                    'noise_init' : 10.,
                    # 'grid_config': {'grid_size': [-1], 
                    #                 'type': 'fixed', # learnable, fixed
                    #                 'dimension_map': 'identity', # latent space: identity, learnable_identity, learnable_map
                    #                 'auto_broadcast_grid_size': True,
                    #                 'squeeze_to_01': False,
                    #                 },
                } # only change dataset config, others use default config

                mfct = controller(HOGP_MF_MODULE, controller_config, mfct_module_config)
                
                with torch.no_grad():
                    # use x->yl_predict for test x+yl -> yh
                    mfct.module.inputs_eval[1] = ct.module.predict_y
                    pass
                non_subset(ct.module, mfct.module)

                mfct.start_train()
                mfct.smart_restore_state(-1)
                mfct.rc_file.write('---> final result\n')
                mfct.rc_file.flush()
                mfct.start_eval({'eval state':'final',
                                'module_name': 'GAR',
                                'subset': str(subset),
                                'cp_record_file': True})
                mfct.rc_file.write('---> end\n\n')
                mfct.rc_file.flush()

    mfct.clear_record()