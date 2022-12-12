import os
import sys

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
demo_name = realpath[-1].rstrip('.py')
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from utils.main_controller import controller
from nn_net.MF_BNN import DeepMFnet


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

if __name__ == '__main__':
    # for _dataset in real_dataset + gen_dataset:
    for _dataset in ['burger_v4_02']:
        for _seed in [0,1,2,3,4]:
            first_fidelity_sample = 32
            
            for second_fidelity_sample in [4, 8, 16, 32]:
            # for start_index in [total_sample-1, total_sample-4, total_sample-8, total_sample-16, total_sample-32]:
                subset = 0.5 * second_fidelity_sample
                module_config = {
                    'dataset': {'name': _dataset,
                    'interp_data': interp_data,

                    'seed': _seed,
                    'train_start_index': 0, 
                    'train_sample': 32, 
                    'eval_start_index': 0,
                    'eval_sample': 128,

                    'inputs_format': ['x[0]', 'x[0]'],
                    'outputs_format': ['y[0]','y[-1]'],

                    'force_2d': True,
                    'x_sample_to_last_dim': False,
                    'y_sample_to_last_dim': False,
                    'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                    },
                    'second_fidelity_sample': second_fidelity_sample,
                    'second_fidelity_start_index': int(first_fidelity_sample - subset),
                    'non_subset': True
                }
                ct = controller(DeepMFnet, {}, module_config, demo_name)
                ct.start_train()

    ct.clear_record()