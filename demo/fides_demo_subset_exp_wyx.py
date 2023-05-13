import os
import sys

import datetime
import time
import torch
import numpy as np

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from modules.gp_module.fides_dec_beta import fides_dec
from modules.gp_module.cigp import CIGP_MODULE
from utils.mlgp_result_record import MLGP_recorder, MLGP_record_parser
from utils.prepare_data import data_preparation


def prepare_data():
    # prepare data
    x = np.load('./data/sample/input.npy')
    y0 = np.load('./data/sample/output_fidelity_0.npy')
    y1 = np.load('./data/sample/output_fidelity_1.npy')
    y2 = np.load('./data/sample/output_fidelity_2.npy')
    data_len = x.shape[0]
    source_shape = [-1, *y0.shape[1:]]

    x = torch.tensor(x).float()
    outputs = [torch.tensor(y0).float(), torch.tensor(y1).float(), torch.tensor(y2).float()]
    # outputs = [torch.tensor(y0).float(), torch.tensor(y2).float()]
    outputs = [y.reshape(data_len, -1) for y in outputs]

    train_inputs = [x[:128,:]]
    train_outputs = [y[:128,:] for y in outputs]
    eval_inputs = [x[128:,:]]
    eval_outputs = [y[128:,:] for y in outputs]

    return train_inputs, train_outputs, eval_inputs, eval_outputs, source_shape


def plot_result(ground_true_y, predict_y, src_shape):
    # plot result
    from visualize_tools.plot_field import plot_container
    from utils.type_define import GP_val_with_bar
    if isinstance(predict_y[0], GP_val_with_bar):
        data_list = [ground_true_y, predict_y[0].get_mean(), (ground_true_y - predict_y[0].get_mean()).abs()]
    else:
        data_list = [ground_true_y, predict_y, (ground_true_y - predict_y).abs()]
    data_list = [_d.reshape(src_shape) for _d in data_list]
    label_list = ['groundtruth', 'predict', 'diff']
    pc = plot_container(data_list, label_list, sample_dim=0)
    pc.plot()


def gp_model_block_test(dataset, exp_config):
    # setting record
    recorder = exp_config['recorder']
    fidelity_num = exp_config['fidelity_num']
    train_samples_num = exp_config['train_samples_num']
    test_samples_num = exp_config['test_samples_num']
    seed = exp_config['seed']
    start_time = time.time()

    # get dataset
    train_inputs, train_outputs, eval_inputs, eval_outputs = dataset

    train_num = [int(train_samples_num * pow(dec_rate, i)) for i in range(fidelity_num)]
    eval_inputs = eval_inputs[0][:test_samples_num]

    m_fid = fides_dec(train_inputs,
                    train_outputs, 
                    eval_inputs,
                    train_begin_index = 0, 
                    train_num = train_num, 
                    fidelity_num = fidelity_num,
                    niteration = 100,
                    learning_rate = 0.01,
                    seed = seed,
                    normal_y_mode = 0)
    yte_mean, yte_var = m_fid.train_mod()
    yte_test = eval_outputs[fidelity_num - 1][0: 128]

    from utils.performance_evaluator import performance_evaluator
    eval_result = performance_evaluator(yte_test, yte_mean, ['rmse', 'r2'])
    eval_result['time'] = time.time()-start_time
    eval_result['train_sample_num'] = train_samples_num
    recorder.record(eval_result)


if __name__ == '__main__':

    # 'Poisson_mfGent_v5', 'Heat_mfGent_v5', 'Burget_mfGent_v5_15', 'TopOP_mfGent_v6', 'plasmonic2_MF'
    # 'maolin1','maolin5','maolin6','maolin7', 'maolin8'
    # 'borehole', 'branin', 'currin'
    data_list = ['maolin1','maolin5','maolin6','maolin7', 'maolin8', 'borehole', 'branin', 'currin']
    fidelity_num = 5
    evaluation_num = 128
    dec_rate = 0.5
    
    for data_name in data_list:
        for seed in [1,2,3,4]:
            # exp_name = os.path.join('exp', 'fides', 'toy_data', str(datetime.date.today()), 'result.txt')
            exp_name = os.path.join('exp', 'fides', data_name, 'dec_' + str(dec_rate), 'result.txt')
            # exp_name = os.path.join('exp', 'fides', data_name, 'fidelity_' + str(fidelity_num), 'result.txt')
            recorder = MLGP_recorder(exp_name, overlap=True)
            recorder.register(['train_sample_num','rmse', 'r2', 'time'])

            # dataset = prepare_data()
            dataset = list(data_preparation(data_name, fidelity_num, seed, 128))
            dataset.append([-1, *dataset[1][0].shape[1:]])

            dec_rate = 0.5
            train_sample_num = [32, 64, 96, 128]
            for _num in train_sample_num:
                exp_config = {
                    'recorder': recorder,
                    'fidelity_num': fidelity_num,
                    'dec_rate': dec_rate, 
                    'train_samples_num': _num, 
                    'test_samples_num': 128,
                    'seed': seed,
                }
                # dataset.append(dataset[-1])         # last one is shape
                gp_model_block_test(dataset[:-1], exp_config)

            recorder.to_csv(seed)
