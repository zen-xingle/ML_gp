import os
import sys

import time
import datetime
import torch
import numpy as np

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from modules.nn_net.dmfal.dmfal import DeepMFnet
from utils import *
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


def dmfal_test(dataset, exp_config):
    # setting record
    recorder = exp_config['recorder']
    fidelity_num = exp_config['fidelity_num']
    start_time = time.time()

    # get dataset
    # train_inputs, train_outputs, eval_inputs, eval_outputs, source_shape = dataset
    train_inputs, train_outputs, eval_inputs, eval_outputs = dataset

    from utils.normalizer import Dateset_normalize_manager
    data_norm_manager = Dateset_normalize_manager(train_inputs, train_outputs)
    train_inputs, train_outputs = data_norm_manager.normalize_all(train_inputs, train_outputs)
    eval_inputs, _ = data_norm_manager.normalize_all(eval_inputs, eval_outputs)

    dmfal_config = {
        # according to original inplement
        # h_w, h_d determine laten dim
        # net_param
        # 'M': 2,
        'nn_param': {
            'hlayers_w': [40],
            'hlayers_d': [2],
            'base_dim': [32],
            'activation': 'relu', # ['tanh','relu','sigmoid']
            # 'out_shape': [(100,1000), (100, 2000)],
            # 'in_shape': [(100, 5)],
        },
    }

    _in_shape = []
    for _i in range(len(train_inputs)):
        _in_shape.append(train_inputs[_i].shape)
    dmfal_config['nn_param']['in_shape'] = _in_shape
    
    _out_shape = []
    for _i in range(fidelity_num):
        _out_shape.append(train_outputs[_i].shape)
    dmfal_config['nn_param']['out_shape'] = _out_shape

    # extend nn_param as fidilety len
    dmfal_config['nn_param']['hlayers_w'] = dmfal_config['nn_param']['hlayers_w'] * fidelity_num
    dmfal_config['nn_param']['hlayers_d'] = dmfal_config['nn_param']['hlayers_d'] * fidelity_num
    dmfal_config['nn_param']['base_dim'] = dmfal_config['nn_param']['base_dim'] * fidelity_num

    dmfal_model = DeepMFnet(dmfal_config)

    # init optimizer, optimizer is also outsider of the model
    lr = 0.001
    params = dmfal_model.get_train_params()['params']
    optimizer = torch.optim.Adam([{'params': _v, 'lr': lr} for _v in params])
    
    max_epoch=exp_config['max_epoch']
    for epoch in range(max_epoch):
        optimizer.zero_grad()
        loss = dmfal_model.compute_loss(train_inputs, train_outputs)
        print('epoch {}/{}, loss_nll: {}'.format(epoch+1, max_epoch, loss.item()), end='\r')
        loss.backward(retain_graph=True)
        optimizer.step()

    # predict
    predict_y = dmfal_model.predict(eval_inputs)
    predict_y = data_norm_manager.denormalize_output(predict_y, fidelity_num-1)
    # plot_result(eval_outputs[-1], predict_y, source_shape)

    from utils.performance_evaluator import performance_evaluator
    eval_result = performance_evaluator(eval_outputs[-1], predict_y, ['rmse', 'r2'])
    eval_result['time'] = time.time()-start_time
    eval_result['train_sample_num'] = train_inputs[0].shape[0]
    recorder.record(eval_result)    



if __name__ == '__main__':
    # 'FlowMix3D_MF', 'MolecularDynamic_MF','SOFC_MF',
    # 'Poisson_mfGent_v5', 'Heat_mfGent_v5', 'Burget_mfGent_v5_15', 'TopOP_mfGent_v6', 'plasmonic2_MF'
    # 'maolin1','maolin5','maolin6','maolin7', 'maolin8', 'borehole', 'branin', 'currin'
    data_list = [ 'plasmonic2_MF']
    fidelity_num = 5
    evaluation_num = 128
    mis_rate = 1

    for data_name in data_list:
        for seed in [0,1,2,3,4]:
            exp_name = os.path.join('exp', 'dmfal', data_name, 'mis_' + str(mis_rate), 'result.txt')

            recorder = MLGP_recorder(exp_name, overlap=True)
            recorder.register(['train_sample_num','rmse', 'r2', 'time'])
            exp_config = {
                'max_epoch': 100,
                'recorder': recorder,
                'fidelity_num': 5,
            }

            dataset = list(data_preparation(data_name, fidelity_num, seed, 128))
            dataset.append([-1, *dataset[1][0].shape[1:]])


            train_sample_num = [32, 64, 96, 128]
            for _num in train_sample_num:    
                # initial random mask
                mask_matrix = []
                sub_dataset = [[], [], [], []]
                xtr_list = []
                ytr_list = []
                yte_list = []
                for fid in range(fidelity_num):
                    mask_tem = np.zeros(128)
                    ones_num = int(_num * pow(mis_rate, fid))
                    mask_tem[:ones_num] = 1
                    np.random.seed(seed * fidelity_num + fid)
                    np.random.shuffle(mask_tem)
                    mask_matrix.append(mask_tem)
                    
                    xtr_exist = []
                    ytr_exist = []
                    for index in range(evaluation_num):
                        if mask_tem[index] == 1:
                            xtr_exist.append(dataset[0][0][index])
                            ytr_exist.append(dataset[1][fid][index])

                    xtr_exist = torch.stack(xtr_exist)
                    ytr_exist = torch.stack(ytr_exist)

                    xtr_list.append(xtr_exist)
                    ytr_list.append(ytr_exist)
                    yte_list.append(dataset[3][fid][:max(evaluation_num, dataset[3][0].shape[0])])
                sub_dataset[0] = xtr_list
                sub_dataset[1] = ytr_list
                sub_dataset[3] = yte_list
                sub_dataset[2] = dataset[2][:max(evaluation_num, dataset[3][0].shape[0])]

                
                dmfal_test(sub_dataset, exp_config)

            recorder.to_csv(seed)