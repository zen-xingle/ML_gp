import os
import sys

import torch
import numpy as np

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('v2')+1])
sys.path.append(realpath)

from modules.gp_module.cigp import CIGP_MODULE


def prepare_data():
    # prepare data
    x = np.load('./data/sample/input.npy')
    # y = np.load('./data/sample/output_fidelity_1.npy')
    y = np.load('./data/sample/output_fidelity_2.npy')
    data_len = x.shape[0]
    source_shape = [-1, *y.shape[1:]]

    x = torch.tensor(x).float()
    y = torch.tensor(y).float()
    # cigp only support 2d input (batch, dim)
    x = x.reshape(data_len, -1)
    y = y.reshape(data_len, -1)
    train_inputs = [x[:128,:]]
    train_outputs = [y[:128,:]]
    eval_inputs = [x[128:,:]]
    eval_outputs = [y[128:,:]]
    return train_inputs, train_outputs, eval_inputs, eval_outputs, source_shape


def plot_result(ground_true_y, predict_y, src_shape):
    # plot result
    from visualize_tools.plot_field import plot_container
    data_list = [ground_true_y, predict_y[0], (ground_true_y - predict_y[0]).abs()]
    data_list = [_d.reshape(src_shape) for _d in data_list]
    label_list = ['groundtruth', 'predict', 'diff']
    pc = plot_container(data_list, label_list, sample_dim=0)
    pc.plot()


def gp_model_block_test():
    # get dataset
    train_inputs, train_outputs, eval_inputs, eval_outputs, source_shape = prepare_data()

    # normalizer now is outsider of the model
    from utils.normalizer import Dateset_normalize_manager
    data_norm_manager = Dateset_normalize_manager(train_inputs, train_outputs)

    # init model
    cigp = CIGP_MODULE()

    # init gp_model_block
    from gp_model_block import GP_model_block
    gp_model_block = GP_model_block()
    gp_model_block.dnm = data_norm_manager
    gp_model_block.gp_model = cigp

    # init optimizer, optimizer is also outsider of the model
    lr_dict = {'kernel': 0.01, 'noise': 0.01}
    params_dict = gp_model_block.get_train_params()
    optimizer_dict = [{'params': params_dict[_key], 'lr': lr_dict[_key]} for _key in params_dict.keys()]
    optimizer = torch.optim.Adam(optimizer_dict)
    
    # start training
    max_epoch=100
    for epoch in range(max_epoch):
        optimizer.zero_grad()
        loss = gp_model_block.compute_loss(train_inputs, train_outputs)
        print('epoch {}/{}, loss_nll: {}'.format(epoch+1, max_epoch, loss.item()), end='\r')
        loss.backward()
        optimizer.step()

    # eval
    print('\n')
    gp_model_block.eval()
    predict_y = gp_model_block.predict(eval_inputs)
    plot_result(eval_outputs[0], predict_y, source_shape)



if __name__ == '__main__':
    gp_model_block_test()