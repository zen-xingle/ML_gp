import torch
import time
import os
import shutil
from copy import deepcopy
import sys

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from utils import mlgp_result_record
from utils.dict_tools import smart_update


default_controller_config = {
    'batch_size': 1, # not implement
    'check_point': [1, 10, 100, 300, 500, 1000, 2500, 5000, 7500, 10000],
    'eval_batch_size': 1, # not implement
    'record_step': 50,
    'max_epoch': 1000,
}


class controller(object):
    def __init__(self, module, controller_config, module_config) -> None:
        self.module_config = module_config
        self.controller_config = smart_update(default_controller_config, controller_config)
        
        self.module = module(module_config)
        if self.module.module_config['cuda']:
            self.module.cuda()
            inputs_name = ['inputs_eval', 'inputs_tr', 'outputs_eval','outputs_tr']
            for _key in inputs_name:
                _p_list = getattr(self.module, _key)
                for i, _p in enumerate(_p_list):
                    _p_list[i] = _p.cuda()

        self.result_list = {}

        self.rc_file = mlgp_result_record.MLGP_recorder('./record.txt',
                                                        overlap=True, 
                                                        append_info={
                                                            'module': str(module),
                                                            'controller_config': self.controller_config, 
                                                            'module_config':self.module.module_config
                                                            })
        self.rc_file.register(['epoch', *self.module.module_config['evaluate_method'], 'time'])

        self.init_time = time.time()
        # self.discriptor = None
        # pass

    def start_train(self):
        for i in range(self.controller_config['max_epoch']):
            self.module.train()

            print('train {}/{}'.format(i, self.controller_config['max_epoch']), end='\r')

            # 每达到record_step步数时, 暂存模型状态
            # if i%self.controller_config['record_step'] == 0 and i!= 0:
                # self.record_state()
                # print('step: {} record state'.format(i))
                # _result = self.start_eval()

            # 达到check_point时, 在验证集上测试效果
            if (i+1) in self.controller_config['check_point']:
                _result = self.start_eval()
                _result['time'] = int(time.time() - self.init_time)
                _result['epoch'] = i+1
                self.rc_file.record(_result)

        
        # if 'quite_reason' not in self.result_list:
        #     self.result_list['quite_reason'] = 'reach max_epoch'
        # print_result(self.result_list)

    def start_eval(self):
        result = self.module.eval()
        return result

    def check_nan_inf(self):
        for _data in self.module.get_params_need_check():
            if torch.isnan(_data).any():
                return True
            if torch.isinf(_data).any():
                return True
        return False

    def record_state(self):
        pass

    def restore_state(self, index):
        pass

    def data_query(self):
        pass
        
    def clear_record(self):
        open('record.txt', 'w').close()