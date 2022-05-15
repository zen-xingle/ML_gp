import torch
import time
import os
import shutil
from copy import deepcopy


def print_result(result):
    for key, _value in result.items():
        if isinstance(_value, dict):
            print('\nfor {}:'.format(key))
            for _dkey, _dvalue in _value.items():
                if isinstance(_dvalue, float):
                    print('{}: '.format(_dkey), '%.5f' % _dvalue)
                else:
                    print('{}: {}'.format(_dkey, _dvalue))
        else:
            print('{}: {}'.format(key, _value))

class result_discriptor:
    def __init__(self, indict, decimal_reserve=5):
        self.interval_space = 6
        self.decimal_reserve = decimal_reserve
        _len_list = []
        self._keys = []
        for key, _value in indict.items():
            self._keys.append(key)
            _len_list.append(max(len(key)+ self.interval_space, 12))
        self.fmt = ['{:'+str(_len)+'}' for _len in _len_list]
        self.fmt = '\t'.join(self.fmt)
        self.first_time = True

    def discript(self, _dict):
        _value_list = ['' for _ in range(len(self._keys))]
        others = ''
        for key, _value in _dict.items():
            if key not in self._keys:
                others = others + '{} : {};'.format(key,_value)
            elif isinstance(_value, int):
                _value_list[self._keys.index(key)] = str(_value)
            else:
                _value_list[self._keys.index(key)] = '%.{}f'.format(self.decimal_reserve) % _value

        word = self.fmt.format(*_value_list)
        if others != '':
            word = word + ' '*5 + others
        if self.first_time:
            self.first_time = False
            word = self.fmt.format(*self._keys) + '\n' + word
        return word

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
        default_controller_config.update(controller_config)
        self.controller_config = default_controller_config
        
        self.module = module(module_config)
        self.param_record_list = []
        self.param_record_list_for_epoch = []
        self.result_list = {}

        self.rc_file = None

        self.init_time = time.time()
        self.discriptor = None
        pass

    def _get_time(self):
        _now = time.time()
        period = _now - self.time
        self.time = _now
        return period

    def start_train(self):
        for i in range(self.controller_config['max_epoch']):
            self.module.train()

            # 检验nan, 
            nan_state = self.check_nan_inf()
            if nan_state:
                self.rc_file.write('  epoch {} reach nan state\n'.format(i+1))
                self.rc_file.flush()
                # self.restore_state(-1)
                # restore_epoch = str((i+1)// self.controller_config['record_step'] *self.controller_config['record_step'])
                # _result = self.start_eval({'epoch':'{}+nan'.format(restore_epoch)})
                # self.result_list['checkpoint' + str(restore_epoch)] = _result
                self.result_list['quite_reason'] = 'nan'
                break

            print('train {}/{}'.format(i, self.controller_config['max_epoch']), end='\r')

            # 每达到record_step步数时, 暂存模型状态
            if i%self.controller_config['record_step'] == 0 and i!= 0:
                self.record_state()
                # print('step: {} record state'.format(i))
                # _result = self.start_eval()

            # 达到check_point时, 在验证集上测试效果
            if (i+1) in self.controller_config['check_point']:
                _result = self.start_eval({'epoch':i+1})
                self.result_list['checkpoint' + str(i+1)] = _result
                self.record_state()
                self.param_record_list_for_epoch.append(deepcopy(self.param_record_list[-1]))

        
        if 'quite_reason' not in self.result_list:
            self.result_list['quite_reason'] = 'reach max_epoch'
        print_result(self.result_list)

    def start_eval(self, additional_dict=None):
        result = self.module.eval()
        if additional_dict is not None:
            additional_dict.update(result)
            result = deepcopy(additional_dict)

        result['time'] = int(time.time() - self.init_time)
        if self.discriptor is None:
            self.discriptor = result_discriptor(result)
        _word = self.discriptor.discript(result)
        self.record_to_file(_word)
        if additional_dict is not None and 'eval state' in additional_dict:
            need = ['module_name', 'eval state', 'cp_record_file']
            save_to_csv = True
            for key in need:
                if key not in additional_dict:
                    save_to_csv = False
                    break
            if additional_dict['eval state'] != 'final':
                save_to_csv = False
            if save_to_csv == True:
                dir_name = './exp/{}'.format(additional_dict['module_name'])
                dataset_name = self.module.module_config['dataset']['name']

                # TODO fix and remove this
                if 'random_shuffle_seed' in self.module.module_config['dataset']:
                    seed = 'Seed[{}]'.format(str(self.module.module_config['dataset']['random_shuffle_seed']))
                else:
                    seed = 'Seed[{}]'.format(str(self.module.module_config['dataset']['seed']))
                intep = 'Interp[{}]'.format(str(self.module.module_config['dataset']['interp_data']))
                if not os.path.exists('./exp'):
                    os.mkdir('./exp')
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                dir_name = os.path.join(dir_name, dataset_name)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)

                csv_file_path = '{}/{}_{}_{}.csv'.format(dir_name, dataset_name, seed, intep)
                if not os.path.exists(csv_file_path):
                    init_csv = True
                else:
                    init_csv = False

                with open(csv_file_path, 'a') as f:
                    _keys_line = []
                    result_line = []
                    for key, value in result.items():
                        _keys_line.append(key)
                        result_line.append(str(value))
                    _keys_line.append('train_sample')
                    result_line.append(str(self.module.module_config['dataset']['train_sample']))
                    if init_csv is True:
                        f.write(','.join(_keys_line) + '\n')
                    f.write(','.join(result_line) + '\n')

                if additional_dict['cp_record_file'] is True:
                    shutil.copy('./record.txt', os.path.join(dir_name, 'Interp[{}]_record.txt'.format(str(self.module.module_config['dataset']['interp_data']))))
        return result

    def _train(self):
        pass

    def _eval(self):
        pass

    def check_nan_inf(self):
        for _data in self.module.get_params_need_check():
            if torch.isnan(_data).any():
                return True
            if torch.isinf(_data).any():
                return True
        return False

    def record_state(self):
        param_dict = self.module.save_state()
        # TODO check memory consumption

        for i,_param in enumerate(param_dict):
            param_dict[i] = _param.data

        # only save one record
        if len(self.param_record_list) > 0:
            self.param_record_list.pop()
        self.param_record_list.append(deepcopy(param_dict))
        # print('save:', param_dict)


    def restore_state(self, index):
        # print('load:', self.param_record_list[index])
        if len(self.param_record_list) == 0:
            print('no record to load')
        else:
            self.module.load_state(self.param_record_list[index])

    def smart_restore_state(self, index):
        self.rc_file.write('---> try to load best state\n')
        # 验证最后一个checkpoint, 及常规暂存的效果, 保留其中更优的一个
        if len(self.param_record_list) == 0:
            print('no record to load')
        else:
            self.module.load_state(self.param_record_list[index])
            _result = self.start_eval({'eval state': 'test_on_restore'})
            self.module.load_state(self.param_record_list_for_epoch[-1])
            _result_for_epoch = self.start_eval({'eval state': 'test_on_last_epoch'})
            if _result['rmse'] > _result_for_epoch['rmse']:
                print('last epoch rmse is better, restore from last epoch')
                self.module.load_state(self.param_record_list_for_epoch[-1])
            else:
                print('last checkpoint is better, restore from last checkpoint')
                self.module.load_state(self.param_record_list[index])

    def data_query(self):
        pass

    def record_to_file(self, line):
        # 记录写入文件
        if self.rc_file is None:
            self.rc_file = open('record.txt', 'a') 
            # self.rc_file = open('record.txt', 'w') # clear old record every time
            # self.rc_file.write('---> {}\n'.format(self.module_config))
            self.rc_file.write('---> module config\n')
            for key, _value in self.module.module_config.items():
                self.rc_file.write('  {}: {}\n'.format(key, _value))
            self.rc_file.write('---> training record\n')

        self.rc_file.write('  {}\n'.format(line.replace('\n', '\n  ')))
        self.rc_file.flush()

    def save_state(self):
        pass
        
    def clear_record(self):
        open('record.txt', 'w').close()