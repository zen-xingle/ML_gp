import os
import sys
realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from utils.mlgp_log import mlgp_log
from copy import deepcopy


def _dict_to_str(tar_dict):
    def _parse_dict(_d, _srt_list, _depth=0):
        _blank = '  '
        for _key, _value in _d.items():
            if isinstance(_value, dict):
                _str.append(_blank*_depth + _key + ':\n')
                _parse_dict(_value, _srt_list, _depth+1)
            else:
                _srt_list.append(_blank*_depth + _key + ': ' + str(_value) + '\n')

    if not isinstance(tar_dict, dict):
        mlgp_log.e('{} is not a dict'.format(tar_dict))

    _str = []
    _parse_dict(tar_dict, _str)
    return _str


class MLGP_recorder:
    def __init__(self, save_path, append_info=None, overlap=False) -> None:
        self.save_path = save_path
        self._f = None
        self._register_state = False
        self._register_len = None

        self._key = None
        self._record_list = []

        if os.path.exists(save_path) and overlap is False:
            mlgp_log.e("[{}] is already exists, create failed, set overlap=True to avoid this check".format(save_path))
            raise RuntimeError()

        if overlap and os.path.exists(save_path):
            self._f = open(save_path, 'a')
            self._f.write('\n\n')
        else:
            self._f = open(save_path, 'w')
        
        self._f.write("@MLGP_recorder@\n")
        if append_info is not None:
            self._write_append_info(append_info)
        self._f.flush()


    def _write_append_info(self, info):
        self._f.write('@append_info@\n')
        if isinstance(info, dict):
            _str = _dict_to_str(info)
            for _s in _str:
                self._f.write(_s)
        elif isinstance(info, list):
            for _s in info:
                self._f.write(str(_s))
        else:
            self._f.write(str(info))
        self._f.write('@append_info@\n')

    def register(self, key_list, re_register=False):
        if self._register_state == True and \
            re_register is False:
            mlgp_log.e("recorder has been register, double register is not allow, unless set overlap=True")
            raise RuntimeError()

        self._key = deepcopy(key_list)
        self._register_len = len(key_list)
        self._register_state = True
        self.record(key_list)


    def record(self, _single_record, check_len=True):
        if not isinstance(_single_record, list) and not isinstance(_single_record, dict):
            mlgp_log.e("MLGP_recorder.record only accept input as dict/list")

        if len(_single_record) != self._register_len and check_len is True:
            mlgp_log.w("record failed, {} len is not match key list: {}".format(len(_single_record), self._key))

        if isinstance(_single_record, dict):
            _backup = deepcopy(_single_record)
            _single_record = [None] * len(_backup)
            for _k, _v in _backup.items():
                _single_record[self._key.index(_k)] = _v

        _single_record = [str(_sr) for _sr in _single_record]
        _single_record[-1] = _single_record[-1] + '\n'

        self._f.write(','.join(_single_record))
        self._f.flush()


if __name__ == '__main__':
    import datetime
    rc = MLGP_recorder('./record_test.txt', {'Function': 'A', 
                                             'Purpose': 'Test',
                                             'time': {
                                                'now': datetime.datetime.today(),
                                                'weekdata': datetime.datetime.today().isoweekday(),
                                             }
                                             })
    rc.register(['epoch','result'])
    rc.record([0, 0.5])
    rc.record({'epoch': 1, 'result': 0.6})
    rc.record({'result': 0.7, 'epoch': 2})