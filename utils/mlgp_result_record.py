import os

from utils.mlgp_log import mlgp_log
from copy import deepcopy


class MLGP_recorder:
    def __init__(self, save_path, overlap=False) -> None:
        self.save_path = save_path
        self._f = None
        self.register_state = False
        self.register_len = None

        self._key = None
        self._record_list = []

        if os.path.exists(save_path) and overlap is False:
            mlgp_log.e("[{}] is already exists, create failed, set overlap=True to avoid this check".format(save_path))
            raise RuntimeError()

        self._f = open(save_path, 'w')


    def register(self, key_list, overlap=False):
        if self.register_state == True and \
            overlap is False:
            mlgp_log.e("recorder has been register, double register is not allow, unless set overlap=True")
            raise RuntimeError()

        self._key = deepcopy(key_list)
        self.register_len = len(key_list)
        self.register_state = True
        self.record(key_list)


    def record(self, *args, check_len=True):
        if len(args) != self.register_len:
            mlgp_log.w("record failed, {} len is ")
        self._f.write(','.join(args))
