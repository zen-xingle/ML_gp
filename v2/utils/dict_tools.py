from copy import deepcopy
from utils.mlgp_log import mlgp_log

def smart_update(dst, src):
    dst = deepcopy(dst)
    src = deepcopy(src)

    mlgp_log.i("\nInput config as follow:")
    for _key, value in src.items():
        mlgp_log.i(_key, ':', '{}'.format(value))
    mlgp_log.i("\n")

    _used_keys = []
    for _key, value in src.items():
        if isinstance(value, dict):
            if _key in dst:
                dst[_key].update(value)
                _used_keys.append(_key)
            else:
                dst[_key] = value
                _used_keys.append(_key)
        else:
            continue
    for _key in _used_keys:
        src.pop(_key)
    dst.update(src)

    mlgp_log.i("\nInput config, after fullfill, meaning what actually feed to the model, as follow:")
    for _key, value in dst.items():
        mlgp_log.i(_key, ':', '{}'.format(value))
    mlgp_log.i("\n")

    return dst