from copy import deepcopy

def smart_update(dst, src):
    dst = deepcopy(dst)
    src = deepcopy(src)
    _used_keys = []
    for _key, value in src.items():
        if isinstance(value, dict):
            dst[_key].update(value)
            _used_keys.append(_key)
        else:
            continue
    for _key in _used_keys:
        src.pop(_key)
    dst.update(src)
    return dst