from copy import deepcopy
from utils.mlgp_log import mlgp_log

def smart_update(target_config, default_config):
    if target_config is None:
        return default_config

    target_config = deepcopy(target_config)
    default_config = deepcopy(default_config)

    mlgp_log.i("\nInput config as follow:")
    for _key, value in default_config.items():
        mlgp_log.i(_key, ':', '{}'.format(value))
    mlgp_log.i("\n")

    _used_keys = []
    for _key, value in default_config.items():
        if isinstance(value, dict):
            if _key in target_config:
                target_config[_key].update(value)
                _used_keys.append(_key)
            else:
                target_config[_key] = value
                _used_keys.append(_key)
        else:
            continue
    for _key in _used_keys:
        default_config.pop(_key)
    target_config.update(default_config)

    mlgp_log.i("\nInput config, after fullfill, meaning what actually feed to the model, as follow:")
    for _key, value in target_config.items():
        mlgp_log.i(_key, ':', '{}'.format(value))
    mlgp_log.i("\n")

    return target_config