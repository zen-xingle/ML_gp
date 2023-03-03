import torch

config_example = {
    'kernel': {
                'K1': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
              },
    # 'mean_function':"None",
    'noise_init' : 1.,
    'exp_restrict': True
}


class GP_script:
    def __init__(self, config: dict) -> None:
        pass

    def set_training_data(dataset: dict):
        pass

    def get_trainable_params():
        pass

    def negative_log_likelihood():
        pass

    def predict(x):
        pass