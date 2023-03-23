import torch


default_config = {
    'noise': 1.,
    'exp_restrict': False,
    'kernel': {
                'K1': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
              },
}


class BASE_GP_MODEL(torch.nn.Module):
    '''
        A gp model is supposed to have at least 3 api functions
            1. predict
            2. compute_loss
            3. get_train_params

        Normally, the gp model hayerparam is set via the config file.
        The config file may have the following params:
            1. noise            (set as float value)
            2. kernel           (kernel config, may have multiple kernel)
            3. exp_restrict     (set as bool value)
    '''
    def __init__(self, gp_model_config) -> None:
        super().__init__()
        self.gp = gp_model_config

        self.noise = None
        self.inputs_tr = None
        self.outputs_tr = None
        self.already_set_train_data = False
        self.kernel_list = None

    def predict(self, inputs):
        pass

    def compute_loss(self, inputs, outputs):
        pass

    def get_train_params(self):
        pass