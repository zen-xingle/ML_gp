import torch
from utils.type_define import *

default_config = {
    'noise': 1.,
    'exp_restrict': False,
    'kernel': {
                'K1': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
              },
}


def check_list_contain_val_with_bar(list):
    for item in list:
        if isinstance(item, GP_val_with_bar):
            return True
    return False

def check_no_fuse_type(list):
    define_type_list = [GP_val_with_bar]
    
    others_count = 0
    define_type_count = [0]* len(define_type_list)

    # count type
    for item in list:
        type_found = False
        for i, _type in enumerate(define_type_list):
            if isinstance(item, _type):
                define_type_count[i] += 1
                type_found = True
                break
        
        if type_found is False:
            others_count += 1

    # assert according not support case
    any_type_exist = set(define_type_count) != {0}
    if any_type_exist is False:
        return
    elif any_type_exist is True:
        if others_count > 0:
            assert False, "GP_val_with_bar can't be fused with other type."
        elif others_count == 0:
            if len(set(define_type_count)) == 1:
                return
            elif len(set(define_type_count)) >= 3:
                assert False, "Multi type can't be fused"
            elif len(set(define_type_count)) == 2:
                if 0 not in set(define_type_count):
                    assert False, "Multi type can't be fused"

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
        check_no_fuse_type(inputs)
        if check_list_contain_val_with_bar(inputs):
            vars = [item.get_var() for item in inputs]
            inputs = [item.get_mean() for item in inputs]
            return self.predict_with_var(inputs, vars)
        else:
            return self.predict_with_var(inputs, None)

    def predict_with_var(self, inputs, vars=None):
        pass

    def compute_loss(self, inputs, outputs):
        check_no_fuse_type(inputs)
        check_no_fuse_type(outputs)

        if check_list_contain_val_with_bar(inputs) != check_list_contain_val_with_bar(outputs):
            assert False, "inputs and outputs should be both GP_val_with_bar or not"

        if check_list_contain_val_with_bar(inputs):
            input_vars = [item.get_var() for item in inputs]
            inputs = [item.get_mean() for item in inputs]
            output_vars = [item.get_var() for item in outputs]
            outputs = [item.get_mean() for item in outputs]
            return self.compute_loss_with_var(inputs, outputs, input_vars, output_vars)
        else:
            return self.compute_loss_with_var(inputs, outputs, None, None)

    def compute_loss_with_var(self, inputs, outputs, input_vars=None, output_vars=None):
        pass

    def get_train_params(self):
        pass