import torch


'''
# DOC: 动态注册类的操作, 有利于兼容性, 但是不利于代码可读性, 对初学者不友好, 目前不启用
def _register_func(self, class_operation, attr, attr_operation):
    def func(self, *args, **kwargs):
        return getattr(getattr(self, attr), attr_operation)(*args, **kwargs)
    _func = func
    setattr(self, class_operation, _func)

def register_standard_operation(self):
    if self.dnm is not None:
        self._register_func('normalize', 'dnm', 'normalize_all')
        self._register_func('denormalize', 'dnm', 'denormalize_outputs')

    if self.gp_model is not None:
        self._register_func('gp_train', 'gp_model', 'train')
        self._register_func('gp_predict', 'gp_model', 'predict')

    if self.pre_process_block is not None:
        self._register_func('pre_process', 'pre_process_block', 'pre_process')
        self._register_func('post_process', 'post_process_block', 'post_process')
    return
'''

class GP_model_block(torch.nn.Module):
    # ======= forward step =======
    #   1.normalize
    #   2.preprocess
    #   3.model predict
    #   4.postprocess
    #   5.denormalize

    # ======= backward step =======
    #   1.normalize
    #   2.preprocess
    #   3.model compute loss

    def __init__(self) -> None:
        super().__init__()
        self.dnm = None
        self.gp_model = None
        self.pre_process_block = None
        self.post_process_block = None

    # @tensor_wrap_to_tensor_with_uncertenty
    def predict(self, inputs):
        inputs = self.dnm.normalize_inputs(inputs)

        if self.pre_process_block is not None:
            gp_inputs, _ = self.pre_process_block.pre_process_at_predict(inputs, None)
        else:
            gp_inputs = inputs

        gp_outputs = self.gp_model.predict(gp_inputs)

        if self.post_process_block is not None:
            _, outputs = self.post_process_block.post_process_at_predict(inputs, gp_outputs)
        else:
            outputs = gp_outputs

        # outputs = self.dnm.denormalize_outputs(outputs)
        outputs = [self.dnm.denormalize_output(outputs[0], 0), outputs[1]]
        return outputs

    def predict_with_detecing_subset(self, inputs):
        #TODO
        return 

    # 当前这种形式会带来额外的内存,计算开销. 优点是自由度高,容易添加新的操作
    def compute_loss(self, inputs, outputs):
        inputs, outputs = self.dnm.normalize_all(inputs, outputs)

        if self.pre_process_block is not None:
            inputs, outputs = self.pre_process_block.pre_process_at_train(inputs, outputs)
        
        loss = self.gp_model.compute_loss(inputs, outputs)
        return loss

    def save_model(self, path):
        # TODO
        pass

    def load_model(self, path):
        # TODO
        pass

    def sync_block_after_train(self):
        # TODO
        pass

    def get_train_params(self):
        params_dict = {}
        params_dict.update(self.gp_model.get_train_params())
        if self.pre_process_block is not None:
            params_dict.update(self.pre_process_block.get_train_params())
        if self.post_process_block is not None:
            params_dict.update(self.post_process_block.get_train_params())
        return params_dict



if __name__ == '__main__':

    pass