import torch

class Tensor_normalizer:
    def __init__(self, tensor_in, dim=0) -> None:
        # default dim is 0, which means the first dim is sample_num dim.
    #     self.dim = dim
    #     if dim == None:
    #         self.mean = inputs.mean()
    #         self.std = inputs.std()
    #     else:
    #         self.mean = inputs.mean(dim=dim)
    #         self.std = inputs.std(dim=dim)

        # default dim is 0, which means the first dim is sample_num dim.
        self.mean = tensor_in.mean(dim=dim, keepdim=True)
        self.std = tensor_in.std(dim=dim, keepdim=True)
        self.dim = dim

    def normalize(self, tensor_in):
        # it should be auto broadcast
        return (tensor_in - self.mean) / (self.std + 1e-8)

    def denormalize(self, tensor_in):
        # it should be auto broadcast
        return tensor_in * self.std + self.mean


class Inputs_normalizer:
    def __init__(self, inputs, dim=0) -> None:
        # default dim is 0, which means the first dim is sample_num dim.
        self.t_n = [Tensor_normalizer(_in, dim) for _in in inputs]

    def normalize(self, inputs):
        # it should be auto broadcast
        return [t_n.normalize(_in) for t_n, _in in zip(self.t_n, inputs)]

    def denormalize(self, inputs):
        # it should be auto broadcast
        return [t_n.denormalize(_in) for t_n, _in in zip(self.t_n, inputs)]


#same as input normalizer
class Outputs_normalizer(Inputs_normalizer):
    def __init__(self, outputs, dim=0) -> None:
        # default dim is 0, which means the first dim is sample_num dim.
        self.t_n = [Tensor_normalizer(_out, dim) for _out in outputs]