import torch

class Normalizer:
    def __init__(self, inputs, dim=0) -> None:
        # default dim is 0, which means the first dim is sample_num dim.
    #     self.dim = dim
    #     if dim == None:
    #         self.mean = inputs.mean()
    #         self.std = inputs.std()
    #     else:
    #         self.mean = inputs.mean(dim=dim)
    #         self.std = inputs.std(dim=dim)

        # default dim is 0, which means the first dim is sample_num dim.
        self.mean = inputs.mean(dim=dim, keepdim=True)
        self.std = inputs.std(dim=dim, keepdim=True)
        self.dim = dim

    def normalize(self, inputs):
        # it should be auto broadcast
        return (inputs - self.mean) / (self.std + 1e-8)

    def denormalize(self, inputs):
        # it should be auto broadcast
        return inputs * self.std + self.mean