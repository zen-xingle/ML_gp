import torch
from utils.mlgp_log import mlgp_log

def nan_hook(self, inp, output):
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output

    for i, out in enumerate(outputs):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            mlgp_log.error("Nan value detect in ", self.__class__.__name__, "output")
            raise RuntimeError(f"NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])


def register_nan_hook(model):
    for submodule in model.modules():
        submodule.register_forward_hook(nan_hook)