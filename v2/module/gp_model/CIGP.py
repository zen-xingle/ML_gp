import torch
from module.gp_model import *

class CIGP(GP_script):
    def __init__(self, config):
        GP_script.__init__(self)
        self.config = config

        self.noise = config['noise_init']
        self.exp_restrict = config['exp_restrict']

    def set_training_data(self, data: dict):
        self.input_list = data['input_list']
        self.output_list = data['output_list']

        # setting kernel list

    def set_inherit_var(self, var_low_fidelity):
        self.var_low_fidelity = var_low_fidelity

    def get_trainable_params(self):
        pass

    def negative_log_likelihood(self):
        # inputs / outputs
        # x: [num, vector_dims]
        # y: [num, vector_dims]
        Sigma = self.kernel_list[0](self.input_list[0], self.input_list[0]) + JITTER * torch.eye(self.input_list[0].size(0), device=list(self.parameters())[0].device)
        if self.config['exp_restrict'] is True:
            _noise = self.noise.exp()
        else:
            _noise = self.noise
        Sigma = Sigma + _noise.pow(-1) * torch.eye(self.input_list[0].size(0), device=list(self.parameters())[0].device)

        L = torch.linalg.cholesky(Sigma)

        gamma = L.inverse() @ self.output_list[0]       # we can use this as an alternative because L is a lower triangular matrix.

        y_num, y_dimension = self.output_list[0].shape
        nll =  0.5 * (gamma ** 2).sum() +  L.diag().log().sum() * y_dimension  \
            + 0.5 * y_num * torch.log(2 * torch.tensor(PI, device=list(self.parameters())[0].device)) * y_dimension

        return nll

    def predict(self, data):
        # avoid changing the original input
        data = deepcopy(data)
    
        with torch.no_grad():
            Sigma = self.kernel_list[0](self.input_list[0], self.input_list[0]) + JITTER * torch.eye(self.input_list[0].size(0), device=list(self.parameters())[0].device)
            if self.config['exp_restrict'] is True:
                _noise = self.noise.exp()
            else:
                _noise = self.noise
            Sigma = Sigma + _noise.pow(-1) * torch.eye(self.input_list[0].size(0), device=list(self.parameters())[0].device)

            kx = self.kernel_list[0](self.input_list[0], data[0])
            L = torch.linalg.cholesky(Sigma)
            LinvKx, _ = torch.triangular_solve(kx, L, upper = False)

            u = kx.t() @ torch.cholesky_solve(self.outputs_tr[0], L)

            var_diag = self.kernel_list[0](data[0], data[0]).diag().view(-1, 1) - (LinvKx**2).sum(dim = 0).view(-1, 1)
            if self.config['exp_restrict'] is True:
                var_diag = var_diag + self.noise.exp().pow(-1)
            else:
                var_diag = var_diag + self.noise.pow(-1)

            return u, var_diag