# Conditional independent Gaussian process for vector output regression based on pytorch
# v10: A stable version. improve over the v02 version to fix nll bug; adapt to torch 1.11.0.
#
# Author: Wei W. Xing (wxing.me)
# Email: wayne.xingle@gmail.com
# Date: 2022-03-23


# %%
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import os

print("cigp torch version:", torch.__version__)
# I use torch (1.11.0) for this work. lower version may not work.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Fixing strange error if run in MacOS 
JITTER = 1e-6
EPS = 1e-10
PI = 3.1415

class cigp(nn.Module):
    def __init__(self, X, Y, normal_mode=1):
        super(cigp, self).__init__()

        #normalize X independently for each dimension
        self.Xmean = X.mean(0)
        self.Xstd = X.std(0)
        self.X = (X - self.Xmean.expand_as(X)) / (self.Xstd.expand_as(X) + EPS)

        # normalize y all together
        self.Ymean = Y.mean()
        self.Ystd = Y.std()  #
        self.Y = (Y - self.Ymean.expand_as(Y)) / (self.Ystd.expand_as(Y) + EPS)
        
        # option 2: normalize y by each dimension
        # self.Ymean = Y.mean(0)
        # self.Yvar = Y.var(0)
        # self.Y = (Y - self.Ymean.expand_as(Y)) / self.Yvar.expand_as(Y)

        self.log_beta = nn.Parameter(torch.ones(1) * 0)   # a large noise 
        self.log_length_scale = nn.Parameter(torch.zeros(X.size(1)))
        self.log_scale = nn.Parameter(torch.zeros(1))

    # define kernel function
    def kernel(self, X, X2): 
        length_scale = torch.exp(self.log_length_scale).view(1, -1)

        X = X / length_scale.expand(X.size(0), length_scale.size(1))
        X2 = X2 / length_scale.expand(X2.size(0), length_scale.size(1))

        X_norm2 = torch.sum(X * X, dim=1).view(-1, 1)
        X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)

        K = -2.0 * X @ X2.t() + X_norm2.expand(X.size(0), X2.size(0)) + X2_norm2.t().expand(X.size(0), X2.size(0))
        K = self.log_scale.exp() * torch.exp(-0.5 * K)

        # X1 = X1 / self.log_length_scale.exp()**2
        # X2 = X2 / self.log_length_scale.exp()**2
        # X1_norm2 = X1 * X1
        # X2_norm2 = X2 * X2

        # K = -2.0 * X1 @ X2.t() + X1_norm2.expand(X1.size(0), X2.size(0)) + X2_norm2.t().expand(X1.size(0), X2.size(0))  #this is the effective Euclidean distance matrix between X1 and X2.
        # K = self.log_scale.exp() * torch.exp(-0.5 * K)
        return K
    
    
    def forward(self, Xte):
        n_test = Xte.size(0)
        Xte = ( Xte - self.Xmean.expand_as(Xte) ) / self.Xstd.expand_as(Xte)
        
        Sigma = self.kernel(self.X, self.X) + self.log_beta.exp().pow(-1) * torch.eye(self.X.size(0)) \
            + JITTER * torch.eye(self.X.size(0))

        kx = self.kernel(self.X, Xte)
        L = torch.cholesky(Sigma)
        LinvKx,_ = torch.triangular_solve(kx, L, upper = False)
        
        # option 1
        mean = kx.t() @ torch.cholesky_solve(self.Y, L)  # torch.linalg.cholesky()
        var_diag = self.log_scale.exp().expand(n_test, 1) \
            - (LinvKx**2).sum(dim = 0).view(-1, 1)
            
        # add the noise uncertainty
        var_diag = var_diag + self.log_beta.exp().pow(-1)
        
        mean = mean * self.Ystd.expand_as(mean) + self.Ymean.expand_as(mean)
        var_diag = var_diag.expand_as(mean) * self.Ystd**2

        return mean, var_diag
    

    def negative_log_likelihood(self):
        y_num, y_dimension = self.Y.shape
        Sigma = self.kernel(self.X, self.X) + self.log_beta.exp().pow(-1) * torch.eye(
            self.X.size(0)) + JITTER * torch.eye(self.X.size(0))
        
        L = torch.linalg.cholesky(Sigma)
        #option 1 (use this if torch supports)
        # Gamma,_ = torch.triangular_solve(self.Y, L, upper = False)
        #option 2
        gamma = L.inverse() @ self.Y       # we can use this as an alternative because L is a lower triangular matrix.
        
        nll =  0.5 * (gamma ** 2).sum() +  L.diag().log().sum() * y_dimension  \
            + 0.5 * y_num * torch.log(2 * torch.tensor(PI)) * y_dimension
        return nll

    def train_adam(self, niteration=10, lr=0.1):
        # adam optimizer
        # uncommont the following to enable
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        for i in range(niteration):
            optimizer.zero_grad()
            # self.update()
            loss = self.negative_log_likelihood()
            loss.backward()
            optimizer.step()
            print('loss_nll:', loss.item())
            # print('iter', i, ' nnl:', loss.item())
            # print('iter', i, 'nnl:{:.5f}'.format(loss.item()))

    def train_bfgs(self, niteration=50, lr=0.1):
        # LBFGS optimizer
        optimizer = torch.optim.LBFGS(self.parameters(), lr=lr)  # lr is very important, lr>0.1 lead to failure
        for i in range(niteration):
            # optimizer.zero_grad()
            # LBFGS
            def closure():
                optimizer.zero_grad()
                # self.update()
                loss = self.negative_log_likelihood()
                loss.backward()
                # print('nll:', loss.item())
                # print('iter', i, ' nnl:', loss.item())
                print('iter', i, 'nnl:{:.5f}'.format(loss.item()))
                return loss

            # optimizer.zero_grad()
            optimizer.step(closure)
        # print('loss:', loss.item())
        
    # TODO: add conjugate gradient method


