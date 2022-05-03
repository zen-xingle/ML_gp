import re
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def _reshape_as_2D(A, sample_last_dim):
    if len(A.shape) > 2:
        sample = A.shape[-1] if sample_last_dim is True else A.shape[0]
        A = A.reshape(-1, sample) if sample_last_dim is True else A.reshape(sample, -1)
    return A


def performance_evaluator(A, B, method_list, sample_last_dim=False):
    if hasattr(A, 'numpy'):
        A = A.detach().numpy()
    if hasattr(B, 'numpy'):
        B = B.detach().numpy()
    A = _reshape_as_2D(A, sample_last_dim)
    B = _reshape_as_2D(B, sample_last_dim)

    result = {}
    for _method in method_list:
        if _method == 'mae':
            result['mae'] = mean_absolute_error(A, B)
        elif _method == 'r2':
            result['r2'] = r2_score(A, B)
        elif _method == 'rmse':
            result['rmse'] = np.sqrt(mean_squared_error(A, B))
    return result