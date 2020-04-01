''' Quantile neural networks '''

from ._estimator import BaseEstimator

import torch

class QuantileNetwork(BaseEstimator, torch.nn.Module):
    def __init__(self):
        raise NotImplementedError
