''' Neural networks '''

from _estimator import BaseQuantileEstimator, BaseBootstrapEstimator
import torch

class QuantileNetwork(BaseQuantileEstimator, torch.nn.Module):
    raise NotImplementedError

class BootstrapNetwork(BaseBootstrapEstimator, torch.nn.Module):
    raise NotImplementedError
