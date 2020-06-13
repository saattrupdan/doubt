''' Doubtful wrapper for PyTorch models '''

import torch

class TorchDoubt(torch.nn.Module):
    def __init__(self):
        super(TorchDoubt, self).__init__()
        raise NotImplementedError
