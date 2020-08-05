''' Doubtful wrapper for PyTorch models '''

# TODO: Implement wrapper, and add documentation and tests

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as D

import pytorch_lightning as pl

import copy
from pathlib import Path

class QuantileLoss(nn.Module):
    ''' Implements the quantile loss function.

    Args:
        uncertainty (float):
            The uncertainty parameter. This uniquely determines the two
            quantiles as `uncertainty` / 2 and 1 + `uncertainty` / 2, 
            respectively. Must be between 0.0 and 1.0, inclusive. Defaults
            to 0.05, yielding quantiles 2.5% and 97.5%.

    Parameters:
        lower (float): The lower quantile, computed as described above.
        upper (float): The upper quantile, computed as described above.
    '''
    def __init__(self, uncertainty: float = 0.05, activation: str = 'relu'):
        super(QuantileLoss, self).__init__()
        self.uncertainty = uncertainty
        self.activation = getattr(F, activation)
        self.lower = uncertainty / 2
        self.upper = 1 - uncertainty / 2
        
    def forward(self, preds, target):
        residuals = target.unsqueeze(1) - preds
        lower_loss = torch.mean(
            self.activation(residuals[:, 0]) * self.lower - \
            self.activation(-residuals[:, 0]) * (self.lower - 1)
        )
        median_loss = torch.abs(residuals[:, 1]).mean()
        upper_loss = torch.mean(
            self.activation(residuals[:, 2]) * self.upper - \
            self.activation(-residuals[:, 2]) * (self.upper - 1)
        )
        return (lower_loss + median_loss + upper_loss) / 3

class QuantileNeuralNetwork(pl.LightningModule):
    ''' Quantile regression wrapper for PyTorch models.



    Args:
        model (PyTorch model): The model that we are wrapping

    Parameters:
        model (PyTorch model): The original model
        lower (PyTorch model): A copy of the original model for predicting
                               the lower quantile
        upper (PyTorch model): A copy of the original model for predicting
                               the upper quantile

    Methods:
        asd

    Examples:
        Wrapping a PyTorch model:
        >>> from doubt.datasets import Concrete
        >>> import torch
        >>> X, y = Concrete().split()
        >>> model = QuantileNeuralNetwork(torch.nn.Sequential(
        ...     torch.nn.Linear(8, 16),
        ...     torch.nn.ReLU(),
        ...     torch.nn.Linear(16, 1)
        ... ))
        >>> model(torch.FloatTensor(X)).shape
        torch.Size([1030, 3])

        The model can be trained by calling `model.fit(X, y)`.
    '''
    def __init__(self, 
        model: nn.Module, 
        uncertainty: float = 0.05, 
        loss_activation: str = 'relu'):
        super(QuantileNeuralNetwork, self).__init__()
        self.model = model
        self.lower = copy.deepcopy(model)
        self.upper = copy.deepcopy(model)
        self.optimizer = None
        self._loss = QuantileLoss(
            uncertainty = uncertainty, 
            activation = loss_activation
        )
    
    def forward(self, x):
        preds = self.model(x)
        lower = preds - self.lower(x) ** 2
        upper = preds + self.upper(x) ** 2
        return torch.cat([lower, preds, upper], dim = 1)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = self._loss(output, target)
        return {'loss': loss}

    def configure_optimizers(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr = 4e-3)
        return self.optimizer

    def fit(self, X, y, batch_size: int = 32, val_size: float = 0.0,
            **kwargs):

        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        dataset = D.TensorDataset(X, y)

        if val_size != 0.0:
            val_size = int(val_size * X.shape[0])
            train_size = X.shape[0] - val_size
            train, val = D.random_split(dataset, [train_size, val_size])

            train_dl = D.DataLoader(train, batch_size, shuffle = True)
            val_dl = D.DataLoader(val, batch_size, shuffle = True)

        else:
            train_dl = D.DataLoader(dataset, batch_size, shuffle = True)
            val_dl = None
        
        trainer = pl.Trainer(**kwargs)
        return trainer.fit(self, train_dataloader = train_dl,
                           val_dataloader = val_dl)
