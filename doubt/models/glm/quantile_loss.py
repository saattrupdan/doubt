'''Implementation of the quantile loss function'''

from typing import Sequence
import numpy as np


def quantile_loss(predictions: Sequence[float],
                  targets: Sequence[float],
                  quantile: float) -> float:
    '''Quantile loss function.

    Args:
        predictions (sequence of floats):
            Model predictions, of shape [n_samples,].
        targets (sequence of floats):
            Target values, of shape [n_samples,].
        quantile (float):
            The quantile we are seeking. Must be between 0 and 1.

    Returns:
        float: The quantile loss.
    '''
    # Convert inputs to NumPy arrays
    target_arr = np.asarray(targets)
    prediction_arr = np.asarray(predictions)

    # Compute the residuals
    res = target_arr - prediction_arr

    # Compute the mean quantile loss
    loss = np.mean(np.maximum(res, np.zeros_like(res)) * quantile +
                   np.maximum(-res, np.zeros_like(res)) * (1 - quantile))

    # Ensure that loss is of type float and return it
    return float(loss)
