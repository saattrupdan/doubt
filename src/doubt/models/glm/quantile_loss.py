"""Implementation of the quantile loss function"""

import numpy as np


def quantile_loss(
    predictions: np.ndarray, targets: np.ndarray, quantile: float
) -> float:
    """Quantile loss function.

    Args:
        predictions (sequence of floats):
            Model predictions, of shape [n_samples,].
        targets (sequence of floats):
            Target values, of shape [n_samples,].
        quantile (float):
            The quantile we are seeking. Must be between 0 and 1.

    Returns:
        float: The quantile loss.
    """
    # Convert inputs to NumPy arrays
    target_arr = np.asarray(targets)
    prediction_arr = np.asarray(predictions)

    # Compute the residuals
    res = target_arr - prediction_arr

    # Compute the mean quantile loss
    loss = np.mean(
        np.maximum(res, np.zeros_like(res)) * quantile
        + np.maximum(-res, np.zeros_like(res)) * (1 - quantile)
    )

    # Ensure that loss is of type float and return it
    return float(loss)


def smooth_quantile_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
    quantile: float,
    alpha: float = 0.4,
) -> float:
    """The smooth quantile loss function from [1].

    Args:
        predictions (sequence of floats):
            Model predictions, of shape [n_samples,].
        targets (sequence of floats):
            Target values, of shape [n_samples,].
        quantile (float):
            The quantile we are seeking. Must be between 0 and 1.
        alpha (float, optional):
            Smoothing parameter. Defaults to 0.4.

    Returns:
        float: The smooth quantile loss.

    Sources:
        [1]: Songfeng Zheng (2011). Gradient Descent Algorithms for Quantile Regression
             With Smooth Approximation. International Journal of Machine Learning and
             Cybernetics.
    """
    # Convert inputs to NumPy arrays
    target_arr = np.asarray(targets)
    prediction_arr = np.asarray(predictions)

    # Compute the residuals
    residuals = target_arr - prediction_arr

    # Compute the smoothened mean quantile loss
    loss = quantile * residuals + alpha * np.log(1 + np.exp(-residuals / alpha))

    # Ensure that loss is of type float and return it
    return float(loss.mean())
