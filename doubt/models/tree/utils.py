'''Utility functions used in tree models'''

import numpy as np
from typing import Sequence, Union, Optional

NumericArray = Sequence[Union[float, int]]


def weighted_percentile(arr: NumericArray,
                        quantile: float,
                        weights: Optional[NumericArray] = None,
                        sorter: Optional[NumericArray] = None):
    '''Returns the weighted percentile of an array.

    See [1] for an explanation of this concept.

    Args:
        arr (array-like):
            Samples at which the quantile should be computed, of
            shape [n_samples,].
        quantile (float):
            Quantile, between 0.0 and 1.0.
        weights (array-like, optional):
            The weights, of shape = (n_samples,). Here weights[i] is the
            weight given to point a[i] while computing the quantile. If
            weights[i] is zero, a[i] is simply ignored during the percentile
            computation. If None then uniform weights will be used. Defaults to
            None.
        sorter (array-like, optional):
            Array of shape [n_samples,], indicating the indices sorting `arr`.
            Thus, if provided, we assume that arr[sorter] is sorted. If None
            then `arr` will be sorted. Defaults to None.

    Returns:
        percentile: float
            Weighted percentile of `arr` at `quantile`.

    Raises:
        ValueError:
            If `quantile` is not between 0.0 and 1.0, or if `arr` and `weights`
            are of different lengths.

    References:
        [1]: https://en.wikipedia.org/wiki/Percentile
                #The_weighted_percentile_method
    '''
    # Ensure that quantile is set properly
    if quantile > 1 or quantile < 0:
        raise ValueError('The quantile should be between 0 and 1.')

    # Set weights to be uniform if not specified
    if weights is None:
        weights = np.ones_like(arr)

    # Ensure that `arr` and `weights` are numpy arrays
    arr = np.asarray(arr, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    # Ensure that `arr` and `weights` are of the same length
    if len(arr) != len(weights):
        raise ValueError('a and weights should have the same length.')

    # If `sorter` is given , then sort `arr` and `weights` using it
    if sorter is not None:
        arr = arr[sorter]
        weights = weights[sorter]

    # Remove all the array (and weight) elements with zero weight
    non_zeros = (weights != 0)
    arr = arr[non_zeros]
    weights = weights[non_zeros]

    # Sort the array if `sorter` is not given
    if sorter is None:
        sorted_indices = np.argsort(arr)
        sorted_arr = arr[sorted_indices]
        sorted_weights = weights[sorted_indices]
    else:
        sorted_arr = arr
        sorted_weights = weights

    # Calculate the partial sum of weights and get the total weight
    sorted_cum_weights = np.cumsum(sorted_weights)
    total = sorted_cum_weights[-1]

    # Calculate the percentile values
    partial_sum = 1. / total
    partial_sum *= sorted_cum_weights - sorted_weights / 2.0

    # Find the spot in `partial_sum` where `quantile` belongs, preserving order
    start = np.searchsorted(partial_sum, quantile) - 1

    # If the spot is the first or last, return the first or last percentiles
    if start == len(sorted_cum_weights) - 1:
        return sorted_arr[-1]
    if start == -1:
        return sorted_arr[0]

    # Find the proportion of which the distance from `partial_sum[start]` to
    # `quantile` is compared to the distance from `partial_sum[start]` to
    # `partial_sum[start + 1]`
    fraction = quantile - partial_sum[start]
    fraction /= partial_sum[start + 1] - partial_sum[start]

    # Add the corresponding proportion from `sorted_arr[start]` to
    # `sorted_arr[start + 1]`, to `sorted_arr[start]`.
    return sorted_arr[start] + fraction * \
        (sorted_arr[start + 1] - sorted_arr[start])
