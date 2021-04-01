'''Quantile regression trees'''

from sklearn.tree import BaseDecisionTree, DecisionTreeRegressor
from sklearn.utils import check_array, check_X_y
from typing import Optional, Union, Sequence
import numpy as np

from .utils import weighted_percentile

FloatArray = Sequence[float]
NumericArray = Sequence[Union[float, int]]


class BaseTreeQuantileRegressor(BaseDecisionTree):

    def predict(self,
                X: NumericArray,
                uncertainty: Optional[float] = None,
                check_input: bool = True):
        '''Predict regression value for X.

        Args:
            X (array-like or sparse matrix):
                The input samples, of shape [n_samples, n_features].
                Internally, it will be converted to `dtype=np.float32` and
                if a sparse matrix is provided to a sparse `csr_matrix`.
            uncertainty (float or None, optional):
                Value ranging from 0 to 1. If None then no prediction intervals
                will be returned. Defaults to None.
            check_input (boolean, optional):
                Allow to bypass several input checking. Don't use this
                parameter unless you know what you do. Defaults to True.

        Returns:
            y (array):
                Array of shape = [n_samples]. If quantile is set to None,
                then return E(Y | X). Else return y such that
                F(Y=y | x) = quantile.
        '''
        # Apply method requires X to be of dtype np.float32
        X = check_array(X, dtype=np.float32, accept_sparse='csc')
        preds = super().predict(X, check_input=check_input)

        if uncertainty is not None:
            lower = uncertainty / 2
            upper = 1 - lower
            intervals = np.empty((X.shape[0], 2))

            X_leaves = self.apply(X)
            unique_leaves = np.unique(X_leaves)

            for leaf in unique_leaves:
                intervals[X_leaves == leaf, 0] = weighted_percentile(
                    self.y_train_[self.y_train_leaves_ == leaf], lower)
                intervals[X_leaves == leaf, 1] = weighted_percentile(
                    self.y_train_[self.y_train_leaves_ == leaf], upper)
            return preds, intervals

        else:
            return preds

    def fit(self,
            X: NumericArray,
            y: NumericArray,
            sample_weight: Optional[NumericArray] = None,
            check_input: bool = True,
            X_idx_sorted: Optional[NumericArray] = None):
        '''Build a decision tree classifier from the training set (X, y).

        Args:
            X (array-like or sparse matrix)
                The training input samples, of shape [n_samples, n_features].
                Internally, it will be converted to `dtype=np.float32` and
                if a sparse matrix is provided to a sparse `csc_matrix`.
            y (array-like):
                The target values (class labels) as integers or strings, of
                shape [n_samples] or [n_samples, n_outputs].
            sample_weight (array-like or None, optional):
                Sample weights of shape = [n_samples]. If None, then samples
                are equally weighted. Splits that would create child nodes
                with net zero or negative weight are ignored while searching
                for a split in each node. Splits are also ignored if they
                would result in any single class carrying a negative weight
                in either child node. Defaults to None.
            check_input (boolean, optional):
                Allow to bypass several input checking. Don't use this
                parameter unless you know what you do. Defaults to True.
            X_idx_sorted (array-like or None, optional):
                The indexes of the sorted training input samples, of shape
                [n_samples, n_features]. If many tree are grown on the same
                dataset, this allows the ordering to be cached between trees.
                If None, the data will be sorted here. Don't use this
                parameter unless you know what to do. Defaults to None.
        '''
        # y passed from a forest is 2-D. This is to silence the annoying
        # data-conversion warnings.
        y = np.asarray(y)
        if np.ndim(y) == 2 and y.shape[1] == 1:
            y = np.ravel(y)

        # Apply method requires X to be of dtype np.float32
        X, y = check_X_y(
            X, y, accept_sparse='csc', dtype=np.float32, multi_output=False)
        super().fit(X, y, sample_weight=sample_weight, check_input=check_input,
                    X_idx_sorted=X_idx_sorted)
        self.y_train_ = y

        # Stores the leaf nodes that the samples lie in.
        self.y_train_leaves_ = self.tree_.apply(X)
        return self


class QuantileRegressionTree(DecisionTreeRegressor, BaseTreeQuantileRegressor):
    '''A decision tree regressor that provides quantile estimates.

    Args:
        criterion (string, optional):
            The function to measure the quality of a split. Supported criteria
            are 'mse' for the mean squared error, which is equal to variance
            reduction as feature selection criterion, and 'mae' for the mean
            absolute error. Defaults to 'mse'.
        splitter (string, optional):
            The strategy used to choose the split at each node. Supported
            strategies are 'best' to choose the best split and 'random' to
            choose the best random split. Defaults to 'best'.
        max_features (int, float, string or None, optional):
            The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a percentage and
              `int(max_features * n_features)` features are considered at each
              split.
            - If 'auto', then `max_features=n_features`.
            - If 'sqrt', then `max_features=sqrt(n_features)`.
            - If 'log2', then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.
            Note: the search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires
            to effectively inspect more than `max_features` features.
            Defaults to None.
        max_depth (int or None, optional):
            The maximum depth of the tree. If None, then nodes are expanded
            until all leaves are pure or until all leaves contain less than
            min_samples_split samples. Defaults to None.
        min_samples_split (int or float, optional):
            The minimum number of samples required to split an internal node:
            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a percentage and
              `ceil(min_samples_split * n_samples)` are the minimum
              number of samples for each split. Defaults to 2.
        min_samples_leaf (int or float, optional):
            The minimum number of samples required to be at a leaf node:
            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a percentage and
              `ceil(min_samples_leaf * n_samples)` are the minimum
              number of samples for each node. Defaults to 1.
        min_weight_fraction_leaf (float, optional):
            The minimum weighted fraction of the sum total of weights (of all
            the input samples) required to be at a leaf node. Samples have
            equal weight when sample_weight is not provided. Defaults to 0.0.
        max_leaf_nodes (int or None, optional):
            Grow a tree with `max_leaf_nodes` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes. Defaults to None.
        random_seed (int, RandomState instance or None, optional):
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random
            number generator; If None, the random number generator is the
            RandomState instance used by `np.random`. Defaults to None.

    Attributes:
        feature_importances_ (array):
            The feature importances, of shape = [n_features]. The higher, the
            more important the feature. The importance of a feature is
            computed as the (normalized) total reduction of the criterion
            brought by that feature. It is also known as the Gini importance.
        max_features_ (int):
            The inferred value of max_features.
        n_features_ (int):
            The number of features when `fit` is performed.
        n_outputs_ (int):
            The number of outputs when `fit` is performed.
        tree_ (Tree object):
            The underlying Tree object.
        y_train_ (array-like):
            Train target values.
        y_train_leaves_ (array-like):
            Cache the leaf nodes that each training sample falls into.
            y_train_leaves_[i] is the leaf that y_train[i] ends up at.
    '''
    def __init__(self,
                 criterion: str = 'mse',
                 splitter: str = 'best',
                 max_features: Optional[Union[int, float, str]] = None,
                 max_depth: Optional[int] = None,
                 min_samples_split: Union[int, float] = 2,
                 min_samples_leaf: Union[int, float] = 1,
                 min_weight_fraction_leaf: float = 0.,
                 max_leaf_nodes: Optional[int] = None,
                 random_seed: Union[int, np.random.RandomState, None] = None):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_seed)
