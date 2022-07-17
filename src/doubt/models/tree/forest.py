"""Quantile regression forests"""

from typing import Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .tree import QuantileRegressionTree


class QuantileRegressionForest:
    """A random forest for regression which can output quantiles as well.

    Args:
        n_estimators (int, optional):
            The number of trees in the forest. Defaults to 100.
        criterion (string, optional):
            The function to measure the quality of a split. Supported criteria are
            'squared_error' for the mean squared error, which is equal to variance
            reduction as feature selection criterion, and 'absolute_error' for the mean
            absolute error. Defaults to 'squared_error'.
        splitter (string, optional):
            The strategy used to choose the split at each node. Supported
            strategies are 'best' to choose the best split and 'random' to
            choose the best random split. Defaults to 'best'.
        max_features (int, float, string or None, optional):
            The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a percentage and
              `int(max_features * n_features)` features are considered at
              each split.
            - If 'auto', then `max_features=n_features`.
            - If 'sqrt', then `max_features=sqrt(n_features)`.
            - If 'log2', then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

            Note: the search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires
            to effectively inspect more than ``max_features`` features.
            Defaults to None.
        max_depth (int or None, optional):
            The maximum depth of the tree. If None, then nodes are expanded
            until all leaves are pure or until all leaves contain less than
            min_samples_split samples. Defaults to None.
        min_samples_split (int or float, optional):
            The minimum number of samples required to split an internal node:

            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a percentage and
              `ceil(min_samples_split * n_samples)` are the minimum number of
              samples for each split. Defaults to 2.

        min_samples_leaf (int or float, optional):
            The minimum number of samples required to be at a leaf node:

            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a percentage and
              `ceil(min_samples_leaf * n_samples)` are the minimum number of
              samples for each node. Defaults to 5.

        min_weight_fraction_leaf (float, optional):
            The minimum weighted fraction of the sum total of weights (of all
            the input samples) required to be at a leaf node. Samples have
            equal weight when sample_weight is not provided. Defaults to 0.0.
        max_leaf_nodes (int or None, optional):
            Grow a tree with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes. Defaults to None.
        n_jobs (int, optional):
            The number of CPU cores used in fitting and predicting. If -1 then
            all available CPU cores will be used. Defaults to -1.
        random_seed (int, RandomState instance or None, optional):
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random
            number generator; If None, the random number generator is the
            RandomState instance used by `np.random`. Defaults to None.
        verbose (bool, optional):
            Whether extra output should be printed during training and
            inference. Defaults to False.

    Examples:
        Fitting and predicting follows scikit-learn syntax::

            >>> from doubt.datasets import Concrete
            >>> X, y = Concrete().split()
            >>> forest = QuantileRegressionForest(random_seed=42, max_leaf_nodes=8)
            >>> forest.fit(X, y).predict(X).shape
            (1030,)
            >>> preds = forest.predict(np.ones(8))
            >>> 16 < preds < 17
            True

        Instead of only returning the prediction, we can also return a
        prediction interval::

            >>> preds, interval = forest.predict(np.ones(8), uncertainty=0.25)
            >>> interval[0] < preds < interval[1]
            True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "squared_error",
        splitter: str = "best",
        max_features: Optional[Union[int, float, str]] = None,
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 5,
        min_weight_fraction_leaf: float = 0.0,
        max_leaf_nodes: Optional[int] = None,
        n_jobs: int = -1,
        random_seed: Optional[int] = None,
        verbose: bool = False,
    ):

        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.verbose = verbose

        self._estimators = n_estimators * [
            QuantileRegressionTree(
                criterion=criterion,
                splitter=splitter,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                random_seed=random_seed,
            )
        ]

    def __repr__(self) -> str:
        txt = "QuantileRegressionForest("
        attributes = [
            "n_estimators",
            "criterion",
            "splitter",
            "max_features",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "min_weight_fraction_leaf",
            "max_leaf_nodes",
            "n_jobs",
            "random_seed",
        ]
        for idx, attr in enumerate(attributes):
            if idx > 0:
                txt += "                         "
            txt += f"{attr}={getattr(self, attr)}"
            if idx < len(attributes) - 1:
                txt += ",\n"
        return txt + ")"

    def fit(self, X, y, **kwargs):
        """Fit decision trees in parallel.

        Args:
            X (array-like or sparse matrix):
                The input samples, of shape [n_samples, n_features].
                Internally, it will be converted to `dtype=np.float32` and
                if a sparse matrix is provided to a sparse `csr_matrix`.
            y (array-like):
                The target values (class labels) as integers or strings, of
                shape [n_samples] or [n_samples, n_outputs].
            verbose (bool or None, optional):
                Whether extra output should be printed during training. If None
                then the initialised value of the `verbose` parameter will be
                used. Defaults to None.
        """
        # Set the verbose argument if it has not been set
        if kwargs.get("verbose") is None:
            verbose = self.verbose
        else:
            verbose = kwargs.get("verbose")

        # Initialise random number generator
        rng = np.random.default_rng(self.random_seed)

        # Store the number of training samples
        n = X.shape[0]

        # Get bootstrap resamples of the data set
        bidxs = rng.choice(n, size=(self.n_estimators, n), replace=True)

        # Set up progress bar if requested
        if verbose:
            itr = tqdm(self._estimators, desc="Fitting trees")
        else:
            itr = self._estimators

        # Fit trees in parallel on the bootstrapped resamples
        with Parallel(n_jobs=self.n_jobs) as parallel:
            self._estimators = parallel(
                delayed(estimator.fit)(X[bidxs[b, :], :], y[bidxs[b, :]])
                for b, estimator in enumerate(itr)
            )

        if verbose:
            itr.close()

        return self

    def predict(
        self,
        X: np.ndarray,
        uncertainty: Optional[float] = None,
        quantiles: Optional[np.ndarray] = None,
        verbose: Optional[bool] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict regression value for X.

        Args:
            X (array-like or sparse matrix):
                The input samples, of shape [n_samples, n_features].
                Internally, it will be converted to `dtype=np.float32` and
                if a sparse matrix is provided to a sparse `csr_matrix`.
            uncertainty (float or None, optional):
                Value ranging from 0 to 1. If None then no prediction intervals
                will be returned. Defaults to None.
            quantiles (sequence of floats or None, optional):
                List of quantiles to output, as an alternative to the
                `uncertainty` argument, and will not be used if that argument
                is set. If None then `uncertainty` is used. Defaults to None.
            verbose (bool or None, optional):
                Whether extra output should be printed during inference. If
                None then the initialised value of the `verbose` parameter will
                be used. Defaults to None.

        Returns:
            Array or pair of arrays:
                Either array with predictions, of shape [n_samples,], or a pair
                of arrays with the first one being the predictions and the
                second one being the desired quantiles/intervals, of shape
                [2, n_samples] if `uncertainty` is not None, and
                [n_quantiles, n_samples] if `quantiles` is not None.
        """
        # Set the verbose argument if it has not been set
        if verbose is None:
            verbose = self.verbose

        # Ensure that X is two-dimensional
        onedim = len(X.shape) == 1
        if onedim:
            X = np.expand_dims(X, 0)

        # Set up progress bar if requested
        if verbose:
            itr = tqdm(self._estimators, desc="Getting tree predictions")
        else:
            itr = self._estimators

        with Parallel(n_jobs=self.n_jobs) as parallel:
            preds = parallel(
                delayed(estimator.predict)(
                    X, uncertainty=uncertainty, quantiles=quantiles
                )
                for estimator in itr
            )

            if uncertainty is not None or quantiles is not None:
                quantile_vals = np.stack([interval for _, interval in preds], axis=0)
                quantile_vals = quantile_vals.mean(0)
                preds = np.stack([pred for pred, _ in preds])
                preds = preds.mean(0)
                if onedim:
                    preds = preds[0]
                    quantile_vals = quantile_vals[0]
                return preds, quantile_vals

            else:
                preds = np.mean(preds, axis=0)
                if onedim:
                    preds = preds[0]
                return preds
