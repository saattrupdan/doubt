''' Quantile Regression Forests '''

from ._estimator import BaseEstimator

from typing import Union, Optional
import numpy as np

from anytree import Node
from anytree.exporter import DotExporter

from functools import partial
import scipy.optimize as opt
from collections import defaultdict
import pandas as pd
from joblib import Parallel, delayed

Number = Union[float, int]

class DecisionTree(BaseEstimator):
    ''' A decision tree for regression.

    Args:
        method (str): 
            The method used. Can be any of the following, defaults to 'cart':
                'cart': Classification and Regression Trees, see [2]
                'prim': Patient Rule Induction Method, see [3]
                'mars': Multivariate Adaptive Regression Splines, see [4]
                'hme': Hierarchical Mixtures of Experts, see [5]
        min_samples_leaf (int):
            The minimum number of unique target values in a leaf. Defaults
            to 5.
        n_jobs (int):
            The number of parallel processes to run at a time. Defaults
            to the total amount of CPU cores available.

    Attributes:
        peeling_ratio (float): 
            The percentage of the training set that will be "peeled" at 
            every step when using the PRIM method. Only relevant when
            `method`='prim'. Defaults to 0.1.

    Methods:
        fit(X, y): 
        predict(X, y):
        plot():


    References:
        .. [1] Hastie, T., Tibshirani, R., & Friedman, J. (2009). 
               The Elements of Statistical Learning: Data Mining, Inference, 
               and Prediction. Springer Science & Business Media. 
        .. [2] Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. 
               (1984). Classification and regression trees. 
               Wadsworth & Brooks. Cole Statistics/Probability Series. 
        .. [3] Friedman, J. H., & Fisher, N. I. (1999). 
               Bump Hunting in High-Dimensional Data. 
               Statistics and Computing, 9(2), 123-143.
        .. [4] Friedman, J. H. (1991). 
               Multivariate Adaptive Regression Splines. 
               The Annals of Statistics, 1-67.
        .. [5] Jordan, M. I., & Jacobs, R. A. (1994). 
               Hierarchical Mixtures of Experts and the EM Algorithm. 
               Neural Computation, 6(2), 181-214.
    Examples:
        A simple example where we fit a decision tree and predict a value 
        from the same training set:

        >>> from doubt.datasets import TehranHousing 
        >>> X, y = TehranHousing().split()
        >>> y = y[:, 0]
        >>> tree = DecisionTree().fit(X, y)
        >>> tree.predict(X[0]).astype(np.int32)
        44952

        We can also predict multiple values at once:

        >>> tree.predict(X[0:3]).shape
        (3,)

        The tree can also be called directly, as an alias for `predict`:

        >>> tree(X[0:3]).shape
        (3,)
    '''
    def __init__(self, method: str = 'cart', min_samples_leaf: int = 5, 
                 n_jobs: int = -1):
        self._root: Node
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs

    @staticmethod
    def _splitting_loss(thres: float, X, y, feat: int):
        ''' Calculate the mean squared loss of a split. '''

        # Get the indices of the features below and above the threshold,
        # which corresponds to the left and right child of the current node
        left = (X[:, feat] <= thres)
        right = ~left

        # If all feature values are below or above the threshold
        # then output infinite loss
        if left.all() or right.all(): return float('inf')

        # Calculate the means of the target values in each child node
        left_mean = y[left].mean()
        right_mean = y[right].mean()

        # Calculate mean squared loss on both child nodes, so that
        # minimising these corresponds to ensuring that the target
        # values within each child are as similar as possible
        left_val = np.mean((y[left] - left_mean) ** 2)
        right_val = np.mean((y[right] - right_mean) ** 2)

        return left_val + right_val

    def _find_threshold(self, X, y, feat_idx):
        ''' Given a feature, find the optimal split threshold for it. '''

        # Initial guess for the optimal threshold, which is required by
        # `scipy.opt.minimize`
        initial_guess = X[:, feat_idx].mean()

        # Find the threshold that minimises the splitting loss
        result = opt.minimize(self._splitting_loss, x0 = initial_guess, 
                              args = (X, y, feat_idx))

        # Get the resulting threshold and the associated loss, the latter
        # of which is used to compare thresholds across multiple features
        threshold = result.x[0]
        loss = result.fun

        return [feat_idx, threshold, loss]

    def _branch(self, X, y, parent: Optional[Node] = None):
        ''' Recursive function that computes the next two child nodes. '''

        # Get the number of rows and features in the data set
        nrows, nfeats = X.shape

        # Compute the best thresholds for all the features in parallel
        result = Parallel(n_jobs = self.n_jobs)(
            delayed(self._find_threshold)(X, y, idx) for idx in range(nfeats)
        )

        # Pull out the feature and threshold with the smallest loss
        arr = np.array(result)
        feat, thres = arr[arr[:, 2].argmin(), 0:2].astype(tuple)
        feat = np.uint16(feat)

        # Define the indices for the two child nodes
        left = X[:, feat] <= thres
        right = X[:, feat] > thres

        # If we have reached a leaf node then store information about
        # the target values and stop the recursion
        if len(np.unique(y[left])) < self.min_samples_leaf or \
            len(np.unique(y[right])) < self.min_samples_leaf:

            name = (f'[{y.min():,.0f}; {y.max():,.0f}]\n'
                    f'n = {nrows}\n'
                    f'n_unique = {len(np.unique(y))}')

            node = Node(name, n = nrows, parent = parent, vals = y)
            return None

        # Define the current node, which by the above conditional can't
        # be a leaf node
        name = f'Is feature {feat} < {thres:.0f}?'
        node = Node(name, n = nrows, parent = parent, feat = feat,
                    thres = thres)

        # If we're at the first step of the recursion then set self._root
        # to be the current node
        if parent is None: self._root = node

        # Define the dataset splittings for the child nodes
        X0 = X[left, :]
        y0 = y[left]
        X1 = X[right, :]
        y1 = y[right]

        # Continue the recursion on the child nodes
        return self._branch(X0, y0, node), self._branch(X1, y1, node)

    def fit(self, X, y):
        ''' Fit a decision tree to the data. '''

        # Call the branching function recursively, which will store all
        # information about the tree structure as well as target values
        # in the leaf nodes
        self._branch(X, y)

        return self

    def export_graph(self, path: str):
        ''' Save an image of the decision tree to disk. '''

        if self._root is None:
            raise RuntimeError('No tree found. You might need to fit it '\
                               'to a data set first?')

        DotExporter(self._root).to_picture(path)
        return self

    def _predict_one(self, x, quantile: Optional[float] = None):
        ''' Perform a prediction for a single input. '''
        node = self._root
        while not node.is_leaf:
            left, right = node.children
            node = left if x[node.feat] <= node.thres else right

        if quantile is None:
            return np.mean(node.vals)
        else:
            return np.quantile(node.vals, quantile)

    def predict(self, X, quantile: Optional[float] = None):
        ''' Predict the response values of a given feature matrix. '''
        onedim = (len(X.shape) == 1)
        if onedim: X = np.expand_dims(X, 0)
        jobs = (delayed(self._predict_one)(x, quantile) for x in X)
        result = np.array(Parallel(n_jobs = self.n_jobs)(jobs))
        return result[0] if onedim else result

class QuantileRandomForest(BaseEstimator):
    def __init__(self):
        raise NotImplementedError
