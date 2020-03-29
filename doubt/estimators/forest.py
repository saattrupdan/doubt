''' Quantile Regression Forests '''

from _estimator import BaseEstimator
from _estimator import BaseQuantileEstimator
from _estimator import BaseBootstrapEstimator

class DecisionTree(BaseEstimator):
    ''' A decision tree for regression.

    Args:
        method (str): 
            The method used. Can be any of the following, defaults to 'cart':
                'cart': Classification and Regression Trees, see [2]
                'prim': Patient Rule Induction Method, see [3]
                'mars': Multivariate Adaptive Regression Splines, see [4]
                'hme': Hierarchical Mixtures of Experts, see [5]

    Attributes:
        peeling_ratio (float): 
            The percentage of the training set that will be "peeled" at 
            every step when using the PRIM method. Only relevant when
            `method`='prim'. Defaults to 0.1.

    Methods:
        x

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
        >>> from doubt.datasets import AutoInsurance
        >>> data = AutoInsurance()
        >>> tree = DecisionTree()
        >>> tree.fit(data)
        >>> tree(80)
        300
        >>> 
    '''
    raise NotImplementedError

class QuantileRandomForest(BaseQuantileEstimator):
    raise NotImplementedError

class BootstrapRandomForest(BaseBootstrapEstimator):
    raise NotImplementedError
