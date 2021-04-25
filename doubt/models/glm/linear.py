'''Quantile linear regression'''

from .quantile_regressor import QuantileRegressor
from sklearn.linear_model import LinearRegression


QuantileLinearRegression = QuantileRegressor(LinearRegression())
