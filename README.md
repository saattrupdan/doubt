# Doubt

*Bringing back uncertainty to machine learning.*

A Python package to include prediction intervals in the predictions of machine
learning models, to quantify their uncertainty.


## Installation

```shell
pip install doubt
```


## Features

- Bootstrap wrapper for all Scikit-Learn and PyTorch models
    - Can also be used to calculate usual bootstrapped statistics of a dataset
- (Linear) Quantile Regression
- Quantile Regression Forests
- A uniform dataset API, with 24 regression datasets and counting


## Quick Start

If you already have a model in Scikit-Learn or PyTorch, then you can simply
wrap it in a `Boot` to enable predicting with prediction intervals:

```python
>>> from sklearn.linear_model import LinearRegression
>>> from doubt.datasets import PowerPlant
>>>
>>> X, y = PowerPlant().split()
>>> clf = Boot(LinearRegression())
>>> clf = linreg.fit(X, y)
>>> clf.predict([10, 30, 1000, 50], uncertainty=0.05)
(481.9203102126274, array([473.43314309, 490.0313962 ]))
```

Alternatively, you can use one of the standalone models with uncertainty
outputs. For instance, a `QuantileRegressionForest`:

```python
>>> from doubt import QuantileRegressionForest as QRF
>>> from doubt.datasets import Concrete
>>> import numpy as np
>>>
>>> X, y = Concrete().split()
>>> clf = QRF(max_leaf_nodes=8)
>>> clf.fit(X, y)
>>> clf.predict(np.ones(8), uncertainty=0.25)
(16.933590347847982, array([ 8.93456428, 26.0664534 ]))
```
