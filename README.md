# Doubt

*Bringing back uncertainty to machine learning.*

______________________________________________________________________
[![PyPI Status](https://badge.fury.io/py/doubt.svg)](https://pypi.org/project/doubt/)
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://saattrupdan.github.io/doubt/doubt.html)
[![License](https://img.shields.io/github/license/saattrupdan/doubt)](https://github.com/saattrupdan/doubt/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/saattrupdan/doubt)](https://github.com/saattrupdan/doubt/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-67%25-yellow.svg)](https://github.com/saattrupdan/doubt/tree/dev/tests)
[![Conference](https://img.shields.io/badge/Conference-AAAI23-blue)](https://arxiv.org/abs/2201.11676)

A Python package to include prediction intervals in the predictions of machine
learning models, to quantify their uncertainty.


## Installation

If you do not already have HDF5 installed, then start by installing that. On MacOS this
can be done using `sudo port install hdf5` after
[MacPorts](https://www.macports.org/install.php) have been installed. On Ubuntu you can
get HDF5 with `sudo apt-get install python-dev python3-dev libhdf5-serial-dev`. After
that, you can install `doubt` with `pip`:

```shell
pip install doubt
```


## Features

- Bootstrap wrapper for all Scikit-Learn models
    - Can also be used to calculate usual bootstrapped statistics of a dataset
- Quantile Regression for all generalised linear models
- Quantile Regression Forests
- A uniform dataset API, with 24 regression datasets and counting


## Quick Start

If you already have a model in Scikit-Learn, then you can simply
wrap it in a `Boot` to enable predicting with prediction intervals:

```python
>>> from sklearn.linear_model import LinearRegression
>>> from doubt import Boot
>>> from doubt.datasets import PowerPlant
>>>
>>> X, y = PowerPlant().split()
>>> clf = Boot(LinearRegression())
>>> clf = clf.fit(X, y)
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

## Citation
```
@inproceedings{mougannielsen2023monitoring,
  title={Monitoring Model Deterioration with Explainable Uncertainty Estimation via Non-parametric Bootstrap},
  author={Mougan, Carlos and Nielsen, Dan Saattrup},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2023}
}
```
