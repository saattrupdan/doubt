# Changelog

All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to
[Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [v4.5.0] - 2023-07-05
### Added
- Now saves the models during training with a `Boot` and reuses those during inference,
  speeding up inference. Thanks to @andrepugni for this contribution!

### Fixed
- Downgraded `tables` to 3.7.x to fix an installation bug.
- Downgraded `scikit-learn` to >=1.1,<1.3, as the decision tree API in v1.3 is
  incompatible with the previous ones. This will be dealt with separately in the
  future.


## [v4.4.1] - 2023-04-23
### Fixed
- When `return_all` is specified in `Boot.predict` and multiple samples have been
  inputted, then it now returns an array of shape `(num_samples, num_boots)` rather
  than the previous `(num_boots, num_samples)`.


## [v4.4.0] - 2023-04-23
### Added
- Added a `return_all` argument to the `Boot.predict` method, which will override the
  `uncertainty` and `quantiles` arguments and return the raw bootstrap distribution
  over which the quantiles would normally be calculated. This allows other uses of the
  bootstrap distribution than for computing prediction intervals.


## [v4.3.1] - 2023-03-20
### Fixed
- Previously, all the trees in `QuantileRegressionForest` were the same. This has now
  been fixed. Thanks to @gugerlir for noticing this!
- The `random_seed` argument in `QuantileRegressionTree` and `QuantileRegressionForest`
  has been changed to `random_state` to be consistent with `DecisionTreeRegressor`, and
  to avoid an `AttributeError` when accessing the estimators of a
  `QuantileRegressionForest`.


## [v4.3.0] - 2022-07-17
### Added
- The `QuantileRegressionForest` now has a `feature_importances_` attribute.


## [v4.2.0] - 2022-07-17
### Changed
- `Boot.fit` and `Boot.predict` methods are now parallelised, speeding up both training
  and prediction time a bit.
- Updated `README` to include generalised linear models, rather than only
  mentioning linear regression.

### Fixed
- Removed mention of `PyTorch` model support, as that has not been implemented
  yet


## [v4.1.0] - 2021-07-26
### Changed
- The `verbose` argument to `QuantileRegressionForest` also displays a progress
  bar during inference now.

### Fixed
- Fixed `QuantileRegressionForest.__repr__`.


## [v4.0.0] - 2021-07-26
### Added
- Added a `verbose` argument to `QuantileRegressionForest`, which displays a
  progress bar during training.

### Changed
- The default value of `QuantileRegressionForest.min_samples_leaf` has changed
  from 1 to 5, to ensure that the quantiles can always be computed sensibly
  with the default setting.

### Fixed
- The `logkow` feature in the `FishBioconcentration` dataset is now converted
into a float, rather than a string.
- Typo in example script in `README`


## [v3.0.0] - 2021-04-25
### Added
- Added `__repr__` to `QuantileRegressor`


## [v3.0.0] - 2021-04-25
### Removed
- `QuantileLinearRegression` has been removed, and `QuantileRegressor` should
  be used instead


## [v2.3.0] - 2021-04-25
### Added
- Added `quantiles` argument to `QuantileRegressionTree` and `Boot`, as an
  alternative to specifying `uncertainty`, if you want to return specific
  quantiles.
- Added general `QuantileRegressor`, which can wrap any general linear model
  for quantile predictions.

### Fixed
- The predictions in `Boot.predict` were based on a fitting of the model to one
  of the bootstrapped datasets. It is now based on the entire dataset, which in
  particular means that the predictions will be deterministic. The intervals
  will still be stochastic, as they should be.

### Changed
- Updated Numpy random number generation to [their new API](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator)
- All residuals in `Boot` are now calculated during fitting, which should
  decrease the prediction times a tiny bit.

### Removed
- Package no longer relies on `statsmodels`


## [v2.2.1] - 2021-04-16
### Fixed
- A handful of docstring style changes to yield a cleaner Sphinx documentation


## [v2.2.0] - 2021-04-16
### Added
- Sphinx documentation


## [v2.1.0] - 2021-04-11
### Added
- Implemented `score` method to `QuantileLinearRegression`, which either
  outputs the mean negative pinball loss function, or the R^2 value
- Added more documentation to `QuantileLinearRegression`

### Changed
- Outputs more informative error message when a singular feature matrix is
  being used with `QuantileLinearRegression`
- Datasets look prettier in notebooks now

### Removed
- Removed docstring comments about closing datasets after use, as this is
  automatic

## [v2.0.2] - 2021-04-09
### Fixed
- Small mistake in the computation of the prediction intervals in
  `Boot.predict`, where the definition of `generalisation` should be the
  difference of the _means_ of the residuals, and not the difference between
  the individual quantiles. Makes a very tiny difference to the prediction
  intervals. Thanks to Bryan Shalloway for catching this mistake.


## [v2.0.1] - 2021-04-04
### Fixed
- `Boot.__repr__` was not working properly


## [v2.0.0] - 2021-04-04
### Added
- Added proper `__repr__` descriptions to all models

### Changed
- Changed the ordering of `Dataset.split` to `X_train`, `X_test`, `y_train`
  and `y_test`, to agree with `scikit-learn`
- Moved some `Dataset` attributes to the private API
