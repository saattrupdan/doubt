# Changelog

All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to
[Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]
### Fixed
The `logkow` feature in the `FishBioconcentration` dataset is now converted
into a float, rather than a string.


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
