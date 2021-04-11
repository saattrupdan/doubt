# Changelog

All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to
[Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]
### Added
-

### Changed
-

### Deprecated
-

### Removed
-

### Fixed
-

### Security
-


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
