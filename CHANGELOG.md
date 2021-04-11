# Change log

## v2.1.0

- Implemented `score` method to `QuantileLinearRegression`, which either
  outputs the mean negative pinball loss function, or the R^2 value
- Outputs more informative error message when a singular feature matrix is
  being used with `QuantileLinearRegression`
- Added more documentation to `QuantileLinearRegression`
- Datasets look prettier in notebooks now
- Removed docstring comments about closing datasets after use, as this is
  automatic

## v2.0.2

- Small mistake in the computation of the prediction intervals in
  `Boot.predict`, where the definition of `generalisation` should be the
  difference of the _means_ of the residuals, and not the difference between
  the individual quantiles. Makes a very tiny difference to the prediction
  intervals. Thanks to Bryan Shalloway for catching this mistake.


## v2.0.1

- `Boot.__repr__` was not working properly


## v2.0.0

- Changed the ordering of `Dataset.split` to `X_train`, `X_test`, `y_train`
  and `y_test`, to agree with `scikit-learn`
- Added proper `__repr__` descriptions to all models
- Moved some attributes to the private API
