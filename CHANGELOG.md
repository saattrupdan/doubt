# Change log

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
