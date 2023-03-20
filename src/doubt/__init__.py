"""
.. include:: ../../README.md
"""

import importlib.metadata

from .models import Boot  # noqa
from .models import QuantileRegressionForest  # noqa
from .models import QuantileRegressionTree  # noqa
from .models import QuantileRegressor  # noqa

# Fetches the version of the package as defined in pyproject.toml
__version__ = importlib.metadata.version(__package__)
