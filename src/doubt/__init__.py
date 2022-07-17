"""
.. include:: ../../README.md
"""

import pkg_resources

from .models import Boot  # noqa
from .models import QuantileRegressionForest  # noqa
from .models import QuantileRegressionTree  # noqa
from .models import QuantileRegressor  # noqa

# Fetches the version of the package as defined in pyproject.toml
__version__ = pkg_resources.get_distribution("doubt").version
