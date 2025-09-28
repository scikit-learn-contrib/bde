# Authors: scikit-learn-contrib developers
# License: BSD 3 clause

from ._template import TemplateClassifier, TemplateEstimator, TemplateTransformer
# from ._version import __version__
from .bde import BdeRegressor, BdeClassifier, Bde, BdePredictor

__version__ = "0.1"

__all__ = [
    "TemplateEstimator",
    "TemplateClassifier",
    "TemplateTransformer",
    "BdeRegressor",
    "BdeClassifier",
    "Bde",
    "BdePredictor",
    "__version__",
]
