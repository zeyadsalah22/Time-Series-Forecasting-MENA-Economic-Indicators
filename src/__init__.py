"""
Time Series Forecasting — MENA Economic Indicators.
Utilities for data collection, model estimation, evaluation, and visualization.
"""

__version__ = "1.0.0"
__author__ = "Zeyad Salah"

from . import data_processing
from . import models
from . import evaluation
from . import visualization

__all__ = ["data_processing", "models", "evaluation", "visualization"]
