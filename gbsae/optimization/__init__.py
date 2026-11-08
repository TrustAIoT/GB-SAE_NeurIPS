"""Optimization-based dictionary initialization adapters, derived from Overcomplete.
"""

from .base import BaseOptimDictionaryLearning
from .sklearn_wrappers import (
    SkPCA,
    SkICA,
    SkNMF,
    SkKMeans,
    SkDictionaryLearning,
    SkSparsePCA,
    SkSVD,
)
