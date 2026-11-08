"""Adapters for optimization-based dictionary decompositions.

Estimators operate on NumPy float32 matrices. Their model-specific constraints
and transform/decode conventions are defined by each wrapper.
"""

import numpy as np
from sklearn.decomposition import (
    PCA,
    NMF,
    FastICA,
    TruncatedSVD,
    SparsePCA,
    DictionaryLearning,
)
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader

from ..base import BaseDictionaryLearning
from gbsae.utils.arrays import to_npf32, unwrap_dataloader


class BaseOptimDictionaryLearning(BaseDictionaryLearning):
    """
    Abstract base class for optimization-based Dictionary Learning models.

    Parameters
    ----------
    n_components : int
        Number of components to learn.
    device : str, optional
        Device to use for tensor computations, by default 'cpu'
    """

    def sanitize_np_input(self, x):
        """Convert a tensor/array or DataLoader to a two-dimensional NumPy float32 matrix.
        
        Dataset batches are concatenated; tuple/list batches contribute their first item.
        """
        if isinstance(x, DataLoader):
            x = unwrap_dataloader(x)
        x = to_npf32(x)
        assert x.ndim == 2, "Input tensor must have 2 dimensions"
        return x

    def sanitize_np_codes(self, z):
        """Convert a tensor/array or DataLoader to a two-dimensional NumPy float32 matrix.
        
        Codes must additionally have n_components columns.
        """
        if isinstance(z, DataLoader):
            z = unwrap_dataloader(z)
        z = to_npf32(z)
        assert (
            z.ndim == 2 and z.shape[1] == self.n_components
        ), "Input tensor must have 2 dimensions and n_components columns"
        return z
