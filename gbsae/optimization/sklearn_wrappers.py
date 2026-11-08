"""
Sklearn-based wrappers for dictionary learning models.
"""

import numpy as np
import torch
from sklearn.decomposition import (
    PCA,
    NMF,
    FastICA,
    TruncatedSVD,
    SparsePCA,
    DictionaryLearning,
)
from sklearn.cluster import KMeans

from .base import BaseOptimDictionaryLearning


class SkPCA(BaseOptimDictionaryLearning):
    """Centered PCA wrapper with torch-facing encode/decode outputs.
    
    encode uses estimator.transform; decode uses inverse_transform, including
    the fitted mean. get_dictionary returns components_, without that mean.
    n_components and extra kwargs configure sklearn PCA; device controls output tensors.
    """

    def __init__(self, n_components, device="cpu", **kwargs):
        super().__init__(n_components, device)
        self.model = PCA(n_components=n_components, **kwargs)

    def encode(self, x):
        """
        Encode the input tensor (the activations) using PCA.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, dims).

        Returns
        -------
        torch.Tensor
            Encoded features (the codes).
        """
        self._assert_fitted()

        # @tfel: see to adapt for online algorithms
        x = self.sanitize_np_input(x)

        z = self.model.transform(x)
        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        return z

    def decode(self, z):
        """
        Decode the input tensor (the codes) using PCA.

        Parameters
        ----------
        z : torch.Tensor
            Encoded tensor (the codes) of shape (batch_size, nb_components).

        Returns
        -------
        torch.Tensor
            Decoded output (the activations).
        """
        self._assert_fitted()

        z = self.sanitize_np_codes(z)

        x_hat = self.model.inverse_transform(z)
        x_hat = torch.tensor(x_hat, dtype=torch.float32, device=self.device)
        return x_hat

    def fit(self, x):
        """
        Fit the PCA model to the input data.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, ...).
        """
        x = self.sanitize_np_input(x)
        self.model.fit(x)
        self._set_fitted()

    def get_dictionary(self):
        """
        Return the learned dictionary components from PCA.

        Returns
        -------
        torch.Tensor
            Dictionary components.
        """
        self._assert_fitted()
        return torch.tensor(
            self.model.components_, dtype=torch.float32, device=self.device
        )


class SkICA(BaseOptimDictionaryLearning):
    """FastICA wrapper with torch-facing outputs.
    
    encode uses estimator.transform. decode computes z @ mixing_.T without
    adding the estimator's fitted mean. get_dictionary returns components_
    (the unmixing matrix), rather than the mixing matrix used by decode.
    """

    def __init__(self, n_components, device="cpu", **kwargs):
        super().__init__(n_components, device)
        self.model = FastICA(n_components=n_components, **kwargs)

    def encode(self, x):
        """
        Encode the input tensor (the activations) using ICA.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, dims).

        Returns
        -------
        torch.Tensor
            Encoded features (the codes).
        """
        self._assert_fitted()

        x = self.sanitize_np_input(x)

        z = self.model.transform(x)
        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        return z

    def decode(self, z):
        """
        Decode the input tensor (the codes) using ICA.

        Parameters
        ----------
        z : torch.Tensor
            Encoded tensor (the codes) of shape (batch_size, nb_components).

        Returns
        -------
        torch.Tensor
            Decoded output (the activations).
        """
        self._assert_fitted()

        z = self.sanitize_np_codes(z)

        x_hat = np.dot(z, self.model.mixing_.T)
        x_hat = torch.tensor(x_hat, dtype=torch.float32, device=self.device)
        return x_hat

    def fit(self, x):
        """
        Fit the ICA model to the input data.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, ...).
        """
        x = self.sanitize_np_input(x)

        self.model.fit(x)
        self._set_fitted()

    def get_dictionary(self):
        """
        Return the learned dictionary components from ICA.

        Returns
        -------
        torch.Tensor
            Dictionary components.
        """
        self._assert_fitted()
        return torch.tensor(
            self.model.components_, dtype=torch.float32, device=self.device
        )


class SkNMF(BaseOptimDictionaryLearning):
    """
    NMF-based Dictionary Learning model.

    Solve the following optimization problem:
    min ||X - ZD||_F^2. s.t Z >=0, D >= 0.

    Parameters
    ----------
    n_components : int
        Number of components to learn.
    device : str, optional
        Device to use for tensor computations, by default 'cpu'
    """

    def __init__(self, n_components, device="cpu", **kwargs):
        super().__init__(n_components, device)
        self.model = NMF(n_components=n_components, **kwargs)

    def encode(self, x):
        """
        Encode the input tensor (the activations) using NMF.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, dims).

        Returns
        -------
        torch.Tensor
            Encoded features (the codes).
        """
        self._assert_fitted()

        x = self.sanitize_np_input(x)
        assert (x >= 0).all(), "Input tensor must be non-negative"

        z = self.model.transform(x)
        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        return z

    def decode(self, z):
        """
        Decode the input tensor (the codes) using NMF.

        Parameters
        ----------
        z : torch.Tensor
            Encoded tensor (the codes) of shape (batch_size, nb_components).

        Returns
        -------
        torch.Tensor
            Decoded output (the activations).
        """
        self._assert_fitted()

        z = self.sanitize_np_codes(z)

        x_hat = np.dot(z, self.model.components_)
        x_hat = torch.tensor(x_hat, dtype=torch.float32, device=self.device)
        return x_hat

    def fit(self, x):
        """
        Fit the NMF model to the input data.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, ...).
        """
        x = self.sanitize_np_input(x)
        assert (x >= 0).all(), "Input tensor must be non-negative"

        self.model.fit(x)
        self._set_fitted()

    def get_dictionary(self):
        """
        Return the learned dictionary components from NMF.

        Returns
        -------
        torch.Tensor
            Dictionary components.
        """
        self._assert_fitted()
        return torch.tensor(
            self.model.components_, dtype=torch.float32, device=self.device
        )


class SkKMeans(BaseOptimDictionaryLearning):
    """KMeans wrapper exposing center distances as codes.
    
    encode returns estimator.transform(X), the distances to every center, not
    one-hot cluster assignments. decode multiplies the supplied code matrix by
    cluster_centers_ without converting distances to assignments. get_dictionary
    returns the center matrix. n_components is the cluster count; device controls
    the returned torch tensors; estimator kwargs are passed through.
    """

    def __init__(self, n_components, device="cpu", **kwargs):
        super().__init__(n_components, device)
        self.model = KMeans(n_clusters=n_components, **kwargs)

    def encode(self, x):
        """
        Encode the input tensor (the activations) using KMeans.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, dims).

        Returns
        -------
        torch.Tensor
            Encoded features (the codes).
        """
        self._assert_fitted()

        x = self.sanitize_np_input(x)

        z = self.model.transform(x)
        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        return z

    def decode(self, z):
        """
        Decode the input tensor (the codes) using KMeans.

        Parameters
        ----------
        z : torch.Tensor
            Encoded tensor (the codes) of shape (batch_size, nb_components).

        Returns
        -------
        torch.Tensor
            Decoded output (the activations).
        """
        self._assert_fitted()

        z = self.sanitize_np_codes(z)

        cluster_centers = self.model.cluster_centers_
        x_hat = np.dot(z, cluster_centers)
        x_hat = torch.tensor(x_hat, dtype=torch.float32, device=self.device)
        return x_hat

    def fit(self, x):
        """
        Fit the KMeans model to the input data.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, ...).
        """
        x = self.sanitize_np_input(x)

        self.model.fit(x)
        self._set_fitted()

    def get_dictionary(self):
        """
        Return the learned dictionary components from KMeans.

        Returns
        -------
        torch.Tensor
            Dictionary components.
        """
        self._assert_fitted()
        return torch.tensor(
            self.model.cluster_centers_, dtype=torch.float32, device=self.device
        )


class SkDictionaryLearning(BaseOptimDictionaryLearning):
    """Sklearn dictionary learning with nonnegative transformed codes.
    
    sparsity maps to alpha (default 1). fit_algorithm defaults to cd,
    transform_algorithm to lasso_cd, and tolerance to tol=1e-3. decode computes
    z @ components_; device controls returned torch tensors.
    """

    def __init__(
        self,
        n_components,
        device="cpu",
        sparsity=1.0,
        fit_algorithm="cd",
        transform_algorithm="lasso_cd",
        tolerance=1e-3,
        **kwargs,
    ):
        super().__init__(n_components, device)
        self.model = DictionaryLearning(
            n_components=n_components,
            alpha=sparsity,
            positive_code=True,
            fit_algorithm=fit_algorithm,
            transform_algorithm=transform_algorithm,
            tol=tolerance,
            **kwargs,
        )

    def encode(self, x):
        """
        Encode the input tensor (the activations) using DictionaryLearning.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, dims).

        Returns
        -------
        torch.Tensor
            Encoded features (the codes).
        """
        self._assert_fitted()

        x = self.sanitize_np_input(x)

        z = self.model.transform(x)
        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        return z

    def decode(self, z):
        """
        Decode the input tensor (the codes) using DictionaryLearning.

        Parameters
        ----------
        z : torch.Tensor
            Encoded tensor (the codes) of shape (batch_size, nb_components).

        Returns
        -------
        torch.Tensor
            Decoded output (the activations).
        """
        self._assert_fitted()

        z = self.sanitize_np_codes(z)

        x_hat = z @ self.model.components_
        x_hat = torch.tensor(x_hat, dtype=torch.float32, device=self.device)
        return x_hat

    def fit(self, x):
        """
        Fit the DictionaryLearning model to the input data.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, ...).
        """
        x = self.sanitize_np_input(x)

        self.model.fit(x)
        self._set_fitted()

    def get_dictionary(self):
        """
        Return the learned dictionary components from DictionaryLearning.

        Returns
        -------
        torch.Tensor
            Dictionary components.
        """
        self._assert_fitted()
        return torch.tensor(
            self.model.components_, dtype=torch.float32, device=self.device
        )


class SkSparsePCA(BaseOptimDictionaryLearning):
    """SparsePCA wrapper using sklearn transform and inverse_transform.
    
    The component matrix is sparse and is not constrained to be orthogonal.
    sparsity maps to alpha (default 1), fit_method to method (default cd), and
    tolerance to tol (default 1e-3). n_components, device and extra estimator
    kwargs select the decomposition size, output device and remaining settings.
    """

    def __init__(
        self,
        n_components,
        device="cpu",
        sparsity=1.0,
        fit_method="cd",
        tolerance=1e-3,
        **kwargs,
    ):
        super().__init__(n_components, device)
        self.model = SparsePCA(
            n_components=n_components,
            alpha=sparsity,
            method=fit_method,
            tol=tolerance,
            **kwargs,
        )

    def encode(self, x):
        """
        Encode the input tensor (the activations) using SparsePCA.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, dims).

        Returns
        -------
        torch.Tensor
            Encoded features (the codes).
        """
        self._assert_fitted()

        x = self.sanitize_np_input(x)

        z = self.model.transform(x)
        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        return z

    def decode(self, z):
        """
        Decode the input tensor (the codes) using SparsePCA.

        Parameters
        ----------
        z : torch.Tensor
            Encoded tensor (the codes) of shape (batch_size, nb_components).

        Returns
        -------
        torch.Tensor
            Decoded output (the activations).
        """
        self._assert_fitted()

        z = self.sanitize_np_codes(z)

        x_hat = self.model.inverse_transform(z)
        x_hat = torch.tensor(x_hat, dtype=torch.float32, device=self.device)
        return x_hat

    def fit(self, x):
        """
        Fit the SparsePCA model to the input data.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, ...).
        """
        x = self.sanitize_np_input(x)

        self.model.fit(x)
        self._set_fitted()

    def get_dictionary(self):
        """
        Return the learned dictionary components from SparsePCA.

        Returns
        -------
        torch.Tensor
            Dictionary components.
        """
        self._assert_fitted()
        return torch.tensor(
            self.model.components_, dtype=torch.float32, device=self.device
        )


class SkSVD(BaseOptimDictionaryLearning):
    """
    SVD-based Dictionary Learning model.

    Solve the following optimization problem:
    min ||X - ZD||_F^2 s.t D is orthogonal.

    Parameters
    ----------
    n_components : int
        Number of components to learn.
    device : str, optional
        Device to use for tensor computations, by default 'cpu'
    """

    def __init__(self, n_components, device="cpu", **kwargs):
        super().__init__(n_components, device)
        self.model = TruncatedSVD(n_components=n_components, **kwargs)

    def encode(self, x):
        """
        Encode the input tensor (the activations) using SVD.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, dims).

        Returns
        -------
        torch.Tensor
            Encoded features (the codes).
        """
        self._assert_fitted()

        x = self.sanitize_np_input(x)

        z = self.model.transform(x)
        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        return z

    def decode(self, z):
        """
        Decode the input tensor (the codes) using SVD.

        Parameters
        ----------
        z : torch.Tensor
            Encoded tensor (the codes) of shape (batch_size, nb_components).

        Returns
        -------
        torch.Tensor
            Decoded output (the activations).
        """
        self._assert_fitted()

        z = self.sanitize_np_codes(z)

        x_hat = self.model.inverse_transform(z)
        x_hat = torch.tensor(x_hat, dtype=torch.float32, device=self.device)
        return x_hat

    def fit(self, x):
        """
        Fit the SVD model to the input data.

        Parameters
        ----------
        x : torch.Tensor or Iterable
            Input tensor of shape (batch_size, ...).
        """
        x = self.sanitize_np_input(x)

        self.model.fit(x)
        self._set_fitted()

    def get_dictionary(self):
        """
        Return the learned dictionary components from TruncatedSVD.

        Returns
        -------
        torch.Tensor
            Dictionary components.
        """
        self._assert_fitted()
        return torch.tensor(
            self.model.components_, dtype=torch.float32, device=self.device
        )
