# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional

import numpy as np
from sympy import tensorproduct
import tensorflow as tf

from ..base import Parameter, TensorType
from ..utils import positive
from ..utils.ops import difference_matrix, square_distance
from .base_bayesian_kernel import ActiveDims, BayesianKernel


class BayesianStationary(BayesianKernel):
    """
    Base class for kernels that are stationary, that is, they only depend on

        d = x - x'

    This class handles 'ard' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
    """

    def __init__(
        self, **kwargs: Any
    ) -> None:
        """
        :param kwargs: accepts `name` and `active_dims`, which is a list or
            slice of indices which controls which columns of X are used (by
            default, all columns are used).
        """
        for kwarg in kwargs:
            if kwarg not in {"name", "active_dims"}:
                raise TypeError(f"Unknown keyword argument: {kwarg}")

        super().__init__(**kwargs)
        #self._validate_ard_active_dims(self.lengthscales)

    '''
    @property
    def ard(self) -> bool:
        """
        Whether ARD behaviour is active.
        """
        ndims: int = self.lengthscales.shape.ndims
        return ndims > 0
    '''

    def scale(self, X: TensorType, lengthscales: TensorType) -> TensorType:
        
        lengthscales = positive(lengthscales)
        X_scaled = X / lengthscales if X is not None else X
        return X_scaled

    def K_diag(self, X: TensorType, variance: TensorType) -> tf.Tensor:

        variance = positive(variance)
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(variance))


class IsotropicBayesianStationary(BayesianStationary):
    """
    Base class for isotropic stationary kernels, i.e. kernels that only
    depend on

        r = ‖x - x'‖

    Derived classes should implement one of:

        K_r2(self, r2): Returns the kernel evaluated on r² (r2), which is the
        squared scaled Euclidean distance Should operate element-wise on r2.

        K_r(self, r): Returns the kernel evaluated on r, which is the scaled
        Euclidean distance. Should operate element-wise on r.
    """

    def K(self, 
        variance: TensorType,
        lengthscales: TensorType,        
        X: TensorType, 
        X2: Optional[TensorType] = None) -> tf.Tensor:
        
        r2 = self.scaled_squared_euclid_dist(lengthscales, X, X2)
        return self.K_r2(r2, variance)

    def K_r2(self, r2: TensorType) -> tf.Tensor:
        if hasattr(self, "K_r"):
            # Clipping around the (single) float precision which is ~1e-45.
            r = tf.sqrt(tf.maximum(r2, 1e-36))
            return self.K_r(r)  # pylint: disable=no-member
        raise NotImplementedError

    def scaled_squared_euclid_dist(
        self, 
        lengthscales: TensorType,
        X: TensorType, 
        X2: Optional[TensorType] = None
    ) -> tf.Tensor:
        """
        Returns ‖(X - X2ᵀ) / ℓ‖², i.e. the squared L₂-norm.
        """
        lengthscales = positive(lengthscales)
        return square_distance(self.scale(X, lengthscales), self.scale(X2, lengthscales))

class BayesianSquaredExponential(IsotropicBayesianStationary):
    """
    The radial basis function (RBF) or squared exponential kernel. The kernel equation is

        k(r) = σ² exp{-½ r²}

    where:
    r   is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ.
    σ²  is the variance parameter

    Functions drawn from a GP with this kernel are infinitely differentiable!
    """

    def K_r2(self, r2: TensorType, variance: TensorType) -> tf.Tensor:
        variance = positive(variance)
        return variance * tf.exp(-0.5 * r2)
