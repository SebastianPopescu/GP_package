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
import tensorflow as tf

from ..base import Parameter, TensorType
from ..utils import positive
from ..utils.ops import difference_matrix, square_distance
from .base_kernel import ActiveDims, Kernel


class Stationary(Kernel):
    """
    Base class for kernels that are stationary, that is, they only depend on

        d = x - x'

    This class handles 'ard' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
    """

    def __init__(
        self, variance: TensorType = 1.0, lengthscales: TensorType = 1.0, **kwargs: Any
    ) -> None:
        """
        :param variance: the (initial) value for the variance parameter.
        :param lengthscales: the (initial) value for the lengthscale
            parameter(s), to induce ARD behaviour this must be initialised as
            an array the same length as the the number of active dimensions
            e.g. [1., 1., 1.]. If only a single value is passed, this value
            is used as the lengthscale of each dimension.
        :param kwargs: accepts `name` and `active_dims`, which is a list or
            slice of indices which controls which columns of X are used (by
            default, all columns are used).
        """
        for kwarg in kwargs:
            if kwarg not in {"name", "active_dims"}:
                raise TypeError(f"Unknown keyword argument: {kwarg}")

        super().__init__(**kwargs)
        self.variance = Parameter(variance, transform=positive())
        self.lengthscales = Parameter(lengthscales, transform=positive())
        self._validate_ard_active_dims(self.lengthscales)

    @property
    def ard(self) -> bool:
        """
        Whether ARD behaviour is active.
        """
        ndims: int = self.lengthscales.shape.ndims
        return ndims > 0

    def scale(self, X: TensorType) -> TensorType:
        X_scaled = X / self.lengthscales if X is not None else X
        return X_scaled

    def K_diag(self, X: TensorType) -> tf.Tensor:
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))


class IsotropicStationary(Stationary):
    """
    Base class for isotropic stationary kernels, i.e. kernels that only
    depend on

        r = ???x - x'???

    Derived classes should implement one of:

        K_r2(self, r2): Returns the kernel evaluated on r?? (r2), which is the
        squared scaled Euclidean distance Should operate element-wise on r2.

        K_r(self, r): Returns the kernel evaluated on r, which is the scaled
        Euclidean distance. Should operate element-wise on r.
    """

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        
        print('----inside K from IsotropicStationary ----')
        print(X2)
        print(X)

        r2 = self.scaled_squared_euclid_dist(X, X2)
        return self.K_r2(r2)

    def K_r2(self, r2: TensorType) -> tf.Tensor:
        if hasattr(self, "K_r"):
            # Clipping around the (single) float precision which is ~1e-45.
            r = tf.sqrt(tf.maximum(r2, 1e-36))
            return self.K_r(r)  # pylint: disable=no-member
        raise NotImplementedError

    def scaled_squared_euclid_dist(
        self, X: TensorType, X2: Optional[TensorType] = None
    ) -> tf.Tensor:
        """
        Returns ???(X - X2???) / ????????, i.e. the squared L???-norm.
        """
        return square_distance(self.scale(X), self.scale(X2))

class SquaredExponential(IsotropicStationary):
    """
    The radial basis function (RBF) or squared exponential kernel. The kernel equation is

        k(r) = ???? exp{-?? r??}

    where:
    r   is the Euclidean distance between the input points, scaled by the lengthscales parameter ???.
    ????  is the variance parameter

    Functions drawn from a GP with this kernel are infinitely differentiable!
    """

    def K_r2(self, r2: TensorType) -> tf.Tensor:
        return self.variance * tf.exp(-0.5 * r2)
