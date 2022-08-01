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

import abc
from ast import Param
from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp
from deprecated import deprecated

from gpflow.base import Module, Parameter, TensorData, TensorType
from gpflow.utilities import positive

import numpy as np

class FourierFeatures1D(Module):
    """
    Abstract base class for RKHS Fourier Features
    'a' and 'b' will define the boundaries where the GP can operate on
    """

    @property
    @abc.abstractmethod
    def num_inducing(self) -> tf.Tensor:
        """
        Returns the number of inducing variables, relevant for example to determine the size of the
        variational distribution.
        """
        raise NotImplementedError

    @deprecated(
        reason="len(iv) should return an `int`, but this actually returns a `tf.Tensor`."
        " Use `iv.num_inducing` instead."
    )
    def __len__(self) -> tf.Tensor:
        return self.num_inducing


class FourierPoints1DBase(FourierFeatures1D):
    def __init__(self, a: TensorData, b: TensorData, M: int, name: Optional[str] = None):
        """
        :param a: lower bound on the interval where the Fourier representation operates
        :param b: upper bound on the interval where the Fourier representation operates
        :param M: number of frequencies to use

        """
        super().__init__(name=name)

        #TODO -- shouldn't these be set to trainable=False?
        self.a = Parameter(a)
        self.b = Parameter(b)
        self.ms = np.arange(M)
        """
        if not isinstance(Z_mean, (tf.Variable, tfp.util.TransformedVariable)):
            Z_mean = Parameter(Z_mean)
        self.Z_mean = Z_mean

        if not isinstance(Z_var, (tf.Variable, tfp.util.TransformedVariable)):
            Z_var = Parameter(value=Z_var, transform=positive())
        self.Z_var = Z_var
        """

    @property
    def num_inducing(self) -> Optional[tf.Tensor]:
        return 2 * tf.shape(self.ms)[0] - 1
    

class FourierPoints1D(FourierPoints1DBase):
    """
    Fourier space frequencies
    """

