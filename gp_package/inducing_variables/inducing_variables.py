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
from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp
from deprecated import deprecated

from ..base import Module, Parameter, TensorData, TensorType
from ..utils import positive


class InducingVariables(Module):
    """
    Abstract base class for inducing variables.
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


class InducingPointsBase(InducingVariables):
    def __init__(self, Z: TensorData, name: Optional[str] = None):
        """
        :param Z: the initial positions of the inducing points, size [M, D]
        """
        super().__init__(name=name)
        if not isinstance(Z, (tf.Variable, tfp.util.TransformedVariable)):
            Z = Parameter(Z)
        self.Z = Z

    @property
    def num_inducing(self) -> Optional[tf.Tensor]:
        return tf.shape(self.Z)[0]


class InducingPoints(InducingPointsBase):
    """
    Real-space inducing points
    """

