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

import tensorflow as tf
import tensorflow_probability as tfp

from typing import Any, Union, Optional

from gpflow.config import default_float
from gpflow.inducing_variables import InducingPoints

from gpflow.base import TensorLike
from gpflow.utilities import to_default_float
from gpflow.kernels import Kernel
from .dispatch import Kuu
import gpflow
from ..inducing_variables import FourierFeatures1D
import numpy as np

BlockDiag = tf.linalg.LinearOperatorBlockDiag
Diag = tf.linalg.LinearOperatorDiag
LowRank = tf.linalg.LinearOperatorLowRankUpdate

@Kuu.register(InducingPoints, Kernel)
def Kuu_kernel_inducingpoints(
    inducing_variable: InducingPoints, kernel: Kernel, *, jitter: float = 0.0
) -> tf.Tensor:
    Kzz = kernel(inducing_variable.Z)
    Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)
    return Kzz

@Kuu.register(FourierFeatures1D, gpflow.kernels.Matern12)
def Kuu_matern12_fourierfeatures1d(inducing_variable, kernel, jitter=None):
    
    #NOTE - nice python-esque iterator
    a, b, ms = (lambda u: (u.a, u.b, u.ms))(inducing_variable)
    omegas = 2.0 * np.pi * ms / (b - a)

    # Cosine block:
    lamb = 1.0 / kernel.lengthscales
    two_or_four = to_default_float(tf.where(omegas == 0, 2.0, 4.0))
    d_cos = (
        (b - a) * (tf.square(lamb) + tf.square(omegas)) / lamb / kernel.variance / two_or_four
    )  # eq. (111)
    v_cos = tf.ones_like(d_cos) / tf.sqrt(kernel.variance)  # eq. (110)
    cosine_block = LowRank(Diag(d_cos, is_positive_definite=True), v_cos[:, None])

    # Sine block:
    omegas = omegas[tf.not_equal(omegas, 0)]  # the sine block does not include omega=0
    d_sin = (
        (b - a) * (tf.square(lamb) + tf.square(omegas)) / lamb / kernel.variance / 4.0
    )  # eq. (113)
    sine_block = Diag(d_sin, is_positive_definite=True)

    return BlockDiag([cosine_block, sine_block])

