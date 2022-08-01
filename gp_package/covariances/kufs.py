import tensorflow as tf
import tensorflow_probability as tfp

from typing import Any, Optional, Union

from gpflow.base import TensorLike, TensorType
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import Kernel
from .dispatch import Kuf
import gpflow
from ..inducing_variables import FourierFeatures1D
import numpy as np


@Kuf.register(InducingPoints, Kernel, TensorLike)
def Kuf_kernel_inducingpoints(
    inducing_variable: InducingPoints, kernel: Kernel, Xnew: TensorType
) -> tf.Tensor:
    return kernel(inducing_variable.Z, Xnew)

@Kuf.register(FourierFeatures1D, gpflow.kernels.Matern12, TensorLike)
def Kuf_matern12_fourierfeatures1d(inducing_variable, kernel, X):
    X = tf.squeeze(X, axis=1)
    a, b, ms = (lambda u: (u.a, u.b, u.ms))(inducing_variable)

    omegas = 2.0 * np.pi * ms / (b - a)
    Kuf_cos = tf.cos(omegas[:, None] * (X[None, :] - a))
    omegas_sin = omegas[tf.not_equal(omegas, 0)]  # don't compute zero frequency
    Kuf_sin = tf.sin(omegas_sin[:, None] * (X[None, :] - a))


    #TODO -- need to read up on this part of the paper; what to do when outside of the [a,b] interval
    # correct Kuf outside [a, b] -- see Table 1
    Kuf_sin = tf.where((X < a) | (X > b), tf.zeros_like(Kuf_sin), Kuf_sin)  # just zero

    left_tail = tf.exp(-tf.abs(X - a) / kernel.lengthscales)[None, :]
    right_tail = tf.exp(-tf.abs(X - b) / kernel.lengthscales)[None, :]
    Kuf_cos = tf.where(X < a, left_tail, Kuf_cos)  # replace with left tail
    Kuf_cos = tf.where(X > b, right_tail, Kuf_cos)  # replace with right tail

    return tf.concat([Kuf_cos, Kuf_sin], axis=0)