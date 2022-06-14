from sre_constants import ANY
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Any, Union, Optional

from ..config import default_float
from ..inducing_variables import InducingPoints
from ..kernels import Kernel, SquaredExponential

def Kuu(
    inducing_variable: InducingPoints, kernel: Kernel, *, jitter: float = 0.0,
    seed : Optional[Any]  = None,
) -> tf.Tensor:

    Kzz = kernel(inducing_variable.Z)
    Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)
    return Kzz