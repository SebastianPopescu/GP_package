import tensorflow as tf

from ..config import default_float
from ..inducing_variables import InducingPoints
from ..kernels import Kernel, SquaredExponential

def Kuu(
    inducing_variable: InducingPoints, kernel: Kernel, *, jitter: float = 0.0
) -> tf.Tensor:
    Kzz = kernel(inducing_variable.Z)
    Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)
    return Kzz