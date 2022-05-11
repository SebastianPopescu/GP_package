import tensorflow as tf

from gp_package.base import TensorType

from ..config import default_float
from ..inducing_variables import InducingPoints
from ..kernels import BayesianKernel, BayesianSquaredExponential

def BayesianKuu(
    inducing_variable: InducingPoints, kernel: BayesianKernel,
    variance: TensorType,
    lengthscales: TensorType,
    *, jitter: float = 0.0
) -> tf.Tensor:
    Kzz = kernel(variance,inducing_variable.Z, lengthscales=lengthscales)
    Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)
    return Kzz