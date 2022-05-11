import tensorflow as tf

from gp_package.base import TensorType

from ...config import default_float
from ...inducing_variables import FallbackSharedIndependentInducingVariables
from ...kernels import SharedIndependent
from gp_package.covariances.bayesian_kuu import BayesianKuu


def Kuus(
    inducing_variable: FallbackSharedIndependentInducingVariables,
    kernel: SharedIndependent,
    variance: TensorType,
    lengthscales: TensorType,
    *,
    jitter: float = 0.0,
) -> tf.Tensor:
    
    Kmm = BayesianKuu(inducing_variable.inducing_variable, kernel.kernel, variance, lengthscales)  # [M, M]
    jittermat = tf.eye(inducing_variable.num_inducing, dtype=Kmm.dtype) * jitter
    return Kmm + jittermat





