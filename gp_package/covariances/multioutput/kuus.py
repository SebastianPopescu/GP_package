import tensorflow as tf

from ...config import default_float
from ...inducing_variables import FallbackSharedIndependentInducingVariables
from ...kernels import SharedIndependent

def _Kuu(inducing_variable, kernel, *, jitter: float = 0.0):
    Kzz = kernel(inducing_variable.Z)
    Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)
    return Kzz

def Kuus(
    inducing_variable: FallbackSharedIndependentInducingVariables,
    kernel: SharedIndependent,
    *,
    jitter: float = 0.0,
) -> tf.Tensor:
    
    Kmm = _Kuu(inducing_variable.inducing_variable, kernel.kernel)  # [M, M]
    jittermat = tf.eye(inducing_variable.num_inducing, dtype=Kmm.dtype) * jitter
    return Kmm + jittermat





