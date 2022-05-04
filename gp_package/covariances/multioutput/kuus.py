import tensorflow as tf

from typing import Any, Union, Optional

from gp_package.covariances.kuu import Kuu

from ...config import default_float
from ...inducing_variables import FallbackSharedIndependentInducingVariables, FallbackSharedIndependentDistributionalInducingVariables
from ...kernels import SharedIndependent

def Kuus(
    inducing_variable: Union[FallbackSharedIndependentInducingVariables,FallbackSharedIndependentDistributionalInducingVariables],
    kernel: SharedIndependent,
    *,
    jitter: float = 0.0,
    seed : Optional[Any] = None
) -> tf.Tensor:


    """
    Warning -- this currently works just with shared variables, hence a single kernel and associated hyperparameters per layer
    """
    
    Kmm = Kuu(inducing_variable.inducing_variable, kernel.kernel, seed = seed)  # [M, M]
    jittermat = tf.eye(inducing_variable.num_inducing, dtype=Kmm.dtype) * jitter
    return Kmm + jittermat





