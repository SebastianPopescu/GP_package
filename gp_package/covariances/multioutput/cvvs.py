import tensorflow as tf

from typing import Any, Union, Optional

from gp_package.covariances.cvv import Cvv

from ...config import default_float
from ...inducing_variables import FallbackSharedIndependentInducingVariables
from ...kernels import SharedIndependent

def Cvvs(
    inducing_variable:FallbackSharedIndependentInducingVariables,
    kernel: SharedIndependent,
    inducing_variable_anchor: FallbackSharedIndependentInducingVariables,
    *,
    jitter: float = 0.0,
    seed : Optional[Any] = None
) -> tf.Tensor:

    """
    Warning -- this currently works just with shared variables, hence a single kernel and associated hyperparameters per layer
    """
    
    _Cvv = Cvv(inducing_variable.inducing_variable, kernel.kernel, inducing_variable_anchor.inducing_variable, seed = seed)  # [M, M]
    jittermat = tf.eye(inducing_variable.num_inducing, dtype=Cvv.dtype) * jitter
    return _Cvv + jittermat





