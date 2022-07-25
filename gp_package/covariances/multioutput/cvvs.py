import tensorflow as tf


from typing import Union
from gpflow.inducing_variables import FallbackSharedIndependentInducingVariables
from gpflow.base import TensorLike
from gpflow.kernels import (
    SharedIndependent,
)
from ..dispatch import Cvv
from typing import Optional, Any

@Cvv.register(FallbackSharedIndependentInducingVariables, FallbackSharedIndependentInducingVariables, SharedIndependent)
def Cvv_shared_shared(
    inducing_variable_u:FallbackSharedIndependentInducingVariables,
    inducing_variable_v:FallbackSharedIndependentInducingVariables,
    kernel: SharedIndependent,
    *,
    jitter: float = 0.0,
    L_Kuu: Optional[tf.Tensor] = None
) -> tf.Tensor:

    """
    Warning -- this currently works just with shared variables, hence a single kernel and associated hyperparameters per layer
    """
    
    _Cvv = Cvv(inducing_variable_u.inducing_variable, inducing_variable_v.inducing_variable, kernel.kernel, L_Kuu = L_Kuu)  # [M, M]
    jittermat = tf.eye(inducing_variable_v.num_inducing, dtype=_Cvv.dtype) * jitter
    return _Cvv + jittermat





