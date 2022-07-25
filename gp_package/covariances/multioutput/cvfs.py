import tensorflow as tf
import tensorflow_probability as tfp

from typing import Any, Optional, Union

from gp_package.covariances.kufs import Kuf

from gpflow.base import TensorLike, TensorType


from gpflow.inducing_variables import (
    InducingPoints,
    SharedIndependentInducingVariables,
)
from gpflow.kernels import (
    MultioutputKernel,
    SharedIndependent,
)

from ..dispatch import Cvf


@Cvf.register(SharedIndependentInducingVariables, SharedIndependentInducingVariables, SharedIndependent, object)
def Kuf_shared_shared(
    inducing_variable_u: SharedIndependentInducingVariables,
    inducing_variable_v: SharedIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: tf.Tensor,
    *,
    L_Kuu: Optional[tf.Tensor] = None
) -> tf.Tensor:

    """
    Warning -- this currently works just with shared variables, hence a single kernel and associated hyperparameters per layer
    """
    return Cvf(inducing_variable_u.inducing_variable, inducing_variable_v.inducing_variable, kernel.kernel, Xnew, L_Kuu = L_Kuu)  # [M, N]


