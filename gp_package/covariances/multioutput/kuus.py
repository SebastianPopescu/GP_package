import tensorflow as tf


from typing import Union
from gpflow.inducing_variables import FallbackSharedIndependentInducingVariables

from gp_package.base import TensorLike


from ...inducing_variables import (
    FallbackSharedIndependentDistributionalInducingVariables,
)
from gpflow.kernels import (
    SharedIndependent,
)
from ..dispatch import Kuu


@Kuu.register(FallbackSharedIndependentInducingVariables, SharedIndependent)
def Kuu_shared_shared(
    inducing_variable: FallbackSharedIndependentInducingVariables,
    kernel: SharedIndependent,
    *,
    jitter: float = 0.0,
) -> tf.Tensor:
    Kmm = Kuu(inducing_variable.inducing_variable, kernel.kernel)  # [M, M]
    jittermat = tf.eye(inducing_variable.num_inducing, dtype=Kmm.dtype) * jitter
    return Kmm + jittermat

@Kuu.register(FallbackSharedIndependentDistributionalInducingVariables, object, SharedIndependent)
def Kuu_distributional_shared_shared(
    inducing_variable: FallbackSharedIndependentDistributionalInducingVariables,
    sampled_inducing_points: TensorLike,
    kernel: SharedIndependent,
    *,
    jitter: float = 0.0,
) -> tf.Tensor:

    Kmm = Kuu(inducing_variable.inducing_variable, sampled_inducing_points, kernel.kernel)  # [M, M]
    jittermat = tf.eye(inducing_variable.num_inducing, dtype=Kmm.dtype) * jitter
    return Kmm + jittermat

