import tensorflow as tf
import tensorflow_probability as tfp

from typing import Any, Optional, Union

from gp_package.covariances.kufs import Kuf

from gpflow.base import TensorLike, TensorType
from ...inducing_variables import SharedIndependentDistributionalInducingVariables


from gpflow.inducing_variables import (
    InducingPoints,
    SharedIndependentInducingVariables,
)
from gpflow.kernels import (
    MultioutputKernel,
    SharedIndependent,
)

from ...kernels import DistributionalSharedIndependent
from ..dispatch import Kuf


@Kuf.register(SharedIndependentInducingVariables, SharedIndependent, object)
def Kuf_shared_shared(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: tf.Tensor,
) -> tf.Tensor:
    return Kuf(inducing_variable.inducing_variable, kernel.kernel, Xnew)  # [M, N]


@Kuf.register(object, SharedIndependentDistributionalInducingVariables, DistributionalSharedIndependent, object, object)
def Kuf_shared_shared(
    sampled_inducing_points: TensorLike,
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: tf.Tensor,
    Xnew_moments: tfp.distributions.MultivariateNormalDiag
) -> tf.Tensor:
    return Kuf(sampled_inducing_points, inducing_variable.inducing_variable, kernel.kernel, Xnew, Xnew_moments)  # [M, N]
