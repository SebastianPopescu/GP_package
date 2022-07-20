import tensorflow as tf
import tensorflow_probability as tfp

from typing import Any, Optional, Union

from gpflow.base import TensorLike, TensorType
from gpflow.inducing_variables import InducingPoints
from ..inducing_variables import DistributionalInducingPoints
from gpflow.kernels import Kernel
from .dispatch import Kuf
from ..kernels import DistributionalKernel



@Kuf.register(InducingPoints, Kernel, TensorLike)
def Kuf_kernel_inducingpoints(
    inducing_variable: InducingPoints, kernel: Kernel, Xnew: TensorType
) -> tf.Tensor:
    return kernel(inducing_variable.Z, Xnew)

@Kuf.register(object, DistributionalInducingPoints, DistributionalKernel, TensorLike, tfp.distributions.MultivariateNormalDiag)
def Kuf_kernel_distributionalinducingpoints( sampled_inducing_points: TensorLike,
    inducing_variable: DistributionalInducingPoints, kernel: DistributionalKernel, Xnew: TensorType, Xnew_moments: tfp.distributions.MultivariateNormalDiag) -> tf.Tensor:

    distributional_inducing_points = inducing_variable.distribution

    return kernel(sampled_inducing_points, distributional_inducing_points, Xnew, Xnew_moments)
