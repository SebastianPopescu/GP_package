import tensorflow as tf

from typing import Any, Optional, Union

from gp_package.covariances.kuf import Kuf

from ...base import TensorLike, TensorType
from ...inducing_variables import SharedIndependentInducingVariables, SharedIndependentDistributionalInducingVariables
from ...kernels import SharedIndependent
import tensorflow_probability as tfp

def Kufs(
    inducing_variable: Union[SharedIndependentInducingVariables,SharedIndependentDistributionalInducingVariables],
    kernel: SharedIndependent,
    Xnew: Union[tf.Tensor, tfp.distributions.MultivariateNormalDiag],
    seed : Optional[Any] = None
) -> tf.Tensor:

    return Kuf(inducing_variable.inducing_variable, kernel.kernel, Xnew, seed = seed)  # [M, N]

