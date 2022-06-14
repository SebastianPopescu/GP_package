import tensorflow as tf

from typing import Any, Optional, Union

from gp_package.covariances.kuf import Kuf

from ...base import TensorLike, TensorType
from ...inducing_variables import SharedIndependentInducingVariables
from ...kernels import SharedIndependent
import tensorflow_probability as tfp

def Kufs(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: tf.Tensor,
    seed : Optional[Any] = None
) -> tf.Tensor:

    """
    Warning -- this currently works just with shared variables, hence a single kernel and associated hyperparameters per layer
    """
    return Kuf(inducing_variable.inducing_variable, kernel.kernel, Xnew, seed = seed)  # [M, N]

