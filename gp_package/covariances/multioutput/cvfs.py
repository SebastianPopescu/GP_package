import tensorflow as tf

from typing import Any, Optional, Union

from gp_package.covariances.cvf import Cvf

from ...base import TensorLike, TensorType
from ...inducing_variables import SharedIndependentInducingVariables
from ...kernels import SharedIndependent
import tensorflow_probability as tfp

def Cvfs(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: tf.Tensor,
    inducing_variable_anchor: SharedIndependentInducingVariables,
    seed : Optional[Any] = None,
) -> tf.Tensor:

    """
    Warning -- this currently works just with shared variables, hence a single kernel and associated hyperparameters per layer
    """
    return Cvf(inducing_variable.inducing_variable,  Xnew, kernel.kernel, inducing_variable_anchor, seed = seed)  # [M, N]

