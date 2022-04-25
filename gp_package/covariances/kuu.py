from sre_constants import ANY
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Any, Union, Optional

from ..config import default_float
from ..inducing_variables import InducingPoints, DistributionalInducingPoints
from ..kernels import Kernel, SquaredExponential

def Kuu(
    inducing_variable: Union[InducingPoints, DistributionalInducingPoints], kernel: Kernel, *, jitter: float = 0.0,
    seed : Optional[Any]  = None,
) -> tf.Tensor:

    if isinstance(inducing_variable, DistributionalInducingPoints):
        # Create instance of tfp.distributions.MultivariateNormalDiag so that it works with underpinning methods from kernel
        distributional_inducing_points = tfp.distributions.MultivariateNormalDiag(loc = inducing_variable.Z_mean,
            scale_diag = tf.sqrt(inducing_variable.Z_var))
        Kzz = kernel(distributional_inducing_points, seed = seed)
    elif isinstance(inducing_variable, InducingPoints):
        Kzz = kernel(inducing_variable.Z)
    Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)
    return Kzz