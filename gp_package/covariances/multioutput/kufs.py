import tensorflow as tf

from ...base import TensorLike, TensorType
from ...inducing_variables import SharedIndependentInducingVariables
from ...kernels import SharedIndependent
from gp_package.covariances.kuf import Kuf

def Kufs(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: tf.Tensor,
) -> tf.Tensor:

    return Kuf(inducing_variable.inducing_variable, kernel.kernel, Xnew)  # [M, N]

