import tensorflow as tf

from ...base import TensorLike, TensorType
from ...inducing_variables import SharedIndependentInducingVariables
from ...kernels import SharedIndependent
from gp_package.covariances.bayesian_kuf import BayesianKuf

def BayesianKufs(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: tf.Tensor,
    variance: TensorType,
    lengthscales: TensorType
) -> tf.Tensor:
    
    return BayesianKuf(inducing_variable.inducing_variable, kernel.kernel, Xnew, variance, lengthscales)  # [M, N]

