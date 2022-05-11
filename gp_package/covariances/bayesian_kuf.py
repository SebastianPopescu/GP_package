import tensorflow as tf

from ..base import TensorLike, TensorType
from ..inducing_variables import InducingPoints
from ..kernels import BayesianKernel, BayesianSquaredExponential

def BayesianKuf(
    inducing_variable: InducingPoints, kernel: BayesianKernel, Xnew: TensorType,
    variance: TensorType, 
    lengthscales: TensorType
) -> tf.Tensor:
    return kernel(variance, inducing_variable.Z, Xnew, lengthscales)

