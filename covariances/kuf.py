import tensorflow as tf

from ..base import TensorLike, TensorType
from ..inducing_variables import InducingPoints
from ..kernels import Kernel, SquaredExponential

def Kuf(
    inducing_variable: InducingPoints, kernel: Kernel, Xnew: TensorType
) -> tf.Tensor:
    return kernel(inducing_variable.Z, Xnew)

