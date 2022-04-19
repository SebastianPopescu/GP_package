import tensorflow as tf

from ...base import TensorLike, TensorType
from ...inducing_variables import SharedIndependentInducingVariables
from ...kernels import SharedIndependent

def _Kuf(inducing_variable, kernel, Xnew):
    print('-- inside _Kuf ---')
    print(Xnew)

    return kernel(inducing_variable.Z, Xnew)

def Kufs(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: tf.Tensor,
) -> tf.Tensor:

    print(' -- inside Kufs ---')
    print(Xnew)
    return _Kuf(inducing_variable.inducing_variable, kernel.kernel, Xnew)  # [M, N]

