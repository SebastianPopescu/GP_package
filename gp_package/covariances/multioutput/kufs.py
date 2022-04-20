import tensorflow as tf

from typing import Union

from gp_package.covariances.kuf import Kuf

from ...base import TensorLike, TensorType
from ...inducing_variables import SharedIndependentInducingVariables, SharedIndependentDistributionalInducingVariables
from ...kernels import SharedIndependent

"""
def _Kuf(inducing_variable, kernel, Xnew):
    return kernel(inducing_variable.Z, Xnew)
"""

def Kufs(
    inducing_variable: Union[SharedIndependentInducingVariables,SharedIndependentDistributionalInducingVariables],
    kernel: SharedIndependent,
    Xnew: tf.Tensor,
) -> tf.Tensor:

    return Kuf(inducing_variable.inducing_variable, kernel.kernel, Xnew)  # [M, N]

