from sre_constants import ANY
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Any, Union, Optional

from gp_package.base import TensorType

from ..config import default_float, default_jitter
from ..inducing_variables import InducingPoints
from ..kernels import Kernel, SquaredExponential

def Cvf(
    inducing_variable: InducingPoints, 
    Xnew: TensorType,
    kernel: Kernel, 
    inducing_variable_anchor: InducingPoints,
    *, jitter: float = 0.0,
    seed : Optional[Any] = None,
    L_Kuu: Optional[tf.Tensor] = None
) -> tf.Tensor:

    Kvf = kernel(inducing_variable.Z, Xnew)

    if not L_Kuu:
        Kuu = kernel(inducing_variable_anchor.Z)
        jittermat = tf.eye(inducing_variable_anchor.num_inducing, dtype=Kuu.dtype) * default_jitter()
        Kuu+= jittermat
        L_Kuu = tf.linalg.cholesky(Kuu)

    Kuv = kernel(inducing_variable_anchor.Z, inducing_variable.Z)
    Kuf = kernel(inducing_variable_anchor.Z, Xnew)

    L_Kuu_inv_Kuv = tf.linalg.triangular_solve(L_Kuu, Kuv)
    L_Kuu_inv_Kuf = tf.linalg.triangular_solve(L_Kuu, Kuf)

    Cvf = Kvf - tf.linalg.matmul(
        L_Kuu_inv_Kuv, L_Kuu_inv_Kuf, transpose_a=True)

    return Cvf