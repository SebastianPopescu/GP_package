import tensorflow as tf
import tensorflow_probability as tfp

from typing import Any, Optional, Union

from gpflow.base import TensorLike, TensorType
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import Kernel
from .dispatch import Cvf
from gpflow.config import default_jitter


@Cvf.register(InducingPoints,InducingPoints, Kernel, TensorLike)
def Cvf_kernel_inducingpoints(
    inducing_variable_u: InducingPoints, 
    inducing_variable_v: InducingPoints, 
    kernel: Kernel, 
    Xnew: TensorType,
    *, 
    jitter: float = 0.0,
    L_Kuu: Optional[tf.Tensor] = None
) -> tf.Tensor:

    Kvf = kernel(inducing_variable_v.Z, Xnew)

    if L_Kuu==None:
        Kuu = kernel(inducing_variable_u.Z)
        jittermat = tf.eye(inducing_variable_u.num_inducing, dtype=Kuu.dtype) * default_jitter()
        Kuu+= jittermat
        L_Kuu = tf.linalg.cholesky(Kuu)

    Kuv = kernel(inducing_variable_u.Z, inducing_variable_v.Z)
    Kuf = kernel(inducing_variable_u.Z, Xnew)

    L_Kuu_inv_Kuv = tf.linalg.triangular_solve(L_Kuu, Kuv)
    L_Kuu_inv_Kuf = tf.linalg.triangular_solve(L_Kuu, Kuf)

    Cvf = Kvf - tf.linalg.matmul(
        L_Kuu_inv_Kuv, L_Kuu_inv_Kuf, transpose_a=True)

    return Cvf