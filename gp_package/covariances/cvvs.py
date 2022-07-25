
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Any, Union, Optional

from gpflow.config import default_float
from gpflow.inducing_variables import InducingPoints

from gpflow.base import TensorLike
from gpflow.kernels import Kernel
from .dispatch import Cvv
from gpflow.config import default_jitter

@Cvv.register(InducingPoints, InducingPoints, Kernel)
def Cvv_kernel_inducingpoints(
    inducing_variable_u: InducingPoints,
    inducing_variable_v: InducingPoints, 
    kernel: Kernel, 
    *, 
    jitter: float = 0.0,
    L_Kuu: Optional[tf.Tensor] = None
    ) -> tf.Tensor:

    Kvv = kernel(inducing_variable_v.Z)

    if L_Kuu ==None:
        Kuu = kernel(inducing_variable_u.Z)
        jittermat = tf.eye(inducing_variable_u.num_inducing, dtype=Kuu.dtype) * default_jitter()
        Kuu+= jittermat
        L_Kuu = tf.linalg.cholesky(Kuu)

    Kuv = kernel(inducing_variable_u.Z, inducing_variable_v.Z)

    L_Kuu_inv_Kuv = tf.linalg.triangular_solve(L_Kuu, Kuv)
    Cvv = Kvv - tf.linalg.matmul(
        L_Kuu_inv_Kuv, L_Kuu_inv_Kuv, transpose_a=True)

    Cvv += jitter * tf.eye(inducing_variable_v.num_inducing, dtype=Cvv.dtype)
    
    return Cvv