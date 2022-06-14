from sre_constants import ANY
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Any, Union, Optional

from ..config import default_float
from ..inducing_variables import InducingPoints
from ..kernels import Kernel, SquaredExponential

def Cvv(
    inducing_variable: InducingPoints, 
    kernel: Kernel, 
    inducing_variable_anchor: InducingPoints,
    *, jitter: float = 0.0,
    seed : Optional[Any] = None,
    L_Kuu: Optional[tf.Tensor] = None
    ) -> tf.Tensor:


    Kvv = kernel(inducing_variable.Z)

    if not L_Kuu:
        Kuu = kernel(inducing_variable_anchor.Z)
        L_Kuu = tf.cholesky(Kuu)

    #Kvv = self.kernel.kernel(self.inducing_variable_v.Z, full_cov=full_cov)
    Kuv = kernel(inducing_variable_anchor.Z, inducing_variable.Z)

    L_Kuu_inv_Kuv = tf.matrix_triangular_solve(L_Kuu, Kuv)
    Cvv = Kvv - tf.linalg.matmul(
        L_Kuu_inv_Kuv, L_Kuu_inv_Kuv, transpose_a=True)

    Cvv += jitter * tf.eye(inducing_variable.num_inducing, dtype=Cvv.dtype)
    return Cvv