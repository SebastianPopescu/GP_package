from numpy import isin
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Any, Optional, Union

from ..base import TensorLike, TensorType
from ..inducing_variables import InducingPoints
from ..kernels import Kernel, SquaredExponential


def Kuf(
    inducing_variable: InducingPoints, kernel: Kernel, 
    Xnew: TensorType,
    seed : Optional[Any] = None) -> tf.Tensor:
        
    return kernel(inducing_variable.Z, Xnew)


