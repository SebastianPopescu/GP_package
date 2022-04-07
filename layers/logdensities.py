
import numpy as np
import tensorflow as tf

from ..base import TensorType
from ..utils import to_default_float


def gaussian(x: TensorType, mu: TensorType, var: TensorType) -> tf.Tensor:
    return -0.5 * (np.log(2 * np.pi) + tf.math.log(var) + tf.square(mu - x) / var)
