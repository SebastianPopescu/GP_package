
import numpy as np
import tensorflow as tf

from ..base import TensorType
from ..utils import to_default_float


def gaussian(x: TensorType, mu: TensorType, var: TensorType) -> tf.Tensor:
    return -0.5 * (np.log(2 * np.pi) + tf.math.log(var) + tf.square(mu - x) / var)


def student_t(x: TensorType, mean: TensorType, scale: TensorType, df: TensorType) -> tf.Tensor:
    df = to_default_float(df)
    const = (
        tf.math.lgamma((df + 1.0) * 0.5)
        - tf.math.lgamma(df * 0.5)
        - 0.5 * (tf.math.log(tf.square(scale)) + tf.math.log(df) + np.log(np.pi))
    )
    return const - 0.5 * (df + 1.0) * tf.math.log(
        1.0 + (1.0 / df) * (tf.square((x - mean) / scale))
    )

