from typing import Collection, Optional

import numpy as np
import tensorflow as tf

from ..base import Module, Parameter, TensorType
from ..config import default_float, default_int
from .base_mean_functions import MeanFunction, Linear


class Identity(Linear):
    """
    y_i = x_i
    """

    # The many type-ignores in this class is because we replace a field in the super class with a
    # property, which mypy doesn't like.

    def __init__(self, input_dim: Optional[int] = None) -> None:
        Linear.__init__(self)
        self.input_dim = input_dim

    def __call__(self, X: TensorType) -> tf.Tensor:
        return X

    @property
    def A(self) -> tf.Tensor:  # type: ignore
        if self.input_dim is None:
            raise ValueError(
                "An input_dim needs to be specified when using the "
                "`Identity` mean function in combination with expectations."
            )
        return tf.eye(self.input_dim, dtype=default_float())

    @property
    def b(self) -> tf.Tensor:  # type: ignore
        if self.input_dim is None:
            raise ValueError(
                "An input_dim needs to be specified when using the "
                "`Identity` mean function in combination with expectations."
            )

        return tf.zeros(self.input_dim, dtype=default_float())

    @A.setter  # type: ignore
    def A(self, A: tf.Tensor) -> None:
        pass

    @b.setter  # type: ignore
    def b(self, b: tf.Tensor) -> None:
        pass


class Constant(MeanFunction):
    def __init__(self, c: TensorType = None) -> None:
        super().__init__()
        c = np.zeros(1) if c is None else c
        self.c = Parameter(c)

    def __call__(self, X: TensorType) -> tf.Tensor:
        tile_shape = tf.concat(
            [tf.shape(X)[:-1], [1]],
            axis=0,
        )
        reshape_shape = tf.concat(
            [tf.ones(shape=(tf.rank(X) - 1), dtype=default_int()), [-1]],
            axis=0,
        )
        return tf.tile(tf.reshape(self.c, reshape_shape), tile_shape)


class Zero(Constant):
    def __init__(self, output_dim: int = 1) -> None:
        Constant.__init__(self)
        self.output_dim = output_dim
        del self.c

    def __call__(self, X: TensorType) -> tf.Tensor:
        output_shape = tf.concat([tf.shape(X)[:-1], [self.output_dim]], axis=0)
        return tf.zeros(output_shape, dtype=X.dtype)

