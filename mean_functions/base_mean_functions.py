
from typing import Collection, Optional

import numpy as np
import tensorflow as tf

from ..base import Module, Parameter, TensorType
from ..config import default_float, default_int

class MeanFunction(Module):
    """
    The base mean function class.
    To implement a mean function, write the __call__ method. This takes a
    tensor X and returns a tensor m(X). In accordance with the GPflow
    standard, each row of X represents one datum, and each row of Y is computed
    independently for each row of X.

    MeanFunction classes can have parameters, see the Linear class for an
    example.
    """

    def __call__(self, X: TensorType) -> tf.Tensor:
        raise NotImplementedError("Implement the __call__ method for this mean function")

    def __add__(self, other: "MeanFunction") -> "MeanFunction":
        return Additive(self, other)

    def __mul__(self, other: "MeanFunction") -> "MeanFunction":
        return Product(self, other)


class Linear(MeanFunction):
    """
    y_i = A x_i + b
    """

    def __init__(self, A: TensorType = None, b: TensorType = None) -> None:
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.

        If X has N rows and D columns, and Y is intended to have Q columns,
        then A must be [D, Q], b must be a vector of length Q.
        """
        MeanFunction.__init__(self)
        A = np.ones((1, 1), dtype=default_float()) if A is None else A
        b = np.zeros(1, dtype=default_float()) if b is None else b
        self.A = Parameter(np.atleast_2d(A))
        self.b = Parameter(b)

    def __call__(self, X: TensorType) -> tf.Tensor:
        return tf.tensordot(X, self.A, [[-1], [0]]) + self.b

class Additive(MeanFunction):
    def __init__(self, first_part: MeanFunction, second_part: MeanFunction) -> None:
        MeanFunction.__init__(self)
        self.add_1 = first_part
        self.add_2 = second_part

    def __call__(self, X: TensorType) -> tf.Tensor:
        return tf.add(self.add_1(X), self.add_2(X))


class Product(MeanFunction):
    def __init__(self, first_part: MeanFunction, second_part: MeanFunction):
        MeanFunction.__init__(self)

        self.prod_1 = first_part
        self.prod_2 = second_part

    def __call__(self, X: TensorType) -> tf.Tensor:
        return tf.multiply(self.prod_1(X), self.prod_2(X))
