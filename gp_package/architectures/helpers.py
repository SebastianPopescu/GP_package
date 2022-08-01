#
# Copyright (c) 2021 The GPflux Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
r"""
This module contains helper functions for constructing :class:`~gpflow.kernels.MultioutputKernel`,
:class:`~gpflow.inducing_variables.MultioutputInducingVariables`,
:class:`~gpflow.mean_functions.MeanFunction`, and :class:`~gpflux.layers.GPLayer` objects.
"""

import tensorflow as tf

import inspect
import warnings
from dataclasses import fields
from typing import List, Optional, Type, TypeVar, Union

import numpy as np

from gpflow.base import default_float
from gpflow.inducing_variables import (
    InducingPoints,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
    MultioutputInducingVariables,
)

from gp_package.inducing_variables.fourier_features import FourierPoints1D

from ..inducing_variables import (
    FourierFeatures1D,
)

from gpflow.kernels import SeparateIndependent, SharedIndependent, Kernel, MultioutputKernel, Stationary


import copy
from typing import Any, Callable, Dict, Mapping, Optional, Pattern, Tuple, Type, TypeVar, Union

M = TypeVar("M", bound=tf.Module)

def deepcopy(input_module: M, memo: Optional[Dict[int, Any]] = None) -> M:
    """
    Returns a deepcopy of the input tf.Module. To do that first resets the caches stored inside each
    tfp.bijectors.Bijector to allow the deepcopy of the tf.Module.

    :param input_module: tf.Module including keras.Model, keras.layers.Layer and gpflow.Module.
    :param memo: passed through to func:`copy.deepcopy`
        (see https://docs.python.org/3/library/copy.html).
    :return: Returns a deepcopy of an input object.
    """
    return copy.deepcopy(reset_cache_bijectors(input_module), memo)  # type: ignore

from gpflow.mean_functions import MeanFunction, Identity, Linear, Zero
from gpflow.utilities import set_trainable

def construct_basic_kernel(
    kernels: Union[Kernel, List[Kernel]],
    output_dim: Optional[int] = None,
    share_hyperparams: bool = False,
) -> MultioutputKernel:
    r"""
    Construct a :class:`~gpflow.kernels.MultioutputKernel` to use
    in :class:`GPLayer`\ s.

    :param kernels: A single kernel or list of :class:`~gpflow.kernels.Kernel`\ s.
        - When a single kernel is passed, the same kernel is used for all
        outputs. Depending on ``share_hyperparams``, the hyperparameters will be
        shared across outputs. You must also specify ``output_dim``.
        - When a list of kernels is passed, each kernel in the list is used on a separate
        output dimension and a :class:`gpflow.kernels.SeparateIndependent` is returned.
    :param output_dim: The number of outputs. This is equal to the number of latent GPs
        in the :class:`GPLayer`. When only a single kernel is specified for ``kernels``,
        you must also specify ``output_dim``. When a list of kernels is specified for ``kernels``,
        we assume that ``len(kernels) == output_dim``, and ``output_dim`` is not required.
    :param share_hyperparams: If `True`, use the type of kernel and the same hyperparameters
        (variance and lengthscales) for the different outputs. Otherwise, the
        same type of kernel (Squared-Exponential, Matern12, and so on) is used for
        the different outputs, but the kernel can have different hyperparameter values for each.
    """
    if isinstance(kernels, list):
        mo_kern = SeparateIndependent(kernels)
    elif not share_hyperparams:
        copies = [deepcopy(kernels) for _ in range(output_dim)]
        mo_kern = SeparateIndependent(copies)
    else:
        mo_kern = SharedIndependent(kernels, output_dim)
    return mo_kern

def construct_basic_inducing_variables(
    num_inducing: Union[int, List[int]],
    input_dim: int,
    output_dim: Optional[int] = None,
    share_variables: bool = False,
    z_init: Optional[np.ndarray] = None,
) -> MultioutputInducingVariables:
    r"""
    #TODO -- 
    """

    if z_init is None:
        warnings.warn(
            "No `z_init` has been specified in `construct_basic_inducing_variables`. "
            "Default initialization using random normal N(0, 1) will be used."
        )

    z_init_is_given = z_init is not None

    if isinstance(num_inducing, list):
        if output_dim is not None:
            # TODO: the following assert may clash with MixedMultiOutputFeatures
            # where the number of independent GPs can differ from the output
            # dimension
            assert output_dim == len(num_inducing)  # pragma: no cover
        assert share_variables is False

        inducing_variables = []
        for i, num_ind_var in enumerate(num_inducing):
            if z_init_is_given:
                assert len(z_init[i]) == num_ind_var
                z_init_i = z_init[i]
            else:
                z_init_i = np.random.uniform(-0.5,0.5, (num_ind_var, input_dim)).astype(dtype=default_float())
            assert z_init_i.shape == (num_ind_var, input_dim)
            inducing_variables.append(InducingPoints(z_init_i))
        return SeparateIndependentInducingVariables(inducing_variables)

    elif not share_variables:
        inducing_variables = []
        for o in range(output_dim):
            if z_init_is_given:
                if z_init.shape != (output_dim, num_inducing, input_dim):
                    raise ValueError(
                        "When not sharing variables, z_init must have shape"
                        "[output_dim, num_inducing, input_dim]"
                    )
                z_init_o = z_init[o]
            else:
                z_init_o = np.random.uniform(-0.5,0.5, (num_inducing, input_dim)).astype(dtype=default_float())
            inducing_variables.append(InducingPoints(z_init_o))
        return SeparateIndependentInducingVariables(inducing_variables)

    else:
        # TODO: should we assert output_dim is None ?

        z_init = (
            z_init
            if z_init_is_given
            else np.random.uniform(-0.5,0.5, (num_inducing, input_dim)).astype(dtype=default_float())
        )
        shared_ip = InducingPoints(z_init)
        return SharedIndependentInducingVariables(shared_ip)


def construct_basic_fourier_features(
    M: Union[int, List[int]],
    input_dim: int,
    output_dim: Optional[int] = None,
    share_variables: bool = False,
    a: Optional[float] = None,
    b: Optional[float] = None
) -> FourierFeatures1D:
    
    r"""
    #TODO -- 
    """
    return FourierPoints1D(a=a, b=b, M=M)



def construct_mean_function(
    X: np.ndarray, D_in: int, D_out: int
) -> MeanFunction:
    """
    Return :class:`gpflow.mean_functions.Identity` when ``D_in`` and ``D_out`` are
    equal. Otherwise, use the principal components of the inputs matrix ``X`` to build a
    :class:`~gpflow.mean_functions.Linear` mean function.

    .. note::
        The returned mean function is set to be untrainable.
        To change this, use :meth:`gpflow.set_trainable`.

    :param X: A data array with the shape ``[N, D_in]`` used to determine the principal
        components to use to create a :class:`~gpflow.mean_functions.Linear` mean function
        when ``D_in != D_out``.
    :param D_in: The dimensionality of the input data (or features) ``X``.
        Typically, this corresponds to ``X.shape[-1]``.
    :param D_out: The dimensionality of the outputs (or targets) ``Y``.
        Typically, this corresponds to ``Y.shape[-1]`` or the number of latent GPs in the layer.
    """
    assert X.shape[-1] == D_in
    if D_in == D_out:
        mean_function = Identity()
    else:
        if D_in > D_out:
            _, _, V = np.linalg.svd(X, full_matrices=False)
            W = V[:D_out, :].T
        else:
            W = np.concatenate([np.eye(D_in), np.zeros((D_in, D_out - D_in))], axis=1)

        assert W.shape == (D_in, D_out)
        mean_function = Linear(W)
        set_trainable(mean_function, False)

    return mean_function



