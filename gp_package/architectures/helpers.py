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

from ..base import default_float
from ..inducing_variables import (
    InducingPoints,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
    MultioutputInducingVariables,
    DistributionalInducingPoints,
    SeparateIndependentDistributionalInducingVariables,
    SharedIndependentDistributionalInducingVariables,
    MultioutputDistributionalInducingVariables
)

from ..kernels import SeparateIndependent, SharedIndependent, Kernel, MultioutputKernel, Stationary

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

from ..layers import GPLayer, DistGPLayer
from ..mean_functions import MeanFunction, Identity, Linear, Zero
from ..utils import set_trainable


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



def construct_basic_hybrid_kernel(
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
    Construct a compatible :class:`~gpflow.inducing_variables.MultioutputInducingVariables`
    to use in :class:`GPLayer`\ s.

    :param num_inducing: The total number of inducing variables, ``M``.
        This parameter can be freely chosen by the user. General advice
        is to set it as high as possible, but smaller than the number of datapoints.
        The computational complexity of the layer is cubic in ``M``.
        If a list is passed, each element in the list specifies the number of inducing
        variables to use for each ``output_dim``.
    :param input_dim: The dimensionality of the input data (or features) ``X``.
        Typically, this corresponds to ``X.shape[-1]``.
        For :class:`~gpflow.inducing_variables.InducingPoints`, this specifies the dimensionality
        of ``Z``.
    :param output_dim: The dimensionality of the outputs (or targets) ``Y``.
        Typically, this corresponds to ``Y.shape[-1]`` or the number of latent GPs.
        The parameter is used to determine the number of inducing variable sets
        to create when a different set is used for each output. The parameter
        is redundant when ``num_inducing`` is a list, because the code assumes
        that ``len(num_inducing) == output_dim``.
    :param share_variables: If `True`, use the same inducing variables for different
        outputs. Otherwise, create a different set for each output. Set this parameter to
        `False` when ``num_inducing`` is a list, because otherwise the two arguments
        contradict each other. If you set this parameter to `True`, you must also specify
        ``output_dim``, because that is used to determine the number of inducing variable
        sets to create.
    :param z_init: Raw values to use to initialise
        :class:`gpflow.inducing_variables.InducingPoints`. If `None` (the default), values
        will be initialised from ``N(0, 1)``. The shape of ``z_init`` depends on the other
        input arguments. If a single set of inducing points is used for all outputs (that
        is, if ``share_variables`` is `True`), ``z_init`` should be rank two, with the
        dimensions ``[M, input_dim]``. If a different set of inducing points is used for
        the outputs (ithat is, if ``num_inducing`` is a list, or if ``share_variables`` is
        `False`), ``z_init`` should be a rank three tensor with the dimensions
        ``[output_dim, M, input_dim]``.
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
                z_init_i = np.random.randn(num_ind_var, input_dim).astype(dtype=default_float())
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
                z_init_o = np.random.randn(num_inducing, input_dim).astype(dtype=default_float())
            inducing_variables.append(InducingPoints(z_init_o))
        return SeparateIndependentInducingVariables(inducing_variables)

    else:
        # TODO: should we assert output_dim is None ?

        z_init = (
            z_init
            if z_init_is_given
            else np.random.randn(num_inducing, input_dim).astype(dtype=default_float())
        )
        shared_ip = InducingPoints(z_init)
        return SharedIndependentInducingVariables(shared_ip)




def construct_basic_distributional_inducing_variables(
    num_inducing: Union[int, List[int]],
    input_dim: int,
    output_dim: Optional[int] = None,
    share_variables: bool = False,
    z_init_mean: Optional[np.ndarray] = None,
    z_init_var: Optional[np.ndarray] = None,
) -> MultioutputDistributionalInducingVariables:
    r"""
    Construct a compatible :class:`~gpflow.inducing_variables.MultioutputInducingVariables`
    to use in :class:`GPLayer`\ s.

    :param num_inducing: The total number of inducing variables, ``M``.
        This parameter can be freely chosen by the user. General advice
        is to set it as high as possible, but smaller than the number of datapoints.
        The computational complexity of the layer is cubic in ``M``.
        If a list is passed, each element in the list specifies the number of inducing
        variables to use for each ``output_dim``.
    :param input_dim: The dimensionality of the input data (or features) ``X``.
        Typically, this corresponds to ``X.shape[-1]``.
        For :class:`~gpflow.inducing_variables.InducingPoints`, this specifies the dimensionality
        of ``Z``.
    :param output_dim: The dimensionality of the outputs (or targets) ``Y``.
        Typically, this corresponds to ``Y.shape[-1]`` or the number of latent GPs.
        The parameter is used to determine the number of inducing variable sets
        to create when a different set is used for each output. The parameter
        is redundant when ``num_inducing`` is a list, because the code assumes
        that ``len(num_inducing) == output_dim``.
    :param share_variables: If `True`, use the same inducing variables for different
        outputs. Otherwise, create a different set for each output. Set this parameter to
        `False` when ``num_inducing`` is a list, because otherwise the two arguments
        contradict each other. If you set this parameter to `True`, you must also specify
        ``output_dim``, because that is used to determine the number of inducing variable
        sets to create.
    :param z_init: Raw values to use to initialise
        :class:`gpflow.inducing_variables.InducingPoints`. If `None` (the default), values
        will be initialised from ``N(0, 1)``. The shape of ``z_init`` depends on the other
        input arguments. If a single set of inducing points is used for all outputs (that
        is, if ``share_variables`` is `True`), ``z_init`` should be rank two, with the
        dimensions ``[M, input_dim]``. If a different set of inducing points is used for
        the outputs (ithat is, if ``num_inducing`` is a list, or if ``share_variables`` is
        `False`), ``z_init`` should be a rank three tensor with the dimensions
        ``[output_dim, M, input_dim]``.
    """

    if z_init_mean is None:
        warnings.warn(
            "No `z_init_mean` has been specified in `construct_basic_inducing_variables`. "
            "Default initialization using random normal N(0, 1) will be used."
        )

    z_init_mean_is_given = z_init_mean is not None

    if z_init_var is None:
        warnings.warn(
            "No `z_init_var` has been specified in `construct_basic_inducing_variables`. "
            "Default initialization using random log-normal N(0, 1) will be used."
        )

    z_init_var_is_given = z_init_var is not None

    if isinstance(num_inducing, list):
        if output_dim is not None:
            # TODO: the following assert may clash with MixedMultiOutputFeatures
            # where the number of independent GPs can differ from the output
            # dimension
            assert output_dim == len(num_inducing)  # pragma: no cover
        assert share_variables is False

        inducing_variables = []
        for i, num_ind_var in enumerate(num_inducing):
            if z_init_mean_is_given:
                assert len(z_init_mean[i]) == num_ind_var
                z_init_mean_i = z_init_mean[i]
            else:
                #z_init_mean_i = np.random.randn(num_ind_var, input_dim).astype(dtype=default_float())
                z_init_mean_i = np.random.uniform(low=-0.5, high=0.5, size=(num_ind_var, input_dim)).astype(dtype=default_float())
            assert z_init_mean_i.shape == (num_ind_var, input_dim)
            
            if z_init_var_is_given:
                assert len(z_init_var[i]) == num_ind_var
                z_init_var_i = z_init_var[i]
            else:
                #z_init_var_i = np.random.lognormal(size=(num_ind_var, input_dim)).astype(dtype=default_float())
                
                z_init_var_i = np.ones((num_ind_var, input_dim)) * 0.0067153485
                z_init_var_i = z_init_var_i.astype(dtype=default_float())
            
            assert z_init_var_i.shape == (num_ind_var, input_dim)
            
            
            inducing_variables.append(DistributionalInducingPoints(z_init_mean_i, z_init_var_i))
        return SeparateIndependentDistributionalInducingVariables(inducing_variables)

    elif not share_variables:
        inducing_variables = []
        for o in range(output_dim):
            
            if z_init_mean_is_given:
                if z_init_mean.shape != (output_dim, num_inducing, input_dim):
                    raise ValueError(
                        "When not sharing variables, z_init_mean must have shape"
                        "[output_dim, num_inducing, input_dim]"
                    )
                z_init_mean_o = z_init_mean[o]
            else:
                #z_init_mean_o = np.random.randn(num_inducing, input_dim).astype(dtype=default_float())
                z_init_mean_o = np.random.uniform(low=-0.5, high=0.5, size=(num_inducing, input_dim)).astype(dtype=default_float())

            if z_init_var_is_given:
                if z_init_var.shape != (output_dim, num_inducing, input_dim):
                    raise ValueError(
                        "When not sharing variables, z_init_mean must have shape"
                        "[output_dim, num_inducing, input_dim]"
                    )
                z_init_var_o = z_init_var[o]
            else:
                #z_init_var_o = np.random.lognormal(num_inducing, input_dim).astype(dtype=default_float())
                z_init_var_o = np.ones((num_inducing, input_dim)) * 0.0067153485
                z_init_var_o = z_init_var_o.astype(dtype=default_float())

            inducing_variables.append(DistributionalInducingPoints(z_init_mean_o, z_init_var_o))
        return SeparateIndependentDistributionalInducingVariables(inducing_variables)

    else:
        # TODO: should we assert output_dim is None ?

        z_init_mean = (
            z_init_mean
            if z_init_mean_is_given
            else np.random.uniform(low=-0.5, high=0.5, size=(num_inducing, input_dim)).astype(dtype=default_float()) #np.random.randn(num_inducing, input_dim).astype(dtype=default_float())
        )
        z_init_var = (
            z_init_var
            if z_init_var_is_given
            else 0.0067153485 * np.ones((num_inducing, input_dim)).astype(dtype=default_float())     #np.random.lognormal(size=(num_inducing, input_dim)).astype(dtype=default_float())
        )
        shared_ip = DistributionalInducingPoints(z_init_mean, z_init_var)
        return SharedIndependentDistributionalInducingVariables(shared_ip)


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


def construct_gp_layer(
    num_data: int,
    num_inducing: int,
    input_dim: int,
    output_dim: int,
    kernel_class: Type[Stationary],
    z_init: Optional[np.ndarray] = None,
    name: Optional[str] = None,
) -> GPLayer:
    """
    Builds a vanilla GP layer with a single kernel shared among all outputs,
        shared inducing point variables and zero mean function.

    :param num_data: total number of datapoints in the dataset, *N*.
        Typically corresponds to ``X.shape[0] == len(X)``.
    :param num_inducing: total number of inducing variables, *M*.
        This parameter can be freely chosen by the user. General advice
        is to pick it as high as possible, but smaller than *N*.
        The computational complexity of the layer is cubic in *M*.
    :param input_dim: dimensionality of the input data (or features) X.
        Typically, this corresponds to ``X.shape[-1]``.
    :param output_dim: The dimensionality of the outputs (or targets) ``Y``.
        Typically, this corresponds to ``Y.shape[-1]``.
    :param kernel_class: The kernel class used by the layer.
        This can be as simple as :class:`gpflow.kernels.SquaredExponential`, or more complex,
        for example, ``lambda **_: gpflow.kernels.Linear() + gpflow.kernels.Periodic()``.
        It will be passed a ``lengthscales`` keyword argument.
    :param z_init: The initial value for the inducing variable inputs.
    :param name: The name for the GP layer.
    """
    lengthscale = float(input_dim) ** 0.5
    base_kernel = kernel_class(lengthscales=np.full(input_dim, lengthscale))
    kernel = construct_basic_kernel(base_kernel, output_dim=output_dim, share_hyperparams=True)
    inducing_variable = construct_basic_inducing_variables(
        num_inducing,
        input_dim,
        output_dim=output_dim,
        share_variables=True,
        z_init=z_init,
    )
    gp_layer = GPLayer(
        kernel=kernel,
        inducing_variable=inducing_variable,
        num_data=num_data,
        mean_function=Zero(),
        name=name,
    )
    return gp_layer

def construct_dist_gp_layer(
    num_data: int,
    num_inducing: int,
    input_dim: int,
    output_dim: int,
    kernel_class: Type[Stationary],
    z_init_mean: Optional[np.ndarray] = None,
    z_init_var: Optional[np.ndarray] = None,
    name: Optional[str] = None,
) -> DistGPLayer:
    """
    Builds a vanilla GP layer with a single kernel shared among all outputs,
        shared inducing point variables and zero mean function.

    :param num_data: total number of datapoints in the dataset, *N*.
        Typically corresponds to ``X.shape[0] == len(X)``.
    :param num_inducing: total number of inducing variables, *M*.
        This parameter can be freely chosen by the user. General advice
        is to pick it as high as possible, but smaller than *N*.
        The computational complexity of the layer is cubic in *M*.
    :param input_dim: dimensionality of the input data (or features) X.
        Typically, this corresponds to ``X.shape[-1]``.
    :param output_dim: The dimensionality of the outputs (or targets) ``Y``.
        Typically, this corresponds to ``Y.shape[-1]``.
    :param kernel_class: The kernel class used by the layer.
        This can be as simple as :class:`gpflow.kernels.SquaredExponential`, or more complex,
        for example, ``lambda **_: gpflow.kernels.Linear() + gpflow.kernels.Periodic()``.
        It will be passed a ``lengthscales`` keyword argument.
    :param z_init: The initial value for the inducing variable inputs.
    :param name: The name for the GP layer.
    """
    lengthscale = float(input_dim) ** 0.5
    base_kernel = kernel_class(lengthscales=np.full(input_dim, lengthscale))
    kernel = construct_basic_hybrid_kernel(base_kernel, output_dim=output_dim, share_hyperparams=True)
    inducing_variable = construct_basic_distributional_inducing_variables(
        num_inducing,
        input_dim,
        output_dim=output_dim,
        share_variables=True,
        z_init_mean=z_init_mean,
        z_init_var=z_init_var,
    )
    dist_gp_layer = DistGPLayer(
        kernel=kernel,
        inducing_variable=inducing_variable,
        num_data=num_data,
        mean_function=Zero(),
        name=name,
    )
    return dist_gp_layer


