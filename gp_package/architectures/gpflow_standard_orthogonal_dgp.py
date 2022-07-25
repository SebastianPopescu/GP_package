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
"""
This module provides :func:`build_constant_input_dim_deep_gp` to build a Deep GP of
arbitrary depth where each hidden layer has the same input dimensionality as the data.
"""

from dataclasses import dataclass
from warnings import WarningMessage

import numpy as np
import tensorflow as tf
from scipy.cluster.vq import kmeans2

from gpflow.kernels import SquaredExponential
from gpflow.likelihoods import Gaussian, Bernoulli, MultiClass

from .helpers import (
    construct_basic_inducing_variables,
    construct_basic_kernel,
    construct_mean_function,
)
from ..layers import OrthogonalSVGP #TODO -- this class needs to be created
from ..models import OrthogonalDeepGP #TODO -- this class needs to be created
from gpflow.mean_functions import Zero, Identity
from gpflow.utilities import set_trainable


@dataclass
class Config:
    """
    The configuration used by :func:`build_constant_input_dim_deep_gp`.
    """

    num_inducing_u: int
    """
    The number of inducing variables, *M*. The Deep GP uses the same number
    of inducing variables in each layer.
    """

    num_inducing_v: int
    """
    The number of inducing variables, *M*. The Deep GP uses the same number
    of inducing variables in each layer.
    """


    inner_layer_qsqrt_factor: float
    """
    A multiplicative factor used to rescale the hidden layers'
    :attr:`~gpflux.layers.GPLayer.q_sqrt`. Typically this value is chosen to be small
    (e.g., 1e-5) to reduce noise at the start of training.
    """

    likelihood_noise_variance: float
    """
    The variance of the :class:`~gpflow.likelihoods.Gaussian` likelihood that is used
    by the Deep GP.
    """

    hidden_layer_size : int
    """
    Support for non-constant input dimension architectures
    """

    task_type: str
    """
    Can either be 'regression' or 'classification'
    """

    dim_output: int
    """
    Mostly to be used for 'classification' option
    """

    num_data: int

    """
    number of training points. To be used in the loss function
    """

    whiten: bool = True
    """
    Determines the parameterisation of the inducing variables.
    If `True`, :math:``p(u) = N(0, I)``, otherwise :math:``p(u) = N(0, K_{uu})``.
    .. seealso:: :attr:`gpflux.layers.GPLayer.whiten`
    """


def _construct_kernel(input_dim: int, is_last_layer: bool, name: str) -> SquaredExponential:
    """
    Return a :class:`gpflow.kernels.SquaredExponential` kernel with ARD lengthscales set to
    2 and a small kernel variance of 1e-6 if the kernel is part of a hidden layer;
    otherwise, the kernel variance is set to 1.0.

    :param input_dim: The input dimensionality of the layer.
    :param is_last_layer: Whether the kernel is part of the last layer in the Deep GP.
    """
    variance = 0.351 if not is_last_layer else 0.351

    # TODO: Looking at this initializing to 2 (assuming N(0, 1) or U[0,1] normalized
    # data) seems a bit weird - that's really long lengthscales? And I remember seeing
    # something where the value scaled with the number of dimensions before
    lengthscales = [0.351] * input_dim
    return SquaredExponential(lengthscales=lengthscales, variance=variance, name = name)


def build_orthogonal_deep_gp(X: np.ndarray, num_layers: int, config: Config) -> OrthogonalDeepGP:
    r"""

    #TODO -- update documentation  here
    Build a Deep GP consisting of ``num_layers`` :class:`GPLayer`\ s.
    All the hidden layers have the same input dimension as the data, that is, ``X.shape[1]``.

    The architecture is largely based on :cite:t:`salimbeni2017doubly`, with
    the most notable difference being that we keep the hidden dimension equal
    to the input dimensionality of the data.

    .. note::
        This architecture might be slow for high-dimensional data.

    .. note::
        This architecture assumes a :class:`~gpflow.likelihoods.Gaussian` likelihood
        for regression tasks. Specify a different likelihood for performing
        other tasks such as classification.

    :param X: The training input data, used to retrieve the number of datapoints and
        the input dimension and to initialise the inducing point locations using k-means. A
        tensor of rank two with the dimensions ``[num_data, input_dim]``.
    :param num_layers: The number of layers in the Deep GP.
    :param config: The configuration for (hyper)parameters. See :class:`Config` for details.
    """
    num_data, input_dim = X.shape
    X_running = X

    gp_layers = []
    centroids_u, _ = kmeans2(X, k=config.num_inducing_u, minit="points")
    centroids_v, _ = kmeans2(X, k=config.num_inducing_v, minit="points")

    for i_layer in range(num_layers):
        is_last_layer = i_layer == num_layers - 1
        D_in = input_dim
        D_out = 1 if is_last_layer else config.hidden_layer_size

        # Pass in kernels, specify output dim (shared hyperparams/variables)

        if i_layer==0:
            inducing_var_u = construct_basic_inducing_variables(
                num_inducing=config.num_inducing_u, input_dim=D_in, share_variables=True, z_init=centroids_u)
            inducing_var_v = construct_basic_inducing_variables(
                num_inducing=config.num_inducing_v, input_dim=D_in, share_variables=True, z_init=centroids_v)
        else:
            inducing_var_u = construct_basic_inducing_variables(
                num_inducing=config.num_inducing_u, input_dim=D_in, share_variables=True, z_init=None)
            inducing_var_v = construct_basic_inducing_variables(
                num_inducing=config.num_inducing_v, input_dim=D_in, share_variables=True, z_init=None)


        kernel = construct_basic_kernel(
            kernels=_construct_kernel(D_in, is_last_layer, f'kernel_layer_{i_layer}'),
            output_dim=D_out,
            share_hyperparams=True
        )

        assert config.whiten is True, "non-whitened case not implemented yet"

        if is_last_layer:
            mean_function = Zero()
            q_sqrt_scaling = 1.0
        else:
            #NOTE -- remaind to put this back to normal after debugging
            mean_function = construct_mean_function(X_running, D_in, D_out)
            X_running = mean_function(X_running)
            if tf.is_tensor(X_running):
                X_running = X_running.numpy()
            #mean_function = Zero()
            q_sqrt_scaling = config.inner_layer_qsqrt_factor

        #TODO -- this class needs to be implemented
        layer = OrthogonalSVGP(
            kernel,
            inducing_var_u,
            inducing_var_v,
            mean_function = mean_function,
            num_latent_gps = D_out)
        layer.q_sqrt_u.assign(layer.q_sqrt_u * q_sqrt_scaling)
        layer.q_sqrt_v.assign(layer.q_sqrt_v * q_sqrt_scaling)
        gp_layers.append(layer)


    if config.task_type=="regression":
        likelihood = Gaussian(config.likelihood_noise_variance)
    elif config.task_type=='classification' and config.dim_output==1:
        likelihood = Bernoulli()
    elif config.task_type=="classification" and config.dim_output>1:
        likelihood = MultiClass(config.dim_output)
    else:
        raise WarningMessage("wrong specification for likelihood")
    
    return OrthogonalDeepGP(gp_layers, likelihood, num_data = config.num_data)
