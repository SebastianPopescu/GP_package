import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Type, Union

import tensorflow as tf

from .. import covariances, kernels, mean_functions

from gpflow.inducing_variables import (
    InducingVariables,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)

from ..inducing_variables import (
    DistributionalInducingVariables,
    SeparateIndependentDistributionalInducingVariables,
    SharedIndependentDistributionalInducingVariables
)

from ..kernels import DistributionalSharedIndependent


from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.utilities import Dispatcher, add_noise_cov
from gpflow.utilities.ops import eye, leading_transpose


from .gpflow_posterior import IndependentPosteriorMultiOutput
from .base_posterior import BasePosterior

get_posterior_class = Dispatcher("get_posterior_class")
from gpflow.kernels import SharedIndependent, SeparateIndependent


"""
#NOTE -- I don't think we need this here
@get_posterior_class.register(kernels.Kernel, InducingVariables)
def _get_posterior_base_case(
    kernel: Kernel, inducing_variable: InducingVariables
) -> Type[BasePosterior]:
    # independent single output
    return IndependentPosteriorSingleOutput
"""

@get_posterior_class.register(
    (SharedIndependent, SeparateIndependent),
    (SeparateIndependentInducingVariables, SharedIndependentInducingVariables),
)
def _get_posterior_independent_mo(
    kernel: Kernel, inducing_variable: InducingVariables
) -> Type[BasePosterior]:
    # independent multi-output
    return IndependentPosteriorMultiOutput



@get_posterior_class.register(
    DistributionalSharedIndependent,
    (SeparateIndependentDistributionalInducingVariables, SharedIndependentDistributionalInducingVariables),
)
def _get_posterior_independent_mo_distributional(
    kernel: Kernel, inducing_variable: DistributionalInducingVariables
) -> Type[BasePosterior]:
    # independent multi-output
    return IndependentPosteriorMultiOutput




"""
#NOTE -- I don't think we need this
def create_posterior(
    kernel: Kernel,
    inducing_variable: InducingVariables,
    q_mu: TensorType,
    q_sqrt: TensorType,
    whiten: bool,
    mean_function: Optional[MeanFunction] = None,
    precompute_cache: Union[PrecomputeCacheType, str, None] = PrecomputeCacheType.TENSOR,
) -> BasePosterior:
    posterior_class = get_posterior_class(kernel, inducing_variable)
    precompute_cache = _validate_precompute_cache_type(precompute_cache)
    return posterior_class(  # type: ignore
        kernel,
        inducing_variable,
        q_mu,
        q_sqrt,
        whiten,
        mean_function,
        precompute_cache=precompute_cache,
    )
"""


