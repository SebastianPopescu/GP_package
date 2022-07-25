import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Type, Union

import tensorflow as tf

from .base_posterior import BaseOrthogonalPosterior, IndependentOrthogonalPosterior
from .gpflow_posterior import IndependentOrthogonalPosteriorMultiOutput

from gpflow.inducing_variables import (
    InducingVariables,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)

from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.utilities import Dispatcher, add_noise_cov
from gpflow.utilities.ops import eye, leading_transpose

get_posterior_class = Dispatcher("get_posterior_class")
from gpflow.kernels import SharedIndependent, SeparateIndependent
from gpflow.posteriors import BasePosterior, IndependentPosteriorMultiOutput


"""
#NOTE -- I don't think we need this here
@get_posterior_class.register(kernels.Kernel, InducingVariables)
def _get_posterior_base_case(
    kernel: Kernel, inducing_variable: InducingVariables
) -> Type[BasePosterior]:
    # independent single output
    return IndependentPosteriorSingleOutput
"""
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
"""

@get_posterior_class.register(
    (SharedIndependent, SeparateIndependent),
    (SeparateIndependentInducingVariables, SharedIndependentInducingVariables),
    (SeparateIndependentInducingVariables, SharedIndependentInducingVariables),
)
def _get_posterior_independent_mo(
    kernel: Kernel, inducing_variable_u: InducingVariables,  inducing_variable_v: InducingVariables
) -> Type[BaseOrthogonalPosterior]:
    # independent multi-output

    #NOTE -- this might create a problem
    return IndependentOrthogonalPosteriorMultiOutput




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


