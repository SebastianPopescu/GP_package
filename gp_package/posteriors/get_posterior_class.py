import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Type, Union

import tensorflow as tf

from gpflow.inducing_variables import (
    InducingVariables,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)

from  ..inducing_variables import FourierFeatures1D

from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.utilities import Dispatcher, add_noise_cov
from gpflow.utilities.ops import eye, leading_transpose

from gp_package.inducing_variables.fourier_features import FourierFeatures1D


from .gpflow_posterior import IndependentPosteriorMultiOutput, IndependentPosteriorSingleOutput
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
    Kernel, FourierFeatures1D
)
def _get_posterior_independent_mo(
    kernel: Kernel, inducing_variable: FourierFeatures1D
) -> Type[BasePosterior]:
    # independent multi-output
    return IndependentPosteriorSingleOutput


