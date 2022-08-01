# Copyright 2016-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Type, Union

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from .. import covariances
from gpflow.base import MeanAndVariance, Module, Parameter, RegressionData, TensorType
from gpflow.conditionals.util import (
    base_conditional,
    base_conditional_with_lm,
    expand_independent_outputs,
    fully_correlated_conditional,
    independent_interdomain_conditional,
    mix_latent_gp,
    separate_independent_conditional_implementation,
)
from gpflow.config import default_float, default_jitter
from ..covariances import Kuf, Kuu
from gpflow.inducing_variables import (
    FallbackSeparateIndependentInducingVariables,
    FallbackSharedIndependentInducingVariables,
    InducingPoints,
    InducingVariables,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.utilities import Dispatcher, add_noise_cov
from gpflow.utilities.ops import eye, leading_transpose
from .base_posterior import IndependentPosterior

class IndependentPosteriorSingleOutput(IndependentPosterior):
    
    # could almost be the same as IndependentPosteriorMultiOutput ...
    def _conditional_fused(
        self, Xnew: TensorType, full_cov: bool = False, full_output_cov: bool = False, detailed_moments: bool = False
    ) -> MeanAndVariance:
        # same as IndependentPosteriorMultiOutput, Shared~/Shared~ branch, except for following
        # line:
        Knn = self.kernel(Xnew, full_cov=full_cov)

        Kmm = Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [M, M]
        Kmn = Kuf(self.X_data, self.kernel, Xnew)  # [M, N]

        fmean, fvar = base_conditional(
            Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
        )  # [N, P],  [P, N, N] or [N, P]
        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)

#NOTE -- we won't make use of this 
class IndependentPosteriorMultiOutput(IndependentPosterior):

    def _conditional_fused(
        self, Xnew: TensorType, *, full_cov: bool = False, full_output_cov: bool = False, detailed_moments: bool = False
    ) -> MeanAndVariance:

        # same as IndependentPosteriorSingleOutput except for following line
        Knn = self.kernel.kernel(Xnew, full_cov=full_cov)
        # we don't call self.kernel() directly as that would do unnecessary tiling

        Kmm = covariances.Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [M, M]
        Kmn = covariances.Kuf(self.X_data, self.kernel, Xnew)  # [M, N]

        fmean, fvar = base_conditional(
            Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
        )  # [N, P],  [P, N, N] or [N, P]

        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)


