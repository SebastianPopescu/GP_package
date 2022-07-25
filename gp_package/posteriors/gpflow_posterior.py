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

from ..conditionals.util import (
    base_orthogonal_conditional
)

from gpflow.config import default_float, default_jitter
from .base_posterior import IndependentOrthogonalPosterior


"""
#TODO -- take care of this
class IndependentPosteriorSingleOutput(IndependentPosterior):
    
    # could almost be the same as IndependentPosteriorMultiOutput ...
    def _conditional_fused(
        self, Xnew: Union[TensorType,tfp.distributions.MultivariateNormalDiag], full_cov: bool = False, full_output_cov: bool = False
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

#TODO -- take care of this
class IndependentPosteriorSingleOutput(IndependentPosterior):
    # could almost be the same as IndependentPosteriorMultiOutput ...
    def _conditional_fused(
        self, Xnew: TensorType, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        # same as IndependentPosteriorMultiOutput, Shared~/Shared~ branch, except for following
        # line:
        Knn = self.kernel(Xnew, full_cov=full_cov)

        Kmm = covariances.Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [M, M]
        Kmn = covariances.Kuf(self.X_data, self.kernel, Xnew)  # [M, N]

        fmean, fvar = base_conditional(
            Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
        )  # [N, P],  [P, N, N] or [N, P]
        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)
"""

class IndependentOrthogonalPosteriorMultiOutput(IndependentOrthogonalPosterior):

    def _conditional_fused_OrthogonalSVGP(
        self, Xnew: TensorType, *, full_cov: bool = False, full_output_cov: bool = False, detailed_moments: bool = False
    ) -> MeanAndVariance:

        Knn = self._get_Kff(Xnew, full_cov=full_output_cov)
        Cnn, L_Kuu = self._get_Cff(Xnew, full_cov=full_output_cov)

        Kmm = covariances.Kuu(self.inducing_variable_u, self.kernel, jitter=default_jitter())  # [M_u, M_u]
        Kmn = covariances.Kuf(self.inducing_variable_u, self.kernel, Xnew)  # [M_U, N]

        Cmm = covariances.Cvv(self.inducing_variable_u, self.inducing_variable_v, self.kernel, jitter=default_jitter(), L_Kuu = L_Kuu)  # [M_v, M_v]
        Cmn = covariances.Cvf(self.inducing_variable_u, self.inducing_variable_v, self.kernel, Xnew, L_Kuu = L_Kuu)  # [M_v, N]

        fmean, fvar = base_orthogonal_conditional(
            Kmn, Kmm, Knn, Cmn, Cmm, Cnn,
            self.q_mu_u, self.q_mu_v, full_cov=full_cov, q_sqrt_u=self.q_sqrt_u, q_sqrt_v=self.q_sqrt_v , white=self.whiten
        )  # [N, P],  [P, N, N] or [N, P]

        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)

