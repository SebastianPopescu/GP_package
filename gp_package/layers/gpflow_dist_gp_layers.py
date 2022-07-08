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


from typing import Optional, Union, Any

import numpy as np
import tensorflow as tf

from .. import kullback_leiblers, posteriors
from ..base import AnyNDArray, InputData, MeanAndVariance, Parameter, RegressionData, TensorLike
from ..conditionals import conditional
from gpflow.config import default_float
from gpflow.inducing_variables import InducingVariables, InducingPoints
from gpflow.kernels import Kernel
from gpflow.likelihoods import Likelihood
from ..mean_functions import MeanFunction
from gpflow.utilities import positive, triangular
from gpflow.models import GPModel

from ..inducing_variables.distributional_inducing_variables import DistributionalInducingVariables

class Distributional_SVGP(GPModel):
    """
    TODO -- update documentation here

    """

    def __init__(
        self,
        kernel: Kernel,
        likelihood: Likelihood, #TODO -- check gpflow.likelihoods.Likelihood
        inducing_variable: DistributionalInducingVariables,
        *,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: int = 1,
        q_diag: bool = False,
        q_mu: Optional[tf.Tensor] = None,
        q_sqrt: Optional[tf.Tensor] = None,
        whiten: bool = True,
        num_data: Optional[tf.Tensor] = None,
    ):
        """
        - kernel, likelihood, inducing_variables, mean_function are appropriate
          GPflow objects
        - num_latent_gps is the number of latent processes to use, defaults to 1
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # init the super class, accept args
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.num_data = num_data
        self.whiten = whiten


        """
        #NOTE -- this won;'t work in this case
        if not isinstance(inducing_variable, InducingVariables):
            #NOTE -- this will nto work in this case as it needs both Z_mean and Z_var
            self.inducing_variable = InducingPoints(inducing_variable)
        else:
            self.inducing_variable = inducing_variable
        """

        self.inducing_variable = inducing_variable
        # init variational parameters
        num_inducing = self.inducing_variable.num_inducing
        #self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag) 

        ###### Introduce variational parameters for q(U) #######
        self.q_mu = Parameter(
            np.random.uniform(-0.5, 0.5, (num_inducing, self.num_latent_gps)), # np.zeros((num_inducing, self.num_latent_gps)),
            dtype=default_float(),
            name=f"{self.name}_q_mu" if self.name else "q_mu",
        )  # [num_inducing, num_latent_gps]

        self.q_sqrt = Parameter(
            np.stack([np.eye(num_inducing) for _ in range(self.num_latent_gps)]),
            transform=triangular(),
            dtype=default_float(),
            name=f"{self.name}_q_sqrt" if self.name else "q_sqrt",
        )  # [num_latent_gps, num_inducing, num_inducing]

    def prior_kl(self, sampled_inducing_points: TensorLike = None) -> tf.Tensor:
        return kullback_leiblers.prior_kl(
            self.inducing_variable, sampled_inducing_points, self.kernel, self.q_mu, self.q_sqrt, whiten=self.whiten
        )

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        
        mu, var = conditional(
            Xnew,
            self.inducing_variable,
            self.kernel,
            self.q_mu,
            q_sqrt=self.q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )

        # tf.debugging.assert_positive(var)  # We really should make the tests pass with this here
        return mu + self.mean_function(Xnew), var

    def __call__(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:

        return self.predict_f(Xnew, full_cov, full_output_cov)

    def maximum_log_likelihood_objective(self, *args: Any, **kwargs: Any) -> tf.Tensor:
        """
        Objective for maximum likelihood estimation. Should be maximized. E.g.
        log-marginal likelihood (hyperparameter likelihood) for GPR, or lower
        bound to the log-marginal likelihood (ELBO) for sparse and variational
        GPs.
        """
        raise NotImplementedError