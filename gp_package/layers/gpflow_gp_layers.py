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
from ..base import AnyNDArray, InputData, MeanAndVariance, Parameter, RegressionData
from ..conditionals import conditional
from gpflow.config import default_float
from gpflow.inducing_variables import InducingVariables, InducingPoints
from gpflow.kernels import Kernel
from gpflow.likelihoods import Likelihood
from ..mean_functions import MeanFunction
from gpflow.utilities import positive, triangular
from gpflow.models import GPModel
#from .util import InducingVariablesLike, inducingpoint_wrapper


InducingVariablesLike = Union[InducingVariables, tf.Tensor, AnyNDArray]
InducingPointsLike = Union[InducingPoints, tf.Tensor, AnyNDArray]


class SVGP(GPModel):
    """
    This is the Sparse Variational GP (SVGP). The key reference is

    ::

      @inproceedings{hensman2014scalable,
        title={Scalable Variational Gaussian Process Classification},
        author={Hensman, James and Matthews, Alexander G. de G. and Ghahramani, Zoubin},
        booktitle={Proceedings of AISTATS},
        year={2015}
      }

    """

    def __init__(
        self,
        kernel: Kernel,
        likelihood: Likelihood,
        inducing_variable: InducingVariablesLike,
        *,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: int = 1,
        q_diag: bool = False,
        q_mu: Optional[tf.Tensor] = None,
        q_sqrt: Optional[tf.Tensor] = None,
        whiten: bool = True,
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
        self.whiten = whiten


        if not isinstance(inducing_variable, InducingVariables):
            self.inducing_variable = InducingPoints(inducing_variable)
        else:
          self.inducing_variable = inducing_variable

        # init variational parameters
        num_inducing = self.inducing_variable.num_inducing
    
        #NOTE -- we don't actuallty make use of kwargs: q_mu and q_sqrt

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

    def prior_kl(self) -> tf.Tensor:
        return kullback_leiblers.prior_kl(
            self.inducing_variable, self.kernel, self.q_mu, self.q_sqrt, whiten=self.whiten
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