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

from gpflow.base import AnyNDArray, InputData, MeanAndVariance, Parameter, RegressionData
from ..conditionals import conditional
from gpflow.config import default_float
from gpflow.inducing_variables import InducingVariables, InducingPoints
from gpflow.kernels import Kernel
from gpflow.likelihoods import Likelihood
from gpflow.mean_functions import MeanFunction, Zero
from gpflow.utilities import positive, triangular
from gpflow.base import Module
#from .util import InducingVariablesLike, inducingpoint_wrapper

from .. import kullback_leiblers


InducingVariablesLike = Union[InducingVariables, tf.Tensor, AnyNDArray]
InducingPointsLike = Union[InducingPoints, tf.Tensor, AnyNDArray]

class OrthogonalSVGP(Module):
    """
    #TODO -- update documentation here 
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
        inducing_variable_u: InducingVariablesLike,
        inducing_variable_v: InducingVariablesLike,
        *,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: int = 1,
        q_diag: bool = False,
        q_mu_u: Optional[tf.Tensor] = None,
        q_sqrt_u: Optional[tf.Tensor] = None,
        q_mu_v: Optional[tf.Tensor] = None,
        q_sqrt_v: Optional[tf.Tensor] = None,
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
        super().__init__()
        self.kernel = kernel
        if mean_function is None:
          mean_function = Zero()
        self.mean_function = mean_function
        self.num_latent_gps = num_latent_gps
        self.whiten = whiten

        if not isinstance(inducing_variable_u, InducingVariables):
            self.inducing_variable_u = InducingPoints(inducing_variable_u)
        else:
          self.inducing_variable_u = inducing_variable_u
        # init variational parameters
        num_inducing_u = self.inducing_variable_u.num_inducing

        if not isinstance(inducing_variable_v, InducingVariables):
            self.inducing_variable_v = InducingPoints(inducing_variable_v)
        else:
          self.inducing_variable_v = inducing_variable_v
        # init variational parameters
        num_inducing_v = self.inducing_variable_v.num_inducing

        ########################################################
        ###### Introduce variational parameters for q(U) #######
        self.q_mu_u = Parameter(
            np.random.uniform(-0.5, 0.5, (num_inducing_u, self.num_latent_gps)), # np.zeros((num_inducing, self.num_latent_gps)),
            dtype=default_float(),
            name=f"{self.name}_q_mu_u" if self.name else "q_mu_u",
        )  # [num_inducing, num_latent_gps]

        self.q_sqrt_u = Parameter(
            np.stack([np.eye(num_inducing_u) for _ in range(self.num_latent_gps)]),
            transform=triangular(),
            dtype=default_float(),
            name=f"{self.name}_q_sqrt_u" if self.name else "q_sqrt_u",
        )  # [num_latent_gps, num_inducing, num_inducing]



        ########################################################
        ###### Introduce variational parameters for q(V) #######
        self.q_mu_v = Parameter(
            np.random.uniform(-0.5, 0.5, (num_inducing_v, self.num_latent_gps)), # np.zeros((num_inducing, self.num_latent_gps)),
            dtype=default_float(),
            name=f"{self.name}_q_mu_v" if self.name else "q_mu_v",
        )  # [num_inducing, num_latent_gps]

        self.q_sqrt_v = Parameter(
            np.stack([np.eye(num_inducing_v) for _ in range(self.num_latent_gps)]),
            transform=triangular(),
            dtype=default_float(),
            name=f"{self.name}_q_sqrt_v" if self.name else "q_sqrt_v",
        )  # [num_latent_gps, num_inducing, num_inducing]


    def prior_kl(self) -> tf.Tensor:
        
        #NOTE -- it returns a list of the KLs
        
        return kullback_leiblers.prior_kl(
            self.inducing_variable_u, self.kernel, self.q_mu_u, self.q_sqrt_u, whiten=self.whiten
        ), kullback_leiblers.prior_kl(
            self.inducing_variable_v, self.kernel, self.q_mu_v, self.q_sqrt_v, whiten=self.whiten
        )

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        
        mu, var = conditional(
            Xnew,
            self.inducing_variable_u,
            self.inducing_variable_v,
            self.kernel,
            self.q_mu_u,
            self.q_mu_v,
            q_sqrt_u=self.q_sqrt_u,
            q_sqrt_v=self.q_sqrt_v,
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
        #return self.predict_f_samples(Xnew, num_samples, full_cov, full_output_cov)

