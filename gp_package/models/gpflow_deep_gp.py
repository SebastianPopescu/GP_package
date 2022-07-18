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

from typing import Optional, Union, List

import numpy as np
import tensorflow as tf

from .. import kullback_leiblers, posteriors
from gpflow.base import AnyNDArray, InputData, MeanAndVariance, Module, Parameter, RegressionData
from ..conditionals import conditional
from ..config import default_float
from gpflow.inducing_variables import InducingVariables
from gpflow.kernels import Kernel
from gpflow.likelihoods import Likelihood
from ..mean_functions import MeanFunction
from gpflow.utilities import positive, triangular
from gpflow.conditionals.util import sample_mvn
from gpflow.models import BayesianModel, ExternalDataTrainingLossMixin
from ..layers import SVGP


class DeepGP(BayesianModel, ExternalDataTrainingLossMixin):

    """
    #TODO -- update documentation
    """

    def __init__(
        self,
        f_layers: List[SVGP],
        likelihood: Likelihood,
        *,
        num_data: Optional[tf.Tensor] = None,
    ):
        """
        # TODO -- update documentation here
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # init the super class, accept args
        #super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.likelihood = likelihood
        self.num_data = num_data
        self.f_layers = f_layers

    def prior_kl_across_layers(self) -> tf.Tensor:
        
        """
        Get list of KL terms for each hidden layer
        """

        list_KL = []

        for layer in self.f_layers:
            list_KL.append(kullback_leiblers.prior_kl(
                layer.inducing_variable, layer.kernel, layer.q_mu, layer.q_sqrt, whiten=layer.whiten)
            )
        return list_KL

    # type-ignore is because of changed method signature:
    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:  # type: ignore
        return self.elbo(data)

    def elbo(self, data: RegressionData, detailed_elbo: bool = False) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = data
        kl = self.prior_kl_across_layers() # NOTE -- this is a list 
        
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl[0].dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl[0].dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        if detailed_elbo:
            return tf.reduce_sum(var_exp) * scale - tf.reduce_sum(kl), tf.reduce_sum(var_exp) * scale, kl        
        else:
            return tf.reduce_sum(var_exp) * scale - tf.reduce_sum(kl)

    def predict_f(
        self, Xnew: InputData, num_samples: int = 1, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:

        features = Xnew
        for layer in self.f_layers:
            mean, cov = layer(features)
        
            ### sampling part ###
            if full_cov:
                # mean: [..., N, P]
                # cov: [..., P, N, N]
                mean_for_sample = tf.linalg.adjoint(mean)  # [..., P, N]
                features = sample_mvn(
                    mean_for_sample, cov, full_cov, num_samples=num_samples
                )  # [..., (S), P, N]
                features = tf.linalg.adjoint(features)  # [..., (S), N, P]
            else:
                # mean: [..., N, P]
                # cov: [..., N, P] or [..., N, P, P]
                features = sample_mvn(
                    mean, cov, full_output_cov, num_samples=num_samples
                )  # [..., (S), N, P]
            features = tf.squeeze(features, axis = 0) 
        
        return mean, cov

        # tf.debugging.assert_positive(var)  # We really should make the tests pass with this here
        #return mu + self.mean_function(Xnew), var

    def _evaluate_layer_wise_deep_gp(
        self, Xnew: InputData, *, num_samples: int = 1, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:

        layer_moments = []

        features = Xnew
        for layer in self.f_layers:
            mean, cov = layer(features)
            layer_moments.append([mean, cov])

            if full_cov:
                # mean: [..., N, P]
                # cov: [..., P, N, N]
                mean_for_sample = tf.linalg.adjoint(mean)  # [..., P, N]
                features = sample_mvn(
                    mean_for_sample, cov, full_cov, num_samples=num_samples
                )  # [..., (S), P, N]
                features = tf.linalg.adjoint(features)  # [..., (S), N, P]
            else:
                # mean: [..., N, P]
                # cov: [..., N, P] or [..., N, P, P]
                features = sample_mvn(
                    mean, cov, full_output_cov, num_samples=num_samples
                )  # [..., (S), N, P]
            features = tf.squeeze(features, axis = 0)            
        return layer_moments

        # tf.debugging.assert_positive(var)  # We really should make the tests pass with this here
        #return mu + self.mean_function(Xnew), var
