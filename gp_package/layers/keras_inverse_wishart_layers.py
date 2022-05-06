# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
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

import abc
from typing import Any, Dict, List, Optional
from numpy import isin

import tensorflow as tf
import tensorflow_probability as tfp
from deprecated import deprecated
from gp_package.config import default_float
from gp_package.inverse_approximations.inverse_approximation import InverseApproximation
from gp_package.math import compute_A_inv_b

from gp_package.utils.bijectors import triangular

from ..base import Module, Parameter, TensorData, TensorType
from ..utils import positive
from ..inducing_variables import MultioutputInducingVariables
from ..kernels import Kernel


class InverseWishartCovariance(Module):
    
    def __init__(self, dof : TensorType, name: Optional[str] = None):
        
        """
        param dof: the initialization value for the restriced intrinsic degree of freedom of the underlying inverse Wishart distribution

        TODO -- document this class; see if we can add other functionalities
        """

        super().__init__(name=name)

        if not isinstance(dof, (tf.Variable, tfp.util.TransformedVariable)):

            dof = Parameter(
                dof,
                transform = positive(),
                dtype=default_float(),
                name=f"{self.name}_dof_inverse_wishart" if self.name else "dof_inverse_wishart"
            )
        self.dof = dof



class Schur_upper_bound_Layer(tfp.layers.DistributionLambda):
    """
    A Schur upper bound Layer. Useful for sampling from it and adding

    # TODO -- update documentation for this function   
    """

    def __init__(
        self,
        inverse_wishart_covariance: InverseWishartCovariance,
        inducing_variable: MultioutputInducingVariables,
        kernel : Kernel,
        *,
        num_samples: Optional[int] = None,
        full_cov: bool = False,
        name: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        :param inverse_wishart_covariance: ....
        :param inducing_variable: The inducing features for this layer.
        :param num_samples: The number of samples to draw when converting the
            :class:`~tfp.layers.DistributionLambda` into a `tf.Tensor`, see
            :meth:`_convert_to_tensor_fn`. Will be stored in the
            :attr:`num_samples` attribute.  If `None` (the default), draw a
            single sample without prefixing the sample shape (see
            :class:`tfp.distributions.Distribution`'s `sample()
            <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution#sample>`_
            method).
        :param full_cov: only works with full_cov = False at the moment since we are missing tfp.distributions.InverseWishart
        :param name: The name of this layer.
        :param verbose: The verbosity mode. Set this parameter to `True`
            to show debug information.
        """

        super().__init__(
            make_distribution_fn=self._make_distribution_fn,
            convert_to_tensor_fn=self._convert_to_tensor_fn,
            dtype=default_float(),
            name=name,
        )

        self.inverse_wishart_covariance = inverse_wishart_covariance
        self.inducing_variable = inducing_variable
        self.num_inducing = self.inducing_variable.num_inducing
        self.num_samples = num_samples

    def call(self, *args: List[Any], **kwargs: Dict[str, Any]) -> tf.Tensor:
        """
        The default behaviour upon calling this layer.

        This method calls the `tfp.layers.DistributionLambda` super-class
        `call` method, which constructs a `tfp.distributions.Distribution`
        for the predictive distributions at the input points
        (see :meth:`_make_distribution_fn`).
        You can pass this distribution to `tf.convert_to_tensor`, which will return
        samples from the distribution (see :meth:`_convert_to_tensor_fn`).
        """
        outputs = super().call(*args, **kwargs)

        return outputs

    def get_Schur_upper_bound(self,
        T : tf.Tensor,
        Kuf : tf.Tensor,
        Kuu : tf.Tensor,
        Kff : tf.Tensor
        ) -> tf.Tensor:

        """
        TODO -- add documentation about this function
        """

        T_Kuf = tf.linalg.matmul(T, Kuf)
        Kuu_T_Kuf = tf.linalg.matmul(Kuu, T_Kuf)
        Schur_upper_bound = Kff + tf.linalg.matmul(T_Kuf, Kuu_T_Kuf, transpose_a = True)
        Schur_upper_bound += -  2. * tf.linalg.matmul(Kuf, T_Kuf, transpose_a=True)

        return Schur_upper_bound


    def _make_distribution_fn(self, 
        inverse_approximation: InverseApproximation,
        T : tf.Tensor,
        Kuf : tf.Tensor,
        Kuu : tf.Tensor,
        Kff : tf.Tensor
    ) -> tfp.distributions.Distribution:
        """
        Construct the posterior distributions for T
        
        :param inverse_approximation: TODO -- document this
        """

        # Remainder: df = dof + M + 1 for notational purposes as in paper   
        df = self.inverse_wishart_covariance.dof + self.num_inducing + 1.

        upper_bound_Schur = self.get_Schur_upper_bound()


        # TODO -- this only works for not self.full_cov atm
        df_inv_gamma = 0.5 * (df +  1.)
        diagonal_posterior_Schur = 0.5 * df * tf.linalg.diag_part(upper_bound_Schur) 
        df_inv_gamma = tf.ones_like(diagonal_posterior_Schur) * df_inv_gamma 
        

        if self.full_cov:
            # TODO -- need to implement this at one point, involves sampling from an InverseWishart distribution
            raise NotImplementedError
        else:

            return tfp.distributions.InverseGamma(
                concentration = df_inv_gamma, 
                scale = diagonal_posterior_Schur, 
                name='InvGamma-UpperBoundSchur')                    
        
    def _convert_to_tensor_fn(self, distribution: tfp.distributions.Distribution) -> tf.Tensor:
        """
        Convert the approximate posterior distribution of q(T) (see
        :meth:`_make_distribution_fn`) to a tensor of :attr:`num_samples`
        samples from that distribution.
        """
        # N input points
        # S = self.num_samples
        # Q = output dimensionality
        if self.num_samples is not None:
            samples = distribution.sample(
                (self.num_samples,)
            )  # [S,N] 
        else:
            samples = distribution.sample()  # [N,]

        # TODO -- need to check if we need to add another dimension at the end 

        return samples

    """
    def sample(self) -> Sample:
        
        #.. todo:: TODO: Document this.
        
        return (
            efficient_sample(
                self.inducing_variable,
                self.kernel,
                self.q_mu,
                q_sqrt=self.q_sqrt,
                whiten=self.whiten,
            )
            # Makes use of the magic __add__ of the Sample class
            + self.mean_function
        )
    """








class Interpolator_Layer(tfp.layers.DistributionLambda):
    """
    An Interpolator Layer. Useful for sampling from it and adding

    # TODO -- update documentation for this function   
    """

    def __init__(
        self,
        inverse_wishart_covariance: InverseWishartCovariance,
        inducing_variable: MultioutputInducingVariables,
        kernel : Kernel,
        Schur : TensorType,
        *,
        num_samples: Optional[int] = None,
        full_cov: bool = False,
        name: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        :param inverse_wishart_covariance: ...
        :param inducing_variable: The inducing features for this layer.
        :param Schur: sampled Schur complement stemming from Upper Bound Schur Layer 
        :param num_samples: The number of samples to draw when converting the
            :class:`~tfp.layers.DistributionLambda` into a `tf.Tensor`, see
            :meth:`_convert_to_tensor_fn`. Will be stored in the
            :attr:`num_samples` attribute.  If `None` (the default), draw a
            single sample without prefixing the sample shape (see
            :class:`tfp.distributions.Distribution`'s `sample()
            <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution#sample>`_
            method).
        :param full_cov: only works with full_cov = False at the moment since we are missing tfp.distributions.InverseWishart
        :param name: The name of this layer.
        :param verbose: The verbosity mode. Set this parameter to `True`
            to show debug information.
        """

        super().__init__(
            make_distribution_fn=self._make_distribution_fn,
            convert_to_tensor_fn=self._convert_to_tensor_fn,
            dtype=default_float(),
            name=name,
        )

        self.inverse_wishart_covariance = inverse_wishart_covariance
        self.inducing_variable = inducing_variable
        self.Schur = Schur
        self.num_inducing = self.inducing_variable.num_inducing
        self.num_samples = num_samples

    def call(self, *args: List[Any], **kwargs: Dict[str, Any]) -> tf.Tensor:
        """
        The default behaviour upon calling this layer.

        This method calls the `tfp.layers.DistributionLambda` super-class
        `call` method, which constructs a `tfp.distributions.Distribution`
        for the predictive distributions at the input points
        (see :meth:`_make_distribution_fn`).
        You can pass this distribution to `tf.convert_to_tensor`, which will return
        samples from the distribution (see :meth:`_convert_to_tensor_fn`).
        """
        outputs = super().call(*args, **kwargs)

        return outputs

    def _make_distribution_fn(self, 
        inverse_approximation: InverseApproximation,
        T : tf.Tensor,
        Kuf : tf.Tensor,
        Kuu : tf.Tensor,
        Kff : tf.Tensor
    ) -> tfp.distributions.Distribution:
        """
        Construct the posterior distributions for T
        
        :param inverse_approximation: TODO -- document this
        """

        # Remainder: df = dof + M + 1 for notational purposes as in paper   
        
        ### TODO -- need the degrees of freedom of the Inverse-Wishart
        df = .dof + self.num_inducing + 1.

        if self.full_cov:
            ### TODO -- need to make sure it works in this case

            ### equations ....

            # return tfp.distributions.MatrixNormalLinearOperator

            raise NotImplementedError

        else:

            batched_diagonal_hadamard_product = tf.multiply(batched_sqrt_Schur, batched_posterior_cholesky_Kmm_inverse) ### shape -- (num_batch, num_inducing, num_inducing)
            mean_interpolator = T_Kuf = tf.linalg.matmul(T, Kuf)

            ### TODO -- need to broadcast Schur to be of size (N,M,M)            
            
            scaled_Schur  = self.Schur / tf.sqrt(df)
            var_interpolator = scaled_Schur * T

            return tfp.distributions.MultivariateNormalDiag(
                loc=None, 
                scale_diag=None, scale_identity_multiplier=None, 
                name='MultivariateNormalDiag'
                )
    
    def _convert_to_tensor_fn(self, distribution: tfp.distributions.Distribution) -> tf.Tensor:
        """
        Convert the approximate posterior distribution of q(T) (see
        :meth:`_make_distribution_fn`) to a tensor of :attr:`num_samples`
        samples from that distribution.
        """
        # N input points
        # S = self.num_samples
        # Q = output dimensionality
        if self.num_samples is not None:
            samples = distribution.sample(
                (self.num_samples,)
            )  # [S,N] 
        else:
            samples = distribution.sample()  # [N,]

        # TODO -- need to check if we need to add another dimension at the end 

        return samples

    """
    def sample(self) -> Sample:
        
        #.. todo:: TODO: Document this.
        
        return (
            efficient_sample(
                self.inducing_variable,
                self.kernel,
                self.q_mu,
                q_sqrt=self.q_sqrt,
                whiten=self.whiten,
            )
            # Makes use of the magic __add__ of the Sample class
            + self.mean_function
        )
    """



