
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sympy import Inverse
import tensorflow as tf
import tensorflow_probability as tfp

from gp_package.inverse_approximations import inverse_approximation
from gp_package.kernels.base_kernel import Kernel

from ..base import Parameter, TensorData, TensorType
from ..config import default_float

from ..kullback_leiblers import standard_kl_T
from ..mean_functions import Identity, MeanFunction

from ..conditionals import conditional_GP
from ..inducing_variables import MultioutputInducingVariables
from ..kernels import MultioutputKernel
from ..utils.bijectors import triangular
from ..inverse_approximations import InverseApproximation

#from gpflux.exceptions import GPLayerIncompatibilityException
from ..math import _cholesky_with_jitter
from gp_package import inducing_variables
#from gpflux.runtime_checks import verify_compatibility
#from gpflux.sampling.sample import Sample, efficient_sample

class Schur_upper_bound_Layer(tfp.layers.DistributionLambda):
    """
    A Schur upper bound Layer. Useful for sampling from it and adding

    # TODO -- update documentation for this function   
    """

    def __init__(
        self,
        inverse_approximation: InverseApproximation,
        inducing_variable: MultioutputInducingVariables,
        kernel : Kernel,
        *,
        num_samples: Optional[int] = None,
        full_cov: bool = False,
        name: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        :param inverse_approximation: The inverse approximation T for this layer.
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

        self.inverse_approximation = inverse_approximation
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
        df = inverse_approximation.dof + self.num_inducing + 1.

        upper_bound_Schur = self.get_Schur_upper_bound()

        df_inv_gamma = 0.5 * (df +  1.)
        diagonal_posterior_Schur = 0.5 * df * tf.linalg.diag_part(upper_bound_Schur) 
        df_inv_gamma = tf.ones_like(diagonal_posterior_Schur) * df_inv_gamma 
        

        if self.full_cov:
            # TODO -- need to implement this at one point
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