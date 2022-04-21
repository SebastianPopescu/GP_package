
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sympy import Inverse
import tensorflow as tf
import tensorflow_probability as tfp

from gp_package.inverse_approximations import inverse_approximation

from ..base import Parameter, TensorType
from ..config import default_float

from ..kullback_leiblers import standard_kl
from ..mean_functions import Identity, MeanFunction

from ..conditionals import conditional_GP
from ..inducing_variables import MultioutputInducingVariables
from ..kernels import MultioutputKernel
from ..utils.bijectors import triangular
from ..inverse_approximations import InverseApproximation

#from gpflux.exceptions import GPLayerIncompatibilityException
from ..math import _cholesky_with_jitter
#from gpflux.runtime_checks import verify_compatibility
#from gpflux.sampling.sample import Sample, efficient_sample

class TLayer(tfp.layers.DistributionLambda):
    """
    A T Layer. Useful for sampling from it and adding   
    """

    def __init__(
        self,
        inverse_approximation: InverseApproximation,
        inducing_variable: MultioutputInducingVariables,
        *,
        num_samples: Optional[int] = None,
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

        This method also adds a layer-specific loss function, given by the KL divergence between
        this layer and the GP prior (scaled to per-datapoint).
        """
        outputs = super().call(*args, **kwargs)

        if kwargs.get("training"):
            #log_prior = tf.add_n([p.log_prior_density() for p in self.kernel.trainable_parameters])
            loss = self.standard_kl() 

        else:
            # TF quirk: add_loss must always add a tensor to compile
            loss = tf.constant(0.0, dtype=default_float())
        self.add_loss(loss)

        # Metric names should be unique; otherwise they get overwritten if you
        # have multiple with the same name
        name = f"{self.name}_standard_kl_Wishart" if self.name else "standard_kl_Wishart"
        self.add_metric(loss, name=name, aggregation="mean")

        return outputs

    def standard_kl(self) -> tf.Tensor:
        r"""
        Returns the KL divergence ``KL[q(T)∥p(T)]`` from the prior ``p(T)`` to
        the variational distribution ``q(T)``.  
        """
        
        #TODO -- import the proper standard_kl for Wishart distributions
        #TODO -- make the right changes to the arguments
        return standard_kl(
            self.inducing_variable, self.kernel, self.q_mu, self.q_sqrt, whiten=self.whiten
        )

    def _make_distribution_fn(
        self, inverse_approximation: InverseApproximation
    ) -> tfp.distributions.Distribution:
        """
        Construct the posterior distributions for T
        
        :param inverse_approximation: TODO -- document this
        """

        # Remainder: df = dof + M + 1 for notational purposes as in paper   
        df = self.inverse_approximation.dof + self.num_inducing + 1.

        return tfp.distributions.WishartTriL(
                df=df, scale_tril=self.inverse_approximation.L_T)
            )  # df: [1,], scale_tril: [M, M]

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
            )  # [S, M, M] 
        else:
            samples = distribution.sample()  # [M,M] 

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