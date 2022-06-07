
from ctypes import set_errno
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ..base import Parameter, TensorType
from ..config import default_float

from ..kullback_leiblers import standard_kl
from ..mean_functions import Identity, MeanFunction

from ..conditionals import conditional_GP
from ..inducing_variables import MultioutputInducingVariables
from ..kernels import MultioutputKernel
from ..utils.bijectors import positive, triangular

#from gpflux.exceptions import GPLayerIncompatibilityException
from ..math import _cholesky_with_jitter
#from gpflux.runtime_checks import verify_compatibility
#from gpflux.sampling.sample import Sample, efficient_sample
from .flow_base_layer import Flow

class ArcSinhFlow(Flow):
    
    """
    -- Arcsinh Flow --

    - smooth flow, however does not recover the identity
    -


    #TODO -- document class
    Essentially implements the following equation:
        fk = a + b * arcsinh[(f0-c)/d]

    """

    init_a: float    
    init_b: float
    init_c: float
    init_d: float
    set_restrictions: Optional[bool]
    add_init_f0: Optional[bool]

    def __init__(
        self,
        init_a: TensorType,
        init_b: TensorType,
        init_c: TensorType,
        init_d: TensorType,
        *,
        set_restrictions: Optional[bool] = False,
        add_init_f0: Optional[bool] = False,
        name: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        #TODO -- document function 
        :param init_a: 
        :param init_b: 
        :param init_c: 
        :param init_d: 
        """

        #########################################################
        self.a = Parameter(
            init_a,
            dtype=default_float(),
            name=f"{self.name}_a" if self.name else "a",
        )  

        self.b = Parameter(
            init_b,
            dtype=default_float(),
            name=f"{self.name}_b" if self.name else "b",
            transform=positive() if set_restrictions else None
        )  

        self.c = Parameter(
            init_c,
            dtype=default_float(),
            name=f"{self.name}_c" if self.name else "c",
        )  

        self.d = Parameter(
            init_d,
            dtype=default_float(),
            name=f"{self.name}_d" if self.name else "d",
            transform=positive() if set_restrictions else None
        )  
        #########################################################

        if add_init_f0:
            set_restrictions = True
        self.set_restrictions = True
        self.add_init_f0 = True

    def forward(self, F: TensorType, X: TensorType = None) -> tf.Tensor:
        
        if self.add_init_f0:

            return F + self.a + self.b* tf.math.asinh( (F - self.c) / self.d)

        else:

            return self.a + self.b* tf.math.asinh( (F - self.c) / self.d)

    def forward_grad(self, F: TensorType) -> tf.Tensor:
        
        return self.b * tf.math.cosh(self.b * tf.math.asinh(x)-self.a)/tf.sqrt(1.+ F**2)


    def inverse(self, F: TensorType) -> tf.Tensor:
        
        return self.c + self.d* tf.math.sinh((F-self.a)/self.b)


    def call(self, inputs: TensorType, *args: List[Any], **kwargs: Dict[str, Any]) -> tf.Tensor:
        """
        The default behaviour upon calling this layer.

        """

        # I think this is getting just the samples from the distribution 
        outputs = super().call(inputs, *args, **kwargs)
        
        if kwargs.get("training"):
            #log_prior = tf.add_n([p.log_prior_density() for p in self.kernel.trainable_parameters])
            loss = self.standard_kl() #- log_prior
            loss_per_datapoint = loss / self.num_data

        else:
            # TF quirk: add_loss must always add a tensor to compile
            loss_per_datapoint = tf.constant(0.0, dtype=default_float())
        self.add_loss(loss_per_datapoint)

        # Metric names should be unique; otherwise they get overwritten if you
        # have multiple with the same name
        name = f"{self.name}_standard_kl" if self.name else "standard_kl"
        self.add_metric(loss_per_datapoint, name=name, aggregation="mean")

        return outputs

    def standard_kl(self) -> tf.Tensor:
        r"""
        Returns the KL divergence ``KL[q(u)∥p(u)]`` from the prior ``p(u)`` to
        the variational distribution ``q(u)``.  If this layer uses the
        :attr:`whiten`\ ed representation, returns ``KL[q(v)∥p(v)]``.
        """
        return standard_kl(
            self.inducing_variable, self.kernel, self.q_mu, self.q_sqrt, whiten=self.whiten
        )

    def _make_distribution_fn(
        self, previous_layer_outputs: TensorType
    ) -> tfp.distributions.Distribution:
        """
        Construct the posterior distributions at the output points of the previous layer,
        depending on :attr:`full_cov` and :attr:`full_output_cov`.

        :param previous_layer_outputs: The output from the previous layer,
            which should be coercible to a `tf.Tensor`
        """
        mean, cov = self.predict(
            previous_layer_outputs,
            full_cov=self.full_cov,
            full_output_cov=self.full_output_cov,
        )

        if self.full_cov and not self.full_output_cov:
            # mean: [N, Q], cov: [Q, N, N]
            return tfp.distributions.MultivariateNormalTriL(
                loc=tf.linalg.adjoint(mean), scale_tril=_cholesky_with_jitter(cov)
            )  # loc: [Q, N], scale: [Q, N, N]
        elif self.full_output_cov and not self.full_cov:
            # mean: [N, Q], cov: [N, Q, Q]
            return tfp.distributions.MultivariateNormalTriL(
                loc=mean, scale_tril=_cholesky_with_jitter(cov)
            )  # loc: [N, Q], scale: [N, Q, Q]
        elif not self.full_cov and not self.full_output_cov:
            # mean: [N, Q], cov: [N, Q]
            return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.sqrt(cov))
        else:
            raise NotImplementedError(
                "The combination of both `full_cov` and `full_output_cov` is not permitted."
            )

    def _convert_to_tensor_fn(self, distribution: tfp.distributions.Distribution) -> tf.Tensor:
        """
        Convert the predictive distributions at the input points (see
        :meth:`_make_distribution_fn`) to a tensor of :attr:`num_samples`
        samples from that distribution.
        Whether the samples are correlated or marginal (uncorrelated) depends
        on :attr:`full_cov` and :attr:`full_output_cov`.
        """
        # N input points
        # S = self.num_samples
        # Q = output dimensionality
        if self.num_samples is not None:
            samples = distribution.sample(
                (self.num_samples,)
            )  # [S, Q, N] if full_cov else [S, N, Q]
        else:
            samples = distribution.sample()  # [Q, N] if full_cov else [N, Q]

        if self.full_cov:
            samples = tf.linalg.adjoint(samples)  # [S, N, Q] or [N, Q]

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