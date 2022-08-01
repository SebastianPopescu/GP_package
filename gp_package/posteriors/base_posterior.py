import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Type, Union

import tensorflow as tf
import tensorflow_probability as tfp

from .. import covariances
from gpflow import kernels
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
from ..covariances import Kuf, Kuu #NOTE -- getting the dispatches essentially
from gpflow.inducing_variables import (
    FallbackSeparateIndependentInducingVariables,
    FallbackSharedIndependentInducingVariables,
    InducingPoints,
    InducingVariables,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from ..inducing_variables import (
    FourierFeatures1D
)


from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.utilities import Dispatcher, add_noise_cov
from gpflow.utilities.ops import eye, leading_transpose


class AbstractPosterior(Module, ABC):
    def __init__(
        self,
        kernel: Kernel,
        X_data: Union[tf.Tensor, InducingVariables, FourierFeatures1D],
        cache: Optional[Tuple[tf.Tensor, ...]] = None,
        mean_function: Optional[MeanFunction] = None,
    ) -> None:
        """
        Users should use `create_posterior` to create instances of concrete
        subclasses of this AbstractPosterior class instead of calling this
        constructor directly. For `create_posterior` to be able to correctly
        instantiate subclasses, developers need to ensure their subclasses
        don't change the constructor signature.
        """
        super().__init__()

        self.kernel = kernel
        self.X_data = X_data
        self.cache = cache
        self.mean_function = mean_function

    def _add_mean_function(self, Xnew: TensorType, mean: TensorType) -> tf.Tensor:
        if self.mean_function is None:
            return mean
        else:
            return mean + self.mean_function(Xnew)

    def fused_predict_f(
        self, Xnew: TensorType, *, full_cov: bool = False, full_output_cov: bool = False,
        detailed_moments: bool = False
    ) -> MeanAndVariance:
        """
        Computes predictive mean and (co)variance at Xnew, including mean_function
        Does not make use of caching
        """
        mean, cov = self._conditional_fused(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov, detailed_moments = detailed_moments
        )
        return self._add_mean_function(Xnew, mean), cov


    @abstractmethod
    def _conditional_fused(
        self, Xnew: TensorType, *, full_cov: bool = False, full_output_cov: bool = False, detailed_moments: bool = False
    ) -> MeanAndVariance:
        """
        Computes predictive mean and (co)variance at Xnew, *excluding* mean_function
        Does not make use of caching
        """

class BasePosterior(AbstractPosterior):
    def __init__(
        self,
        kernel: Kernel,
        inducing_variable: Union[InducingVariables, FourierFeatures1D],
        q_mu: tf.Tensor,
        q_sqrt: tf.Tensor,
        whiten: bool = True,
        mean_function: Optional[MeanFunction] = None,
    ):

        super().__init__(kernel, inducing_variable, mean_function=mean_function)
        self.whiten = whiten
        self.q_mu = q_mu
        self.q_sqrt = q_sqrt
        #self._set_qdist(q_mu, q_sqrt)

    """
    @property
    def q_mu(self) -> tf.Tensor:
        return self._q_dist.q_mu

    @property
    def q_sqrt(self) -> tf.Tensor:
        return self._q_dist.q_sqrt


    def _set_qdist(self, q_mu: TensorType, q_sqrt: TensorType) -> tf.Tensor:
        if q_sqrt is None:
            self._q_dist = _DeltaDist(q_mu)
        elif len(q_sqrt.shape) == 2:  # q_diag
            self._q_dist = _DiagNormal(q_mu, q_sqrt)
        else:
            self._q_dist = _MvNormal(q_mu, q_sqrt)
    """

class IndependentPosterior(BasePosterior):
    
    def _post_process_mean_and_cov(
        self, mean: TensorType, cov: TensorType, full_cov: bool, full_output_cov: bool
    ) -> MeanAndVariance:
        return mean, expand_independent_outputs(cov, full_cov, full_output_cov)

    def _get_Kff(self, Xnew: Union[TensorType,tfp.distributions.MultivariateNormalDiag], full_cov: bool) -> tf.Tensor:

        # TODO: this assumes that Xnew has shape [N, D] and no leading dims

        if isinstance(self.kernel, kernels.SeparateIndependent):
            # NOTE calling kernel(Xnew, full_cov=full_cov, full_output_cov=False) directly would
            # return
            # if full_cov: [P, N, N] -- this is what we want
            # else: [N, P] instead of [P, N] as we get from the explicit stack below
            Kff = tf.stack([k(Xnew, full_cov=full_cov) for k in self.kernel.kernels], axis=0)
        elif isinstance(self.kernel, kernels.MultioutputKernel):
            # effectively, SharedIndependent path
            Kff = self.kernel.kernel(Xnew, full_cov=full_cov)
            # NOTE calling kernel(Xnew, full_cov=full_cov, full_output_cov=False) directly would
            # return
            # if full_cov: [P, N, N] instead of [N, N]
            # else: [N, P] instead of [N]
        else:
            #NOTE -- this is what's gonna be called
            # standard ("single-output") kernels
            Kff = self.kernel(Xnew, full_cov=full_cov)  # [N, N] if full_cov else [N]

        return Kff



