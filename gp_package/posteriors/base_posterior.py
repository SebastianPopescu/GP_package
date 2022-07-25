import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Type, Union

import tensorflow as tf
import tensorflow_probability as tfp

from gpflow import kernels
from gpflow.mean_functions import MeanFunction
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

from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.utilities import Dispatcher, add_noise_cov
from gpflow.utilities.ops import eye, leading_transpose


class AbstractOrthogonalPosterior(Module, ABC):
    def __init__(
        self,
        kernel: Kernel,
        X_data: Union[tf.Tensor, InducingVariables],
        V_data: Union[tf.Tensor, InducingVariables],
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
        self.inducing_variable_u = X_data
        self.inducing_variable_v = V_data
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
        mean, cov = self._conditional_fused_OrthogonalSVGP(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov, detailed_moments = detailed_moments
        )
        return self._add_mean_function(Xnew, mean), cov


    @abstractmethod
    def _conditional_fused_OrthogonalSVGP(
        self, Xnew: TensorType, *, full_cov: bool = False, full_output_cov: bool = False, detailed_moments: bool = False
    ) -> MeanAndVariance:
        """
        someting
        """
    

class BaseOrthogonalPosterior(AbstractOrthogonalPosterior):
    def __init__(
        self,
        kernel: Kernel,
        inducing_variable_u: InducingVariables,
        inducing_variable_v: InducingVariables,
        q_mu_u: tf.Tensor,
        q_mu_v: tf.Tensor,
        q_sqrt_u: tf.Tensor,
        q_sqrt_v: tf.Tensor,
        whiten: bool = True,
        mean_function: Optional[MeanFunction] = None,
    ):

        super().__init__(kernel, inducing_variable_u, inducing_variable_v, mean_function=mean_function)
        self.whiten = whiten
        self.q_mu_u = q_mu_u
        self.q_sqrt_u = q_sqrt_u
        self.q_mu_v = q_mu_v
        self.q_sqrt_v = q_sqrt_v        
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

class IndependentOrthogonalPosterior(BaseOrthogonalPosterior):
    
    def _post_process_mean_and_cov(
        self, mean: TensorType, cov: TensorType, full_cov: bool, full_output_cov: bool
    ) -> MeanAndVariance:
        return mean, expand_independent_outputs(cov, full_cov, full_output_cov)

    def _get_Kff(self, Xnew: TensorType, full_cov: bool) -> tf.Tensor:

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
            # standard ("single-output") kernels
            Kff = self.kernel(Xnew, full_cov=full_cov)  # [N, N] if full_cov else [N]

        return Kff


    def _get_Cff(self, Xnew: TensorType, full_cov: bool) -> tf.Tensor:

        # TODO: this assumes that Xnew has shape [N, D] and no leading dims

        if isinstance(self.kernel, kernels.SeparateIndependent):
            
            # TODO -- need to finish this at one point
            # NOTE calling kernel(Xnew, full_cov=full_cov, full_output_cov=False) directly would
            # return
            # if full_cov: [P, N, N] -- this is what we want
            # else: [N, P] instead of [P, N] as we get from the explicit stack below
            #Kff = tf.stack([k(Xnew, full_cov=full_cov) for k in self.kernel.kernels], axis=0)

            raise NotImplementedError

        elif isinstance(self.kernel, kernels.MultioutputKernel):
            # effectively, SharedIndependent path
            Kff = self.kernel.kernel(Xnew, full_cov=full_cov)
            # NOTE calling kernel(Xnew, full_cov=full_cov, full_output_cov=False) directly would
            # return
            # if full_cov: [P, N, N] instead of [N, N]
            # else: [N, P] instead of [N]

            Kuu = self.kernel.kernel(self.inducing_variable_u.inducing_variable.Z, full_cov=True)
            jittermat = tf.eye(self.inducing_variable_u.inducing_variable.num_inducing, dtype=Kuu.dtype) * default_jitter()
            Kuu+= jittermat
            L_Kuu = tf.linalg.cholesky(Kuu) #NOTE -- I think this is the first time we compute the Choleksy decomposition


            #Kvv = self.kernel.kernel(self.inducing_variable_v.Z, full_cov=full_cov)
            Kuf = self.kernel.kernel(self.inducing_variable_u.inducing_variable.Z, Xnew)

            L_Kuu_inv_Kuf = tf.linalg.triangular_solve(L_Kuu, Kuf)

            # compute the covariance due to the conditioning
            if full_cov:
                Cff = Kff - tf.linalg.matmul(L_Kuu_inv_Kuf, L_Kuu_inv_Kuf, transpose_a=True)  # [..., N, N]
                #num_func = tf.shape(self.q_mu_u)[-1]
                #N = tf.shape(Kuf)[-1]
                #cov_shape = [num_func, N, N]
                #Cff = tf.broadcast_to(tf.expand_dims(Cff, -3), cov_shape)  # [..., R, N, N]
            else:
                Cff = Kff - tf.reduce_sum(tf.square(L_Kuu_inv_Kuf), -2)  # [..., N]
                #num_func = tf.shape(self.q_mu_u)[-1]
                #N = tf.shape(Kuf)[-1]
                #cov_shape = [num_func, N]  # [..., R, N]
                #Cff = tf.broadcast_to(tf.expand_dims(Cff, -2), cov_shape)  # [..., R, N]

        else:
            # standard ("single-output") kernels
            Kuu = self.kernel(self.inducing_variable_u.Z, full_cov=True)
            L_Kuu = tf.linalg.cholesky(Kuu) #NOTE -- I think this is the first time we compute the Choleksy decomposition

            #Kvv = self.kernel.kernel(self.inducing_variable_v.Z, full_cov=full_cov)
            Kuf = self.kernel(self.inducing_variable_u.Z, Xnew)

            L_Kuu_inv_Kuf = tf.linalg.triangular_solve(L_Kuu, Kuf)
            
            if full_cov:
                Cff = Kff - tf.linalg.matmul(
                    L_Kuu_inv_Kuf, L_Kuu_inv_Kuf, transpose_a=True)
            else:
                Cff = Kff - tf.reduce_sum(tf.square(L_Kuu_inv_Kuf), -2)  # [..., N]
                #NOTE -- I don't think this is necessary
                #Cff = Cff[:,tf.newaxis] # [..., N, 1]

        return Cff, L_Kuu




