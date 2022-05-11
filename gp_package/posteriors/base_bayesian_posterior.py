import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Type, Union

import tensorflow as tf

from ..base import MeanAndVariance, Module, Parameter, RegressionData, TensorType
from ..conditionals.utils_conditionals import *
from ..config import default_float, default_jitter
from ..covariances import BayesianKuf, BayesianKuu, BayesianKuus, BayesianKufs
from ..inducing_variables import (
    InducingPoints,
    InducingVariables,
    SharedIndependentInducingVariables,
)
from ..kernels import *
from ..mean_functions import MeanFunction

class AbstractPosterior(Module, ABC):
    def __init__(
        self,
        kernel: BayesianKernel,
        X_data: Union[tf.Tensor, InducingVariables],
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
        self, Xnew: TensorType, 
        U: TensorType,
        variance: TensorType,
        lengthscales: TensorType,
        full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Computes predictive mean and (co)variance at Xnew, including mean_function
        Does not make use of caching
        """
        mean, cov = self._conditional_fused(
            Xnew, U, variance, lengthscales,full_cov=full_cov, full_output_cov=full_output_cov
        )
        return self._add_mean_function(Xnew, mean), cov

    @abstractmethod
    def _conditional_fused(
        self, Xnew: TensorType, 
        U: TensorType,
        variance: TensorType,
        lengthscales: TensorType,
        full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Computes predictive mean and (co)variance at Xnew, *excluding* mean_function
        Does not make use of caching
        """


class BaseBayesianPosterior(AbstractPosterior):
    def __init__(
        self,
        kernel: BayesianKernel,
        inducing_variable: InducingVariables,
        whiten: bool = True,
        mean_function: Optional[MeanFunction] = None,
    ):

        super().__init__(kernel, inducing_variable, mean_function=mean_function)
        self.whiten = whiten


class IndependentBayesianPosterior(BaseBayesianPosterior):
    
    def _post_process_mean_and_cov(
        self, mean: TensorType, cov: TensorType, full_cov: bool, full_output_cov: bool
    ) -> MeanAndVariance:
        return mean, expand_independent_outputs(cov, full_cov, full_output_cov)

    def _get_Kff(self, Xnew: TensorType, variance: TensorType, full_cov: bool) -> tf.Tensor:

        # TODO: this assumes that Xnew has shape [N, D] and no leading dims

        if isinstance(self.kernel, SeparateIndependent):
            # NOTE calling kernel(Xnew, full_cov=full_cov, full_output_cov=False) directly would
            # return
            # if full_cov: [P, N, N] -- this is what we want
            # else: [N, P] instead of [P, N] as we get from the explicit stack below
            Kff = tf.stack([k(variance, Xnew, full_cov=full_cov) for k in self.kernel.kernels], axis=0)
        elif isinstance(self.kernel, MultioutputKernel):
            # effectively, SharedIndependent path
            Kff = self.kernel.kernel(variance, Xnew, full_cov=full_cov)
            # NOTE calling kernel(Xnew, full_cov=full_cov, full_output_cov=False) directly would
            # return
            # if full_cov: [P, N, N] instead of [N, N]
            # else: [N, P] instead of [N]
        else:
            # standard ("single-output") kernels
            Kff = self.kernel(variance, Xnew, full_cov=full_cov)  # [N, N] if full_cov else [N]

        return Kff



'''
class IndependentPosteriorSingleOutput(IndependentPosterior):
    
    # could almost be the same as IndependentPosteriorMultiOutput ...
    def _conditional_fused(
        self, Xnew: TensorType, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        # same as IndependentPosteriorMultiOutput, Shared~/Shared~ branch, except for following
        # line:
        Knn = self.kernel(Xnew, full_cov=full_cov)

        Kmm = Kuu(self.X_data, self.kernel, jitter=default_jitter())  # [M, M]
        Kmn = Kuf(self.X_data, self.kernel, Xnew)  # [M, N]

        fmean, fvar = base_conditional(
            Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
        )  # [N, P],  [P, N, N] or [N, P]
        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)
'''

class IndependentBayesianPosteriorMultiOutput(IndependentBayesianPosterior):

    def _conditional_fused(
        self, 
        Xnew: TensorType, 
        variance: TensorType, 
        lengthscales: TensorType,
        f: TensorType, 
        full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        
        if isinstance(self.X_data, SharedIndependentInducingVariables) and isinstance(
            self.kernel, SharedIndependent):
            # same as IndependentPosteriorSingleOutput except for following line
            Knn = self.kernel.kernel(variance, Xnew, full_cov=full_cov)
            # we don't call self.kernel() directly as that would do unnecessary tiling

            Kmm = BayesianKuus(self.X_data, self.kernel, variance, lengthscales, jitter=default_jitter())  # [M, M]
            Kmn = BayesianKufs(self.X_data, self.kernel, Xnew, variance, lengthscales)  # [M, N]

            fmean, fvar = base_bayesian_conditional(
                Kmn, Kmm, Knn, f, full_cov=full_cov, white=self.whiten
            )  # [N, P],  [P, N, N] or [N, P]
        else:
            raise NotImplementedError

        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)




