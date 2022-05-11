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

"""
Likelihoods are another core component of GPflow. This describes how likely the
data is under the assumptions made about the underlying latent functions
p(Y|F). Different likelihoods make different
assumptions about the distribution of the data, as such different data-types
(continuous, binary, ordinal, count) are better modelled with different
likelihood assumptions.

Use of any likelihood other than Gaussian typically introduces the need to use
an approximation to perform inference, if one isn't already needed. A
variational inference and MCMC models are included in GPflow and allow
approximate inference with non-Gaussian likelihoods. An introduction to these
models can be found :ref:`here <implemented_models>`. Specific notebooks
illustrating non-Gaussian likelihood regressions are available for
`classification <notebooks/classification.html>`_ (binary data), `ordinal
<notebooks/ordinal.html>`_ and `multiclass <notebooks/multiclass.html>`_.

Creating new likelihoods
----------
Likelihoods are defined by their
log-likelihood. When creating new likelihoods, the
:func:`logp <gpflow.likelihoods.Likelihood.logp>` method (log p(Y|F)), the
:func:`conditional_mean <gpflow.likelihoods.Likelihood.conditional_mean>`,
:func:`conditional_variance
<gpflow.likelihoods.Likelihood.conditional_variance>`.

In order to perform variational inference with non-Gaussian likelihoods a term
called ``variational expectations``, ∫ q(F) log p(Y|F) dF, needs to
be computed under a Gaussian distribution q(F) ~ N(μ, Σ).

The :func:`variational_expectations <gpflow.likelihoods.Likelihood.variational_expectations>`
method can be overriden if this can be computed in closed form, otherwise; if
the new likelihood inherits
:class:`Likelihood <gpflow.likelihoods.Likelihood>` the default will use
Gauss-Hermite numerical integration (works well when F is 1D
or 2D), if the new likelihood inherits from
:class:`MonteCarloLikelihood <gpflow.likelihoods.MonteCarloLikelihood>` the
integration is done by sampling (can be more suitable when F is higher dimensional).
"""

import abc
import warnings
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import tensorflow as tf
import numpy as np

from .logdensities import gaussian, student_t
from ..base import MeanAndVariance, Parameter, TensorType
from ..utils import positive
from .base_likelihood_layers import ScalarLikelihood



class Gaussian(ScalarLikelihood):
    r"""
    The Gaussian likelihood is appropriate where uncertainties associated with
    the data are believed to follow a normal distribution, with constant
    variance.

    Very small uncertainties can lead to numerical instability during the
    optimization process. A lower bound of 1e-6 is therefore imposed on the
    likelihood variance by default.
    """

    DEFAULT_VARIANCE_LOWER_BOUND = 1e-6

    def __init__(
        self,
        variance: float = 1.0,
        variance_lower_bound: float = DEFAULT_VARIANCE_LOWER_BOUND,
        **kwargs: Any,
    ) -> None:
        """
        :param variance: The noise variance; must be greater than
            ``variance_lower_bound``.
        :param variance_lower_bound: The lower (exclusive) bound of ``variance``.
        :param kwargs: Keyword arguments forwarded to :class:`ScalarLikelihood`.
        """
        super().__init__(**kwargs)

        if variance <= variance_lower_bound:
            raise ValueError(
                f"The variance of the Gaussian likelihood must be strictly greater than {variance_lower_bound}"
            )

        self.variance = Parameter(variance, transform=positive(lower=variance_lower_bound))

    def _scalar_log_prob(self, F: TensorType, Y: TensorType) -> tf.Tensor:
        return gaussian(Y, F, self.variance)

    def _conditional_mean(self, F: TensorType) -> tf.Tensor:  # pylint: disable=R0201
        return tf.identity(F)

    def _conditional_variance(self, F: TensorType) -> tf.Tensor:
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    def _predict_mean_and_var(self, Fmu: TensorType, Fvar: TensorType) -> MeanAndVariance:
        return tf.identity(Fmu), Fvar + self.variance

    def _predict_log_density(self, Fmu: TensorType, Fvar: TensorType, Y: TensorType) -> tf.Tensor:
        return tf.reduce_sum(gaussian(Y, Fmu, Fvar + self.variance), axis=-1)

    def _variational_expectations(
        self, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        return tf.reduce_sum(
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(self.variance)
            - 0.5 * ((Y - Fmu) ** 2 + Fvar) / self.variance,
            axis=-1,
        )





class StudentT(ScalarLikelihood):
    def __init__(self, scale: float = 1.0, df: float = 3.0, **kwargs: Any) -> None:
        """
        :param scale float: scale parameter
        :param df float: degrees of freedom
        """
        super().__init__(**kwargs)
        self.df = df
        self.scale = Parameter(scale, transform=positive())

    def _scalar_log_prob(self, F: TensorType, Y: TensorType) -> tf.Tensor:
        return student_t(Y, F, self.scale, self.df)

    def _conditional_mean(self, F: TensorType) -> tf.Tensor:
        return F

    def _conditional_variance(self, F: TensorType) -> tf.Tensor:
        var = (self.scale ** 2) * (self.df / (self.df - 2.0))
        return tf.fill(tf.shape(F), tf.squeeze(var))







