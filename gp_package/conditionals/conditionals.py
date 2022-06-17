from typing import Optional, Union

import tensorflow as tf
import tensorflow_probability as tfp

from ..inducing_variables import InducingVariables
from ..kernels import Kernel
from ..base import MeanAndVariance
from ..posteriors import IndependentPosteriorSingleOutput, IndependentPosteriorMultiOutput, IndependentOrthogonalPosteriorSingleOutput, IndependentOrthogonalPosteriorMultiOutput, IndependentHeteroskedasticOrthogonalPosteriorMultiOutput

def conditional_GP(
    Xnew: tf.Tensor,
    inducing_variable: InducingVariables,
    kernel: Kernel,
    f: tf.Tensor,
    *,
    full_cov: bool = False,
    full_output_cov: bool = False,
    q_sqrt: Optional[tf.Tensor] = None,
    white: bool = False,
) -> MeanAndVariance:
    """
    Single-output GP conditional.

    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [M, M]
    - Kuf: [M, N]
    - Kff: [N, N]

    Further reference
    -----------------
    - See `gpflow.conditionals._dense_conditional` (below) for a detailed explanation of
      conditional in the single-output case.
    - See the multiouput notebook for more information about the multiouput framework.

    Parameters
    ----------
    :param Xnew: data matrix, size [N, D].
    :param f: data matrix, [M, R]
    :param full_cov: return the covariance between the datapoints
    :param full_output_cov: return the covariance between the outputs.
           NOTE: as we are using a single-output kernel with repetitions
                 these covariances will be zero.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size [M, R] or [R, M, M].
    :param white: boolean of whether to use the whitened representation
    :return:
        - mean:     [N, R]
        - variance: [N, R], [R, N, N], [N, R, R] or [N, R, N, R]
        Please see `gpflow.conditional._expand_independent_outputs` for more information
        about the shape of the variance, depending on `full_cov` and `full_output_cov`.
    """
        
    posterior =  IndependentPosteriorMultiOutput(
        kernel,
        inducing_variable,
        f,
        q_sqrt,
        whiten=white,
        mean_function=None,
    )

    return posterior.fused_predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)

def conditional_orthogonal_GP(
    Xnew: tf.Tensor,
    inducing_variable_u: InducingVariables,
    inducing_variable_v: InducingVariables,
    kernel: Kernel,
    f_u: tf.Tensor,
    f_v: tf.Tensor,
    *,
    full_cov: bool = False,
    full_output_cov: bool = False,
    q_sqrt_u: Optional[tf.Tensor] = None,
    q_sqrt_v: Optional[tf.Tensor] = None,
    white: bool = False,
) -> MeanAndVariance:
    """
    Single-output orthogonal GP conditional.

    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [M_u, M_u]
    - Kuf: [M_u, N]
    - Kff: [N, N]

    - Cvv: [M_v, M_v]
    - Cvf: [M_v, N]
    - Cff: [N, N]


    Further reference
    -----------------
    - See `gpflow.conditionals._dense_conditional` (below) for a detailed explanation of
      conditional in the single-output case.
    - See the multiouput notebook for more information about the multiouput framework.

    Parameters
    ----------
    :param Xnew: data matrix, size [N, D].
    :param f: data matrix, [M, R]
    :param full_cov: return the covariance between the datapoints
    :param full_output_cov: return the covariance between the outputs.
           NOTE: as we are using a single-output kernel with repetitions
                 these covariances will be zero.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size [M, R] or [R, M, M].
    :param white: boolean of whether to use the whitened representation
    :return:
        - mean:     [N, R]
        - variance: [N, R], [R, N, N], [N, R, R] or [N, R, N, R]
        Please see `gpflow.conditional._expand_independent_outputs` for more information
        about the shape of the variance, depending on `full_cov` and `full_output_cov`.
    """
    
    posterior =  IndependentOrthogonalPosteriorMultiOutput(
        kernel,
        inducing_variable_u,
        inducing_variable_v,
        f_u,
        q_sqrt_u,
        f_v,
        q_sqrt_v,
        whiten=white,
        mean_function=None,
    )

    return posterior.fused_predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)


def conditional_heteroskedastic_orthogonal_GP(
    Xnew: tf.Tensor,
    inducing_variable_u: InducingVariables,
    inducing_variable_v: InducingVariables,
    kernel: Kernel,
    f_u: tf.Tensor,
    f_v: tf.Tensor,
    *,
    full_cov: bool = False,
    full_output_cov: bool = False,
    q_sqrt_u: Optional[tf.Tensor] = None,
    q_sqrt_v: Optional[tf.Tensor] = None,
    white: bool = False,
) -> MeanAndVariance:
    """
    Single-output orthogonal GP conditional.

    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [M_u, M_u]
    - Kuf: [M_u, N]
    - Kff: [N, N]

    - Cvv: [M_v, M_v]
    - Cvf: [M_v, N]
    - Cff: [N, N]


    Further reference
    -----------------
    - See `gpflow.conditionals._dense_conditional` (below) for a detailed explanation of
      conditional in the single-output case.
    - See the multiouput notebook for more information about the multiouput framework.

    Parameters
    ----------
    :param Xnew: data matrix, size [N, D].
    :param f: data matrix, [M, R]
    :param full_cov: return the covariance between the datapoints
    :param full_output_cov: return the covariance between the outputs.
           NOTE: as we are using a single-output kernel with repetitions
                 these covariances will be zero.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size [M, R] or [R, M, M].
    :param white: boolean of whether to use the whitened representation
    :return:
        - mean:     [N, R]
        - variance: [N, R], [R, N, N], [N, R, R] or [N, R, N, R]
        Please see `gpflow.conditional._expand_independent_outputs` for more information
        about the shape of the variance, depending on `full_cov` and `full_output_cov`.
    """
    
    posterior =  IndependentHeteroskedasticOrthogonalPosteriorMultiOutput(
        kernel,
        inducing_variable_u,
        inducing_variable_v,
        f_u,
        q_sqrt_u,
        f_v,
        q_sqrt_v,
        whiten=white,
        mean_function=None,
    )

    return posterior.fused_predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)

