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

# -*- coding: utf-8 -*-

import numpy as np
from typing import List
from sklearn.metrics import pairwise_kernels
import tensorflow as tf
from packaging.version import Version

from ..base import TensorType
from ..config import default_float, default_jitter
from ..covariances import Kuu
from ..inducing_variables import InducingVariables
from ..kernels import Kernel
from ..utils import to_default_float

import tensorflow_probability as tfp

def integral_gaussian_products( q_mu_1: TensorType, 
    q_cov_1: TensorType, 
    q_mu_2: TensorType, 
    q_cov_2: TensorType) -> tf.Tensor:

    """
    # TODO -- finish documenting this function
    param: q_mu_1  -- shape [M,D]
    param: q_cov_1 -- shape [D, M, M]
 
    param: q_mu_2  -- shape [M,D]
    param: q_cov_2 -- shape [D, M, M]
    """

    L_q_cov  = tf.linalg.cholesky(q_cov_1 + q_cov_2)

    q_mu = tf.transpose(q_mu)[:, :, None]   # [D, M, 1]
    m = tf.transpose(m)[:, :, None] # [D, M, 1]
    diff = q_mu - m # [D, M, 1]

    alpha = tf.linalg.triangular_solve(L_q_cov, diff, lower=True)  # [D, M, 1] 
    
    # Mahalanobis term: μqᵀ Σp⁻¹ μq
    mahalanobis = tf.reduce_sum(tf.square(alpha), axis = [1,2]) # [D, ]

    # Constant term: M
    constant = to_default_float(tf.shape(q_mu_1)[1])

    pi = to_default_float(np.pi)

    L_q_cov_diag = tf.linalg.diag_part(L_q_cov)  # [M, D]

    # Log-determinant of the covariance of q(x):
    logdet_qcov = tf.reduce_sum(tf.math.log(tf.square(L_q_cov_diag)), axis = 0) # [D, ]

    # TODO -- check if this is the correct way
    tf.debugging.assert_shapes([
        (q_mu_1, ('D', 'M', 'M')),
        (q_mu_2, ('D', 'M', 'M')),
        (L_q_cov, ('D', 'M', 'M'))
        (alpha, ('D', 'M', '1')),
        (mahalanobis, ('D', )),
        (L_q_cov_diag, ('M', 'D')),
        (logdet_qcov, ('D', ))        
    ])

    return tf.exp(-0.5 * mahalanobis) / tf.sqrt(constant * 2. * pi * logdet_qcov) # [D, ]





def multivariate_normal_entropy(q_mu: TensorType, 
    q_cov: TensorType) -> tf.Tensor:

    """
    # TODO -- finish documenting this function
    param: q_mu  -- shape [M, D]
    param: q_cov -- shape [D, M, M]
    """

    # Constant term: M
    constant = to_default_float(tf.shape(q_mu)[1])
    other_constant = to_default_float(tf.shape(q_mu)[0])

    euler_number = tf.math.exp(1)
    pi = to_default_float(np.pi)

    L_q_cov = tf.linalg.cholesky(q_cov) # [D, M, M]
    L_q_cov_diag = tf.linalg.diag_part(L_q_cov)  # [M, D]

    # Log-determinant of the covariance of q(x):
    logdet_qcov = tf.reduce_sum(tf.math.log(tf.square(L_q_cov_diag)), axis = 0) # [D, ]

    # TODO -- check if this is the correct way
    tf.debugging.assert_shapes([
        (q_mu, ('M', 'D')),
        (q_cov, ('D', 'M', 'M')),
        (L_q_cov, ('D', 'M', 'M'))
        (L_q_cov_diag, ('M', 'D')),
        (logdet_qcov, ('D', ))        
    ])

    # TODO -- what is the shape of the output
    return 0.5 * logdet_qcov + 0.5 * other_constant * constant * tf.math.log(2.0 * pi * euler_number) 


def upper_bound_kl_gmm(
    posterior_dist: List[tfp.distributions.MultivariateNormalFullCovariance],
    prior_dist: List[tfp.distributions.MultivariateNormalFullCovariance]
) -> tf.Tensor:

    samples_posterior = len(posterior_dist)
    samples_prior = len(prior_dist)


    q_mu_posterior = [dist.loc[None, :] for dist in posterior_dist]
    q_cov_posterior = [dist.covariance_matrix[None, :] for dist in posterior_dist]

    q_mu_posterior = tf.concat(q_mu_posterior, axis = 0) # -- shape [?,?]
    q_cov_posterior = tf.concat(q_cov_posterior, axis = 0) # -- shape [?,?]

    q_mu_prior = [dist.loc[None, :] for dist in prior_dist]
    q_cov_prior = [dist.covariance_matrix[None, :] for dist in prior_dist]

    q_mu_prior = tf.concat(q_mu_prior, axis = 0) # -- shape [?,?]
    q_cov_prior = tf.concat(q_cov_prior, axis = 0) # -- shape [?,?]

    # TODO -- need to introduce the rest of the upper bound 

    z_a_alpha = integral_gaussian_products( 
        q_mu_1 = tf.repeat(q_mu_posterior, repeats = samples_posterior, axis = -1), 
        q_cov_1 = tf.repeat(q_cov_posterior, repeats= samples_posterior, axis = 0),
        q_mu_2 = tf.tile(q_mu_posterior, [1, samples_posterior]), 
        q_cov_2 = tf.tile(q_cov_posterior, [samples_posterior, 1, 1])
    )

    z_a_alpha = tf.reshape(z_a_alpha, [samples_posterior, samples_posterior])

    pairwise_kl = gauss_kl(
        q_mu = tf.repeat(q_mu_posterior, repeats = samples_prior, axis = -1), 
        q_cov= tf.repeat(q_cov_posterior, repeats= samples_prior, axis = 0), 
        m = tf.tile(q_mu_prior, [1, samples_posterior]), 
        K = tf.tile(q_cov_prior, [samples_posterior, 1, 1])
    )

    upper_part = tf.reduce_sum( (1. / samples_posterior) * z_a_alpha, axis = -1)
    bottom_part = tf.reduce_sum( (1. / samples_prior)  * tf.exp(- pairwise_kl), axis = -1)
    fraction = tf.log(upper_part / bottom_part)

    return  tf.reduce_sum((1. / samples_posterior) * fraction) + tf.reduce_sum(multivariate_normal_entropy(q_mu_posterior, q_cov_posterior))
    

# TODO -- finish writing this function 
def gauss_kl(
    q_mu: TensorType, q_cov: TensorType, 
    m: TensorType, K: TensorType, *, K_cholesky: TensorType = None
    ) -> tf.Tensor:

    """
    #TODO -- finish documenting this function 
    param q_mu:                   -- shape (M, D)
    param q_cov:                  -- shape (D, M, M)
    param m:                      -- shape (M, D)
    param K:                      -- shape (D, M, M)

    Compute the KL divergence KL[q || p] between

          q(x) = N(q_mu, q_sqrt^2)
    and
          p(x) = N(m, K)   
          
    We assume L multiple independent distributions, given by the columns of
    q_mu and the first or last dimension of q_cov. Returns the *sum* of the
    divergences.

    q_mu is a matrix ([M, L]), each column contains a mean.

    q_sqrt can be a 3D tensor ([L, M, M]), each matrix within is a lower
        triangular square-root matrix of the covariance of q.
    q_sqrt can be a matrix ([M, L]), each column represents the diagonal of a
        square-root matrix of the covariance of q.

    K is the covariance of p (positive-definite matrix).  The K matrix can be
    passed either directly as `K`, or as its Cholesky factor, `K_cholesky`.  In
    either case, it can be a single matrix [M, M], in which case the sum of the
    L KL divergences is computed by broadcasting, or L different covariances
    [L, M, M].

    Note: if no K matrix is given (both `K` and `K_cholesky` are None),
    `gauss_kl` computes the KL divergence from p(x) = N(0, I) instead.
    """

    if (K is not None) and (K_cholesky is not None):
        raise ValueError(
            "Ambiguous arguments: gauss_kl() must only be passed one of `K` or `K_cholesky`."
        )

    is_diag = len(q_cov.shape) == 2
    if is_diag:
        q_sqrt = tf.sqrt(q_cov)
    else:
        q_sqrt = tf.linalg.cholesky(q_cov)

    shape_constraints = [
        (q_mu, ["M", "L"]),
        (q_sqrt, (["M", "L"] if is_diag else ["L", "M", "M"])),
    ]

    if K is not None:
        shape_constraints.append((K, (["L", "M", "M"] if len(K.shape) == 3 else ["M", "M"])))
    else:
        shape_constraints.append(
            (K_cholesky, (["L", "M", "M"] if len(K_cholesky.shape) == 3 else ["M", "M"]))
        )
    tf.debugging.assert_shapes(shape_constraints, message="gauss_kl() arguments")

    M, L = tf.shape(q_mu)[0], tf.shape(q_mu)[1]

    if K is not None:
        Lp = tf.linalg.cholesky(K)  # [L, M, M] or [M, M]
    elif K_cholesky is not None:
        Lp = K_cholesky  # [L, M, M] or [M, M]

    is_batched = len(Lp.shape) == 3

    q_mu = tf.transpose(q_mu)[:, :, None] if is_batched else q_mu  # [L, M, 1] or [M, L]
    m = tf.transpose(m)[:, :, None] if is_batched else m  # [L, M, 1] or [M, L]
    diff = q_mu - m
    alpha = tf.linalg.triangular_solve(Lp, diff, lower=True)  # [L, M, 1] or [M, L]

    if is_diag:
        Lq = Lq_diag = q_sqrt
        Lq_full = tf.linalg.diag(tf.transpose(q_sqrt))  # [L, M, M]
    else:
        Lq = Lq_full = tf.linalg.band_part(q_sqrt, -1, 0)  # force lower triangle # [L, M, M]
        Lq_diag = tf.linalg.diag_part(Lq)  # [M, L]

    # Mahalanobis term: μqᵀ Σp⁻¹ μq
    mahalanobis = tf.reduce_sum(tf.square(alpha))

    # Constant term: - L * M
    constant = -to_default_float(tf.size(q_mu, out_type=tf.int64))

    # Log-determinant of the covariance of q(x):
    logdet_qcov = tf.reduce_sum(tf.math.log(tf.square(Lq_diag)))

    # Trace term: tr(Σp⁻¹ Σq)

    if is_diag and not is_batched:
        # K is [M, M] and q_sqrt is [M, L]: fast specialisation
        LpT = tf.transpose(Lp)  # [M, M]
        Lp_inv = tf.linalg.triangular_solve(
            Lp, tf.eye(M, dtype=default_float()), lower=True
        )  # [M, M]
        K_inv = tf.linalg.diag_part(tf.linalg.triangular_solve(LpT, Lp_inv, lower=False))[
            :, None
        ]  # [M, M] -> [M, 1]
        trace = tf.reduce_sum(K_inv * tf.square(q_sqrt))
    else:
        if is_batched or Version(tf.__version__) >= Version("2.2"):
            Lp_full = Lp
        else:
            # workaround for segfaults when broadcasting in TensorFlow<2.2
            Lp_full = tf.tile(tf.expand_dims(Lp, 0), [L, 1, 1])
        LpiLq = tf.linalg.triangular_solve(Lp_full, Lq_full, lower=True)
        trace = tf.reduce_sum(tf.square(LpiLq))

    twoKL = mahalanobis + constant - logdet_qcov + trace

    # Log-determinant of the covariance of p(x):

    log_sqdiag_Lp = tf.math.log(tf.square(tf.linalg.diag_part(Lp)))
    sum_log_sqdiag_Lp = tf.reduce_sum(log_sqdiag_Lp)
    # If K is [L, M, M], num_latent_gps is no longer implicit, no need to multiply the single kernel logdet
    scale = 1.0 if is_batched else to_default_float(L)
    twoKL += scale * sum_log_sqdiag_Lp

    tf.debugging.assert_shapes([(twoKL, ())], message="gauss_kl() return value")  # returns scalar
    return 0.5 * twoKL
