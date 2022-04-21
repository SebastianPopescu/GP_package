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

import tensorflow as tf
from packaging.version import Version

from ..base import TensorType
from ..config import default_float, default_jitter
from ..covariances import Kuu
from ..inducing_variables import InducingVariables
from ..kernels import Kernel
from ..utils import to_default_float

from typing import Optional


def standard_kl(
    inducing_variable: InducingVariables,
    kernel: Kernel,
    q_mu: TensorType,
    q_sqrt: TensorType,
    Sigma_mm_inverse: Optional[TensorType] = None,
    L_Sigma_mm_inverse: Optional[TensorType] = None,
    whiten: bool = False) -> tf.Tensor:

    """
    TODO -- document function
    """

    if whiten:
        return gauss_kl_inverse_free(q_mu, q_sqrt, None, None)
    else:
        #K = Kuu(inducing_variable, kernel, jitter=default_jitter())  # [P, M, M] or [M, M]
        return gauss_kl_inverse_free(q_mu, q_sqrt, Sigma_mm_inverse, L_Sigma_mm_inverse)

def gauss_kl_inverse_free(
    q_mu: TensorType, 
    q_sqrt: TensorType, 
    Sigma_mm_inverse: TensorType,
    L_Sigma_mm_inverse: TensorType) -> tf.Tensor:
    """
    Compute the inverse-free KL divergence KL[q || p] between

          q(x) = N(q_mu, q_sqrt^2)
    and
          p(x) = N(0, K)    if Sigma_mm_inverse is not None
          p(x) = N(0, I)    if Sigma_mm_inverse is None

    We assume L multiple independent distributions, given by the columns of
    q_mu and the first or last dimension of q_sqrt. Returns the *sum* of the
    divergences.

    q_mu is a matrix ([M, L]), each column contains a mean.

    q_sqrt can be a 3D tensor ([L, M, M]), each matrix within is a lower
        triangular square-root matrix of the covariance of q.
    q_sqrt can be a matrix ([M, L]), each column represents the diagonal of a
        square-root matrix of the covariance of q.

    TODO -- finish documenting this function

    Note: if no K matrix is given (both `K` and `K_cholesky` are None),
    `gauss_kl` computes the KL divergence from p(x) = N(0, I) instead.
    """

    is_white = (Sigma_mm_inverse is None) and (L_Sigma_mm_inverse is None)
    is_diag = len(q_sqrt.shape) == 2

    shape_constraints = [
        (q_mu, ["M", "L"]),
        (q_sqrt, (["M", "L"] if is_diag else ["L", "M", "M"])),
    ]
    if not is_white:
        shape_constraints.append((Sigma_mm_inverse, (["L", "M", "M"] if len(K.shape) == 3 else ["M", "M"])))
        
        shape_constraints.append(
                (L_Sigma_mm_inverse, (["L", "M", "M"] if len(L_Sigma_mm_inverse.shape) == 3 else ["M", "M"])))

    tf.debugging.assert_shapes(shape_constraints, message="gauss_kl() arguments")

    M, L = tf.shape(q_mu)[0], tf.shape(q_mu)[1]

    if is_white:
        alpha = q_mu  # [M, L]
    else:

        is_batched = len(Sigma_mm_inverse.shape) == 3

        q_mu = tf.transpose(q_mu)[:, :, None] if is_batched else q_mu  # [L, M, 1] or [M, L]
        alpha = tf.linalg.matmul(L_Sigma_mm_inverse, q_mu, lower=True)  # [L, M, 1] or [M, L]

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
    if is_white:
        trace = tf.reduce_sum(tf.square(Lq))
    else:
        if is_diag and not is_batched:
            # Sigma_mm_inverse is [M, M] and q_sqrt is [M, L]: fast specialisation
            Sigma_mm_inverse_diag = tf.linalg.diag_part(Sigma_mm_inverse)[
                :, None
            ]  # [M, M] -> [M, 1]
            trace = tf.reduce_sum(Sigma_mm_inverse_diag * tf.square(q_sqrt))
        else:
            LpiLq = tf.linalg.matmul(L_Sigma_mm_inverse, Lq_full)
            trace = tf.reduce_sum(tf.square(LpiLq))

    twoKL = mahalanobis + constant - logdet_qcov + trace

    # Log-determinant of the covariance of p(x):
    if not is_white:
        log_sqdiag_Lp = tf.math.log(tf.square(tf.linalg.diag_part(L_Sigma_mm_inverse)))
        sum_log_sqdiag_Lp = tf.reduce_sum(log_sqdiag_Lp)
        # If K is [L, M, M], num_latent_gps is no longer implicit, no need to multiply the single kernel logdet
        scale = 1.0 if is_batched else to_default_float(L)
        twoKL += scale * sum_log_sqdiag_Lp

    tf.debugging.assert_shapes([(twoKL, ())], message="gauss_kl() return value")  # returns scalar
    return 0.5 * twoKL
