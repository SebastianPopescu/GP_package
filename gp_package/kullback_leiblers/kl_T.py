
# -*- coding: utf-8 -*-

from ast import Param
import tensorflow as tf
from packaging.version import Version

from gp_package.inverse_approximations.inverse_approximation import InverseApproximation
from gp_package.utils.ops import condition

from ..base import Parameter, TensorType
from ..config import default_float, default_jitter
from ..covariances import Kuu
from ..covariances.multioutput import Kuus
from ..inducing_variables import InducingVariables
from ..kernels import Kernel
from ..utils import to_default_float
from typing import Union

######################################################
###### KL-div between Wishart Distributions ##########
######################################################

def standard_kl_T(
    inverse_approximation : InverseApproximation, 
    df_p : TensorType, 
    inducing_variable : InducingVariables,
    kernel : Kernel, 
    g : tf.Tensor, 
    use_diagnostics : bool = False) -> tf.Tensor:

    """
    Compute the KL divergence KL[q || p] between

          q(x) = W(df_q, L_T L_T^{t})
    and
          p(x) = W(df_p, K_{uu}^{-1})

    TODO -- update the documentation for this function
    """
        
    q_Kuu_inv = tf.linalg.matmul(inverse_approximation.L_T, inverse_approximation.L_T, transpose_b = True)

    L_T_inv = inverse_approximation.get_cholesky_inverse()
    q_Kuu = tf.linalg.matmul(L_T_inv, L_T_inv, transpose_a = True)

    q_Kuu = condition(q_Kuu) 
    q_Kuu_tiled = q_Kuu[tf.newaxis, :]
    q_Kuu_tiled = tf.tile(q_Kuu_tiled, [g.get_shape().as_list()[0], 1, 1])
    
    q_Kuu_inv_tiled = q_Kuu_inv[tf.newaxis,:]
    q_Kuu_inv_tiled = tf.tile(q_Kuu_inv_tiled, [tf.shape(g)[0], 1, 1])

    Kuu = Kuus(inducing_variable.Z, kernel, jitter=default_jitter())  # [M, M]

    Kuu_operator = tf.linalg.LinearOperatorFullMatrix(
        matrix = condition(Kuu), is_non_singular=True, is_self_adjoint=True, is_positive_definite=True,
        is_square=True, name='LinearOperatorFullMatrixKuu')
    use_this = tf.stop_gradient(q_Kuu_inv)
    posterior_Kuu_inv_operator = tf.linalg.LinearOperatorFullMatrix(
        matrix = use_this, is_non_singular=True, is_self_adjoint=True, is_positive_definite=True,
        is_square=True, name='LinearOperatorFullMatrixKuuinv')

    output_cg = tf.linalg.experimental.conjugate_gradient(
        operator = Kuu_operator, rhs = g, preconditioner= posterior_Kuu_inv_operator, 
        tol=1e-05, max_iter=tf.shape(Kuu)[0],
        name='conjugate_gradient')

    conj_grad_solution = output_cg[1][:,tf.newaxis]
    print('_________________________________________')
    print('Conjugate Gradient solution')
    print(conj_grad_solution)

    conj_grad_solution = tf.linalg.matmul(tf.linalg.matmul(tf.expand_dims(g, axis=1), q_Kuu_tiled),conj_grad_solution)
    
    log_det_Kuu_lower_bound = - tf.reduce_mean(conj_grad_solution)
    log_det_Kuu_lower_bound +=  inducing_variable.num_inducing
    log_det_posterior_Kuu = - 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(inverse_approximation.L_T)))
    log_det_Kuu_lower_bound += log_det_posterior_Kuu
    log_det_Kuu_lower_bound = 0.5 * df_p * log_det_Kuu_lower_bound

    kl_term = - log_det_Kuu_lower_bound   
    kl_term+=  0.5 * inverse_approximation.dof * log_det_posterior_Kuu
 
    ### Stochastic trace term estimation ###
    trace_term_hutch_first_part = tf.linalg.matmul(g, condition(Kuu)) ### shape (num_hutch_samples, M)
    trace_term_hutch_second_part = tf.linalg.matmul(q_Kuu_inv, g, transpose_b = True) ### shape (M, num_hutch_samples)
    trace_term_hutch  = tf.multiply(trace_term_hutch_first_part, tf.transpose(trace_term_hutch_second_part)) ### shape (num_hutch_samples, M)
    trace_term_hutch  = tf.reduce_mean(tf.reduce_sum(trace_term_hutch, axis = 1))

    
    kl_term+= 0.5 * inverse_approximation.dof * ( trace_term_hutch - inducing_variable.num_inducing )

    return kl_term
