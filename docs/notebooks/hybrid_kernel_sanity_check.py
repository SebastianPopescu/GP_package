import tensorflow as tf
from gp_package.kernels import Hybrid
from gp_package.utils.ops import wasserstein_2_distance, wasserstein_2_distance_simplified 
import numpy as np
import tensorflow_probability as tfp

def wasserstein_2_distance_gaussian_kernel(X1_mean, X2_mean, X1_var, X2_var):
      
    ### X1_mean -- (num_1, num_dim) ###
    ### X2_mean -- (num_2, num_dim) ###
    ### X1_var -- (num_1, num_dim) ###
    ### X2_var -- (num_2, num_dim) ###
    ### log_lengthscale -- shape (dim_input, )

    
    num_1 = tf.shape(X1_mean)[0]
    num_2 = tf.shape(X2_mean)[0]

    X1_mean = tf.expand_dims(X1_mean, axis=-1)
    X2_mean = tf.expand_dims(X2_mean, axis=-1)

    X1_var = tf.expand_dims(X1_var, axis=-1)
    X2_var = tf.expand_dims(X2_var, axis=-1)

    X1_mean = tf.tile(X1_mean, [1, 1, num_2])
    X1_var = tf.tile(X1_var, [1, 1, num_2])

    X2_mean = tf.tile(X2_mean, [1, 1, num_1])
    X2_var = tf.tile(X2_var, [1, 1, num_1])

    X2_mean = tf.transpose(X2_mean, perm=[2, 1, 0])
    X2_var = tf.transpose(X2_var, perm=[2, 1, 0])

    ### X1_mean -- (num_1, num_dim, num_2) ###
    ### X2_mean -- (num_1, num_dim, num_2) ###
    ### X1_var -- (num_1, num_dim, num_2) ###
    ### X2_var -- (num_1, num_dim, num_2) ###

    ### calculate the W-2-squared distance ###

    w2_distance_normed_part = tf.square(X1_mean - X2_mean)
    w2_distance_inside_trace = X1_var + X2_var - 2.0 * tf.sqrt(tf.multiply(X1_var, X2_var))
    w2_distance = w2_distance_normed_part + w2_distance_inside_trace

    return tf.reduce_sum(w2_distance, axis = 1)


if __name__ == "__main__":

    mu1_mean = np.random.randn(5,1)
    mu1_var = np.random.lognormal(size = (5,1))

    mu2_mean = np.random.randn(5,1)
    mu2_var = np.random.lognormal(size = (5,1))

    print(mu1_mean.shape)
    print(mu1_var.shape)

    old_w2 = wasserstein_2_distance_gaussian_kernel(mu1_mean, mu1_mean, mu1_var, mu1_var)
    print(old_w2)

    ## create tfp distributions for GPflow-esque kernels 

    mu1 = tfp.distributions.MultivariateNormalDiag(loc = mu1_mean, scale_diag = np.sqrt(mu1_var))
    mu2 = tfp.distributions.MultivariateNormalDiag(loc = mu2_mean, scale_diag = np.sqrt(mu2_var))

    new_w2 = tf.reduce_sum(wasserstein_2_distance(mu1,mu1), axis = -1, keepdims=False)
    print(new_w2)

    new_w2_v2 = tf.reduce_sum(wasserstein_2_distance_simplified(mu1,mu1), axis = -1, keepdims=False)
    print(new_w2_v2)
