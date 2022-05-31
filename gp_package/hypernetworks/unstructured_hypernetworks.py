from typing import Callable, Optional, Tuple

import tensorflow as tf


def basic_unc_hypernetwork(noise_dim, output_dim, num_inducing, input_dim, dim_layer):

    """
    #TODO -- write documentation for this function
    """

    # Functional API implementation

    input_noise = tf.keras.layers.InputLayer(input_shape=(noise_dim,), dtype=tf.float64)
    x = tf.keras.layers.Dense(250, activation=tf.nn.relu, kernel_initializer="glorot_normal")(input_noise)
    x = tf.keras.layers.Dense(250, activation=tf.nn.relu, kernel_initializer="glorot_normal")(x)
    x = tf.keras.layers.Dense(250, activation=tf.nn.relu, kernel_initializer="glorot_normal")(x)
    x = tf.keras.layers.Dense(output_dim, activation=None, kernel_initializer="glorot_normal")(x)

    ### extract sampled_U ###
    sampled_U  = tf.slice(x, [0,0], [-1, num_inducing * dim_layer])

    ### extract sampled kernel variance ###
    sampled_kernel_variance  = tf.slice(x, [0,num_inducing], [-1, 1])
    sampled_kernel_variance = tf.math.softplus(
        sampled_kernel_variance
    )

    ### extract sampled lengthscales ###
    sampled_lengthscales  = tf.slice(x, [0,num_inducing+1], [-1, input_dim])
    sampled_lengthscales = tf.math.softplus(
        sampled_lengthscales
    )

    return sampled_U, sampled_kernel_variance, sampled_lengthscales


