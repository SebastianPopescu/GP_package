# -*- coding: utf-8 -*-
from __future__ import print_function, division
#from subprocess import HIGH_PRIORITY_CLASS
import matplotlib as mpl


mpl.use('Agg')
import tensorflow as tf
import numpy as np
from collections import defaultdict
import random
import argparse
import matplotlib.pyplot as plt
import sys
import os
DTYPE=tf.float32
import seaborn as sns
from sklearn.cluster import  KMeans
from matplotlib import rcParams
import itertools
from scipy.stats import norm
import pandas as pd
import scipy
sys.setrecursionlimit(10000)

from time import perf_counter
from gpflow.base import TensorType

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tf.keras.backend.set_floatx("float64")


from sklearn.linear_model import LinearRegression
import gpflow
from gp_package.models import *
from gp_package.layers import *
from gp_package.kernels import *
from gp_package.inducing_variables import *
from gp_package.architectures import Config, build_dist_deep_gp
from typing import Callable, Tuple, Optional
from functools import wraps
from tensorflow_probability.python.util.deferred_tensor import TensorMetaClass

from .misc import LikelihoodOutputs, batch_predict
from .plotting_functions import get_regression_detailed_plot
from gpflow.ci_utils import ci_niter

def produce_regression_plots(model, num_epoch, start_point, end_point, dataset_name, file_name):

    cmd = 'mkdir -p ./docs/my_figures/'+dataset_name+'/'
    where_to_save = f'./docs/my_figures/'+dataset_name+'/'
    os.system(cmd)

    input_space = np.linspace(start_point, end_point, 500).reshape((-1,1))
    input_space = input_space.astype(np.float64)

    # Get predictive mean and variance (both parametric/non-parametric)at hidden layers

    f_mean_overall = defaultdict()
    f_var_overall = defaultdict()
    for current_layer in range(NUM_LAYERS):
        f_mean_overall[current_layer] = []
        f_var_overall[current_layer] = []

    for nvm in range(100):

        preds = model._evaluate_layer_wise_deep_gp(input_space)  

        for current_layer in range(NUM_LAYERS):
            current_preds = preds[current_layer]

            f_mean = current_preds[0]
            f_var = current_preds[1]
            
            f_mean_overall[current_layer].append(f_mean)
            f_var_overall[current_layer].append(f_var)

    for current_layer in range(NUM_LAYERS):

        f_mean_overall[current_layer] = tf.concat(f_mean_overall[current_layer], axis = 1)
        f_var_overall[current_layer] = tf.concat(f_var_overall[current_layer], axis = 1)

        f_mean_overall[current_layer] = tf.reduce_mean(f_mean_overall[current_layer], axis = 1)
        f_var_overall[current_layer] = tf.reduce_mean(f_var_overall[current_layer], axis = 1)

        f_mean_overall[current_layer] = f_mean_overall[current_layer].numpy()
        f_var_overall[current_layer] = f_var_overall[current_layer].numpy()

    
    get_regression_detailed_plot(
        num_layers = NUM_LAYERS,
        X_training = x_training,
        Y_training = y_training,
        where_to_save = where_to_save,
        mean = f_mean_overall,
        var = f_var_overall, 
        name_file =  file_name+f'_{num_epoch}.png',
        x_margin = X_MARGIN,
        y_margin = Y_MARGIN,
        X_test = input_space
        )


def optimization_step(model: DeepGP, batch: Tuple[tf.Tensor, tf.Tensor], optimizer):
    
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        loss = model.training_loss(batch)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def simple_training_loop(model: DistDeepGP, 
    num_batches_per_epoch: int,
    train_dataset,
    optimizer,
    epochs: int = 1, 
    logging_epoch_freq: int = 10, 
    plotting_epoch_freq: int = 10
    ):

    tf_optimization_step = tf.function(optimization_step)

    for epoch in range(epochs):
        
        batches = iter(train_dataset)
        
        for _ in range(ci_niter(num_batches_per_epoch)):
            
            tf_optimization_step(model, next(batches), optimizer)

        epoch_id = epoch + 1
        if epoch_id % logging_epoch_freq == 0:
            _elbo = model.elbo(data, True)
            tf.print(f"Epoch {epoch_id}: ELBO (train) {_elbo[0]}- Exp. ll. (train) {_elbo[1]}- KLs (train) {_elbo[2]}")
        
        if epoch_id % plotting_epoch_freq == 0:
            produce_regression_plots(model, epoch_id, x_training.min() - X_MARGIN, x_training.max() + X_MARGIN, 'motor', '_dist_dgp_')



if __name__ == '__main__':

    #####################################################
    ########### Get Motor data ##########################
    #####################################################


    def motorcycle_data():
        """ Return inputs and outputs for the motorcycle dataset. We normalise the outputs. """
        import pandas as pd
        df = pd.read_csv("/home/sebastian.popescu/Desktop/my_code/GP_package/docs/notebooks/data/motor.csv", index_col=0)
        X, Y = df["times"].values.reshape(-1, 1), df["accel"].values.reshape(-1, 1)
        Y = (Y - Y.mean()) / Y.std()
        X /= X.max()
        return X, Y

    X_data, Y_data = motorcycle_data()
    num_data, d_xim = X_data.shape


    np.random.seed(7)
    lista = np.arange(X_data.shape[0])
    np.random.shuffle(lista)
    cutoff = int(num_data * 0.8)
    index_training = lista[:cutoff]
    index_testing = lista[cutoff:]

    x_values_training_np = X_data[index_training,...]
    y_values_training_np = Y_data[index_training,...]
    
    print('----- size of training dataset -------')
    print(x_values_training_np.shape)
    print(y_values_training_np.shape)
    x_values_testing_np = X_data[index_testing,...]
    y_values_testing_np = Y_data[index_testing,...]
    
    print('------- size of testing dataset ---------')
    print(x_values_testing_np.shape)
    print(y_values_testing_np.shape)

    x_training = x_values_training_np.reshape((-1, d_xim)).astype(np.float64)
    x_testing = x_values_testing_np.reshape((-1, d_xim)).astype(np.float64)

    y_training = y_values_training_np.reshape((-1, 1)).astype(np.float64)
    y_testing = y_values_testing_np.reshape((-1, 1)).astype(np.float64)

    ###############################################################
    ########### Create model and train it #########################
    ###############################################################

    #train_dataset = tf.data.Dataset.from_tensor_slices((x_training, y_training)).shuffle(buffer_size=900 + 1).batch(32)

    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(x, y, s=2, label="data")
    xx = np.linspace(-1, 2, 101)[:, np.newaxis]
    # ax.plot(xx,  _f(xx), c='k')
    """

    NUM_INDUCING = 10
    HIDDEN_DIMS = 1
    NUM_LAYERS = 2
    X_MARGIN = 0.5
    Y_MARGIN = 0.1
    BATCH_SIZE = 32
    NUM_EPOCHS = 5000


    ### TRAIN MODEL ###
    config = Config(
        num_inducing=NUM_INDUCING, inner_layer_qsqrt_factor=1e-5, likelihood_noise_variance=1e-2, whiten=True, 
        hidden_layer_size=HIDDEN_DIMS, num_data = x_training.shape[0]
    )

    dist_deep_gp: DistDeepGP = build_dist_deep_gp(x_training, num_layers = NUM_LAYERS, config = config)

    data = (x_training, y_training)

    optimizer = tf.optimizers.Adam()
    #training_loss = deep_gp.training_loss_closure(
    #    data
    #    )  # We save the compiled closure in a variable so as not to re-compile it each step
    #optimizer.minimize(training_loss, deep_gp.trainable_variables)  # Note that this does a single step
    NUM_BATCHES_PER_EPOCH = int(x_training.shape[0] / BATCH_SIZE)

    if x_training.shape[0] % BATCH_SIZE !=0:
        NUM_BATCHES_PER_EPOCH+=1

    batched_dataset = tf.data.Dataset.from_tensor_slices(data).batch(BATCH_SIZE)

    simple_training_loop(model= dist_deep_gp, 
        num_batches_per_epoch = NUM_BATCHES_PER_EPOCH,
        train_dataset = batched_dataset,
        optimizer = optimizer,
        epochs = NUM_EPOCHS, 
        logging_epoch_freq = 10, 
        plotting_epoch_freq = 10
    )

    """

    batched_dataset = tf.data.Dataset.from_tensor_slices(data).batch(BATCH_SIZE)
    training_loss = deep_gp.training_loss_closure(iter(batched_dataset))

    optimizer.minimize(training_loss, deep_gp.trainable_variables)  # Note that this does a single step
    """


    """
    model = dist_deep_gp.as_training_model()
    model.compile(tf.optimizers.Adam(1e-2))

    for epoch_iterator in range(5):

        history = model.fit({"inputs": x_training, "targets": y_training}, epochs=int(10 * epoch_iterator), verbose=1)

        ########### Get results on testing set and produce plots #########################
        #model_testing = dist_deep_gp.as_prediction_model()

        produce_regression_plots(epoch_iterator, x_training.min() - X_MARGIN, x_training.max() + X_MARGIN)

    """