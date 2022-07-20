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

from tensorflow_probability.python.layers.internal import distribution_tensor_coercible as dtc
tf.keras.backend.set_floatx("float64")

from gp_package.models import *
from gp_package.layers import *
from gp_package.kernels import *
from gp_package.inducing_variables import *
from gp_package.architectures import Config, build_deep_gp
from typing import Callable, Tuple, Optional
from functools import wraps
from tensorflow_probability.python.util.deferred_tensor import TensorMetaClass

from .misc import LikelihoodOutputs, batch_predict
from .plotting_functions import get_classification_detailed_plot
from gpflow.ci_utils import ci_niter

def produce_classification_plots(model, num_epoch, dataset_name, file_name):

    cmd = 'mkdir -p ./docs/my_figures/'+dataset_name+'/'
    where_to_save = './docs/my_figures/'+dataset_name+'/'
    os.system(cmd)

    #expanded_space = np.linspace(-4.0, 7.5, 500).reshape((-1,1))

    xx, yy = np.mgrid[-5:5:.1, -5:5:.1]
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid = grid.astype(np.float64)

    # Get predictive mean and variance (both parametric/non-parametric)at hidden layers

    f_mean_overall = defaultdict()
    f_var_overall = defaultdict()
    for current_layer in range(NUM_LAYERS):
        f_mean_overall[current_layer] = []
        f_var_overall[current_layer] = []

    for nvm in range(100):

        preds = model._evaluate_layer_wise_deep_gp(grid)  

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

    
    get_classification_detailed_plot(
        num_layers = NUM_LAYERS,
        X_training = x_training,
        Y_training = y_training,
        where_to_save = where_to_save,
        f_mean_overall = f_mean_overall,
        f_var_overall = f_var_overall, 
        name_file = file_name+f'_{num_epoch}.png'
        )



def optimization_step(model: DeepGP, batch: Tuple[tf.Tensor, tf.Tensor], optimizer):
    
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        loss = model.training_loss(batch)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def simple_training_loop(model: DeepGP, 
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
            produce_classification_plots(model, epoch_id,  'banana', '_dgp_')

if __name__ == '__main__':

    #####################################################
    ########### Get Banana data #########################
    #####################################################

    df = pd.read_csv('~/Desktop/my_code/Dist_DGPS/datasets/banana.csv', header=None)
    data = df.values
    X_data = data[:,:2]
    Y_data = data[:,-1].reshape((-1,1)) - 1.0
    
    plt.scatter(x =X_data[:,0], y = X_data[:,1], c = Y_data.ravel())
    plt.show()

   # plt.scatter(x,y)
    plt.savefig('/tmp/banana_dataset.png')
    plt.close()

    np.random.seed(7)
    lista = np.arange(X_data.shape[0])
    np.random.shuffle(lista)
    index_training = lista[:4000]
    index_testing = lista[4000:]

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

    x_training = x_values_training_np.reshape((-1,2)).astype(np.float64)
    x_testing = x_values_testing_np.reshape((-1,2)).astype(np.float64)

    y_training = y_values_training_np.reshape((-1,1)).astype(np.float64)
    y_testing = y_values_testing_np.reshape((-1,1)).astype(np.float64)


    ###############################################################
    ########### Create model and train it #########################
    ###############################################################

    NUM_INDUCING = 16
    HIDDEN_DIMS = 1
    NUM_LAYERS = 2
    BATCH_SIZE = 32
    NUM_EPOCHS = 5000

    ### TRAIN MODEL ###
    config = Config(
        num_inducing=NUM_INDUCING, inner_layer_qsqrt_factor=1e-5, likelihood_noise_variance=1e-2, whiten=True, 
        hidden_layer_size=HIDDEN_DIMS, num_data = x_training.shape[0], task_type = "classification", dim_output=1
    )

    deep_gp: DeepGP = build_deep_gp(x_training, num_layers = NUM_LAYERS, config = config)

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

    simple_training_loop(model= deep_gp, 
        num_batches_per_epoch = NUM_BATCHES_PER_EPOCH,
        train_dataset = batched_dataset,
        optimizer = optimizer,
        epochs = NUM_EPOCHS, 
        logging_epoch_freq = 10, 
        plotting_epoch_freq = 10
    )