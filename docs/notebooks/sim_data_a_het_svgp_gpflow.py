from random import random, uniform
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow_probability as tfp
from sklearn.neighbors import KernelDensity

from gp_package.base import TensorType
tf.keras.backend.set_floatx("float64")

from dataclasses import dataclass

from gp_package.models import *
from gp_package.layers import *
from gp_package.kernels import *
from gp_package.inducing_variables import *
from gp_package.architectures import build_constant_input_dim_het_deep_gp
from typing import Callable, Tuple, Optional
from functools import wraps
from tensorflow_probability.python.util.deferred_tensor import TensorMetaClass

import os
from scipy.stats import norm, uniform, multivariate_normal
import gpflow as gpf


#############################################################################################################################################
### create simulated data a from "Expectation Propagation for NOnstationary heteroscedastic gaussian process regression" by Tolvanen,2014 ###
#############################################################################################################################################

X = uniform.rvs(loc = -8.0, scale = 16.0, size = 200, random_state = 7)

f_tilda = 5.0 * np.sin(X)

first_prob = norm.pdf(
    x = X, loc = -2.5, scale = 1.)

second_prob = norm.pdf(
    x = X, loc = 2.5, scale = 1.)
f_sigma = first_prob + second_prob

third_prob = norm.pdf(
    x = X, loc = -8., scale = 3.0)
fourth_prob = norm.pdf(
    x = X, loc = 8., scale = 3.0)

sigma = 0.02 + third_prob #+ fourth_prob

print(sigma)
#noise = np.random.normal(loc = np.zeros_like(sigma), scale = np.sqrt(sigma))
#print(noise)

#epsi = tfp.distributions.MultivariateNormalDiag(loc = tf.zeros_like(sigma), scale_diag = tf.sqrt(sigma))
#sampled_epsi = epsi.sample()

noise = multivariate_normal.rvs(mean = np.zeros_like(sigma), cov = np.diag(sigma), size = 1, random_state=7)

#noise = multivariate_normal.rvs(mean = np.zeros_like(sigma), cov = np.diag(np.ones_like(sigma)*0.01), size = 1, random_state=7)

print(noise)
print(f_sigma)
print(f_tilda)

Y = f_sigma * f_tilda + noise

X = X.reshape((-1,1))
Y = Y.reshape((-1,1))


num_data, d_xim = X.shape

X_MARGIN, Y_MARGIN = 2.0, 0.5
fig, ax = plt.subplots()
ax.scatter(X, Y, marker='x', color='k');
ax.set_ylim(Y.min() - Y_MARGIN, Y.max() + Y_MARGIN);
ax.set_xlim(X.min() - X_MARGIN, X.max() + X_MARGIN);
plt.savefig('./sim_data_a_dataset.png')
plt.close()

#########################################
#########################################
######## GPflow code ####################
#########################################
#########################################

likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
    distribution_class=tfp.distributions.Normal,  # Gaussian Likelihood
    scale_transform=tfp.bijectors.Exp(),  # Exponential Transform
)

print(f"Likelihood's expected latent_dim: {likelihood.latent_dim}")

kernel = gpf.kernels.SeparateIndependent(
    [
        gpf.kernels.SquaredExponential(),  # This is k1, the kernel of f1
        gpf.kernels.SquaredExponential(),  # this is k2, the kernel of f2
    ]
)


M = 20  # Number of inducing variables for each f_i

# Initial inducing points position Z
Z = np.linspace(X.min(), X.max(), M)[:, None]  # Z must be of shape [M, 1]

inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
    [
        gpf.inducing_variables.InducingPoints(Z),  # This is U1 = f1(Z1)
        gpf.inducing_variables.InducingPoints(Z),  # This is U2 = f2(Z2)
    ]
)

model = gpf.models.SVGP(
    kernel=kernel,
    likelihood=likelihood,
    inducing_variable=inducing_variable,
    num_latent_gps=likelihood.latent_dim,
)

data = (X, Y)
loss_fn = model.training_loss_closure(data)

gpf.utilities.set_trainable(model.q_mu, False)
gpf.utilities.set_trainable(model.q_sqrt, False)

variational_vars = [(model.q_mu, model.q_sqrt)]
natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

adam_vars = model.trainable_variables
adam_opt = tf.optimizers.Adam(0.01)

@tf.function
def optimisation_step():
    natgrad_opt.minimize(loss_fn, variational_vars)
    adam_opt.minimize(loss_fn, adam_vars)

epochs = 100
log_freq = 20

for epoch in range(1, epochs + 1):
    optimisation_step()

#########################################

Ymean, Yvar = model.predict_y(X)
Ymean = Ymean.numpy().squeeze()
Ystd = tf.sqrt(Yvar).numpy().squeeze()


fig, ax = plt.subplots()
num_data_test = 200
X_test = np.linspace(X.min() - X_MARGIN, X.max() + X_MARGIN, num_data_test).reshape(-1, 1)
#out = model(X_test)
#out = model(X_test)
NUM_TESTING = X_test.shape[0]

### Multi-sample case ##
# NOTE -- we just tile X_test NUM_SAMPLES times

NUM_SAMPLES = 1

X_test_tiled = np.tile(X_test, (NUM_SAMPLES,1))
mu, var = model.predict_y(X_test_tiled)

mu = mu.numpy().squeeze()
var = var.numpy().squeeze()

print(' ---- size of predictions ----')
print(mu.shape)
print(var.shape)

mu = np.mean(mu.reshape((NUM_SAMPLES, NUM_TESTING)), axis = 0)
var = np.mean(var.reshape((NUM_SAMPLES, NUM_TESTING)), axis = 0)

print(' ---- size of predictions ----')
print(mu.shape)
print(var.shape)

X_test = X_test.squeeze()

for i in [1, 2]:
    lower = mu - i * np.sqrt(var)
    upper = mu + i * np.sqrt(var)
    ax.fill_between(X_test, lower, upper, color="C1", alpha=0.3)

ax.set_ylim(Y.min() - Y_MARGIN, Y.max() + Y_MARGIN)
ax.set_xlim(X.min() - X_MARGIN, X.max() + X_MARGIN)
ax.plot(X, Y, "kx", alpha=0.5)
ax.plot(X_test, mu, "C1")
ax.set_xlabel('time')
ax.set_ylabel('acc')
plt.savefig(f"./figures/sim_data_a_dataset_het_svgp_gpflow.png")
plt.close()




