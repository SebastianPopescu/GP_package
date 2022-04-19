import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow_probability as tfp
from sklearn.neighbors import KernelDensity
tf.keras.backend.set_floatx("float64")

from gp_package.models import *
from gp_package.layers import *
from gp_package.kernels import *
from gp_package.inducing_variables import *
from gp_package.architectures import Config, build_constant_input_dim_deep_gp


def motorcycle_data():
    """ Return inputs and outputs for the motorcycle dataset. We normalise the outputs. """
    import pandas as pd
    df = pd.read_csv("/home/sebastian.popescu/Desktop/my_code/GP_package/docs/notebooks/data/motor.csv", index_col=0)
    X, Y = df["times"].values.reshape(-1, 1), df["accel"].values.reshape(-1, 1)
    Y = (Y - Y.mean()) / Y.std()
    X /= X.max()
    return X, Y

X, Y = motorcycle_data()
num_data, d_xim = X.shape

X_MARGIN, Y_MARGIN = 0.1, 0.5
fig, ax = plt.subplots()
ax.scatter(X, Y, marker='x', color='k');
ax.set_ylim(Y.min() - Y_MARGIN, Y.max() + Y_MARGIN);
ax.set_xlim(X.min() - X_MARGIN, X.max() + X_MARGIN);
plt.savefig('./simple_dataset.png')
plt.close()

config = Config(
    num_inducing=25, inner_layer_qsqrt_factor=1e-5, likelihood_noise_variance=1e-2, whiten=True, hidden_layer_size=X.shape[1]
)
deep_gp: DeepGP = build_constant_input_dim_deep_gp(X, num_layers=2, config=config)

outputs, f_outputs = deep_gp.call(X, Y, True)
print(outputs)
print(f_outputs)

"""
model = deep_gp.as_training_model()
model.compile(tf.optimizers.Adam(1e-2))

history = model.fit({"inputs": X, "targets": Y}, epochs=int(1e3), verbose=1)
fig, ax = plt.subplots()
ax.plot(history.history["loss"])
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.savefig('./simple_dataset_loss_during_training.png')
plt.close()

fig, ax = plt.subplots()
num_data_test = 200
X_test = np.linspace(X.min() - X_MARGIN, X.max() + X_MARGIN, num_data_test).reshape(-1, 1)
model = deep_gp.as_prediction_model()
out = model(X_test)

mu = out.y_mean.numpy().squeeze()
var = out.y_var.numpy().squeeze()
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
plt.savefig('./simple_dataset_predictions_testing.png')
plt.close()
"""


