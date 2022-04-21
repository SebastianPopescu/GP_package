import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow_probability as tfp
from sklearn.neighbors import KernelDensity
tf.keras.backend.set_floatx("float64")
import os

from gp_package.models import *
from gp_package.layers import *
from gp_package.kernels import *
from gp_package.inducing_variables import *
from gp_package.architectures import Config, build_constant_input_dim_dist_deep_gp


class ToyData1D(object):
    def __init__(self, train_x, train_y, test_x, normalize=False, 
                 dtype=np.float64):
        self.train_x = np.array(train_x, dtype=dtype)[:, None]
        self.train_y = np.array(train_y, dtype=dtype)[:, None]
        self.n_train = self.train_x.shape[0]
        self.test_x = np.array(test_x, dtype=dtype)[:, None]
        self.x_min = np.min(test_x)
        self.x_max = np.max(test_x)
        self.n_test = self.test_x.shape[0]
        if normalize:
            self.normalize()

    def normalize(self):
        self.mean_x = np.mean(self.train_x, axis=0, keepdims=True)
        self.std_x = np.std(self.train_x, axis=0, keepdims=True) + 1e-6
        self.mean_y = np.mean(self.train_y, axis=0, keepdims=True)
        self.std_y = np.std(self.train_y, axis=0, keepdims=True) + 1e-6

        for x in [self.train_x, self.test_x]:
            x -= self.mean_x
            x /= self.std_x

        for x in [self.x_min, self.x_max]:
            x -= self.mean_x.squeeze()
            x /= self.std_x.squeeze()

        self.train_y -= self.mean_y
        self.train_y /= self.std_y

    
def load_snelson_data(n=100, dtype=np.float64):
    def _load_snelson(filename):
        with open(os.path.join("/home/sebastian.popescu/Desktop/my_code/GP_package/docs/notebooks","data", "snelson", filename), "r") as f:
            return np.array([float(i) for i in f.read().strip().split("\n")],
                            dtype=dtype)

    train_x = _load_snelson("train_inputs")
    train_y = _load_snelson("train_outputs")
    test_x = _load_snelson("test_inputs")
    perm = np.random.permutation(train_x.shape[0])
    train_x = train_x[perm][:n]
    train_y = train_y[perm][:n]
    return ToyData1D(train_x, train_y, test_x=test_x)


if __name__=="__main__":

    toy = load_snelson_data(n=100)
    X, Y = toy.train_x, toy.train_y
    test_x = toy.test_x

    num_data, d_xim = X.shape

    X_MARGIN, Y_MARGIN = 0.1, 0.5
    fig, ax = plt.subplots()
    ax.scatter(X, Y, marker='x', color='k');
    ax.set_ylim(Y.min() - Y_MARGIN, Y.max() + Y_MARGIN);
    ax.set_xlim(X.min() - X_MARGIN, X.max() + X_MARGIN);
    plt.savefig('./simple_dataset.png')
    plt.close()


    """
    NUM_INDUCING = 20

    kernel = SquaredExponential()
    inducing_variable = InducingPoints(
        np.linspace(X.min(), X.max(), NUM_INDUCING).reshape(-1, 1)
    )

    NUM_LAYERS = 2
    gp_layers = []

    for l in range(NUM_LAYERS):
        gp_layers.append(GPLayer(
        kernel, inducing_variable, num_data=num_data, num_latent_gps=1, name = f'layer_{l}'))


    likelihood_layer = LikelihoodLayer(Gaussian(0.1))

    single_layer_dgp = DeepGP(gp_layers, likelihood_layer, num_data=X.shape[0])
    """

    config = Config(
        num_inducing=10, inner_layer_qsqrt_factor=1e-1, likelihood_noise_variance=1e-2, whiten=True, hidden_layer_size=X.shape[1]
    )
    dist_deep_gp: DistDeepGP = build_constant_input_dim_dist_deep_gp(X, num_layers=2, config=config)

    model = dist_deep_gp.as_training_model()
    model.compile(tf.optimizers.Adam(1e-2))

    history = model.fit({"inputs": X, "targets": Y}, epochs=int(1), verbose=1)
    fig, ax = plt.subplots()
    ax.plot(history.history["loss"])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.savefig('./simple_dataset_loss_during_training.png')
    plt.close()

    fig, ax = plt.subplots()
    num_data_test = 200
    X_test = np.linspace(X.min() - X_MARGIN, X.max() + X_MARGIN, num_data_test).reshape(-1, 1)
    model = dist_deep_gp.as_prediction_model()
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



