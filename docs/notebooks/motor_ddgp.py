import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow_probability as tfp
from sklearn.neighbors import KernelDensity
from gp_package.base import TensorType
tf.keras.backend.set_floatx("float64")

from gp_package.models import *
from gp_package.layers import *
from gp_package.kernels import *
from gp_package.inducing_variables import *
from gp_package.architectures import Config, build_constant_input_dim_dist_deep_gp
from typing import Callable, Tuple, Optional
from functools import wraps
from tensorflow_probability.python.util.deferred_tensor import TensorMetaClass


class LikelihoodOutputs(tf.Module, metaclass=TensorMetaClass):
    """
    This class encapsulates the outputs of a :class:`~gpflux.layers.LikelihoodLayer`.

    It contains the mean and variance of the marginal distribution of the final latent
    :class:`~gpflux.layers.GPLayer`, as well as the mean and variance of the likelihood.

    This class includes the `TensorMetaClass
    <https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/util/deferred_tensor.py#L81>`_
    to make objects behave as a `tf.Tensor`. This is necessary so that it can be
    returned from the `tfp.layers.DistributionLambda` Keras layer.
    """

    def __init__(
        self,
        f_mean: TensorType,
        f_var: TensorType,
        y_mean: Optional[TensorType],
        y_var: Optional[TensorType],
    ):
        super().__init__(name="likelihood_outputs")

        self.f_mean = f_mean
        self.f_var = f_var
        self.y_mean = y_mean
        self.y_var = y_var

    def _value(
        self, dtype: tf.dtypes.DType = None, name: str = None, as_ref: bool = False
    ) -> tf.Tensor:
        return self.f_mean

    @property
    def shape(self) -> tf.Tensor:
        return self.f_mean.shape

    @property
    def dtype(self) -> tf.dtypes.DType:
        return self.f_mean.dtype


def batch_predict(
    predict_callable: Callable[[np.ndarray], Tuple[np.ndarray, ...]], batch_size: int = 1000
) -> Callable[[np.ndarray], Tuple[np.ndarray, ...]]:
    """
    Simple wrapper that transform a full dataset predict into batch predict.
    :param predict_callable: desired predict function that we want to wrap so it's executed in
     batch fashion.
    :param batch_size: how many predictions to do within single batch.
    """
    if batch_size <= 0:
        raise ValueError(f"Batch size has to be positive integer!")

    @wraps(predict_callable)
    def wrapper(x: np.ndarray) -> Tuple[np.ndarray, ...]:
        batches_f_mean = []
        batches_f_var = []
        batches_y_mean = []
        batches_y_var = []
        for x_batch in tf.data.Dataset.from_tensor_slices(x).batch(
            batch_size=batch_size, drop_remainder=False
        ):
            batch_predictions = predict_callable(x_batch)
            batches_f_mean.append(batch_predictions.f_mean)
            batches_f_var.append(batch_predictions.f_var)
            batches_y_mean.append(batch_predictions.y_mean)
            batches_y_var.append(batch_predictions.y_var)

        return LikelihoodOutputs(
            tf.concat(batches_f_mean, axis=0),
            tf.concat(batches_f_var, axis=0),
            tf.concat(batches_y_mean, axis=0),
            tf.concat(batches_y_var, axis=0)
        )

    return wrapper

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
dist_deep_gp: DistDeepGP = build_constant_input_dim_dist_deep_gp(X, num_layers=2, config=config, dim_output=1)

model = dist_deep_gp.as_training_model()
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
model = dist_deep_gp.as_prediction_model()
#out = model(X_test)
out = batch_predict(model)(X_test)


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



