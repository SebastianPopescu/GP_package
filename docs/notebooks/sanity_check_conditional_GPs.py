from gp_package.base import TensorType
from gp_package.kernels.stationary_kernels import Hybrid
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow_probability as tfp
from sklearn.neighbors import KernelDensity
tf.keras.backend.set_floatx("float64")
import os
from typing import Callable, Optional, Tuple, Any
from gp_package.kernels import SquaredExponential, Kernel
from gp_package.inducing_variables import InducingPoints, DistributionalInducingPoints
from typing import Union
from gp_package.base import Parameter
from gp_package.utils import triangular

def base_conditional(
    Kmn: tf.Tensor,
    Kmm: tf.Tensor,
    Knn: tf.Tensor,
    f: tf.Tensor,
    *,
    full_cov: bool = False,
    q_sqrt: Optional[tf.Tensor] = None,
    white: bool = False,
) -> Tuple[tf.Tensor, tf.Tensor]:

    Lm = tf.linalg.cholesky(Kmm)
    return base_conditional_with_lm(
        Kmn=Kmn, Lm=Lm, Knn=Knn, f=f, full_cov=full_cov, q_sqrt=q_sqrt, white=white
    )

def base_conditional_with_lm(
    Kmn: tf.Tensor,
    Lm: tf.Tensor,
    Knn: tf.Tensor,
    f: tf.Tensor,
    *,
    full_cov: bool = False,
    q_sqrt: Optional[tf.Tensor] = None,
    white: bool = False,
) -> Tuple[tf.Tensor, tf.Tensor]:

    # compute kernel stuff
    num_func = tf.shape(f)[-1]  # R
    N = tf.shape(Kmn)[-1]
    M = tf.shape(f)[-2]

    # get the leading dims in Kmn to the front of the tensor
    # if Kmn has rank two, i.e. [M, N], this is the identity op.
    K = tf.rank(Kmn)
    perm = tf.concat(
        [
            tf.reshape(tf.range(1, K - 1), [K - 2]),  # leading dims (...)
            tf.reshape(0, [1]),  # [M]
            tf.reshape(K - 1, [1]),
        ],
        0,
    )  # [N]
    Kmn = tf.transpose(Kmn, perm)  # [..., M, N]

    shape_constraints = [
        (Kmn, [..., "M", "N"]),
        (Lm, ["M", "M"]),
        (Knn, [..., "N", "N"] if full_cov else [..., "N"]),
        (f, ["M", "R"]),
    ]
    if q_sqrt is not None:
        shape_constraints.append(
            (q_sqrt, (["M", "R"] if q_sqrt.shape.ndims == 2 else ["R", "M", "M"]))
        )
    tf.debugging.assert_shapes(
        shape_constraints,
        message="base_conditional() arguments "
        "[Note that this check verifies the shape of an alternative "
        "representation of Kmn. See the docs for the actual expected "
        "shape.]",
    )

    leading_dims = tf.shape(Kmn)[:-2]

    # Compute the projection matrix A
    Lm = tf.broadcast_to(Lm, tf.concat([leading_dims, tf.shape(Lm)], 0))  # [..., M, M]
    A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)  # [..., M, N]

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - tf.linalg.matmul(A, A, transpose_a=True)  # [..., N, N]
        cov_shape = tf.concat([leading_dims, [num_func, N, N]], 0)
        fvar = tf.broadcast_to(tf.expand_dims(fvar, -3), cov_shape)  # [..., R, N, N]
    else:
        fvar = Knn - tf.reduce_sum(tf.square(A), -2)  # [..., N]
        cov_shape = tf.concat([leading_dims, [num_func, N]], 0)  # [..., R, N]
        fvar = tf.broadcast_to(tf.expand_dims(fvar, -2), cov_shape)  # [..., R, N]

    print('GPflow-esque fvar_distributional', fvar)
    # another backsubstitution in the unwhitened case
    if not white:
        A = tf.linalg.triangular_solve(tf.linalg.adjoint(Lm), A, lower=False)

    # construct the conditional mean
    f_shape = tf.concat([leading_dims, [M, num_func]], 0)  # [..., M, R]
    f = tf.broadcast_to(f, f_shape)  # [..., M, R]
    fmean = tf.linalg.matmul(A, f, transpose_a=True)  # [..., N, R]

    if q_sqrt is not None:
        q_sqrt_dims = q_sqrt.shape.ndims
        if q_sqrt_dims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # [R, M, N]
        elif q_sqrt_dims == 3:
            L = tf.linalg.band_part(q_sqrt, -1, 0)  # force lower triangle # [R, M, M]
            L_shape = tf.shape(L)
            L = tf.broadcast_to(L, tf.concat([leading_dims, L_shape], 0))

            shape = tf.concat([leading_dims, [num_func, M, N]], axis=0)
            A_tiled = tf.broadcast_to(tf.expand_dims(A, -3), shape)
            LTA = tf.linalg.matmul(L, A_tiled, transpose_a=True)  # [R, M, N]
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.shape.ndims))

        if full_cov:
            fvar = fvar + tf.linalg.matmul(LTA, LTA, transpose_a=True)  # [R, N, N]
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), -2)  # [R, N]
            print('GPflow-esque fvar_epistemic', tf.reduce_sum(tf.square(LTA), -2))

    if not full_cov:
        fvar = tf.linalg.adjoint(fvar)  # [N, R]

    shape_constraints = [
        (Kmn, [..., "M", "N"]),  # tensor included again for N dimension
        (f, [..., "M", "R"]),  # tensor included again for R dimension
        (fmean, [..., "N", "R"]),
        (fvar, [..., "R", "N", "N"] if full_cov else [..., "N", "R"]),
    ]
    tf.debugging.assert_shapes(shape_constraints, message="base_conditional() return values")

    return fmean, fvar


def conditional_GP(q_mu, q_sqrt, Knn, Kmn, Kmm,
    white=True, full_cov=False):

    """
    TODO -- document this function
    """
    print(' ******* conditional GP ****************')

    Lm = tf.linalg.cholesky(Kmm)
    A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)
    
    if full_cov:
        fvar_distributional = Knn - tf.matmul(A, A, transpose_a=True)
    else:
        fvar_distributional = Knn[:,tf.newaxis] - tf.transpose(tf.reduce_sum(tf.square(A), 0, keepdims=True))
        #print('fvar_distributional ', fvar_distributional)
    
    # another backsubstitution in the unwhitened case
    if not white:
        A = tf.linalg.triangular_solve(tf.transpose(Lm), A, lower=False)

    fmean = tf.matmul(A, q_mu, transpose_a = True) 

    if full_cov:
        LTA= tf.matmul(q_sqrt,A, transpose_a = True)
        fvar_epistemic = tf.matmul(LTA,LTA,transpose_a=True)
    else:
        A = tf.tile(tf.expand_dims(A, axis=0),[ 1,1,1])
        LTA= tf.matmul(q_sqrt,A, transpose_a = True)
        fvar_epistemic = tf.transpose(tf.reduce_sum(tf.square(LTA), 1, keepdims=False))
        #print("fvar_epistemic", fvar_epistemic)

    fvar = fvar_epistemic + fvar_distributional

    return fmean, fvar


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


def Kuu(
    inducing_variable: Union[InducingPoints, DistributionalInducingPoints], kernel: Kernel, *, jitter: float = 0.0,
    seed : Optional[Any]  = None,
) -> tf.Tensor:

    if isinstance(inducing_variable, DistributionalInducingPoints):
        # Create instance of tfp.distributions.MultivariateNormalDiag so that it works with underpinning methods from kernel
        distributional_inducing_points = tfp.distributions.MultivariateNormalDiag(loc = inducing_variable.Z_mean,
            scale_diag = tf.sqrt(inducing_variable.Z_var))
        Kzz = kernel(distributional_inducing_points, seed = seed)
    elif isinstance(inducing_variable, InducingPoints):
        Kzz = kernel(inducing_variable.Z)
    Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)
    return Kzz

def Kuf(
    inducing_variable: Union[InducingPoints,DistributionalInducingPoints], kernel: Kernel, 
    Xnew: Union[TensorType,tfp.distributions.MultivariateNormalDiag],
    seed : Optional[Any] = None) -> tf.Tensor:
    
    if isinstance(inducing_variable,DistributionalInducingPoints):
        # Create instance of tfp.distributions.MultivariateNormalDiag so that it works with underpinning methods from kernel

        assert isinstance(Xnew, tfp.distributions.MultivariateNormalDiag)

        distributional_inducing_points = tfp.distributions.MultivariateNormalDiag(loc = inducing_variable.Z_mean,
            scale_diag = tf.sqrt(inducing_variable.Z_var))
        
        return kernel(distributional_inducing_points, Xnew, seed = seed)
    
    elif isinstance(inducing_variable, InducingPoints):    
        
        return kernel(inducing_variable.Z, Xnew)


if __name__=="__main__":

    toy = load_snelson_data(n=100)
    X, Y = toy.train_x, toy.train_y
    test_x = toy.test_x

    num_data, d_xim = X.shape

    NUM_INDUCING = 10

    kernel_euclidean = SquaredExponential()
    inducing_variable_euclidean = InducingPoints(
        np.linspace(X.min(), X.max(), NUM_INDUCING).reshape(-1, 1))

    print('-----------------------------------------')
    print('---------- Layer 1 info -----------------')
    print('-----------------------------------------')


    Kuu_euclidean = Kuu(inducing_variable_euclidean, kernel_euclidean, jitter = 1e-6)
    print("Kuu euclidean -- ", Kuu_euclidean)
    Kuf_euclidean = Kuf(inducing_variable_euclidean, kernel_euclidean, X[:5,...])
    print("Kuf euclidean -- ", Kuf_euclidean)
    Kff_euclidean = kernel_euclidean(X[:5,...], full_cov=False)
    print("Kff euclidean -- ", Kff_euclidean)

    ###### Introduce variational parameters for q(U) #######
    q_mu = Parameter(
        np.random.uniform(-0.5, 0.5, (NUM_INDUCING, 1)), # np.zeros((num_inducing, 1)),
        name="q_mu_EUCLIDEAN",
    )  # [num_inducing, num_latent_gps]

    q_sqrt = Parameter(
        np.stack([1e-1 * np.eye(NUM_INDUCING) for _ in range(1)]),
        transform=triangular(),
        name="q_sqrt_EUCLIDEAN",
    )  # [num_latent_gps, num_inducing, num_inducing]

    f1_mean, f1_var = base_conditional(Kuf_euclidean, Kuu_euclidean, Kff_euclidean, q_mu, full_cov = False, q_sqrt = q_sqrt, white = True)
    f1_mean_v2, f1_var_v2 = conditional_GP(q_mu, q_sqrt, Kff_euclidean, Kuf_euclidean, Kuu_euclidean, white=True, full_cov=False)


    print('f1_mean -- ', f1_mean)
    print('f1_var -- ', f1_var)

    print('f1_mean_v2 -- ', f1_mean_v2)
    print('f1_var_v2 -- ', f1_var_v2)

    tf.debugging.assert_equal(f1_mean, f1_mean_v2, message = "problem with f1_mean")
    tf.debugging.assert_equal(f1_var, f1_var_v2, message = "problem with f1_var")

    print('-----------------------------------------')
    print('---------- Layer 2 info -----------------')
    print('-----------------------------------------')


    lcl_seed = np.random.randint(1e5)
    kernel_hyrbid = Hybrid()
    z_init_mean = np.random.uniform(low=-0.5, high=0.5, size=(NUM_INDUCING, 1))
    z_init_var = 0.0067153485 * np.ones((NUM_INDUCING, 1))
    inducing_variable_distributional = DistributionalInducingPoints(z_init_mean, z_init_var)

    F1_dist = tfp.distributions.MultivariateNormalDiag(loc = f1_mean, scale_diag = tf.sqrt(f1_var))

    Kuu_wass = Kuu(inducing_variable_distributional, kernel_hyrbid, jitter = 1e-6, seed = lcl_seed)
    print("Kuu hybrid -- ", Kuu_wass)
    Kuf_wass = Kuf(inducing_variable_distributional, kernel_hyrbid, F1_dist, seed = lcl_seed)
    print("Kuf hybrid -- ", Kuf_wass)
    Kff_wass = kernel_hyrbid(F1_dist, full_cov=False)
    print("Kff hybrid -- ", Kff_wass)

    ###### Introduce variational parameters for q(U) #######
    q_mu = Parameter(
        np.random.uniform(-0.5, 0.5, (NUM_INDUCING, 1)), # np.zeros((num_inducing, 1)),
        name="q_mu_wass",
    )  # [num_inducing, num_latent_gps]

    q_sqrt = Parameter(
        np.stack([1e-1 * np.eye(NUM_INDUCING) for _ in range(1)]),
        transform=triangular(),
        name="q_sqrt_wass",
    )  # [num_latent_gps, num_inducing, num_inducing]

    f2_mean, f2_var = base_conditional(Kuf_wass, Kuu_wass, Kff_wass, q_mu, full_cov = False, q_sqrt = q_sqrt, white = True)
    f2_mean_v2, f2_var_v2 = conditional_GP(q_mu, q_sqrt, Kff_wass, Kuf_wass, Kuu_wass, True, False)

    tf.debugging.assert_equal(f2_mean, f2_mean_v2, message = "problem with f2_mean")
    tf.debugging.assert_equal(f2_var, f2_var_v2, message = "problem with f2_var")

    print('f2_mean -- ', f2_mean)
    print('f2_var -- ', f2_var)

    print('f2_mean_v2 -- ', f2_mean_v2)
    print('f2_var_v2 -- ', f2_var_v2)
