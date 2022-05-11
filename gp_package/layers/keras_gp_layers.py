
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ..base import Parameter, TensorType
from ..config import default_float

from ..kullback_leiblers import standard_kl
from ..mean_functions import Identity, MeanFunction

from ..conditionals import conditional_GP
from ..inducing_variables import MultioutputInducingVariables
from ..kernels import MultioutputKernel
from ..utils.bijectors import triangular

#from gpflux.exceptions import GPLayerIncompatibilityException
from ..math import _cholesky_with_jitter
#from gpflux.runtime_checks import verify_compatibility
#from gpflux.sampling.sample import Sample, efficient_sample

class GPLayer(tfp.layers.DistributionLambda):
    """
    A sparse variational multioutput GP layer. This layer holds the kernel,
    inducing variables and variational distribution, and mean function.
    """

    num_data: int
    whiten: bool
    num_samples: Optional[int]
    full_cov: bool
    full_output_cov: bool
    q_mu: Parameter
    q_sqrt: Parameter

    def __init__(
        self,
        kernel: MultioutputKernel,
        inducing_variable: MultioutputInducingVariables,
        num_data: int,
        num_latent_gps: int,
        mean_function: Optional[MeanFunction] = None,
        *,
        num_samples: Optional[int] = None,
        full_cov: bool = False,
        full_output_cov: bool = False,
        whiten: bool = True,
        name: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        :param kernel: The multioutput kernel for this layer.
        :param inducing_variable: The inducing features for this layer.
        :param num_data: The number of points in the training dataset (see :attr:`num_data`).
        :param mean_function: The mean function that will be applied to the
            inputs. Default: :class:`~gpflow.mean_functions.Identity`.

            .. note:: The Identity mean function requires the input and output
                dimensionality of this layer to be the same. If you want to
                change the dimensionality in a layer, you may want to provide a
                :class:`~gpflow.mean_functions.Linear` mean function instead.

        :param num_samples: The number of samples to draw when converting the
            :class:`~tfp.layers.DistributionLambda` into a `tf.Tensor`, see
            :meth:`_convert_to_tensor_fn`. Will be stored in the
            :attr:`num_samples` attribute.  If `None` (the default), draw a
            single sample without prefixing the sample shape (see
            :class:`tfp.distributions.Distribution`'s `sample()
            <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution#sample>`_
            method).
        :param full_cov: Sets default behaviour of calling this layer
            (:attr:`full_cov` attribute):
            If `False` (the default), only predict marginals (diagonal
            of covariance) with respect to inputs.
            If `True`, predict full covariance over inputs.
        :param full_output_cov: Sets default behaviour of calling this layer
            (:attr:`full_output_cov` attribute):
            If `False` (the default), only predict marginals (diagonal
            of covariance) with respect to outputs.
            If `True`, predict full covariance over outputs.
        :param num_latent_gps: The number of (latent) GPs in the layer
            (which can be different from the number of outputs, e.g. with a
            :class:`~gpflow.kernels.LinearCoregionalization` kernel).
            This is used to determine the size of the
            variational parameters :attr:`q_mu` and :attr:`q_sqrt`.
            If possible, it is inferred from the *kernel* and *inducing_variable*.
        :param whiten: If `True` (the default), uses the whitened parameterisation
            of the inducing variables; see :attr:`whiten`.
        :param name: The name of this layer.
        :param verbose: The verbosity mode. Set this parameter to `True`
            to show debug information.
        """

        super().__init__(
            make_distribution_fn=self._make_distribution_fn,
            convert_to_tensor_fn=self._convert_to_tensor_fn,
            dtype=default_float(),
            name=name,
        )

        self.kernel = kernel
        self.inducing_variable = inducing_variable
        self.num_data = num_data

        if mean_function is None:
            mean_function = Identity()
            if verbose:
                warnings.warn(
                    "Beware, no mean function was specified in the construction of the `GPLayer` "
                    "so the default `gpflow.mean_functions.Identity` is being used. "
                    "This mean function will only work if the input dimensionality "
                    "matches the number of latent Gaussian processes in the layer."
                )
        self.mean_function = mean_function

        self.full_output_cov = full_output_cov
        self.full_cov = full_cov
        self.whiten = whiten
        self.verbose = verbose
        num_inducing = self.inducing_variable.num_inducing
        self.num_latent_gps = num_latent_gps

        ###### Introduce variational parameters for q(U) #######
        self.q_mu = Parameter(
            np.zeros((num_inducing, self.num_latent_gps)),
            dtype=default_float(),
            name=f"{self.name}_q_mu" if self.name else "q_mu",
        )  # [num_inducing, num_latent_gps]

        self.q_sqrt = Parameter(
            np.stack([np.eye(num_inducing) for _ in range(self.num_latent_gps)]),
            transform=triangular(),
            dtype=default_float(),
            name=f"{self.name}_q_sqrt" if self.name else "q_sqrt",
        )  # [num_latent_gps, num_inducing, num_inducing]

        self.num_samples = num_samples

    def predict(
        self,
        inputs: TensorType,
        *,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Make a prediction at N test inputs for the Q outputs of this layer,
        including the mean function contribution.

        The covariance and its shape is determined by *full_cov* and *full_output_cov* as follows:

        +--------------------+---------------------------+--------------------------+
        | (co)variance shape | ``full_output_cov=False`` | ``full_output_cov=True`` |
        +--------------------+---------------------------+--------------------------+
        | ``full_cov=False`` | [N, Q]                    | [N, Q, Q]                |
        +--------------------+---------------------------+--------------------------+
        | ``full_cov=True``  | [Q, N, N]                 | [N, Q, N, Q]             |
        +--------------------+---------------------------+--------------------------+

        :param inputs: The inputs to predict at, with a shape of [N, D], where D is
            the input dimensionality of this layer.
        :param full_cov: Whether to return full covariance (if `True`) or
            marginal variance (if `False`, the default) w.r.t. inputs.
        :param full_output_cov: Whether to return full covariance (if `True`)
            or marginal variance (if `False`, the default) w.r.t. outputs.

        :returns: posterior mean (shape [N, Q]) and (co)variance (shape as above) at test points
        """
        mean_function = self.mean_function(inputs)
        mean_cond, cov = conditional_GP(
            inputs,
            self.inducing_variable,
            self.kernel,
            self.q_mu,
            q_sqrt=self.q_sqrt,
            full_cov=full_cov,
            full_output_cov=full_output_cov,
            white=self.whiten,
        )

        return mean_cond + mean_function, cov

    def call(self, inputs: TensorType, *args: List[Any], **kwargs: Dict[str, Any]) -> tf.Tensor:
        """
        The default behaviour upon calling this layer.

        This method calls the `tfp.layers.DistributionLambda` super-class
        `call` method, which constructs a `tfp.distributions.Distribution`
        for the predictive distributions at the input points
        (see :meth:`_make_distribution_fn`).
        You can pass this distribution to `tf.convert_to_tensor`, which will return
        samples from the distribution (see :meth:`_convert_to_tensor_fn`).

        This method also adds a layer-specific loss function, given by the KL divergence between
        this layer and the GP prior (scaled to per-datapoint).
        """
        outputs = super().call(inputs, *args, **kwargs)

        if kwargs.get("training"):
            #log_prior = tf.add_n([p.log_prior_density() for p in self.kernel.trainable_parameters])
            loss = self.standard_kl() #- log_prior
            loss_per_datapoint = loss / self.num_data

        else:
            # TF quirk: add_loss must always add a tensor to compile
            loss_per_datapoint = tf.constant(0.0, dtype=default_float())
        self.add_loss(loss_per_datapoint)

        # Metric names should be unique; otherwise they get overwritten if you
        # have multiple with the same name
        name = f"{self.name}_standard_kl" if self.name else "standard_kl"
        self.add_metric(loss_per_datapoint, name=name, aggregation="mean")

        return outputs

    def standard_kl(self) -> tf.Tensor:
        r"""
        Returns the KL divergence ``KL[q(u)∥p(u)]`` from the prior ``p(u)`` to
        the variational distribution ``q(u)``.  If this layer uses the
        :attr:`whiten`\ ed representation, returns ``KL[q(v)∥p(v)]``.
        """
        return standard_kl(
            self.inducing_variable, self.kernel, self.q_mu, self.q_sqrt, whiten=self.whiten
        )

    def _make_distribution_fn(
        self, previous_layer_outputs: TensorType
    ) -> tfp.distributions.Distribution:
        """
        Construct the posterior distributions at the output points of the previous layer,
        depending on :attr:`full_cov` and :attr:`full_output_cov`.

        :param previous_layer_outputs: The output from the previous layer,
            which should be coercible to a `tf.Tensor`
        """
        mean, cov = self.predict(
            previous_layer_outputs,
            full_cov=self.full_cov,
            full_output_cov=self.full_output_cov,
        )

        if self.full_cov and not self.full_output_cov:
            # mean: [N, Q], cov: [Q, N, N]
            return tfp.distributions.MultivariateNormalTriL(
                loc=tf.linalg.adjoint(mean), scale_tril=_cholesky_with_jitter(cov)
            )  # loc: [Q, N], scale: [Q, N, N]
        elif self.full_output_cov and not self.full_cov:
            # mean: [N, Q], cov: [N, Q, Q]
            return tfp.distributions.MultivariateNormalTriL(
                loc=mean, scale_tril=_cholesky_with_jitter(cov)
            )  # loc: [N, Q], scale: [N, Q, Q]
        elif not self.full_cov and not self.full_output_cov:
            # mean: [N, Q], cov: [N, Q]
            return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.sqrt(cov))
        else:
            raise NotImplementedError(
                "The combination of both `full_cov` and `full_output_cov` is not permitted."
            )

    def _convert_to_tensor_fn(self, distribution: tfp.distributions.Distribution) -> tf.Tensor:
        """
        Convert the predictive distributions at the input points (see
        :meth:`_make_distribution_fn`) to a tensor of :attr:`num_samples`
        samples from that distribution.
        Whether the samples are correlated or marginal (uncorrelated) depends
        on :attr:`full_cov` and :attr:`full_output_cov`.
        """
        # N input points
        # S = self.num_samples
        # Q = output dimensionality
        if self.num_samples is not None:
            samples = distribution.sample(
                (self.num_samples,)
            )  # [S, Q, N] if full_cov else [S, N, Q]
        else:
            samples = distribution.sample()  # [Q, N] if full_cov else [N, Q]

        if self.full_cov:
            samples = tf.linalg.adjoint(samples)  # [S, N, Q] or [N, Q]

        return samples

    """
    def sample(self) -> Sample:
        
        #.. todo:: TODO: Document this.
        
        return (
            efficient_sample(
                self.inducing_variable,
                self.kernel,
                self.q_mu,
                q_sqrt=self.q_sqrt,
                whiten=self.whiten,
            )
            # Makes use of the magic __add__ of the Sample class
            + self.mean_function
        )
    """


class Bayesian_GPLayer(tfp.layers.DistributionLambda):
    """
    A sparse variational multioutput GP layer. This layer holds the kernel,
    inducing variables and variational distribution, and mean function.

    It it full Bayesian, meaning that we only have access to samples from U and theta
    """

    num_data: int
    whiten: bool
    num_samples: Optional[int]
    full_cov: bool
    full_output_cov: bool
    f: TensorType 
    theta: TensorType

    def __init__(
        self,
        kernel: MultioutputKernel,
        inducing_variable: MultioutputInducingVariables,
        num_data: int,
        num_latent_gps: int,
        mean_function: Optional[MeanFunction] = None,
        *,
        num_samples: Optional[int] = None,
        full_cov: bool = False,
        full_output_cov: bool = False,
        whiten: bool = True,
        name: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        :param kernel: The multioutput kernel for this layer.
        :param inducing_variable: The inducing features for this layer.
        :param num_data: The number of points in the training dataset (see :attr:`num_data`).
        :param mean_function: The mean function that will be applied to the
            inputs. Default: :class:`~gpflow.mean_functions.Identity`.

            .. note:: The Identity mean function requires the input and output
                dimensionality of this layer to be the same. If you want to
                change the dimensionality in a layer, you may want to provide a
                :class:`~gpflow.mean_functions.Linear` mean function instead.

        :param num_samples: The number of samples to draw when converting the
            :class:`~tfp.layers.DistributionLambda` into a `tf.Tensor`, see
            :meth:`_convert_to_tensor_fn`. Will be stored in the
            :attr:`num_samples` attribute.  If `None` (the default), draw a
            single sample without prefixing the sample shape (see
            :class:`tfp.distributions.Distribution`'s `sample()
            <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution#sample>`_
            method).
        :param full_cov: Sets default behaviour of calling this layer
            (:attr:`full_cov` attribute):
            If `False` (the default), only predict marginals (diagonal
            of covariance) with respect to inputs.
            If `True`, predict full covariance over inputs.
        :param full_output_cov: Sets default behaviour of calling this layer
            (:attr:`full_output_cov` attribute):
            If `False` (the default), only predict marginals (diagonal
            of covariance) with respect to outputs.
            If `True`, predict full covariance over outputs.
        :param num_latent_gps: The number of (latent) GPs in the layer
            (which can be different from the number of outputs, e.g. with a
            :class:`~gpflow.kernels.LinearCoregionalization` kernel).
            This is used to determine the size of the
            variational parameters :attr:`q_mu` and :attr:`q_sqrt`.
            If possible, it is inferred from the *kernel* and *inducing_variable*.
        :param whiten: If `True` (the default), uses the whitened parameterisation
            of the inducing variables; see :attr:`whiten`.
        :param name: The name of this layer.
        :param verbose: The verbosity mode. Set this parameter to `True`
            to show debug information.
        """

        super().__init__(
            make_distribution_fn=self._make_distribution_fn,
            convert_to_tensor_fn=self._convert_to_tensor_fn,
            dtype=default_float(),
            name=name,
        )

        self.kernel = kernel
        self.inducing_variable = inducing_variable
        self.num_data = num_data

        if mean_function is None:
            mean_function = Identity()
            if verbose:
                warnings.warn(
                    "Beware, no mean function was specified in the construction of the `GPLayer` "
                    "so the default `gpflow.mean_functions.Identity` is being used. "
                    "This mean function will only work if the input dimensionality "
                    "matches the number of latent Gaussian processes in the layer."
                )
        self.mean_function = mean_function

        self.full_output_cov = full_output_cov
        self.full_cov = full_cov
        self.whiten = whiten
        self.verbose = verbose
        num_inducing = self.inducing_variable.num_inducing
        self.num_latent_gps = num_latent_gps

        '''
        ### Remainder -- this is not required here as U values are sampled from a hypernetwork
        ###### Introduce variational parameters for q(U) #######
        self.q_mu = Parameter(
            np.zeros((num_inducing, self.num_latent_gps)),
            dtype=default_float(),
            name=f"{self.name}_q_mu" if self.name else "q_mu",
        )  # [num_inducing, num_latent_gps]

        self.q_sqrt = Parameter(
            np.stack([np.eye(num_inducing) for _ in range(self.num_latent_gps)]),
            transform=triangular(),
            dtype=default_float(),
            name=f"{self.name}_q_sqrt" if self.name else "q_sqrt",
        )  # [num_latent_gps, num_inducing, num_inducing]
        '''

        self.num_samples = num_samples

    def predict(
        self,
        inputs: TensorType,
        theta: list[Any],
        U: TensorType,
        *,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Make a prediction at N test inputs for the Q outputs of this layer,
        including the mean function contribution.

        The covariance and its shape is determined by *full_cov* and *full_output_cov* as follows:

        +--------------------+---------------------------+--------------------------+
        | (co)variance shape | ``full_output_cov=False`` | ``full_output_cov=True`` |
        +--------------------+---------------------------+--------------------------+
        | ``full_cov=False`` | [N, Q]                    | [N, Q, Q]                |
        +--------------------+---------------------------+--------------------------+
        | ``full_cov=True``  | [Q, N, N]                 | [N, Q, N, Q]             |
        +--------------------+---------------------------+--------------------------+

        :param inputs: The inputs to predict at, with a shape of [N, D], where D is
            the input dimensionality of this layer.
        :param full_cov: Whether to return full covariance (if `True`) or
            marginal variance (if `False`, the default) w.r.t. inputs.
        :param full_output_cov: Whether to return full covariance (if `True`)
            or marginal variance (if `False`, the default) w.r.t. outputs.
        :param theta: list of kernel hyperparameters:
            first element is kernel variance
            second element is lengthscales
        :param U: sampled values for U

        :returns: posterior mean (shape [N, Q]) and (co)variance (shape as above) at test points
        """
        mean_function = self.mean_function(inputs)
        mean_cond, cov = conditional_Bayesian_GP(
            inputs,
            self.inducing_variable,
            self.kernel,
            theta,
            U,
            full_cov=full_cov,
            full_output_cov=full_output_cov,
            white=self.whiten,
        )

        return mean_cond + mean_function, cov

    def call(self, inputs: TensorType, *args: List[Any], **kwargs: Dict[str, Any]) -> tf.Tensor:
        """
        The default behaviour upon calling this layer.

        This method calls the `tfp.layers.DistributionLambda` super-class
        `call` method, which constructs a `tfp.distributions.Distribution`
        for the predictive distributions at the input points
        (see :meth:`_make_distribution_fn`).
        You can pass this distribution to `tf.convert_to_tensor`, which will return
        samples from the distribution (see :meth:`_convert_to_tensor_fn`).

        This method also adds a layer-specific loss function, given by the KL divergence between
        this layer and the GP prior (scaled to per-datapoint).
        """
        outputs = super().call(inputs, *args, **kwargs)

        if kwargs.get("training"):
            #log_prior = tf.add_n([p.log_prior_density() for p in self.kernel.trainable_parameters])
            loss = self.standard_kl() #- log_prior
            loss_per_datapoint = loss / self.num_data

        else:
            # TF quirk: add_loss must always add a tensor to compile
            loss_per_datapoint = tf.constant(0.0, dtype=default_float())
        self.add_loss(loss_per_datapoint)

        # Metric names should be unique; otherwise they get overwritten if you
        # have multiple with the same name
        name = f"{self.name}_standard_kl" if self.name else "standard_kl"
        self.add_metric(loss_per_datapoint, name=name, aggregation="mean")

        return outputs

    def standard_kl(self) -> tf.Tensor:
        r"""
        Returns the KL divergence ``KL[q(u)∥p(u)]`` from the prior ``p(u)`` to
        the variational distribution ``q(u)``.  If this layer uses the
        :attr:`whiten`\ ed representation, returns ``KL[q(v)∥p(v)]``.
        """
        return standard_kl(
            self.inducing_variable, self.kernel, self.q_mu, self.q_sqrt, whiten=self.whiten
        )

    def _make_distribution_fn(
        self, previous_layer_outputs: TensorType
    ) -> tfp.distributions.Distribution:
        """
        Construct the posterior distributions at the output points of the previous layer,
        depending on :attr:`full_cov` and :attr:`full_output_cov`.

        :param previous_layer_outputs: The output from the previous layer,
            which should be coercible to a `tf.Tensor`
        """
        mean, cov = self.predict(
            previous_layer_outputs,
            full_cov=self.full_cov,
            full_output_cov=self.full_output_cov,
        )

        if self.full_cov and not self.full_output_cov:
            # mean: [N, Q], cov: [Q, N, N]
            return tfp.distributions.MultivariateNormalTriL(
                loc=tf.linalg.adjoint(mean), scale_tril=_cholesky_with_jitter(cov)
            )  # loc: [Q, N], scale: [Q, N, N]
        elif self.full_output_cov and not self.full_cov:
            # mean: [N, Q], cov: [N, Q, Q]
            return tfp.distributions.MultivariateNormalTriL(
                loc=mean, scale_tril=_cholesky_with_jitter(cov)
            )  # loc: [N, Q], scale: [N, Q, Q]
        elif not self.full_cov and not self.full_output_cov:
            # mean: [N, Q], cov: [N, Q]
            return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.sqrt(cov))
        else:
            raise NotImplementedError(
                "The combination of both `full_cov` and `full_output_cov` is not permitted."
            )

    def _convert_to_tensor_fn(self, distribution: tfp.distributions.Distribution) -> tf.Tensor:
        """
        Convert the predictive distributions at the input points (see
        :meth:`_make_distribution_fn`) to a tensor of :attr:`num_samples`
        samples from that distribution.
        Whether the samples are correlated or marginal (uncorrelated) depends
        on :attr:`full_cov` and :attr:`full_output_cov`.
        """
        # N input points
        # S = self.num_samples
        # Q = output dimensionality
        if self.num_samples is not None:
            samples = distribution.sample(
                (self.num_samples,)
            )  # [S, Q, N] if full_cov else [S, N, Q]
        else:
            samples = distribution.sample()  # [Q, N] if full_cov else [N, Q]

        if self.full_cov:
            samples = tf.linalg.adjoint(samples)  # [S, N, Q] or [N, Q]

        return samples

    """
    def sample(self) -> Sample:
        
        #.. todo:: TODO: Document this.
        
        return (
            efficient_sample(
                self.inducing_variable,
                self.kernel,
                self.q_mu,
                q_sqrt=self.q_sqrt,
                whiten=self.whiten,
            )
            # Makes use of the magic __add__ of the Sample class
            + self.mean_function
        )
    """