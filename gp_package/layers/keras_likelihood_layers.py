from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.util.deferred_tensor import TensorMetaClass

from ..config import default_float
from ..base import TensorType, Module
from .base_likelihood_layers import Likelihood
from .explicit_likelihood_layers import Gaussian, StudentT

class LikelihoodLayer(tf.keras.layers.Layer):
    r"""
    A Keras layer that wraps a GPflow :class:`~gpflow.likelihoods.Likelihood`. This layer expects a
    `tfp.distributions.MultivariateNormalDiag` as its input, describing ``q(f)``.
    When training, calling this class computes the negative variational expectation
    :math:`-\mathbb{E}_{q(f)}[\log p(y|f)]` and adds it as a layer loss.
    When not training, it computes the mean and variance of ``y`` under ``q(f)``
    using :meth:`~gpflow.likelihoods.Likelihood.predict_mean_and_var`.

    .. note::

        Use **either** this `LikelihoodLayer` (together with
        `gpflux.models.DeepGP`) **or** `LikelihoodLoss` (e.g. together with a
        `tf.keras.Sequential` model). Do **not** use both at once because
        this would add the loss twice.
    """

    def __init__(self, likelihood: Likelihood):
        super().__init__(dtype=default_float())
        self.likelihood = likelihood

    def call(
        self,
        inputs: tfp.distributions.MultivariateNormalDiag,
        targets: Optional[TensorType] = None,
        training: bool = None,
    ) -> "LikelihoodOutputs":
        """
        When training (``training=True``), this method computes variational expectations
        (data-fit loss) and adds this information as a layer loss.
        When testing (the default), it computes the posterior mean and variance of ``y``.

        :param inputs: The output distribution of the previous layer. This is currently
            expected to be a :class:`~tfp.distributions.MultivariateNormalDiag`;
            that is, the preceding :class:`~gpflux.layers.GPLayer` should have
            ``full_cov=full_output_cov=False``.
        :returns: a `LikelihoodOutputs` tuple with the mean and variance of ``f`` and,
            if not training, the mean and variance of ``y``.

        .. todo:: Turn this layer into a
            :class:`~tfp.layers.DistributionLambda` as well and return the
            correct :class:`~tfp.distributions.Distribution` instead of a tuple
            containing mean and variance only.
        """

        assert isinstance(inputs, tfp.distributions.MultivariateNormalDiag)



        #if isinstance(self.likelihood, Gaussian):
        F_mean = inputs.loc
        F_var = inputs.scale.diag ** 2
        
        if training:
            assert targets is not None
            
            loss_per_datapoint = tf.reduce_mean(
                -self.likelihood.variational_expectations(F_mean, F_var, targets)
            )
            Y_mean = Y_var = None
        else:
            loss_per_datapoint = tf.constant(0.0, dtype=default_float())
            Y_mean, Y_var = self.likelihood.predict_mean_and_var(F_mean, F_var)

        """
        #NOTE -- I don't think this is necessary
        #TODO -- should probably delete this 

        elif isinstance(self.likelihood, StudentT):

            if training:
                assert targets is not None
                
                # TODO -- don't we need to use tf.reduce_mean on this??
                loss_per_datapoint = -self.likelihood._log_prob(inputs, targets)
                Y_mean = Y_var = None

            else:
                loss_per_datapoint = tf.constant(0.0, dtype=default_float())

            F_mean = inputs.loc
            F_var = inputs.scale.diag ** 2

        else:

            #NOTE -- this is currently covering the Heteroskedastic likelihood case 
            if training:
                assert targets is not None
                
                # TODO -- don't we need to use tf.reduce_mean on this??
                loss_per_datapoint = tf.reduce_mean(-self.likelihood.log_prob(inputs, targets))
                Y_mean = Y_var = None

            else:
                loss_per_datapoint = tf.constant(0.0, dtype=default_float())
                Y_mean = Y_var = None

            # TODO -- this needs to be changed here
            #NOTE -- these will actually be samples

            F_samples = inputs.sample()

            F_mean = tf.slice(F_samples, [0,0], [-1,1])
            F_var =  tf.exp(tf.slice(F_samples, [0,1], [-1,1]))

        """

        self.add_loss(loss_per_datapoint)

        return LikelihoodOutputs(F_mean, F_var, Y_mean, Y_var)




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
