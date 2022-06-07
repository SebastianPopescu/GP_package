
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ..base import Parameter, TensorType
from ..config import default_float

#from gpflux.exceptions import GPLayerIncompatibilityException
from ..math import _cholesky_with_jitter


from ..base import MeanAndVariance, Module, TensorType
import abc

class Flow(Module, metaclass=abc.ABCMeta):
    
    """
    A base class for a Flow. 
    This layer holds the kernel,
    inducing variables and variational distribution, and mean function.
    """

    """
    # TODO add stuff here below --->>

    num_data: int
    whiten: bool
    num_samples: Optional[int]
    full_cov: bool
    full_output_cov: bool
    q_mu: Parameter
    q_sqrt: Parameter
    """

    def __init__(
        self,

        *,
        name: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        #TODO -- document arguments to __init__ method

        """

        """
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

        """

    @abc.abstractmethod
    def forward(self, F: TensorType, X: TensorType = None) -> tf.Tensor:
        raise NotImplementedError("Not Implemented in explicit Flow class")

    @abc.abstractmethod
    def inverse(self, F: TensorType) -> tf.Tensor:
        raise NotImplementedError("Not Implemented in explicit Flow class")


    @abc.abstractmethod
    def forward_grad(self, F: TensorType) -> tf.Tensor:
        
        # TODO -- I think we just need to use a gradient tape and that's it for automatic differentiation
        raise NotImplementedError("Not Implemented in explicit Flow class")


    @abc.abstractmethod
    def inverse_grad(self, F: TensorType) -> tf.Tensor:
        
        # TODO -- I think we just need to use a gradient tape and that's it for automatic differentiation
        raise NotImplementedError("Not Implemented in explicit Flow class")


