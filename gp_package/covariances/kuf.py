from numpy import isin
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Union

from gp_package.inducing_variables.distributional_inducing_variables import DistributionalInducingPoints

from ..base import TensorLike, TensorType
from ..inducing_variables import InducingPoints
from ..kernels import Kernel, SquaredExponential

def Kuf(
    inducing_variable: Union[InducingPoints,DistributionalInducingPoints], kernel: Kernel, 
    Xnew: Union[TensorType,tfp.distributions.MultivariateNormalDiag]) -> tf.Tensor:
    
    if isinstance(inducing_variable,DistributionalInducingPoints):
        # Create instance of tfp.distributions.MultivariateNormalDiag so that it works with underpinning methods from kernel

        assert isinstance(Xnew, tfp.distributions.MultivariateNormalDiag)

        distributional_inducing_points = tfp.distributions.MultivariateNormalDiag(loc = inducing_variable.Z_mean,
            scale_diag = tf.sqrt(inducing_variable.Z_var))
        
        return kernel(distributional_inducing_points, Xnew)
    
    elif isinstance(inducing_variable, InducingPoints):    

        print('*********************')
        print('--- inside Kuf ------')
        print(Xnew) 
        
        return kernel(inducing_variable.Z, Xnew)

