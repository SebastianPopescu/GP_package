# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Optional
from numpy import isin

import tensorflow as tf
import tensorflow_probability as tfp
from deprecated import deprecated
from gp_package.config import default_float

from gp_package.utils.bijectors import triangular

from ..base import Module, Parameter, TensorData, TensorType
from ..utils import positive

class InverseApproximation(Module):
    
    def __init__(self, L_T : TensorData, dof : TensorType, name: Optional[str] = None):
        
        """
        param L_T: the initialization of the Cholesky of T (acts as replacement for Kuu^{-1})
        param dof: the initialization value for the restriced intrinsic degree of freedom of the underlying Wishart distribution
        """

        super().__init__(name=name)
        if not isinstance(L_T, (tf.Variable, tfp.util.TransformedVariable)):
        
            L_T = Parameter(
                L_T,
                transform = triangular(),
                dtype=default_float(),
                name=f"{self.name}_L_T" if self.name else "L_T"
            )
        self.L_T = L_T

        if not isinstance(dof, (tf.Variable, tfp.util.TransformedVariable)):

            dof = Parameter(
                dof,
                transform = positive(),
                dtype=default_float(),
                name=f"{self.name}_dof" if self.name else "dof"
            )
        self.dof = dof






