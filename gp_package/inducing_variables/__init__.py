from .inducing_variables import InducingPoints, InducingVariables
from .distributional_inducing_variables import DistributionalInducingPoints, DistributionalInducingVariables
from .multioutput import (
    FallbackSeparateIndependentInducingVariables,
    FallbackSharedIndependentInducingVariables,
    MultioutputInducingVariables,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
    FallbackSeparateIndependentDistributionalInducingVariables,
    FallbackSharedIndependentDistributionalInducingVariables,
    MultioutputDistributionalInducingVariables,
    SeparateIndependentDistributionalInducingVariables,
    SharedIndependentDistributionalInducingVariables,
)



__all__ = [
    "FallbackSeparateIndependentInducingVariables",
    "FallbackSharedIndependentInducingVariables",
    "InducingPoints",
    "InducingVariables",
    "DistributionalInducingPoints",
    "DistributionalInducingVariables",
    "MultioutputInducingVariables",
    "SeparateIndependentInducingVariables",
    "SharedIndependentInducingVariables",
    "inducing_variables",
    "distributional_inducing_variables",
    "FallbackSeparateIndependentDistributionalInducingVariables",
    "FallbackSharedIndependentDistributionalInducingVariables",
    "MultioutputDistributionalInducingVariables",
    "SeparateIndependentDistributionalInducingVariables",
    "SharedIndependentDistributionalInducingVariables",
]
