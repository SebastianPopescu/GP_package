from .gpflow_posterior import IndependentPosteriorMultiOutput
from .base_posterior import BasePosterior
from .get_posterior_class import get_posterior_class


__all__ = [
    "BasePosterior",
    "get_posterior_class",
    "IndependentPosteriorMultiOutput",
]