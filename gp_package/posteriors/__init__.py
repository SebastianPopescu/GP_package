from .gpflow_posterior import IndependentOrthogonalPosteriorMultiOutput
from .base_posterior import BaseOrthogonalPosterior
from .get_posterior_class import get_posterior_class


__all__ = [
    "BaseOrthogonalPosterior",
    "get_posterior_class",
    "IndependentOrthogonalPosteriorMultiOutput",
]