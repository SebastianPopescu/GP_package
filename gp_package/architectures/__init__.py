from .gpflow_standard_dgp import build_deep_gp
from .gpflow_standard_vff_dgp import Config, build_deep_vff_gp

__all__ = [
    "Config",
    "build_deep_gp",
    "build_deep_vff_gp"
]