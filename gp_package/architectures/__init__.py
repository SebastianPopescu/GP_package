from .standard_dgp import Config, build_constant_input_dim_deep_gp
from .standard_ddgp import build_constant_input_dim_dist_deep_gp

__all__ = [
    "Config",
    "build_constant_input_dim_deep_gp",
    "build_constant_input_dim_dist_deep_gp"
]