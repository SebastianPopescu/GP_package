from .standard_dgp import Config, build_constant_input_dim_deep_gp
from .standard_orthogonal_dgp import build_constant_input_dim_orthogonal_deep_gp

__all__ = [
    "Config",
    "build_constant_input_dim_deep_gp",
    "build_constant_input_dim_orthogonal_deep_gp"
]