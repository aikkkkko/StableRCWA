"""各向同性磁性介质的 1D、2D 与圆锥 RCWA 实验分支。"""

from .rcwa1d_mag import Layer, Stack, solve, field_at, field_map, eps_mu_profile, eps_profile
from .rcwa2d_mag import Lattice, Layer2D, Stack2D, solve2d, field_at as field_at_2d, field_xy, eps_mu_map
from .rcwa_conical_mag import (
    ResultConicalMag, solve_conical, sweep_phi, order_index as conical_order_index,
    field_at as field_at_conical, field_map as field_map_conical,
    eps_mu_profile as eps_mu_profile_conical, eps_profile as eps_profile_conical,
)

__all__ = [
    "Layer", "Stack", "solve", "field_at", "field_map",
    "eps_mu_profile", "eps_profile",
    "Lattice", "Layer2D", "Stack2D", "solve2d", "field_at_2d", "field_xy", "eps_mu_map",
    "ResultConicalMag", "solve_conical", "sweep_phi", "conical_order_index",
    "field_at_conical", "field_map_conical", "eps_mu_profile_conical", "eps_profile_conical",
]
