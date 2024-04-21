from typing import Tuple
from dataclasses import dataclass


@dataclass
class Domain:
    """
    For now I am making assumption that:
        1. x_domain == y_domain
        2. x_points == y_points
        3. The mesh is normalized, xy_domain is [0, 1]

    """
    XY_DOMAIN: Tuple[float, float] = (0, 1.0)
    T_DOMAIN: Tuple[float, float] = (0, 0.5)

    N_POINTS: int = 15
    T_POINTS: int = 15

    N_POINTS_PLOT: int = 50
