from typing import Tuple
from dataclasses import dataclass

from utils.singleton import Singleton

"""
For now I am making assumption that:
    1. x_domain == y_domain
    2. x_points == y_points
    3. For mesh the xy_domain is always [0, 1]

"""
@dataclass
class Domain(metaclass=Singleton):
    XY_DOMAIN: Tuple[float, float] = (0, 1.0)
    T_DOMAIN: Tuple[float, float] = (0, 0.5)

    N_POINTS: int = 30
    T_POINTS: int = 30
    
    N_POINTS_PLOT: int = 50

