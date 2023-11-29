from typing import Tuple
from dataclasses import dataclass

from utils.singleton import Singleton

@dataclass
class Domain(metaclass=Singleton):
    X_DOMAIN: Tuple[float, float] = (0, 1.0)
    Y_DOMAIN: Tuple[float, float] = (0, 1.0)
    T_DOMAIN: Tuple[float, float] = (0, 0.5)
    X_POINTS: int = 30
    Y_POINTS: int = 30
    T_POINTS: int = 30
