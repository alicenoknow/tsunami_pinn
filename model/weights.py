from dataclasses import dataclass

@dataclass
class Weights:
    WEIGHT_RESIDUAL: float = 0.05       # Weight of residual part of loss function
    WEIGHT_INITIAL: float = 2.5         # Weight of initial part of loss function
    WEIGHT_BOUNDARY: float = 0.0005     # Weight of boundary part of loss function