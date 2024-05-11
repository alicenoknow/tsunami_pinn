from dataclasses import dataclass, fields
from enum import Enum
import json
import os


class LossFunction(Enum):
    BASE = "BASE"
    MIX = "MIX"
    RELO = "RELO"
    SOFTADAPT = "SOFTADAPT"


@dataclass
class SimulationParameters:
    _shared_state = {}

    # Model
    LAYERS: int = 12
    NEURONS_PER_LAYER: int = 100
    LOSS: LossFunction = LossFunction.BASE
    MODEL_PATH: str = ""

    # Training
    RUN_NUM: int = 0
    EPOCHS: int = 200
    LEARNING_RATE: float = 0.00015
    SAVE_BEST_CLB: bool = True
    VISUALIZE: bool = True
    REPORT: bool = True
    CLIP_GRAD: bool = False

    # Simulation
    GRAVITY: float = 9.81
    MESH: str = ""
    DIR: str = "results"

    # Initial condition
    BASE_HEIGHT: float = 0.0  # Base water level
    DECAY_RATE: float = 120  # The rate of decay, how quickly func decreases with distance
    PEAK_HEIGHT: float = 0.4  # The height of the function's peak
    X_DIVISOR: float = 5  # The divisor used to calculate the x-coord of the center of the function
    Y_DIVISOR: float = 2  # The divisor used to calculate the y-coord of the center of the function

    # Initial weights
    INITIAL_WEIGHT_RESIDUAL: float = 0.5       # Weight of residual part of loss function
    INITIAL_WEIGHT_INITIAL: float = 30         # Weight of initial part of loss function
    INITIAL_WEIGHT_BOUNDARY: float = 0.05     # Weight of boundary part of loss function

    def __init__(self, **kwargs):
        self.__dict__ = self._shared_state

        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_json(self, json_file):
        if json_file is None or not os.path.isfile(json_file):
            return

        with open(json_file, 'r') as f:
            config = json.load(f)

        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def set_cli_args(self, args):
        for field in fields(self):
            arg_value = getattr(args, field.name.lower())
            if arg_value is not None:
                setattr(self, field.name, arg_value)

    def save_params(self):
        file_path = os.path.join(self.DIR, f"run_{self.RUN_NUM}", "config.json")
        params_to_save = {field.name: getattr(self, field.name) for field in fields(self)}
        with open(file_path, 'w') as json_file:
            json.dump(params_to_save, json_file, indent=4)
