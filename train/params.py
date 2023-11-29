from dataclasses import dataclass
import os


@dataclass
class SimulationParameters:
    # Model params
    LAYERS: int = 12
    NEURONS_PER_LAYER: int = 100

    # Training params
    RUN_NUM: int = 0
    EPOCHS: int = 200
    LEARNING_RATE: float = 0.00015

    SAVE_BEST_CLB = True
    VISUALIZE = True
    REPORT = True
    CLIP_GRAD = False

    # Simulation params
    GRAVITY: float = 9.81
    MESH = os.path.join("data", "val_square_UTM_translated_1.inp")

