import argparse
from dataclasses import fields
import logging
import sys
import os
import torch

from conditions.initial import make_initial_condition
from environment.mesh_env import MeshEnvironment
from environment.simple_env import SimpleEnvironment
from loss.loss import Loss
from loss.relo_loss import ReloLoss
from loss.wave_equations import wave_equation, wave_equation_simplified
from model.pinn import PINN
from train.params import LossFunction, SimulationParameters
from train.training import Training

"""
Takes config file name.
Individual fields can be overridden by CLI arguments.
e.g. run.py --config base.json --layers 8
"""

logger = logging.getLogger()


def setup_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='JSON config file name')

    for field in fields(SimulationParameters):
        if type(field.default) == bool:
            parser.add_argument(f'--{field.name.lower()}',
                                dest=field.name.lower(), action='store_true')
            parser.add_argument(f'--no-{field.name.lower()}',
                                dest=field.name.lower(), action='store_false')
            parser.set_defaults(visualize=True)
        else:
            parser.add_argument(f'--{field.name.lower()}', type=type(field.default))

    args = parser.parse_args()

    params.set_json(args.config)
    params.set_cli_args(args)
    params.save_params()


def setup_logger(params):
    log_format = '[%(levelname)s] %(message)s'
    log_dir = os.path.join(params.DIR, f"run_{params.RUN_NUM}")
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(
        os.path.join(
            log_dir,
            "run.log"),
        "w+")
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)  # Set the logger's level to the lowest level (DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info("Logger set up successfully")


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")
    return device


def setup_environment(params, device):
    if params.MESH:
        logger.info(f"Simulation environment: {params.MESH}")
        return MeshEnvironment(params.MESH, device), wave_equation

    logger.info("Simulation environment: no mesh")
    return SimpleEnvironment(device), wave_equation_simplified


def setup_loss(environment,
               params,
               initial_condition,
               wave_equation,):
    logger.info(f"Loss: {params.LOSS}")

    if params.LOSS == LossFunction.RELO:
        return ReloLoss(environment, initial_condition, wave_equation)
    return Loss(environment, initial_condition, wave_equation)


def run():
    params = SimulationParameters()
    setup_params(params)
    setup_logger(params)
    device = setup_device()

    environment, wave_equation = setup_environment(params, device)
    initial_condition = make_initial_condition(params.BASE_HEIGHT,
                                               params.DECAY_RATE,
                                               params.PEAK_HEIGHT,
                                               params.X_DIVISOR,
                                               params.Y_DIVISOR)

    pinn = PINN(params.LAYERS, params.NEURONS_PER_LAYER, device).to(device)
    loss = setup_loss(
        environment,
        params,
        initial_condition,
        wave_equation,
    )

    training = Training(pinn, loss, environment, initial_condition)
    training.start()


if __name__ == "__main__":
    run()
