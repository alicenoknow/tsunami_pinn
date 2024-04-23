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
from loss.softadapt_loss import SoftAdaptLoss
from loss.wave_equations import wave_equation, wave_equation_simplified
from train.params import LossFunction, SimulationParameters
from visualization.plotting import plot_all

"""
Takes config file name.
Individual fields can be overridden by CLI arguments.
e.g. evaluate.py --config base.json --model_path ./best.pt --dir evaluation
"""

logger = logging.getLogger()


def setup_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='JSON config file name')

    for field in fields(SimulationParameters):
        parser.add_argument(f'--{field.name.lower()}', type=type(field.default))

    args = parser.parse_args()

    params.set_json(args.config)
    params.set_cli_args(args)


def setup_logger(params):
    log_format = '[%(levelname)s] %(message)s'
    file_handler = logging.FileHandler(os.path.join(params.DIR, "run.log"), "w")
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
    elif params.LOSS == LossFunction.SOFTADAPT:
        return SoftAdaptLoss(environment, initial_condition, wave_equation)
    elif params.LOSS == LossFunction.MIX:
        return SoftAdaptLoss(environment, initial_condition, wave_equation)
    return Loss(environment, initial_condition, wave_equation)


def visualize_results(params, model, environment, initial_condition):
    plot_all(params.DIR,
             model,
             environment,
             initial_condition)


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

    logger.info(f'Loading model: {params.MODEL_PATH})')
    pinn = torch.load(params.MODEL_PATH).cuda()

    loss = setup_loss(
        environment,
        params,
        initial_condition,
        wave_equation,
    )

    logger.info('Evaluating model')
    total_loss, initial_loss, residual_loss, boundary_loss = loss.verbose(pinn)

    logger.info(f'Total loss: \t{total_loss:.5f} ({total_loss:.3E})')
    logger.info(f'Interior loss: \t{initial_loss:.5f} ({initial_loss:.3E})')
    logger.info(f'Initial loss: \t{residual_loss:.5f} ({residual_loss:.3E})')
    logger.info(f'Boundary loss: \t{boundary_loss:.5f} ({boundary_loss:.3E})')

    logger.info('Visualizing results')
    visualize_results(params, pinn, environment, initial_condition, )


if __name__ == "__main__":
    run()
