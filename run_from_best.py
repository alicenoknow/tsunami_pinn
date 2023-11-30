import torch

from conditions.initial import initial_condition
from environment.mesh_env import MeshEnvironment
from loss.loss import Loss
from loss.wave_equations import wave_equation_simplified
from model.pinn import PINN
from model.weights import Weights
from train.params import SimulationParameters
from train.training import Training
from environment.simple_env import SimpleEnvironment

import os


if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: ", device)

    params = SimulationParameters(EPOCHS=1, RUN_NUM=1, SAVE_BEST_CLB=False)
    weights = Weights()
    environment = MeshEnvironment(params.MESH, device) if params.MESH else SimpleEnvironment(device)

    model = torch.load(os.path.join(f"results", f"run_{params.RUN_NUM}", f"best_{params.RUN_NUM}.pt"))


    loss = Loss(
        environment,
        weights,
        params,
        initial_condition,
        wave_equation_simplified,
    )

    losses = loss.verbose(model)

    training = Training(model, loss, params, environment, weights)
    _, loss, loss_r, loss_i, loss_b = training.start()