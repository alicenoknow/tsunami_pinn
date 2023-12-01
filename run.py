import torch
import os

from conditions.initial import initial_condition
from environment.mesh_env import MeshEnvironment
from loss.loss import Loss
from loss.wave_equations import wave_equation
from model.pinn import PINN
from model.weights import Weights
from train.params import SimulationParameters
from train.training import Training
from environment.simple_env import SimpleEnvironment


if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: ", device)

    params = SimulationParameters(RUN_NUM=5, EPOCHS=50_000, LAYERS=6, NEURONS_PER_LAYER=120, MESH=os.path.join("data", "val_square_UTM_translated_5.inp"))
    weights = Weights()
    environment = MeshEnvironment(params.MESH, device) if params.MESH else SimpleEnvironment(device)
    pinn = PINN(params.LAYERS, params.NEURONS_PER_LAYER, device).to(device)

    loss = Loss(
        environment,
        weights,
        params,
        initial_condition,
        wave_equation,
    )

    training = Training(pinn, loss, params, environment, weights)
    _, loss, loss_r, loss_i, loss_b = training.start()
