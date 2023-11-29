import torch

from conditions.initial import initial_condition
from environment.mesh_env import MeshEnvironment
from loss.loss import Loss
from loss.wave_equations import wave_equation_simplified
from model.pinn import PINN
from model.weights import Weights
from train.params import SimulationParameters
from train.training import Training
from environment.domain import Domain
from environment.simple_env import SimpleEnvironment


if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: ", device)

    params = SimulationParameters(EPOCHS=4)
    weights = Weights()
    pinn = PINN(params.LAYERS, params.NEURONS_PER_LAYER, device).to(device)
    environment = MeshEnvironment(params.MESH, device) if params.MESH else SimpleEnvironment(device)


    loss = Loss(
        environment,
        Domain(),
        weights,
        params,
        initial_condition,
        wave_equation_simplified,
    )

    training = Training(pinn, loss, params, environment, weights)
    _, loss, loss_r, loss_i, loss_b = training.start()