import torch
import os

from conditions.initial import initial_condition
from environment.mesh_env import MeshEnvironment
from loss.relo_loss import ReloLoss
from loss.wave_equations import wave_equation, wave_equation_simplified
from model.pinn import PINN
from model.weights import Weights
from train.params import SimulationParameters
from train.training import Training
from environment.simple_env import SimpleEnvironment

# TODO evaluate model from file and visualize
# TODO train script


if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: ", device)
    run_num = 33

    for _ in range(0,1):

        # params = SimulationParameters(RUN_NUM=run_num+40, EPOCHS=150_000, NEURONS_PER_LAYER=300, LAYERS=3,  MESH=os.path.join("data", f"val_square_UTM_translated_{run_num}.inp"))
        params = SimulationParameters(RUN_NUM=run_num, EPOCHS=450_000, NEURONS_PER_LAYER=300, LAYERS=4, MESH=None)

        weights = Weights()
        environment = MeshEnvironment(params.MESH, device) if params.MESH else SimpleEnvironment(device)
        pinn = PINN(params.LAYERS, params.NEURONS_PER_LAYER, device).to(device)
        # pinn = torch.load(os.path.join(f"results", f"run_{run_num}", f"best_{run_num}.pt"))

        loss = ReloLoss(
            environment,
            weights,
            params,
            initial_condition,
            wave_equation=wave_equation_simplified if params.MESH is None else wave_equation,
        )

        training = Training(pinn, loss, params, environment, weights)
        _, loss, loss_r, loss_i, loss_b = training.start()
