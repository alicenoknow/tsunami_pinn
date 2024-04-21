import torch
import os

from conditions.initial import make_initial_condition
from environment.mesh_env import MeshEnvironment
from loss.loss import Loss
from loss.wave_equations import wave_equation, wave_equation_simplified
from model.pinn import PINN
from train.params import SimulationParameters
from train.training import Training
from environment.simple_env import SimpleEnvironment


if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: ", device)

    for run_num in range(7, 11):
        
        mesh = os.path.join("data", f"val_square_UTM_translated_{run_num}.inp")


        # params = SimulationParameters(RUN_NUM=run_num, EPOCHS=150_000, NEURONS_PER_LAYER=300, LAYERS=3,  MESH=os.path.join("data", f"val_square_UTM_translated_{run_num}.inp"))
        params = SimulationParameters(RUN_NUM=run_num+70, EPOCHS=150_000, NEURONS_PER_LAYER=300, LAYERS=3, MESH=mesh, BASE_HEIGHT=123)

        environment = MeshEnvironment(params.MESH, device) if params.MESH else SimpleEnvironment(device)
        pinn = PINN(params.LAYERS, params.NEURONS_PER_LAYER, device).to(device)
        # pinn = torch.load(os.path.join(f"results", f"run_{run_num}", f"best_{run_num}.pt"))
        initial_condition = make_initial_condition(params.BASE_HEIGHT, params.DECAY_RATE, params.PEAK_HEIGHT, params.X_DIVISOR, params.Y_DIVISOR)

        loss = Loss(
            environment,
            initial_condition,
            wave_equation=wave_equation_simplified if params.MESH is None else wave_equation,
        )

        training = Training(pinn, loss, params, environment)
        _, loss, loss_r, loss_i, loss_b = training.start()
