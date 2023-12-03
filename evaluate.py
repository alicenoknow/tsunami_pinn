import torch
import os
import sys

from conditions.initial import initial_condition
from environment.mesh_env import MeshEnvironment
from loss.loss import Loss
from loss.wave_equations import wave_equation, wave_equation_simplified
from model.pinn import PINN
from model.weights import Weights
from train.params import SimulationParameters
from train.training import Training
from environment.simple_env import SimpleEnvironment



if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: ", device)

    run_num = sys.argv[1]
    model_path = sys.argv[2]
    mesh_path = sys.argv[3]
    results_dir = sys.argv[4]

    params = SimulationParameters(RUN_NUM=run_num, MESH=mesh_path, DIR=results_dir, EPOCHS=0)
    weights = Weights()
    environment = MeshEnvironment(params.MESH, device) if params.MESH else SimpleEnvironment(device)
    pinn = torch.load(model_path).cuda()

    loss = Loss(
        environment,
        weights,
        params,
        initial_condition,
        wave_equation=wave_equation if params.MESH else wave_equation_simplified,
    )

    training = Training(pinn, loss, params, environment, weights)
    training.evaluate()

