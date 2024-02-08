from typing import Callable
from environment.env import SimulationEnvironment
from environment.mesh_env import MeshEnvironment
from environment.simple_env import SimpleEnvironment
from train.params import SimulationParameters
from visualization.plotting import plot_color, plot_3D
import matplotlib.pyplot as plt
import os
import sys
import torch

# sample initial cond
def initial_condition(x: torch.Tensor, y: torch.Tensor, xy_length: float) -> torch.Tensor:
    base_height = 0.5
    alpha = 120
    r = torch.sqrt((x-xy_length/5)**2 + (y-xy_length/2)**2)
    res = 0.4 * torch.exp(-(r)**2 * alpha) + base_height
    return res

def plot_initial(environment: SimulationEnvironment,
                          initial_condition: Callable,
                          mesh: str = None) -> None:
    title = "Initial condition"
    n_points_plot = environment.domain.N_POINTS_PLOT
    length = environment.domain.XY_DOMAIN[1]
    
    x, y, t = environment.get_initial_points(n_points_plot, requires_grad=False)

    z = initial_condition(x, y, length)
    
    fig1 = plot_color(z, x, y, n_points_plot, f"{title}")
    plt.show()
    fig2 = plot_3D(z, x, y, n_points_plot, length, mesh, f"{title}", limit=1)
    plt.show()

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mesh = None if len(sys.argv) <= 1 else os.path.join("data", f"val_square_UTM_translated_{sys.argv[1]}.inp")
    params = SimulationParameters(MESH=mesh)
    environment = MeshEnvironment(params.MESH, device) if params.MESH else SimpleEnvironment(device)

    plot_initial(environment, initial_condition, mesh)

    
   