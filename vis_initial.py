import os
import sys
import torch
import matplotlib.pyplot as plt

from typing import Callable

from conditions.initial import make_initial_condition
from environment.env import SimulationEnvironment
from environment.mesh_env import MeshEnvironment
from environment.simple_env import SimpleEnvironment
from train.params import SimulationParameters
from visualization.plotting import plot_color, plot_3D, plot_3D_top_view


def plot_initial(environment: SimulationEnvironment,
                 initial_condition: Callable) -> None:
    title = "Initial condition"
    n_points_plot = environment.domain.N_POINTS_PLOT
    length = environment.domain.XY_DOMAIN[1]

    x, y, _ = environment.get_initial_points(n_points_plot, requires_grad=False)
    z = initial_condition(x, y, length)

    plot_color(z, x, y, n_points_plot, title)
    plt.show()

    plot_3D(z, x, y, n_points_plot, length, environment, title)
    plt.show()

    fig = plot_3D_top_view(z, x, y, n_points_plot, environment, title)
    fig.show()


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mesh = None if len(sys.argv) <= 1 else os.path.join(
        "data", f"val_square_UTM_translated_{sys.argv[1]}.inp")
    params = SimulationParameters(MESH=mesh)
    environment = MeshEnvironment(params.MESH, device) if params.MESH else SimpleEnvironment(device)
    initial_condition = make_initial_condition(
        params.BASE_HEIGHT,
        params.DECAY_RATE,
        params.PEAK_HEIGHT,
        params.X_DIVISOR,
        params.Y_DIVISOR)

    plot_initial(environment, initial_condition)
