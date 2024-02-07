import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from matplotlib.animation import FuncAnimation
from mayavi import mlab
from typing import Callable, Optional
from environment.env import SimulationEnvironment
from environment.mesh_utils import dump_points

from model.pinn import PINN

use_mayavi = True

def create_gif(save_path: str,
               run: int,
               total_time: float, 
               step: float=0.01, 
               duration: float=0.1) -> None:
    time_values = np.arange(0, total_time, step)
    frames = []
    for idx in range(len(time_values)):
        image = imageio.v2.imread(os.path.join(save_path, f"run_{run}", "img", "img_{:03d}.png".format(idx)))
        frames.append(image)

    imageio.mimsave(os.path.join(save_path, f"run_{run}", f"tsunami_{run}.gif"), frames, duration=duration)


def plot_initial_condition(save_path: str,
                          environment: SimulationEnvironment,
                          pinn: PINN,
                          initial_condition: Callable,
                          run_num: int,
                          mesh: str = None) -> None:
    
    title = "Initial condition"
    n_points_plot = environment.domain.N_POINTS_PLOT
    length = environment.domain.XY_DOMAIN[1]
    
    x, y, t = environment.get_initial_points(n_points_plot, requires_grad=False)

    z = initial_condition(x, y, length)
    
    fig1 = plot_color(z, x, y, n_points_plot, f"{title} - exact")
    fig2 = plot_3D(z, x, y, n_points_plot, length, mesh, f"{title} - exact")
    
    z = pinn(x, y, t)
    
    fig3 = plot_color(z, x, y, n_points_plot, f"{title} - PINN")    
    fig4 = plot_3D(z, x, y, n_points_plot, length, mesh, f"{title} - PINN")
    
    c1 = fig1.canvas
    c2 = fig2.canvas
    c3 = fig3.canvas
    c4 = fig4.canvas

    c1.draw()
    c2.draw()
    c3.draw()
    c4.draw()

    a1 = np.array(c1.buffer_rgba())
    a2 = np.array(c2.buffer_rgba())
    a3 = np.array(c3.buffer_rgba())
    a4 = np.array(c4.buffer_rgba())
    a12 = np.vstack((a1,a2))
    a34 = np.vstack((a3, a4))
    a = np.hstack((a12, a34))

    fig,ax = plt.subplots(figsize=(100, 100), dpi=100)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()
    ax.matshow(a)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"run_{run_num}", "initial.png"))


def plot_color(z: torch.Tensor, x: torch.Tensor, y: torch.Tensor, n_points_plot: int, title, figsize=(8, 6), dpi=100, cmap="viridis"):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    z_raw = z.detach().cpu().numpy()
    x_raw = x.detach().cpu().numpy()
    y_raw = y.detach().cpu().numpy()
    X = x_raw.reshape(n_points_plot, n_points_plot)
    Y = y_raw.reshape(n_points_plot, n_points_plot)
    Z = z_raw.reshape(n_points_plot, n_points_plot)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    c = ax.pcolormesh(X, Y, Z, cmap=cmap)
    fig.colorbar(c, ax=ax)

    return fig

def plot_3D(z: torch.Tensor, x: torch.Tensor, y: torch.Tensor, n_points_plot: int, length: int, mesh_file: Optional[str], title: str, figsize=(8, 6), limit=1):  
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    z_raw = z.detach().cpu().numpy()
    x_raw = x.detach().cpu().numpy()
    y_raw = y.detach().cpu().numpy()
    X = x_raw.reshape(n_points_plot, n_points_plot)
    Y = y_raw.reshape(n_points_plot, n_points_plot)
    Z = z_raw.reshape(n_points_plot, n_points_plot)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axes.set_zlim3d(bottom=0, top=limit)
    ax.plot_surface(X, Y, Z, alpha=0.8)

    if mesh_file is not None:
        # based on mesh file
        x_floor, y_floor, z_floor = dump_points(mesh_file)
        ax.plot_trisurf(x_floor, y_floor, z_floor, linewidth=0.2, alpha=0.8)

    else:
        # based on floor function
        x_floor = torch.linspace(0.0, length, steps=n_points_plot)
        y_floor = torch.linspace(0.0, length, steps=n_points_plot)
        z_floor = torch.zeros((n_points_plot, n_points_plot))
        # for x_idx, _ in enumerate(x_floor):
        #     for y_idx, _ in enumerate(y_floor):
        #         z_floor[x_idx, y_idx] = 0
        x_floor = torch.tile(x_floor, (n_points_plot, 1))
        y_floor = torch.tile(y_floor, (n_points_plot, 1)).T
        ax.plot_surface(np.array(x_floor), np.array(y_floor), np.array(z_floor), color='green', alpha=0.8)

    return fig

def plot_3D_mayavi(z: torch.Tensor, x: torch.Tensor, y: torch.Tensor, n_points_plot: int, length: int, mesh_file: str, title: str, limit=1):
    fig = mlab.figure(figure=None, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), engine=None, size=(700, 700))
    
    z_raw = z.detach().cpu().numpy()
    x_raw = x.detach().cpu().numpy()
    y_raw = y.detach().cpu().numpy()
    X = x_raw.reshape(n_points_plot, n_points_plot)
    Y = y_raw.reshape(n_points_plot, n_points_plot)
    Z = z_raw.reshape(n_points_plot, n_points_plot)
    
    mlab.surf(X, Y, Z, color=(0.01, 0.27, 0.92), opacity=0.8, warp_scale=0.5)

    if mesh_file is not None:
        # based on mesh file
        import meshio
        mesh = meshio.avsucd.read(mesh_file)
        vx, vy, vz = dump_points(mesh_file)
        triangles = mesh.cells_dict['triangle']
        mlab.triangular_mesh(vx, vy, vz, triangles, color=(0.88,0.75,0.05),)
        mlab.triangular_mesh(vx, vy, vz, triangles, representation='wireframe')

    else:
        # based on floor function
        x_floor = torch.linspace(0.0, length, steps=n_points_plot)
        y_floor = torch.linspace(0.0, length, steps=n_points_plot)
        z_floor = torch.zeros((n_points_plot, n_points_plot))
        mlab.surf(x_floor, y_floor, z_floor, color=(1, 0.8, 0), opacity=0.8)

    # mlab.title(title)
    mlab.xlabel("x")
    mlab.ylabel("y")
    mlab.zlabel("z")
    mlab.axes(color=(0, 0, 0), extent=[0, 1, 0, 1, 0, 3])
    mlab.view(distance=7)
    return fig

def plot_frame(save_path: str,
                environment: SimulationEnvironment,
                pinn: PINN,
                run_num: int,
                idx: int,
                t_value: float,
                mesh: str = None) -> None:
    
    n_points_plot = environment.domain.N_POINTS_PLOT
    length = environment.domain.XY_DOMAIN[1]
    x, y, t = environment.get_initial_points(n_points_plot, requires_grad=False)

    t = torch.full_like(x, t_value)
    z = pinn(x, y, t)
    fig1 = plot_color(z, x, y, n_points_plot, f"PINN for t = {t_value}")
    fig2 = plot_3D(z, x, y, n_points_plot, length, mesh, f"PINN for t = {t_value}")
    plt.savefig(os.path.join(save_path, f"run_{run_num}", "img", "img_{:03d}.png".format(idx)))
    # mlab.savefig(os.path.join(save_path, f"run_{run_num}", "img", "img_{:03d}.png".format(idx)))
    # plt.show()
    plt.close(fig1)
    plt.close(fig2)

def plot_simulation_by_frame(save_path: str,
                             pinn: PINN, 
                             environment: SimulationEnvironment,
                             run_num: int,
                             time_step:float=0.01,
                             mesh: str = None) -> None:
    t_max = environment.domain.T_DOMAIN[1]    
    time_values = np.arange(0, t_max, time_step)

    for idx, t_value in enumerate(time_values):
        plot_frame(save_path=save_path,
                   environment=environment,
                   pinn=pinn,
                   run_num=run_num,
                   idx=idx,
                   t_value=t_value,
                   mesh=mesh)

def running_average(y, window: int=100):
    cumsum = np.cumsum(np.insert(y, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)

def plot_running_average(save_path: str, loss_values, title: str, path: str, run_num: int):
    average_loss = running_average(loss_values, window=100)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.plot(average_loss)
    ax.set_yscale('log')
    
    fig.savefig(os.path.join(save_path, f"run_{run_num}", f"{path}.png"))
