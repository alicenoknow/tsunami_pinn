import imageio
import jinja2
import matplotlib.pyplot as plt
import numpy as np
import torch

from datetime import datetime
from matplotlib.animation import FuncAnimation
from typing import List, TypedDict, Callable
from xhtml2pdf import pisa


class ReportContext(TypedDict):
    num: int
    date: datetime
    WEIGHT_RESIDUAL: float 
    WEIGHT_INITIAL: float 
    WEIGHT_BOUNDARY: float
    LAYERS: int
    NEURONS_PER_LAYER: int
    EPOCHS: int 
    LEARNING_RATE: float
    total_loss: float
    residual_loss: float
    initial_loss: float
    boundary_loss: float
    img1: str
    img2: str
    img3: str
    img4: str


def create_report(context: ReportContext, 
                  env_path: str, 
                  template_path: str, 
                  report_title: str) -> None:
    template_loader = jinja2.FileSystemLoader(env_path)
    template_env = jinja2.Environment(loader=template_loader)

    template = template_env.get_template(template_path)
    output_text = template.render(context)
    
    with open(f'./results/{report_title}', "w+b") as out_pdf_file_handle:
        pisa.CreatePDF(src=output_text, dest=out_pdf_file_handle)
        
def create_gif(total_time: float, 
               title: str, 
               step: float=0.01, 
               base_dir: str=".", 
               duration: float=0.1) -> None:
    time_values = np.arange(0, total_time, step)
    frames = []
    for idx in range(len(time_values)):
        image = imageio.v2.imread(base_dir + '/img/img_{:03d}.png'.format(idx))
        frames.append(image)

    imageio.mimsave(f'{base_dir}/results/{title}.gif', frames, duration=duration)

def get_initial_points(x_domain: List[float], 
                       y_domain: List[float], 
                       t_domain: List[float], 
                       n_points: int, 
                       device=torch.device("cpu"), 
                       requires_grad=True):
    x_linspace = torch.linspace(x_domain[0], x_domain[1], n_points)
    y_linspace = torch.linspace(y_domain[0], y_domain[1], n_points)
    x_grid, y_grid = torch.meshgrid( x_linspace, y_linspace, indexing="ij")
    x_grid = x_grid.reshape(-1, 1).to(device)
    x_grid.requires_grad = requires_grad
    y_grid = y_grid.reshape(-1, 1).to(device)
    y_grid.requires_grad = requires_grad
    t0 = torch.full_like(x_grid, t_domain[0], requires_grad=requires_grad)
    return (x_grid, y_grid, t0)

def plot_intial_condition(x_domain: List[float], 
                          y_domain: List[float], 
                          t_domain: List[float],
                          length: float,
                          pinn: 'PINN',
                          initial_condition: Callable,
                          floor: Callable,
                          n_points: int) -> None:
    title = "Initial condition"
    
    x, y, _ = get_initial_points(x_domain, y_domain, t_domain, n_points, requires_grad=False)
    z = initial_condition(x, y)
    
    fig = plot_color(z, x, y, n_points, n_points, f"{title} - exact")
    fig = plot_3D(z, x, y, n_points, n_points, length, floor, f"{title} - exact", n_points)
    
    t_value = 0.0
    t = torch.full_like(x, t_value)
    z = pinn(x, y, t)
    
    fig = plot_color(z, x, y, n_points, n_points, f"{title} - PINN")    
    fig = plot_3D(z, x, y, n_points, n_points, length, floor, f"{title} - PINN", n_points)
    

def plot_frame(x_domain: List[float], 
               y_domain: List[float], 
               t_domain: List[float], 
               pinn: 'PINN', 
               idx: int, 
               t_value: float, 
               n_points: int,
               length: float,
               floor: Callable,
               base_dir: str=".") -> None:
    x, y, _ = get_initial_points(x_domain, y_domain, t_domain, n_points, requires_grad=False)
    t = torch.full_like(x, t_value)
    z = pinn(x, y, t)
    fig = plot_color(z, x, y, n_points, n_points, f"PINN for t = {t_value}")
    fig = plot_3D(z, x, y, n_points, n_points, length, floor, f"PINN for t = {t_value}", n_points)
    plt.savefig(base_dir + '/img/img_{:03d}.png'.format(idx))


def plot_simulation_by_frame(total_time: float, 
                             x_domain: List[float], 
                             y_domain: List[float], 
                             t_domain: List[float], 
                             pinn: 'PINN', 
                             n_points: int,
                             length: float,
                             floor: Callable,
                             step:float=0.01) -> None:
    time_values = np.arange(0, total_time, step)

    for idx, t_value in enumerate(time_values):
        plot_frame(x_domain=x_domain, 
                   y_domain=y_domain, 
                   t_domain=t_domain,
                   pinn=pinn,
                   idx=idx,
                   t_value=t_value,
                   n_points=n_points,
                   length=length,
                   floor=floor)

def running_average(y, window: int=100):
    cumsum = np.cumsum(np.insert(y, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)

def plot_running_average(loss_values, title: str, path: str):
    average_loss = running_average(loss_values, window=100)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.plot(average_loss)
    ax.set_yscale('log')
    
    fig.savefig(f'./results/{path}.png')

def plot_solution(pinn: 'PINN', x: torch.Tensor, t: torch.Tensor, figsize=(8, 6), dpi=100):

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x_raw = torch.unique(x).reshape(-1, 1)
    t_raw = torch.unique(t)

    def animate(i):
        if not i % 10 == 0:
            t_partial = torch.ones_like(x_raw) * t_raw[i]
            f_final = f(pinn, x_raw, t_partial)
            ax.clear()
            ax.plot(
                x_raw.detach().numpy(), f_final.detach().numpy(), label=f"Time {float(t[i])}"
            )
            ax.set_ylim(-1, 1)
            ax.legend()

    n_frames = t_raw.shape[0]
    return FuncAnimation(fig, animate, frames=n_frames, interval=100, repeat=False)

def plot_color(z: torch.Tensor, x: torch.Tensor, y: torch.Tensor, n_points_x, n_points_t, title, figsize=(8, 6), dpi=100, cmap="viridis"):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    z_raw = z.detach().cpu().numpy()
    x_raw = x.detach().cpu().numpy()
    y_raw = y.detach().cpu().numpy()
    X = x_raw.reshape(n_points_x, n_points_t)
    Y = y_raw.reshape(n_points_x, n_points_t)
    Z = z_raw.reshape(n_points_x, n_points_t)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    c = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=0, vmax=3)
    fig.colorbar(c, ax=ax)

    return fig

def plot_3D(z: torch.Tensor, x: torch.Tensor, y: torch.Tensor, n_points_x, n_points_t, length: float, floor: Callable, title: str, n_points_plot: int, figsize=(8, 6), dpi=100, limit=3):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    z_raw = z.detach().cpu().numpy()
    x_raw = x.detach().cpu().numpy()
    y_raw = y.detach().cpu().numpy()
    X = x_raw.reshape(n_points_x, n_points_t)
    Y = y_raw.reshape(n_points_x, n_points_t)
    Z = z_raw.reshape(n_points_x, n_points_t)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axes.set_zlim3d(bottom=0, top=limit)

    c = ax.plot_surface(X, Y, Z)

    x_floor = torch.linspace(0.0, length, n_points_plot)
    y_floor = torch.linspace(0.0, length, n_points_plot)
    z_floor = torch.zeros((n_points_plot, n_points_plot))
    for x_idx, x_coord in enumerate(x_floor):
        for y_idx, y_coord in enumerate(y_floor):
            z_floor[x_idx, y_idx] = floor(x_idx, y_idx)
    x_floor = torch.tile(x_floor, (n_points_plot, 1))
    y_floor = torch.tile(y_floor, (n_points_plot, 1)).T
    f = ax.plot_surface(x_floor, y_floor, z_floor, color='green', alpha=0.7)

    return fig