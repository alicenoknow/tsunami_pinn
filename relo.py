## Pawel Maczuga and Maciej Paszynski 2023

from typing import Callable, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_NUM = 43

LENGTH = 1. # Domain size in x axis. Always starts at 0
TOTAL_TIME = .25 # Domain size in t axis. Always starts at 0
N_POINTS = 20 # Number of in single asxis
N_POINTS_PLOT = 128 # Number of points in single axis used in plotting

WEIGHT_RESIDUAL = 0.01 #0.03 # Weight of residual part of loss function
WEIGHT_INITIAL = 1800.0 # 1.0 # Weight of initial part of loss function
WEIGHT_BOUNDARY = 0.01 # 0.0005 # Weight of boundary part of loss function
GRAVITY = 9.81

LAYERS = 4
NEURONS_PER_LAYER = 150
EPOCHS = 100_000
LEARNING_RATE = 0.001

DIR = "./results/flat/relo"

NAME="relo_1800_t05"

class PINN(nn.Module):
    """Simple neural network accepting two features as input and returning a single output

    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh()):

        super().__init__()

        self.layer_in = nn.Linear(3, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 1)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act

    def forward(self, x, y, t):

        x_stack = torch.cat([x, y, t], dim=1)
        out = self.act(self.layer_in(x_stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        logits = self.layer_out(out)

        return logits

    def device(self):
        return next(self.parameters()).device


def f(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""
    return pinn(x, y, t)


def df(output: torch.Tensor, input: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""
    df_value = output
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            input,
            grad_outputs=torch.ones_like(input),
            create_graph=True,
            retain_graph=True,
        )[0]

    return df_value


def dfdt(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, order: int = 1):
    f_value = f(pinn, x, y, t)
    return df(f_value, t, order=order)


def dfdx(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, order: int = 1):
    f_value = f(pinn, x, y, t)
    return df(f_value, x, order=order)

def dfdy(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, order: int = 1):
    f_value = f(pinn, x, y, t)
    return df(f_value, y, order=order)


def get_boundary_points(x_domain, y_domain, t_domain, n_points, device = torch.device("cpu"), requires_grad=True):
    """
         .+------+
       .' |    .'|
      +---+--+'  |
      |   |  |   |
    y |  ,+--+---+
      |.'    | .' t
      +------+'
         x
    """
    x_linspace = torch.linspace(x_domain[0], x_domain[1], n_points)
    y_linspace = torch.linspace(y_domain[0], y_domain[1], n_points)
    t_linspace = torch.linspace(t_domain[0], t_domain[1], n_points)

    x_grid, t_grid = torch.meshgrid( x_linspace, t_linspace, indexing="ij")
    y_grid, _      = torch.meshgrid( y_linspace, t_linspace, indexing="ij")

    x_grid = x_grid.reshape(-1, 1).to(device)
    x_grid.requires_grad = requires_grad
    y_grid = y_grid.reshape(-1, 1).to(device)
    y_grid.requires_grad = requires_grad
    t_grid = t_grid.reshape(-1, 1).to(device)
    t_grid.requires_grad = requires_grad

    x0 = torch.full_like(t_grid, x_domain[0], requires_grad=requires_grad)
    x1 = torch.full_like(t_grid, x_domain[1], requires_grad=requires_grad)
    y0 = torch.full_like(t_grid, y_domain[0], requires_grad=requires_grad)
    y1 = torch.full_like(t_grid, y_domain[1], requires_grad=requires_grad)

    down    = (x_grid, y0,     t_grid)
    up      = (x_grid, y1,     t_grid)
    left    = (x0,     y_grid, t_grid)
    right   = (x1,     y_grid, t_grid)

    return down, up, left, right


def get_initial_points(x_domain, y_domain, t_domain, n_points, device = torch.device("cpu"), requires_grad=True):
    x_linspace = torch.linspace(x_domain[0], x_domain[1], n_points)
    y_linspace = torch.linspace(y_domain[0], y_domain[1], n_points)
    x_grid, y_grid = torch.meshgrid( x_linspace, y_linspace, indexing="ij")
    x_grid = x_grid.reshape(-1, 1).to(device)
    x_grid.requires_grad = requires_grad
    y_grid = y_grid.reshape(-1, 1).to(device)
    y_grid.requires_grad = requires_grad
    t0 = torch.full_like(x_grid, t_domain[0], requires_grad=requires_grad)
    return (x_grid, y_grid, t0)

def get_interior_points(x_domain, y_domain, t_domain, n_points, device = torch.device("cpu"), requires_grad=True):
    t_step = t_domain[1] / n_points
    x_raw = torch.linspace(x_domain[0], x_domain[1], steps=n_points, requires_grad=requires_grad)
    y_raw = torch.linspace(y_domain[0], y_domain[1], steps=n_points, requires_grad=requires_grad)
    t_raw = torch.linspace(t_domain[0]+t_step, t_domain[1], steps=n_points, requires_grad=requires_grad)
    grids = torch.meshgrid(x_raw, y_raw, t_raw, indexing="ij")

    #W tym miejscu wczytujemy punkty z pliku siatki
    # Czy moglbys napisac tutaj wczytywanie punktow z pliku siatki
    x = grids[0].reshape(-1, 1).to(device)
    y = grids[1].reshape(-1, 1).to(device)
    t = grids[2].reshape(-1, 1).to(device)

    return x, y, t


import torch.nn.functional as F

class Loss:
    def __init__(
        self,
        x_domain: Tuple[float, float],
        y_domain: Tuple[float, float],
        t_domain: Tuple[float, float],
        n_points: int,
        initial_condition: Callable,
        floor: Callable,
        weight_r: float = 1.0,
        weight_b: float = 1.0,
        weight_i: float = 1.0,
        verbose: bool = False,
    ):
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.t_domain = t_domain
        self.n_points = n_points
        self.initial_condition = initial_condition
        self.floor = floor
        self.weight_r = weight_r
        self.weight_b = weight_b
        self.weight_i = weight_i

        self.epoch = 0
        self.alpha = 0.99
        self.temperature = 0.5
        self.rho = 0.9999
        self.call_count = torch.tensor(0, requires_grad=False, dtype=torch.int16)

        self.lambdas = [torch.tensor(self.weight_r, requires_grad=False),
                        torch.tensor(self.weight_r, requires_grad=False)]
        self.last_losses = [torch.tensor(1., requires_grad=False) for _ in range(2)]
        self.init_losses = [torch.tensor(1., requires_grad=False) for _ in range(2)]

        self.residual_history = []
        self.boundary_history = []


    def residual_loss(self, pinn: PINN):
        x, y, t = get_interior_points(self.x_domain, self.y_domain, self.t_domain, self.n_points, pinn.device())
        u = f(pinn, x, y, t)
        z = self.floor(x, y)
        loss = dfdt(pinn, x, y, t, order=2) - \
                      GRAVITY * ( dfdx(pinn, x, y, t) ** 2 + \
                      (u-z) * dfdx(pinn, x, y, t, order=2) + \
                      dfdy(pinn, x, y, t) ** 2 + \
                      (u-z) * dfdy(pinn, x, y, t, order=2)
                      )
        # casual_weights = torch.exp(-2 * t)
        # return (casual_weights * loss).pow(2).mean()
        return loss.pow(2).mean()


    def initial_loss(self, pinn: PINN):
        x, y, t = get_initial_points(self.x_domain, self.y_domain, self.t_domain, self.n_points*5, pinn.device())
        pinn_init = self.initial_condition(x, y)
        loss = f(pinn, x, y, t) - pinn_init
        return loss.pow(2).mean()

    def boundary_loss(self, pinn: PINN):
        down, up, left, right = get_boundary_points(self.x_domain, self.y_domain, self.t_domain, self.n_points*3, pinn.device())
        x_down,  y_down,  t_down    = down
        x_up,    y_up,    t_up      = up
        x_left,  y_left,  t_left    = left
        x_right, y_right, t_right   = right

        loss_down  = dfdy( pinn, x_down,  y_down,  t_down  )
        loss_up    = dfdy( pinn, x_up,    y_up,    t_up    )
        loss_left  = dfdx( pinn, x_left,  y_left,  t_left  )
        loss_right = dfdx( pinn, x_right, y_right, t_right )

        return loss_down.pow(2).mean()  + \
            loss_up.pow(2).mean()    + \
            loss_left.pow(2).mean()  + \
            loss_right.pow(2).mean()

    def verbose(self, pinn: PINN, epoch=0):
        """
        Returns all parts of the loss function

        Not used during training! Only for checking the results later.
        """
        residual_loss = self.residual_loss(pinn)
        initial_loss = self.initial_loss(pinn)
        boundary_loss = self.boundary_loss(pinn)

        if self.epoch < 0:
            loss = 0.0001 * residual_loss + 5 * initial_loss + 0.0001 * boundary_loss
        else:	

            losses = [residual_loss, boundary_loss]

            alpha = torch.where(self.call_count == 0,
                            torch.tensor(1.),
                            torch.where(self.call_count == 1,
                                        torch.tensor(0.),
                                        torch.tensor(self.alpha)))
            rho = torch.where(self.call_count == 0,
                          torch.tensor(1.),
                          torch.where(self.call_count == 1,
                                      torch.tensor(1.),
                                      (torch.rand(()) < self.rho).to(torch.float32)))

            EPS = 1e-5
            # compute new lambdas w.r.t. the losses in the previous iteration
            lambdas_hat = [losses[i] / (self.last_losses[i] * self.temperature + EPS)
                       for i in range(len(losses))]
            lambdas_hat = F.softmax(torch.tensor(lambdas_hat) -
                                torch.max(torch.tensor(lambdas_hat)), dim=0) * len(losses)
            # compute new lambdas w.r.t. the losses in the first iteration
            init_lambdas_hat = [losses[i] /
                            (self.init_losses[i] *
                             self.temperature +
                             EPS) for i in range(len(losses))]
            init_lambdas_hat = F.softmax(torch.tensor(init_lambdas_hat) -
                                     torch.max(torch.tensor(init_lambdas_hat)), dim=0) * len(losses)

            # use rho for deciding, whether a random look back should be performed
            new_lambdas = [(rho * alpha * self.lambdas[i]
                        + (1 - rho) * alpha * init_lambdas_hat[i]
                        + (1 - alpha) * lambdas_hat[i])
                       for i in range(len(losses))]
            self.lambdas = [lam.clone().detach().requires_grad_(False) for lam in new_lambdas]
            # compute weighted loss
            loss = torch.sum(torch.stack([lam * loss for lam, loss in zip(self.lambdas, losses)]))

            # store current losses in self.last_losses to be accessed in the next iteration
            self.last_losses = [loss.clone().detach().requires_grad_(False) for loss in losses]
            # in first iteration, store losses in self.init_losses to be accessed in next iterations
            first_iteration = (self.call_count < 1).to(torch.float32)
            self.init_losses = [(loss * first_iteration + init_loss * (1 - first_iteration)).clone(
            ).detach().requires_grad_(False) for init_loss, loss in zip(self.init_losses, losses)]
            loss += self.weight_i * initial_loss
            self.call_count += 1
        self.epoch += 1

        return loss, residual_loss, initial_loss, boundary_loss

    def __call__(self, pinn: PINN, epoch=0):
        """
        Allows you to use instance of this class as if it was a function:

        ```
            >>> loss = Loss(*some_args)
            >>> calculated_loss = loss(pinn)
        ```
        """
        return self.verbose(pinn, epoch)

def train_model(
    nn_approximator: PINN,
    loss_fn: Callable,
    learning_rate: int = 0.01,
    max_epochs: int = 1_000
) -> PINN:

    optimizer = torch.optim.Adam(nn_approximator.parameters(), lr=learning_rate, )
    loss_values = []
    residual_loss_values = []
    initial_loss_values = []
    boundary_loss_values = []

    start_time = time.time()
    best_loss = 1000000000

    for epoch in range(max_epochs):

        try:

            loss: torch.Tensor = loss_fn(nn_approximator, epoch)
            optimizer.zero_grad()
            loss[0].backward()
            # TODO
            # torch.nn.utils.clip_grad_value_(nn_approximator.parameters(), 0.7)
            optimizer.step()

            loss_values.append(loss[0].item())
            residual_loss_values.append(loss[1].item())
            initial_loss_values.append(loss[2].item())
            boundary_loss_values.append(loss[3].item())
            
            if loss[0].item() < best_loss:
                best_loss = loss[0].item() 
                torch.save(nn_approximator.state_dict(), f"{DIR}/best_{NAME}.pt")
            
            if (epoch + 1) % 1000 == 0:
                epoch_time = time.time() - start_time
                start_time = time.time()

                print(f"Epoch: {epoch + 1} - Loss: {float(loss[0].item()):>7f}, Residual Loss: {float(loss[1].item()):>7f}, Initital Loss: {float(loss[2].item()):>7f}, Boundary Loss: {float(loss[3].item()):>7f}")

        except KeyboardInterrupt:
            break

    return nn_approximator, np.array(loss_values), np.array(residual_loss_values), np.array(initial_loss_values), np.array(boundary_loss_values)

def initial_condition(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r = torch.sqrt((x-LENGTH/2)**2 + (y-LENGTH/2)**2)
    res = 0.5 * torch.exp(-(r)**2 * 50) + 1
    return res
def floor(x, y):
    """Get the sea floor value"""
    return 0



def plot_solution(pinn: PINN, x: torch.Tensor, t: torch.Tensor, figsize=(8, 6), dpi=100):

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
    c = ax.pcolormesh(X, Y, Z, cmap=cmap)
    fig.colorbar(c, ax=ax)

    return fig

def plot_3D(z: torch.Tensor, x: torch.Tensor, y: torch.Tensor, n_points_x, n_points_t, title, figsize=(8, 6), dpi=100, limit=5):
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
    ax.axes.set_zlim3d(bottom=0, top=2)

    c = ax.plot_surface(X, Y, Z)

    x_floor = torch.linspace(0.0, LENGTH, 50)
    y_floor = torch.linspace(0.0, LENGTH, 50)
    z_floor = torch.zeros((50, 50))
    for x_idx, x_coord in enumerate(x_floor):
        for y_idx, y_coord in enumerate(y_floor):
            z_floor[x_idx, y_idx] = floor(x_coord, y_coord)
    x_floor = torch.tile(x_floor, (50, 1))
    y_floor = torch.tile(y_floor, (50, 1)).T
    f = ax.plot_surface(x_floor, y_floor, z_floor, color='green', alpha=0.7)

    return fig

def running_average(y, window=100):
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
    
    fig.savefig(f'{DIR}/{path}.png')

pinn = PINN(LAYERS, NEURONS_PER_LAYER, act=nn.Tanh()).to(device)

x_domain = [0.0, LENGTH]
y_domain = [0.0, LENGTH]
t_domain = [0.0, TOTAL_TIME]

# train the PINN
loss_fn = Loss(
    x_domain=x_domain,
    y_domain=y_domain,
    t_domain=t_domain,
    n_points=N_POINTS,
    initial_condition=initial_condition,
    floor=floor,
    weight_r=WEIGHT_RESIDUAL,
    weight_b=WEIGHT_BOUNDARY,
    weight_i=WEIGHT_INITIAL
)

pinn_trained, loss_values, residual_loss_values, initial_loss_values, boundary_loss_values = train_model(
    pinn, loss_fn=loss_fn, learning_rate=LEARNING_RATE, max_epochs=EPOCHS)

pinn = pinn.cpu()
losses = loss_fn.verbose(pinn)
print(f'Total loss: \t{losses[0]:.5f} ({losses[0]:.3E})')
print(f'Interior loss: \t{losses[1]:.5f} ({losses[1]:.3E})')
print(f'Initial loss: \t{losses[2]:.5f} ({losses[2]:.3E})')
print(f'Bondary loss: \t{losses[3]:.5f} ({losses[3]:.3E})')

text_content_with_data = f"""
Total loss: \t{losses[0]:.5f} ({losses[0]:.3E})
Interior loss: \t{losses[1]:.5f} ({losses[1]:.3E})
Initial loss: \t{losses[2]:.5f} ({losses[2]:.3E})
Bondary loss: \t{losses[3]:.5f} ({losses[3]:.3E})
"""

losses_file = f"{DIR}/losses_{NAME}.txt"
with open(losses_file, 'w+') as file:
    file.write(text_content_with_data)



torch.save(pinn_trained.state_dict(), f"{DIR}/best_{NAME}.pt")

plot_running_average(loss_values, "Loss function (runnig average)", f"loss_relo_{NAME}")
plot_running_average(residual_loss_values, "Residual loss function (running average)", f"residual_loss_relo_{NAME}")
plot_running_average(initial_loss_values, "Initial loss function (running average)", f"initial_loss_relo_{NAME}")
plot_running_average(boundary_loss_values, "Boundary loss function (running average)", f"boundary_loss_relo_{NAME}")


def plot_frame(x_domain: List[float], 
               y_domain: List[float], 
               t_domain: List[float], 
               pinn: PINN, 
               idx: int, 
               t_value: float, 
               n_points: int, 
               base_dir: str=".") -> None:
    x, y, _ = get_initial_points(x_domain, y_domain, t_domain, n_points, requires_grad=False)
    t = torch.full_like(x, t_value)
    z = pinn(x, y, t)
    # fig = plot_color(z, x, y, n_points, n_points, f"PINN for t = {t_value}")
    # fig = plot_3D(z, x, y, n_points, n_points, f"PINN for t = {t_value}")
    # plt.savefig(base_dir + '/img/img_{:03d}.png'.format(idx))
    np.save(f"{DIR}/data_{NAME}_{idx}.npy", z.detach().cpu().numpy())



def plot_simulation_by_frame(total_time: float, 
                             x_domain: List[float], 
                             y_domain: List[float], 
                             t_domain: List[float], 
                             pinn: PINN, 
                             n_points: int, 
                             step:float=0.01) -> None:
    time_values = np.arange(0, total_time, step)

    for idx, t_value in enumerate(time_values):
        plot_frame(x_domain=x_domain, 
                   y_domain=y_domain, 
                   t_domain=t_domain,
                   pinn=pinn,
                   idx=idx,
                   t_value=t_value,
                   n_points=n_points)

plot_simulation_by_frame(total_time=TOTAL_TIME,
                   x_domain=x_domain, 
                   y_domain=y_domain, 
                   t_domain=t_domain,
                   pinn=pinn,
                   n_points=N_POINTS_PLOT)
