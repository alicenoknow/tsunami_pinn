import meshio
import numpy as np
import torch
import matplotlib.pyplot as plt

from datetime import datetime
from torch import nn
from scipy.interpolate import griddata
from typing import Callable, Tuple, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

LENGTH = 1. # Domain size in x axis. Always starts at 0
TOTAL_TIME = .25 # Domain size in t axis. Always starts at 0
N_POINTS = 20 # Number of in single asxis
N_POINTS_PLOT = 128 # Number of points in single axis used in plotting

WEIGHT_RESIDUAL = 0.01 # 0.03 # Weight of residual part of loss function
WEIGHT_INITIAL = 10.0 # 1.0 # Weight of initial part of loss function
WEIGHT_BOUNDARY = 0.001 # 0.0005 # Weight of boundary part of loss function
GRAVITY = 9.81

LAYERS = 4
NEURONS_PER_LAYER = 150
EPOCHS = 100_000
LEARNING_RATE = 0.001

MESH_FILENAME = "data/val_square_UTM_translated_10.inp"

NAME = "underwater_lbfgs_wr01wi10wb001"
DIR = "./results/under/lbfgs"

t_domain = [0, TOTAL_TIME]

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
        x_stack = torch.cat([x, y, t], dim=1).to(device)
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


def dfdt(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, f_val=None, order: int = 1):
    f_value = f_val if f_val is not None else f(pinn, x, y, t)
    # f_value = f(pinn, x, y, t)
    return df(f_value, t, order=order)


def dfdx(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, f_val=None, order: int = 1):
    # f_value = f(pinn, x, y, t)
    f_value = f_val if f_val is not None else f(pinn, x, y, t)
    return df(f_value, x, order=order)

def dfdy(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, f_val=None, order: int = 1):
    # f_value = f(pinn, x, y, t)
    f_value = f_val if f_val is not None else f(pinn, x, y, t)
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

    x_grid, t_grid = torch.meshgrid(x_linspace, t_linspace, indexing="ij")
    y_grid, _      = torch.meshgrid(y_linspace, t_linspace, indexing="ij")

    x_grid = x_grid.reshape(-1, 1).to(device)
    y_grid = y_grid.reshape(-1, 1).to(device)
    t_grid = t_grid.reshape(-1, 1).to(device)
    
    x_grid.requires_grad = requires_grad
    y_grid.requires_grad = requires_grad
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

def get_interior_points_mesh(t_domain, n_points, device=torch.device("cpu"), requires_grad=True):
    x_raw, y_raw, z_raw = dump_points(MESH_FILENAME)
    step = t_domain[1] / n_points
    t_raw = torch.linspace(t_domain[0]+step, t_domain[1], steps=n_points)
    x_grid, t_grid = torch.meshgrid(x_raw, t_raw, indexing="ij")
    y_grid, _      = torch.meshgrid(y_raw, t_raw, indexing="ij")
    z_grid, _      = torch.meshgrid(z_raw, t_raw, indexing="ij")
    x = x_grid.reshape(-1, 1).to(device)
    y = y_grid.reshape(-1, 1).to(device)
    z = z_grid.reshape(-1, 1).to(device)
    t = t_grid.reshape(-1, 1).to(device)
    x.requires_grad = True
    y.requires_grad = True
    z.requires_grad = True
    t.requires_grad = True
    return x, y, z, t


def dump_points(filename):
    mesh = meshio.avsucd.read(filename)
    points = torch.tensor(mesh.points, dtype=torch.float32)
    x,y,z = points.transpose(0,1)
    #-> translate into [0,1]
    min_x, min_y, min_z = torch.min(x), torch.min(y), torch.min(z)
    max_x, max_y, max_z = torch.max(x), torch.max(y), torch.max(z)
    x = (x - min_x) / (max_x - min_x)
    y = (y - min_y) / (max_y - min_y)
    z = (z - min_z) / (max_z - min_z)

    z *= 0.1

    return x,y,z

def get_initial_points(x_domain: List[float], 
                       y_domain: List[float], 
                       t_domain: List[float], 
                       n_points: int, 
                       device=torch.device("cpu"), 
                       requires_grad=True):
    x_linspace = torch.linspace(x_domain[0], x_domain[1], n_points)
    y_linspace = torch.linspace(y_domain[0], y_domain[1], n_points)
    
    x_grid, y_grid = torch.meshgrid(x_linspace, y_linspace, indexing="ij")
    
    x_grid = x_grid.reshape(-1, 1).to(device)
    y_grid = y_grid.reshape(-1, 1).to(device)
    
    x_grid.requires_grad = requires_grad
    y_grid.requires_grad = requires_grad
    
    t0 = torch.full_like(x_grid, t_domain[0], requires_grad=requires_grad)
    return (x_grid, y_grid, t0)

x_raw, y_raw, z_raw = dump_points(MESH_FILENAME)
x_interior, y_interior, z_interior, t_interior = get_interior_points_mesh(t_domain, N_POINTS, device)
x_domain = [0, 1]
y_domain = [0, 1]

x_initial, y_initial, t_initial = get_initial_points(x_domain, y_domain, t_domain, 5*N_POINTS, device)
down, up, left, right = get_boundary_points(x_domain, y_domain, t_domain, 3*N_POINTS, device)


LENGTH = x_domain[1]


class Loss:
    def __init__(
        self,
        x_domain: Tuple[float, float],
        y_domain: Tuple[float, float],
        t_domain: Tuple[float, float],
        n_points: int,
        initial_condition: Callable,
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
        self.weight_r = weight_r
        self.weight_b = weight_b
        self.weight_i = weight_i
        self.epoch = 0

    def residual_loss(self, pinn: PINN):
        x,y,z,t = x_interior, y_interior, z_interior, t_interior
        u = f(pinn, x, y, t)

        loss = dfdt(pinn, x, y, t, u, order=2) - \
              GRAVITY * ((dfdx(pinn, x, y, t, u)- dzdx) * dfdx(pinn, x, y, t, u) + \
              (u-z) * dfdx(pinn, x, y, t, u, order=2) + \
              (dfdy(pinn, x, y, t, u) - dzdy) * dfdy(pinn, x, y, t, u) + \
              (u-z) * dfdy(pinn, x, y, t, u, order=2))

        #casual_weights = torch.exp(-2 * t)
        #return (casual_weights * loss).pow(2).mean()
        return loss.pow(2).mean()

    def initial_loss(self, pinn: PINN):
        x, y, t = x_initial, y_initial, t_initial #get_initial_points(self.x_domain, self.y_domain, self.t_domain, self.n_points, pinn.device())
        pinn_init = self.initial_condition(x, y)
        loss = f(pinn, x, y, t) - pinn_init
        return loss.pow(2).mean()

    def boundary_loss(self, pinn: PINN):
        # down, up, left, right = get_boundary_points(self.x_domain, self.y_domain, self.t_domain, self.n_points, pinn.device())
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

    def verbose(self, pinn: PINN, only_initial=False):
        """
        Returns all parts of the loss function

        Not used during training! Only for checking the results later.
        """
        residual_loss = self.residual_loss(pinn)
        initial_loss = self.initial_loss(pinn)
        boundary_loss = self.boundary_loss(pinn)

        if self.epoch < 0:
            final_loss = \
                0.0 * residual_loss + \
                self.weight_i * initial_loss + \
                0.0 * boundary_loss
        else:
            final_loss = \
                self.weight_r * residual_loss + \
                self.weight_i * initial_loss + \
                self.weight_b * boundary_loss # 5, 1000 i 1?, 0.0005

        #final_loss = \
        #    self.weight_r * residual_loss + \
        #    self.weight_i * initial_loss + \
        #    self.weight_b * boundary_loss # 5, 1000 i 1?, 0.0005

        self.epoch += 1

        return final_loss, residual_loss, initial_loss, boundary_loss

    def __call__(self, pinn: PINN, only_initial=False):
        """
        Allows you to use the instance of this class as if it were a function:

        ```
            >>> loss = Loss(*some_args)
            >>> calculated_loss = loss(pinn)
        ```
        """
        return self.verbose(pinn, only_initial)


def interpolate_plane(x, y, z):
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
    points = np.vstack((x.numpy(), y.numpy())).T
    values = z.numpy()

    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

    if np.any(np.isnan(grid_z)):
        grid_z = griddata(points, values, (grid_x, grid_y), method='nearest')

    return grid_x, grid_y, grid_z


def calculate_derivatives(grid_x, grid_y, grid_z):
    dz_dx, dz_dy = np.gradient(grid_z, grid_x[1, 0] - grid_x[0, 0], grid_y[0, 1] - grid_y[0, 0])
    return dz_dx, dz_dy


def interpolate_derivatives_to_mesh_points(x, y, grid_x, grid_y, dz_dx, dz_dy):
    points = np.vstack((x.numpy(), y.numpy())).T
    dz_dx_mesh = griddata((grid_x.flatten(), grid_y.flatten()),
                          dz_dx.flatten(), points, method='cubic')
    dz_dy_mesh = griddata((grid_x.flatten(), grid_y.flatten()),
                          dz_dy.flatten(), points, method='cubic')

    dz_dx_tensor = torch.tensor(dz_dx_mesh, dtype=torch.float32)
    dz_dy_tensor = torch.tensor(dz_dy_mesh, dtype=torch.float32)

    return dz_dx_tensor, dz_dy_tensor


def calculate_partial_derivatives(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, device):
    grid_x, grid_y, grid_z = interpolate_plane(x, y, z)
    dz_dx, dz_dy = calculate_derivatives(grid_x, grid_y, grid_z)
    dz_dx_tensor, dz_dy_tensor = interpolate_derivatives_to_mesh_points(
        x, y, grid_x, grid_y, dz_dx, dz_dy)
    #return dz_dx_tensor.to(device), dz_dy_tensor.to(device)
    return torch.tensor(np.tile(dz_dx_tensor, N_POINTS)).reshape(-1, 1).to(device), torch.tensor(np.tile(dz_dy_tensor, N_POINTS)).reshape(-1, 1).to(device)

x_raw, y_raw, z_raw = dump_points(MESH_FILENAME)
dzdx, dzdy = calculate_partial_derivatives(x_raw, y_raw, z_raw, device)

def train_model(
    nn_approximator: PINN,
    loss_fn: Callable,
    learning_rate: int = 0.01,
    max_epochs: int = 1_000
) -> PINN:

    optimizer = torch.optim.Adam(nn_approximator.parameters(), lr=learning_rate)
    loss_values = []
    residual_loss_values = []
    initial_loss_values = []
    boundary_loss_values = []
    top_loss = 100000000

    for epoch in range(max_epochs):
        try:
            loss: torch.Tensor = loss_fn(nn_approximator)
            optimizer.zero_grad()
            loss[0].backward()
            optimizer.step()

            if loss[0].item() < top_loss:
                torch.save(nn_approximator, f"{DIR}/best_{NAME}.pt")
                top_loss = loss[0].item()

            loss_values.append(loss[0].item())
            residual_loss_values.append(loss[1].item())
            initial_loss_values.append(loss[2].item())
            boundary_loss_values.append(loss[3].item())
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch: {epoch + 1} - Loss: {float(loss[0].item()):>7f}, Residual Loss: {float(loss[1].item()):>7f}, Initital Loss: {float(loss[2].item()):>7f}, Boundary Loss: {float(loss[3].item()):>7f}")

        except KeyboardInterrupt:
            break

    return nn_approximator, np.array(loss_values), np.array(residual_loss_values), np.array(initial_loss_values), np.array(boundary_loss_values)

def train_lbfgs(loss_fn, model):
    loss_values, residual_loss_values, initial_loss_values, boundary_loss_values = [], [], [], []
    
    lbfgs_optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, history_size=100, line_search_fn="strong_wolfe", max_iter=20)

    def closure():
        model.train()
        loss = loss_fn(model)
        lbfgs_optimizer.zero_grad()
        loss[0].backward()
        return loss[0]
        
    for epoch in range(5000):
        try:
            total_loss = lbfgs_optimizer.step(closure).item()
            loss = loss_fn(model)
                
            loss_values.append(total_loss)
            residual_loss_values.append(loss[1].item())
            initial_loss_values.append(loss[2].item())
            boundary_loss_values.append(loss[3].item())

        except KeyboardInterrupt:
            break
            
    return (np.array(loss_values), 
            np.array(residual_loss_values), 
            np.array(initial_loss_values), 
            np.array(boundary_loss_values))


def initial_condition(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r = torch.sqrt((x-LENGTH/2)**2 + (y-LENGTH/2)**2)
    res = 0.5 * torch.exp(-(r)**2 * 50) + 1
    return res

pinn = PINN(LAYERS, NEURONS_PER_LAYER, act=nn.Tanh()).to(device)

# train the PINN
loss_fn = Loss(
    x_domain=x_domain,
    y_domain=y_domain,
    t_domain=t_domain,
    n_points=N_POINTS,
    initial_condition=initial_condition,
    weight_r=WEIGHT_RESIDUAL,
    weight_b=WEIGHT_BOUNDARY,
    weight_i=WEIGHT_INITIAL
)

pinn_trained, loss_values, residual_loss_values, initial_loss_values, boundary_loss_values = train_model(
    pinn, loss_fn=loss_fn, learning_rate=LEARNING_RATE, max_epochs=EPOCHS)

loss_values_lbfgs, residual_loss_values_lbfgs, initial_loss_values_lbfgs, boundary_loss_values_lbfgs = train_lbfgs(loss_fn, pinn_trained)
loss_values = np.concatenate((loss_values, loss_values_lbfgs))
residual_loss_values = np.concatenate((residual_loss_values, residual_loss_values_lbfgs))
initial_loss_values = np.concatenate((initial_loss_values, initial_loss_values_lbfgs))
boundary_loss_values = np.concatenate((boundary_loss_values, boundary_loss_values_lbfgs))

losses = loss_fn.verbose(pinn)

print(f'Total loss: \t{losses[0]:.5f} ({losses[0]:.3E})')
print(f'Interior loss: \t{losses[1]:.5f} ({losses[1]:.3E})')
print(f'Initial loss: \t{losses[2]:.5f} ({losses[2]:.3E})')
print(f'Boundary loss: \t{losses[3]:.5f} ({losses[3]:.3E})')

text_content_with_data = f"""
Total loss: \t{losses[0]:.5f} ({losses[0]:.3E})
Interior loss: \t{losses[1]:.5f} ({losses[1]:.3E})
Initial loss: \t{losses[2]:.5f} ({losses[2]:.3E})
Bondary loss: \t{losses[3]:.5f} ({losses[3]:.3E})
"""

losses_file = f"{DIR}/losses_{NAME}.txt"
with open(losses_file, 'w+') as file:
    file.write(text_content_with_data)



def plot_frame(x_domain: List[float], 
               y_domain: List[float], 
               t_domain: List[float], 
               pinn: 'PINN', 
               idx: int, 
               t_value: float, 
               n_points: int,
               length: float,
               base_dir: str=".") -> None:
    x, y, _ = get_initial_points(x_domain, y_domain, t_domain, n_points, requires_grad=False)
    t = torch.full_like(x, t_value)
    z = pinn(x, y, t)
    np.save(f"{DIR}/data_{NAME}_{idx}.npy", z.detach().cpu().numpy())


def plot_simulation_by_frame(total_time: float, 
                             x_domain: List[float], 
                             y_domain: List[float], 
                             t_domain: List[float], 
                             pinn: 'PINN', 
                             n_points: int,
                             length: float,
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
                   length=length)

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
    
    fig.savefig(f'{DIR}/{path}.png')

plot_running_average(loss_values, "Loss function (runnig average)", f"total_loss_uw_{NAME}")
plot_running_average(residual_loss_values, "Residual loss function (running average)", f"residual_loss_uw_{NAME}")
plot_running_average(initial_loss_values, "Initial loss function (running average)", f"initial_loss_uw_{NAME}")
plot_running_average(boundary_loss_values, "Boundary loss function (running average)", f"boundary_loss_uw_{NAME}")

plot_simulation_by_frame(total_time=TOTAL_TIME,
                   x_domain=x_domain, 
                   y_domain=y_domain, 
                   t_domain=t_domain,
                   pinn=pinn_trained,
                   n_points=N_POINTS_PLOT,
                   length=LENGTH)
