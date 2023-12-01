import torch
import numpy as np

from environment.domain import Domain
from environment.env import SimulationEnvironment
from environment.mesh_utils import calculate_partial_derivatives, dump_points


class MeshEnvironment(SimulationEnvironment):

    def __init__(self, mesh_filename: str,
                    device=torch.device("cpu")) -> None:
        super().__init__()
        self.domain = Domain()
        self.device = device
        self.mesh_filename = mesh_filename

        self.x_raw, self.y_raw, self.z_raw = dump_points(self.mesh_filename)
        self.interior_points = self.get_interior_points()
        self.initial_points = self.get_initial_points()
        self.boundary_points = self.get_boundary_points()
        self.partial_x, self.partial_y = calculate_partial_derivatives(self.x_raw, self.y_raw, self.z_raw)
    
    def get_initial_points(self, n_points: int=None, requires_grad=True):
        if n_points:
            return self.get_initial_points_n(n_points, requires_grad)
        x_raw, y_raw = self.x_raw, self.y_raw
        x_grid = x_raw.to(self.device)
        y_grid = y_raw.to(self.device)
        
        x_grid.requires_grad = requires_grad
        y_grid.requires_grad = requires_grad

        x_grid = x_grid.reshape(-1, 1).to(self.device)
        y_grid = y_grid.reshape(-1, 1).to(self.device)
        t0 = torch.full_like(x_grid, self.domain.T_DOMAIN[0], requires_grad=requires_grad)

        return (x_grid, y_grid, t0)

    def get_initial_points_n(self, n_points: int=None, requires_grad=True):
        x_linspace, y_linspace, _ = self._generate_linespaces_n(n_points, requires_grad)
        
        x_grid, y_grid = torch.meshgrid(x_linspace, y_linspace, indexing="ij")
        
        x_grid = x_grid.reshape(-1, 1).to(self.device)
        y_grid = y_grid.reshape(-1, 1).to(self.device)

        
        t0 = torch.full_like(x_grid, self.domain.T_DOMAIN[0], requires_grad=requires_grad)
        return (x_grid, y_grid, t0)

    def get_boundary_points(self, requires_grad=True):
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
        x_linspace, y_linspace, t_linspace = self._generate_linespaces_n(int(np.sqrt(self.domain.N_POINTS)))

        x_grid, t_grid = torch.meshgrid(x_linspace, t_linspace, indexing="ij")
        y_grid, _      = torch.meshgrid(y_linspace, t_linspace, indexing="ij")

        x_grid = x_grid.reshape(-1, 1).to(self.device)
        y_grid = y_grid.reshape(-1, 1).to(self.device)
        t_grid = t_grid.reshape(-1, 1).to(self.device)
        
        x_grid.requires_grad = requires_grad
        y_grid.requires_grad = requires_grad
        t_grid.requires_grad = requires_grad

        x0 = torch.full_like(t_grid, self.domain.XY_DOMAIN[0], requires_grad=requires_grad)
        x1 = torch.full_like(t_grid, self.domain.XY_DOMAIN[1], requires_grad=requires_grad)
        y0 = torch.full_like(t_grid, self.domain.XY_DOMAIN[0], requires_grad=requires_grad)
        y1 = torch.full_like(t_grid, self.domain.XY_DOMAIN[1], requires_grad=requires_grad)

        down    = (x_grid, y0,     t_grid)
        up      = (x_grid, y1,     t_grid)
        left    = (x0,     y_grid, t_grid)
        right   = (x1,     y_grid, t_grid)

        return down, up, left, right
    
    def get_interior_points(self, requires_grad=True):
        x_raw, y_raw, z_raw = self.x_raw, self.y_raw, self.z_raw
        t_raw = torch.linspace(self.domain.T_DOMAIN[0], self.domain.T_DOMAIN[1], steps=self.domain.T_POINTS)
        
        x_grid, t_grid = torch.meshgrid(x_raw, t_raw, indexing="ij")
        y_grid, _      = torch.meshgrid(y_raw, t_raw, indexing="ij")
        z_grid, _      = torch.meshgrid(z_raw, t_raw, indexing="ij")

        x = x_grid.reshape(-1, 1).to(self.device)
        y = y_grid.reshape(-1, 1).to(self.device)
        z = z_grid.reshape(-1, 1).to(self.device)
        t = t_grid.reshape(-1, 1).to(self.device)

        x.requires_grad = requires_grad
        y.requires_grad = requires_grad
        z.requires_grad = requires_grad
        t.requires_grad = requires_grad

        self.domain.XY_DOMAIN = [x.min().item(), x.max().item()]
        self.domain.N_POINTS = x.size()[0] // self.domain.T_POINTS

        return x, y, z, t

    def _generate_linespaces(self, requires_grad=False):
        x_domain, y_domain, t_domain = self.domain.XY_DOMAIN, self.domain.XY_DOMAIN, self.domain.T_DOMAIN
        x_points, y_points, t_points = self.domain.N_POINTS, self.domain.N_POINTS, self.domain.T_POINTS

        x_linspace = torch.linspace(x_domain[0], x_domain[1], steps=x_points, requires_grad=requires_grad)
        y_linspace = torch.linspace(y_domain[0], y_domain[1], steps=y_points, requires_grad=requires_grad)
        t_linspace = torch.linspace(t_domain[0], t_domain[1], steps=t_points, requires_grad=requires_grad)
        return x_linspace, y_linspace, t_linspace
    
    def _generate_linespaces_n(self, n: int, requires_grad=False):
        x_domain, y_domain, t_domain = self.domain.XY_DOMAIN, self.domain.XY_DOMAIN, self.domain.T_DOMAIN
        x_points, y_points, t_points = n, n, self.domain.T_POINTS

        x_linspace = torch.linspace(x_domain[0], x_domain[1], steps=x_points, requires_grad=requires_grad)
        y_linspace = torch.linspace(y_domain[0], y_domain[1], steps=y_points, requires_grad=requires_grad)
        t_linspace = torch.linspace(t_domain[0], t_domain[1], steps=t_points, requires_grad=requires_grad)
        return x_linspace, y_linspace, t_linspace