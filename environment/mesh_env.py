import torch
from typing import List
from environment.domain import Domain
from environment.env import SimulationEnvironment
from environment.mesh_utils import dump_points


class MeshEnvironment(SimulationEnvironment):

    def __init__(self, mesh_filename: str,
                    device=torch.device("cpu")) -> None:
        super().__init__()
        self.device = device
        self.mesh_filename = mesh_filename

        self.initial_points = self.get_initial_points()
        self.boundary_points = self.get_boundary_points()
        self.interior_points = self.get_interior_points()
    
    def get_initial_points(self, requires_grad=True):
        x_raw, y_raw, _ = dump_points(self.mesh_filename)
        x_grid = x_raw.to(self.device)
        y_grid = y_raw.to(self.device)
        
        x_grid.requires_grad = requires_grad
        y_grid.requires_grad = requires_grad

        x_grid = x_grid.reshape(-1, 1).to(self.device)
        y_grid = y_grid.reshape(-1, 1).to(self.device)
        t0 = torch.full_like(x_grid, Domain().T_DOMAIN[0], requires_grad=requires_grad)

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
        x_linspace, y_linspace, t_linspace = self._generate_linespaces()

        x_grid, t_grid = torch.meshgrid(x_linspace, t_linspace, indexing="ij")
        y_grid, _      = torch.meshgrid(y_linspace, t_linspace, indexing="ij")

        x_grid = x_grid.reshape(-1, 1).to(self.device)
        y_grid = y_grid.reshape(-1, 1).to(self.device)
        t_grid = t_grid.reshape(-1, 1).to(self.device)
        
        x_grid.requires_grad = requires_grad
        y_grid.requires_grad = requires_grad
        t_grid.requires_grad = requires_grad

        x0 = torch.full_like(t_grid, Domain().X_DOMAIN[0], requires_grad=requires_grad)
        x1 = torch.full_like(t_grid, Domain().X_DOMAIN[1], requires_grad=requires_grad)
        y0 = torch.full_like(t_grid, Domain().Y_DOMAIN[0], requires_grad=requires_grad)
        y1 = torch.full_like(t_grid, Domain().Y_DOMAIN[1], requires_grad=requires_grad)

        down    = (x_grid, y0,     t_grid)
        up      = (x_grid, y1,     t_grid)
        left    = (x0,     y_grid, t_grid)
        right   = (x1,     y_grid, t_grid)

        return down, up, left, right
    
    def get_interior_points(self, requires_grad=True):
        x_raw, y_raw, z_raw = dump_points(self.mesh_filename)
        t_raw = torch.linspace(Domain().T_DOMAIN[0], Domain().T_DOMAIN[1], steps=Domain().T_POINTS)
        
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

        Domain().X_DOMAIN = [x.min().item(), x.max().item()]
        Domain().Y_DOMAIN = [y.min().item(), y.max().item()]

        Domain().X_POINTS = x.size()[0] // Domain().T_POINTS
        Domain().Y_POINTS = y.size()[0] // Domain().T_POINTS

        return x, y, z, t

    def _generate_linespaces(self, requires_grad=False):
        x_domain, y_domain, t_domain = Domain().X_DOMAIN, Domain().Y_DOMAIN, Domain().T_DOMAIN
        x_points, y_points, t_points = Domain().X_POINTS, Domain().Y_POINTS, Domain().T_POINTS

        x_linspace = torch.linspace(x_domain[0], x_domain[1], steps=x_points, requires_grad=requires_grad)
        y_linspace = torch.linspace(y_domain[0], y_domain[1], steps=y_points, requires_grad=requires_grad)
        t_linspace = torch.linspace(t_domain[0], t_domain[1], steps=t_points, requires_grad=requires_grad)
        return x_linspace, y_linspace, t_linspace