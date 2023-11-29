import torch

from environment.domain import Domain
from environment.env import SimulationEnvironment

class SimpleEnvironment(SimulationEnvironment):

    def __init__(self, device=torch.device("cpu")) -> None:
        super().__init__()
        self.domain = Domain()
        self.device = device
        self.x_domain = self.domain.X_DOMAIN
        self.y_domain = self.domain.Y_DOMAIN
        self.t_domain = self.domain.T_DOMAIN
        self.x_points = self.domain.X_POINTS
        self.y_points = self.domain.Y_POINTS
        self.t_points = self.domain.T_POINTS

        self.initial_points = self.get_initial_points(self.domain)
        self.boundary_points = self.get_boundary_points()
        self.interior_points = self.get_interior_points()

    def get_initial_points(self, new_domain: Domain, requires_grad=True):
        domain = new_domain if new_domain is not None else self.domain

        x_linspace, y_linspace, _ = self._generate_linespaces(domain, requires_grad)
        
        x_grid, y_grid = torch.meshgrid(x_linspace, y_linspace, indexing="ij")
        
        x_grid = x_grid.reshape(-1, 1).to(self.device)
        y_grid = y_grid.reshape(-1, 1).to(self.device)

        
        t0 = torch.full_like(x_grid, self.t_domain[0], requires_grad=requires_grad)
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
        x_linspace, y_linspace, t_linspace = self._generate_linespaces(self.domain, requires_grad)

        x_grid, t_grid = torch.meshgrid(x_linspace, t_linspace, indexing="ij")
        y_grid, _      = torch.meshgrid(y_linspace, t_linspace, indexing="ij")

        x_grid = x_grid.reshape(-1, 1).to(self.device)
        y_grid = y_grid.reshape(-1, 1).to(self.device)
        t_grid = t_grid.reshape(-1, 1).to(self.device)

        x0 = torch.full_like(t_grid, self.x_domain[0], requires_grad=requires_grad)
        x1 = torch.full_like(t_grid, self.x_domain[1], requires_grad=requires_grad)
        y0 = torch.full_like(t_grid, self.y_domain[0], requires_grad=requires_grad)
        y1 = torch.full_like(t_grid, self.y_domain[1], requires_grad=requires_grad)

        down    = (x_grid, y0,     t_grid)
        up      = (x_grid, y1,     t_grid)
        left    = (x0,     y_grid, t_grid)
        right   = (x1,     y_grid, t_grid)

        return down, up, left, right

    def get_interior_points(self, requires_grad=True):
        x_raw, y_raw, t_raw = self._generate_linespaces(self.domain, requires_grad)
        grids = torch.meshgrid(x_raw, y_raw, t_raw, indexing="ij")

        x = grids[0].reshape(-1, 1).to(self.device)
        y = grids[1].reshape(-1, 1).to(self.device)
        t = grids[2].reshape(-1, 1).to(self.device)
        return x, y, 0, t

    def _generate_linespaces(self, domain: Domain, requires_grad=False):
        x_domain, y_domain, t_domain = domain.X_DOMAIN, domain.Y_DOMAIN, domain.T_DOMAIN
        x_points, y_points, t_points = domain.X_POINTS, domain.Y_POINTS, domain.T_POINTS

        x_linspace = torch.linspace(x_domain[0], x_domain[1], steps=x_points, requires_grad=requires_grad)
        y_linspace = torch.linspace(y_domain[0], y_domain[1], steps=y_points, requires_grad=requires_grad)
        t_linspace = torch.linspace(t_domain[0], t_domain[1], steps=t_points, requires_grad=requires_grad)
        return x_linspace, y_linspace, t_linspace