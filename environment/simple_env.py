from typing import List
import torch

from environment.domain import Domain
from environment.env import SimulationEnvironment


class SimpleEnvironment(SimulationEnvironment):

    def __init__(self, device=torch.device("cpu")) -> None:
        super().__init__()
        self.domain = Domain()
        self.device = device

        self.initial_points = self.get_initial_points()
        self.boundary_points = self.get_boundary_points()
        self.interior_points = self.get_interior_points()

    def get_initial_points(self, n_points: int = None, requires_grad=True) -> List[torch.Tensor]:
        x_linspace, y_linspace, _ = self._generate_linespaces(n_points, requires_grad)

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
        x_linspace, y_linspace, t_linspace = self._generate_linespaces(requires_grad=requires_grad)

        x_grid, t_grid = torch.meshgrid(x_linspace, t_linspace, indexing="ij")
        y_grid, _ = torch.meshgrid(y_linspace, t_linspace, indexing="ij")

        x_grid = self._reshape_and_to_device(x_grid)
        y_grid = self._reshape_and_to_device(y_grid)
        t_grid = self._reshape_and_to_device(t_grid)

        x0 = torch.full_like(t_grid, self.domain.XY_DOMAIN[0], requires_grad=requires_grad)
        x1 = torch.full_like(t_grid, self.domain.XY_DOMAIN[1], requires_grad=requires_grad)
        y0 = torch.full_like(t_grid, self.domain.XY_DOMAIN[0], requires_grad=requires_grad)
        y1 = torch.full_like(t_grid, self.domain.XY_DOMAIN[1], requires_grad=requires_grad)

        down = (x_grid, y0, t_grid)
        up = (x_grid, y1, t_grid)
        left = (x0, y_grid, t_grid)
        right = (x1, y_grid, t_grid)

        return down, up, left, right

    def get_interior_points(self, requires_grad: bool = True):
        x_raw, y_raw, t_raw = self._generate_linespaces(requires_grad=requires_grad)
        grids = torch.meshgrid(x_raw, y_raw, t_raw, indexing="ij")

        x = self._reshape_and_to_device(grids[0])
        y = self._reshape_and_to_device(grids[1])
        t = self._reshape_and_to_device(grids[2])
        return x, y, 0, t

    def _reshape_and_to_device(
            self,
            tensor: torch.Tensor,
            requires_grad: bool = False) -> torch.Tensor:
        tensor = tensor.reshape(-1, 1).to(self.device)
        if requires_grad:
            tensor.requires_grad = requires_grad
        return tensor

    def _generate_linespaces(self, n_points: int = None, requires_grad=False):
        x_domain, y_domain = self.domain.XY_DOMAIN, self.domain.XY_DOMAIN
        t_domain = self.domain.T_DOMAIN
        xy_points, t_points = self.domain.N_POINTS, self.domain.T_POINTS
        n_points_linspace = n_points if n_points else xy_points

        x_linspace = torch.linspace(
            x_domain[0],
            x_domain[1],
            steps=n_points_linspace,
            requires_grad=requires_grad)
        y_linspace = torch.linspace(
            y_domain[0],
            y_domain[1],
            steps=n_points_linspace,
            requires_grad=requires_grad)
        t_linspace = torch.linspace(
            t_domain[0],
            t_domain[1],
            steps=t_points,
            requires_grad=requires_grad)
        return x_linspace, y_linspace, t_linspace
