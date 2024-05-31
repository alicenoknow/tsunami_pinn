from typing import Optional, Tuple
import torch

from environment.domain import Domain
from environment.env import SimulationEnvironment
from environment.mesh_utils import calculate_partial_derivatives, dump_points


class MeshEnvironment(SimulationEnvironment):
    def __init__(self, mesh_filename: str, device=torch.device("cpu")) -> None:
        super().__init__()
        self.domain = Domain()
        self.device = device

        self.x_raw, self.y_raw, self.z_raw = dump_points(mesh_filename, True)
        self.interior_points = self.get_interior_points()
        self.initial_points = self.get_initial_points(90)  # TODO: parametrize
        self.boundary_points = self.get_boundary_points(30)
        self.partial_x, self.partial_y = calculate_partial_derivatives(
            self.x_raw, self.y_raw, self.z_raw)

    def get_initial_points(self,
                           n_points: int = None,
                           requires_grad=True) -> Tuple[torch.Tensor,
                                                        torch.Tensor,
                                                        torch.Tensor]:
        """
        If no n_points is provided: then uses points from mesh and returns:
        - x_grid: [[x1], [x2], [x3], ...]
        - y_grid: [[y1], [y2], [y3], ...]
        - t0: [[0], [0], ...] -> number of points

        otherwise takes n_points and creates grids n_points x n_points:
        - x_grid: [[x1], [x2], [x3], ...] size: (n_points x n_points) x 1
        - y_grid: [[y1], [y2], [y3], ...] size: (n_points x n_points) x 1
        - t0: [[0], [0], ...] size: (n_points x n_points) x 1
        """

        if n_points:
            x_linspace, y_linspace, _ = self._generate_linespaces_n(n_points, requires_grad)
            x_grid, y_grid = torch.meshgrid(x_linspace, y_linspace, indexing="ij")
            x_grid = self._reshape_and_to_device(x_grid)
            y_grid = self._reshape_and_to_device(y_grid)
        else:
            x_grid = self._reshape_and_to_device(self.x_raw, requires_grad)
            y_grid = self._reshape_and_to_device(self.y_raw, requires_grad)

        t0 = torch.full_like(x_grid, self.domain.T_DOMAIN[0], requires_grad=requires_grad)
        return (x_grid, y_grid, t0)

    def get_boundary_points(self, n_points=None, requires_grad=True):
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
        x_linspace, y_linspace, t_linspace = self._generate_linespaces_n(n_points)

        x_grid, t_grid = torch.meshgrid(x_linspace, t_linspace, indexing="ij")
        y_grid, _ = torch.meshgrid(y_linspace, t_linspace, indexing="ij")

        x_grid = self._reshape_and_to_device(x_grid, requires_grad)
        y_grid = self._reshape_and_to_device(y_grid, requires_grad)
        t_grid = self._reshape_and_to_device(t_grid, requires_grad)

        x0 = torch.full_like(t_grid, self.domain.XY_DOMAIN[0], requires_grad=requires_grad)
        x1 = torch.full_like(t_grid, self.domain.XY_DOMAIN[1], requires_grad=requires_grad)
        y0 = torch.full_like(t_grid, self.domain.XY_DOMAIN[0], requires_grad=requires_grad)
        y1 = torch.full_like(t_grid, self.domain.XY_DOMAIN[1], requires_grad=requires_grad)

        down = (x_grid, y0, t_grid)
        up = (x_grid, y1, t_grid)
        left = (x0, y_grid, t_grid)
        right = (x1, y_grid, t_grid)

        # for each time step, for each point: t_points x n_points
        # down ([[x1], [x1], ..., [x2], [x2], ...], [0, ...], [[t1], [t2], ..., [t1], [t2], ..])
        return down, up, left, right

    def get_interior_points(self, requires_grad=True):
        # Vector with T_POINTS values, equally distributed in T_DOMAIN
        t_raw = torch.linspace(
            self.domain.T_DOMAIN[0],
            self.domain.T_DOMAIN[1],
            steps=self.domain.T_POINTS)

        # x_grid: (n_points x T_POINTS) [[x1, x1, ...], [x2, x2, ...], ...]
        # y_grid: (n_points x T_POINTS) [[y1, y1, ...], [y2, y2, ...], ...]
        # z_grid: (n_points x T_POINTS) [[z1, z1, ...], [z2, z2, ...], ...]
        # t_grid: (n_points x T_POINTS) [[t1, t2, ...], [t1, t2, ...], ...]
        x_grid, t_grid = torch.meshgrid(self.x_raw, t_raw, indexing="ij")
        y_grid, _ = torch.meshgrid(self.y_raw, t_raw, indexing="ij")
        z_grid, _ = torch.meshgrid(self.z_raw, t_raw, indexing="ij")

        x = self._reshape_and_to_device(x_grid, requires_grad)
        y = self._reshape_and_to_device(y_grid, requires_grad)
        z = self._reshape_and_to_device(z_grid, requires_grad)
        t = self._reshape_and_to_device(t_grid, requires_grad)

        self.domain.XY_DOMAIN = [x.min().item(), x.max().item()]
        self.domain.N_POINTS = x.size()[0] // self.domain.T_POINTS

        # sizes: (n_points x T_POINTS) x 1, e.g. [[x1], [x1], ..., [x2], [x2], ...]
        return x, y, z, t

    def _reshape_and_to_device(self, tensor: torch.Tensor, requires_grad: Optional[bool] = None) -> torch.Tensor:
        tensor = tensor.reshape(-1, 1).to(self.device)
        if requires_grad is not None:
            tensor.requires_grad = requires_grad
        return tensor

    def _generate_linespaces_n(self,
                               n_points: int,
                               requires_grad=False) -> Tuple[torch.Tensor,
                                                             torch.Tensor,
                                                             torch.Tensor]:
        n_points_linspace = n_points if n_points else self.domain.N_POINTS

        x_linspace = torch.linspace(
            self.domain.XY_DOMAIN[0],
            self.domain.XY_DOMAIN[1],
            steps=n_points_linspace,
            requires_grad=requires_grad)
        y_linspace = torch.linspace(
            self.domain.XY_DOMAIN[0],
            self.domain.XY_DOMAIN[1],
            steps=n_points_linspace,
            requires_grad=requires_grad)
        t_linspace = torch.linspace(
            self.domain.XY_DOMAIN[0],
            self.domain.XY_DOMAIN[1],
            steps=self.domain.T_POINTS,
            requires_grad=requires_grad)

        return x_linspace, y_linspace, t_linspace
