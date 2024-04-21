from abc import ABC, abstractmethod
from typing import List
import torch


class SimulationEnvironment(ABC):
    """
    Abstract base class for a simulation environment.
    """

    def __init__(self):
        self.domain = None
        self.initial_points = None
        self.boundary_points = None
        self.interior_points = None

    @abstractmethod
    def get_initial_points(self, n_points: int, requires_grad=True) -> List[torch.Tensor]:
        """
        Abstract method to get the initial points.

        Parameters:
        n_points (int): The number of points.
        requires_grad (bool): Whether the points require gradients.

        Returns:
        List[torch.Tensor]: The interior points.
        """
        pass

    @abstractmethod
    def get_boundary_points(self, requires_grad=True):
        """
        Abstract method to get the boundary points.

        Parameters:
        requires_grad (bool): Whether the points require gradients.

        Returns:
        List[torch.Tensor]: The boundary points.

            .+------+
          .' |    .'|
         +---+--+'  |
         |   |  |   |
       y |  ,+--+---+
         |.'    | .' t
         +------+'
            x
        """
        pass

    @abstractmethod
    def get_interior_points(self, requires_grad=True):
        """
        Abstract method to get the interior points.

        Parameters:
        requires_grad (bool): Whether the points require gradients.

        Returns:
        List[torch.Tensor]: The interior points.
        """
        pass
