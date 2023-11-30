from abc import ABC, abstractmethod
from typing import List

class SimulationEnvironment(ABC):
    def __init__(self):
        self.domain = None
        self.initial_points = None
        self.boundary_points = None
        self.interior_points = None

    @abstractmethod
    def get_initial_points(self, n_points: int, requires_grad=True):
        pass

    @abstractmethod
    def get_boundary_points(self, requires_grad=True):
        pass

    @abstractmethod
    def get_interior_points(self, requires_grad=True):
        pass