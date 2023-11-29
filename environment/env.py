from abc import ABC, abstractmethod
from typing import List

from utils.singleton import Singleton

class SimulationEnvironment(ABC):
    def __init__(self):
        self.initial_points = None
        self.boundary_points = None
        self.interior_points = None

    @abstractmethod
    def get_initial_points(self):
        pass

    @abstractmethod
    def get_boundary_points(self):
        pass

    @abstractmethod
    def get_interior_points(self):
        pass