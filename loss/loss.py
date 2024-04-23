from typing import Callable
import torch
from environment.env import SimulationEnvironment
from loss.wave_equations import dfdx, dfdy, f
from model.pinn import PINN
from train.params import SimulationParameters


class Loss:
    def __init__(
        self,
        environment: SimulationEnvironment,
        initial_condition: Callable,
        wave_equation: Callable,
    ):
        self.environment = environment
        self.wave_equation = wave_equation
        self.initial_condition = initial_condition

        self.params = SimulationParameters()

    def residual_loss(self, pinn: PINN):
        """
        Calculates the residual loss for the given PINN.

        The residual loss is calculated as
        the mean squared error of the wave equation's output
        at the interior points.

        Args:
            pinn (PINN): The PINN for which to calculate the residual loss.

        Returns:
            torch.Tensor: The calculated residual loss.
        """
        x, y, z, t = self.environment.interior_points
        loss = self.wave_equation(pinn, x, y, z, t, self.params.GRAVITY, self.environment)

        return loss.pow(2).mean()

    def initial_loss(self, pinn: PINN):
        """
        Calculates the initial loss for the given PINN.

        The initial loss is calculated as the difference between
        the PINN's output and the initial condition
        at the initial points.

        Args:
            pinn (PINN): The PINN for which to calculate the initial loss.

        Returns:
            torch.Tensor: The calculated initial loss.
        """
        x, y, t = self.environment.initial_points

        length = self.environment.domain.XY_DOMAIN[1]
        pinn_initial = self.initial_condition(x, y, length)

        return (f(pinn, x, y, t) - pinn_initial).pow(2).mean()

    def boundary_loss(self, pinn: PINN):
        """
        Calculates the boundary loss for the given PINN.

        The boundary loss is calculated as
        the mean squared error of the derivatives of the PINN's output
        with respect to x and y at the boundary points.

        Args:
            pinn (PINN): The PINN for which to calculate the boundary loss.

        Returns:
        torch.Tensor: The calculated boundary loss.
        """
        down, up, left, right = self.environment.boundary_points

        x_down, y_down, t_down = down
        x_up, y_up, t_up = up
        x_left, y_left, t_left = left
        x_right, y_right, t_right = right

        loss_down = dfdy(pinn, x_down, y_down, t_down)
        loss_up = dfdy(pinn, x_up, y_up, t_up)
        loss_left = dfdx(pinn, x_left, y_left, t_left)
        loss_right = dfdx(pinn, x_right, y_right, t_right)

        return sum(map(torch.Tensor.mean,
                       (loss_down.pow(2),
                        loss_up.pow(2),
                        loss_left.pow(2),
                        loss_right.pow(2))))

    def verbose(self, pinn: PINN):
        """
        Returns all parts of the loss function for the given PINN.

        This method is not used during training, only for checking the results later.

        Args:
            pinn (PINN): The PINN for which to calculate the loss components.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: The calculated residual loss and initial loss.
        """
        residual_loss = self.residual_loss(pinn)
        initial_loss = self.initial_loss(pinn)
        boundary_loss = self.boundary_loss(pinn)

        final_loss = \
            self.params.INITIAL_WEIGHT_RESIDUAL * residual_loss + \
            self.params.INITIAL_WEIGHT_INITIAL * initial_loss + \
            self.params.INITIAL_WEIGHT_BOUNDARY * boundary_loss

        return final_loss, residual_loss, initial_loss, boundary_loss

    def __call__(self, pinn: PINN, epoch=None):
        """
        Allows you to use the instance of this class as if it were a function:

        ```
            >>> loss = Loss(*some_args)
            >>> calculated_loss = loss(pinn)
        ```
        """
        return self.verbose(pinn)
