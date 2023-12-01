from typing import Callable

from environment.env import SimulationEnvironment
from loss.wave_equations import dfdx, dfdy, f
from model.pinn import PINN
from model.weights import Weights
from train.params import SimulationParameters


class Loss:
    def __init__(
        self,
        environment: SimulationEnvironment,
        weights: Weights,
        params: SimulationParameters,
        initial_condition: Callable,
        wave_equation: Callable,
    ):
        self.environment = environment
        self.weights = weights
        self.params = params

        self.wave_equation = wave_equation
        self.initial_condition = initial_condition

    def residual_loss(self, pinn: PINN):
        x, y, z, t = self.environment.interior_points
        loss = self.wave_equation(pinn, x, y, z, t, self.params.GRAVITY, self.environment)

        return loss.pow(2).mean()

    def initial_loss(self, pinn: PINN):
        x, y, t = self.environment.initial_points
        length = self.environment.domain.XY_DOMAIN[1]
        pinn_init = self.initial_condition(x, y, length)
        loss = f(pinn, x, y, t) - pinn_init

        return loss.pow(2).mean()

    def boundary_loss(self, pinn: PINN):
        down, up, left, right = self.environment.boundary_points
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

    def verbose(self, pinn: PINN):
        """
        Returns all parts of the loss function

        Not used during training! Only for checking the results later.
        """
        residual_loss = self.residual_loss(pinn)
        initial_loss = self.initial_loss(pinn)
        boundary_loss = self.boundary_loss(pinn)

        final_loss = \
            self.weights.WEIGHT_RESIDUAL * residual_loss + \
            self.weights.WEIGHT_INITIAL * initial_loss + \
            self.weights.WEIGHT_BOUNDARY * boundary_loss

        return final_loss, residual_loss, initial_loss, boundary_loss

    def __call__(self, pinn: PINN):
        """
        Allows you to use the instance of this class as if it were a function:

        ```
            >>> loss = Loss(*some_args)
            >>> calculated_loss = loss(pinn)
        ```
        """
        return self.verbose(pinn)