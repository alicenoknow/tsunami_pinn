import torch


def make_initial_condition(
        base_height: float,
        decay_rate: float,
        peak_height: float,
        x_divisor: float,
        y_divisor: float):
    """
    Creates an initial condition function for a 2D Gaussian distribution.

    Parameters:
    base_height (float): The base height of the distribution.
    decay_rate (float): The decay rate of the distribution.
    peak_height (float): The peak height of the distribution.
    x_divisor (float): The divisor for the x-coordinate of the distribution's center.
    y_divisor (float): The divisor for the y-coordinate of the distribution's center.

    Returns:
    Callable: A function that takes x and y coordinates and a length,
    and returns a 2D Gaussian distribution.
    """
    def initial_condition(x: torch.Tensor, y: torch.Tensor, xy_length: float) -> torch.Tensor:
        x_center = xy_length / x_divisor
        y_center = xy_length / y_divisor

        r = torch.sqrt((x - x_center)**2 + (y - y_center)**2)
        return peak_height * torch.exp(-(r)**2 * decay_rate) + base_height

    return initial_condition
