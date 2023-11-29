import torch


def initial_condition(x: torch.Tensor, y: torch.Tensor, x_length: float, y_length: float) -> torch.Tensor:
    r = torch.sqrt((x-x_length/2)**2 + (y-y_length/2)**2)
    res = 2 * torch.exp(-(r)**2 * 30) + 2
    return res