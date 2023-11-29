import torch


def initial_condition(x: torch.Tensor, y: torch.Tensor, xy_length: float) -> torch.Tensor:
    r = torch.sqrt((x-xy_length/2)**2 + (y-xy_length/2)**2)
    res = 2 * torch.exp(-(r)**2 * 30) + 2
    return res