import torch


# def initial_condition(x: torch.Tensor, y: torch.Tensor, xy_length: float) -> torch.Tensor:
#     r = torch.sqrt((x-xy_length/2)**2 + (y-xy_length/2)**2)
#     res = 2 * torch.exp(-(r)**2 * 30) + 0.8
#     return res

def initial_condition(x: torch.Tensor, y: torch.Tensor, xy_length: float) -> torch.Tensor:
    base_height = 0.5
    alpha = 120
    r = torch.sqrt((x-xy_length/5)**2 + (y-xy_length/2)**2)
    res = 0.4 * torch.exp(-(r)**2 * alpha) + base_height
    return res