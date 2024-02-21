import torch


def initial_condition(x: torch.Tensor, y: torch.Tensor, xy_length: float) -> torch.Tensor:
    r = torch.sqrt((x-xy_length/2)**2 + (y-xy_length/2)**2)
    res = 0.5 * torch.exp(-(r)**2 * 50) + 0.4
    return res

# def initial_condition(x: torch.Tensor, y: torch.Tensor, xy_length: float) -> torch.Tensor:
#     base_height = 0.5
#     k = 120
#     n = 0.4
#     x_center = xy_length/5
#     y_center = xy_length/2

#     r = torch.sqrt((x-x_center)**2 + (y-y_center)**2)
#     res = n * torch.exp(-(r)**2 * k) + base_height
#     return res

def initial_condition(x: torch.Tensor, y: torch.Tensor, xy_length: float) -> torch.Tensor:
    base_height = 0.5476
    k = 120
    n = 0.4
    x_center = xy_length/5
    y_center = xy_length/2

    r = torch.sqrt((x-x_center)**2 + (y-y_center)**2)
    res = n * torch.exp(-(r)**2 * k) + base_height
    return res