import torch
from typing import Union


def f(pinn: "PINN",  # noqa: F821 # type: ignore
      x: torch.Tensor,
      y: torch.Tensor,
      t: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""
    return pinn(x, y, t)


def df(output: torch.Tensor, input: torch.Tensor, order: int = 1) -> torch.Tensor:
    """    
    Compute the nth-order derivative of a tensor with respect to the input tensor.

    Args:
        output (torch.Tensor): The output tensor, usually a result of some computation.
        input (torch.Tensor): The input tensor with respect to which the derivative is computed.
        order (int): The order of the derivative to compute. Default is 1 (first derivative).

    Returns:
        torch.Tensor: The computed nth-order derivative tensor.
    """
    df_value = output
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            input,
            grad_outputs=torch.ones_like(input),
            create_graph=True,
            retain_graph=True,
        )[0]

    return df_value


def dfdt(pinn: "PINN",  # noqa: F821 # type: ignore
         x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
         u: Union[torch.Tensor, None] = None, order: int = 1):
    """u is a precalculated f() value"""
    u = u if u is not None else f(pinn, x, y, t)
    return df(u, t, order=order)


def dfdx(pinn: "PINN",  # noqa: F821 # type: ignore
         x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
         u: Union[torch.Tensor, None] = None, order: int = 1):
    u = u if u is not None else f(pinn, x, y, t)
    return df(u, x, order=order)


def dfdy(pinn: "PINN",  # noqa: F821 # type: ignore
         x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
         u: Union[torch.Tensor, None] = None, order: int = 1):
    u = u if u is not None else f(pinn, x, y, t)
    return df(u, y, order=order)


# Simplified for z(x,y) = 0
def wave_equation_simplified(
        pinn: "PINN",  # noqa: F821 # type: ignore
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        G: float,
        _env: "SimulationEnvironment"):  # noqa: F821 # type: ignore
    u = f(pinn, x, y, t)
    return dfdt(pinn, x, y, t, u, order=2) - \
        G * (dfdx(pinn, x, y, t, u) ** 2 + dfdy(pinn, x, y, t, u) ** 2 +
             u * (dfdx(pinn, x, y, t, u, order=2) + dfdy(pinn, x, y, t, u, order=2)))


def wave_equation(
        pinn: "PINN",  # noqa: F821 # type: ignore
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        G: float,
        env: "SimulationEnvironment"):  # noqa: F821 # type: ignore
    u = f(pinn, x, y, t)
    dzdx = env.partial_x(x, y).to(env.device)
    dzdy = env.partial_y(x, y).to(env.device)

    return dfdt(pinn, x, y, t, u, order=2) - \
        G * ((u - z) * (dfdx(pinn, x, y, t, u, order=2) + dfdy(pinn, x, y, t, u, order=2)) +
             (dfdx(pinn, x, y, t, u) - dzdx) * dfdx(pinn, x, y, t, u) +
             (dfdy(pinn, x, y, t, u) - dzdy) * dfdy(pinn, x, y, t, u))
