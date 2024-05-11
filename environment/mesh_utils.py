import meshio
import numpy as np
import torch

from sympy import diff, symbols, simplify, lambdify
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def dump_points(filename: str, z_relative_to_x: bool = True):
    """
    Reads triangular mesh from file.

    Values are normalized with respect to x, to keep relation between dimensions.
    Flag z_relative_to_x determines whether z is scaled to (0, 1) separately or with respect to x.

    Returns vectors x, y, z where (x[i], y[i], z[i]) was the original point in mesh.
    """
    mesh = meshio.avsucd.read(filename)
    points = torch.tensor(mesh.points, dtype=torch.float32)

    x, y, z = points.transpose(0, 1)

    min_x, min_y, min_z = torch.min(x), torch.min(y), torch.min(z)
    max_x, max_y, max_z = torch.max(x), torch.max(y), torch.max(z)

    x = (x - min_x) / (max_x - min_x)
    y = (y - min_x) / (max_x - min_x)

    if z_relative_to_x:
        z = z / (max_x - min_x)
    else:
        z = z / (max_z - min_z)

    return x, y, z


def calculate_partial_derivatives(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    x, y, z = x.detach().numpy(), y.detach().numpy(), z.detach().numpy()

    degree = 3
    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(np.column_stack((x, y)))
    lin_reg = LinearRegression()
    lin_reg.fit(x_poly, z)

    x_values = np.linspace(np.min(x), np.max(x), 100)
    y_values = np.linspace(np.min(y), np.max(y), 100)
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)

    x_mesh_flat = x_mesh.ravel()
    y_mesh_flat = y_mesh.ravel()
    xy_mesh = np.column_stack((x_mesh_flat, y_mesh_flat))
    xy_poly = poly_features.transform(xy_mesh)

    lin_reg.predict(xy_poly)

    coefficients = lin_reg.coef_
    intercept = lin_reg.intercept_
    x, y = symbols('x y')
    terms = [intercept]
    for i in range(degree):
        for j in range(degree):
            if i + j <= degree:
                terms.append(coefficients[i * (degree + 1) + j] * x**i * y**j)

    polynomial_eq = simplify(sum(terms))

    partial_derivative_x = diff(polynomial_eq, x)  # Partial derivative w.r.t. x
    partial_derivative_y = diff(polynomial_eq, y)

    f_x = lambdify((x, y), partial_derivative_x, modules="numpy")
    f_y = lambdify((x, y), partial_derivative_y, modules="numpy")

    return f_x, f_y
