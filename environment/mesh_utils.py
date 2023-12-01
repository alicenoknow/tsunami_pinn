import meshio
import numpy as np
import torch

from sympy import diff, symbols, simplify, lambdify
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def dump_points(filename: str):
    mesh = meshio.avsucd.read(filename)
    points = torch.tensor(mesh.points, dtype=torch.float32)
    x,y,z = points.transpose(0,1)

    min_x, min_y, min_z = torch.min(x), torch.min(y), torch.min(z)
    max_x, max_y, max_z = torch.max(x), torch.max(y), torch.max(z)

    x = (x - min_x) / (max_x - min_x)
    y = (y - min_y) / (max_y - min_y)
    z = (z - min_z) / (max_z - min_z)

    return x,y,z

# TODO take x_val, y_val and z_val as arguments
def floor(x, y, mesh):
    x_val, y_val, z_val = dump_points(mesh)
    indices = np.where((x_val == x) & (y_val == y))[0]

    if len(indices) > 0:
        return z_val[indices[0]]
    else:
        return None
    


def calculate_partial_derivatives(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    x,y,z  = np.array(x), np.array(y), np.array(z)

    degree = 3
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(np.column_stack((x, y)))
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, z)

    x_values = np.linspace(np.min(x), np.max(x), 100)
    y_values = np.linspace(np.min(y), np.max(y), 100)
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    z_mesh = np.zeros_like(x_mesh)

    x_mesh_flat = x_mesh.ravel()
    y_mesh_flat = y_mesh.ravel()
    xy_mesh = np.column_stack((x_mesh_flat, y_mesh_flat))
    xy_poly = poly_features.transform(xy_mesh)

    z_mesh_flat = lin_reg.predict(xy_poly)
    z_mesh = z_mesh_flat.reshape(x_mesh.shape)

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