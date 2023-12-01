# import torch

# from conditions.initial import initial_condition
# from environment.mesh_env import MeshEnvironment
# from loss.loss import Loss
# from loss.wave_equations import wave_equation_simplified
# from model.pinn import PINN
# from model.weights import Weights
# from train.params import SimulationParameters
# from train.training import Training
# from environment.simple_env import SimpleEnvironment


import numpy as np

def estimate_partial_derivatives(vertices, epsilon=1e-6):
    # Create an empty array to store partial derivatives for each vertex
    partial_derivatives = np.zeros((len(vertices), 3))  # Three columns for dx, dy, dz
    
    for i, (x, y, z) in enumerate(vertices):
        # Perturb x coordinate and calculate z values
        z_perturbed_x_plus = z_perturbed_x_minus = z
        z_perturbed_x_plus += epsilon
        z_perturbed_x_minus -= epsilon
        
        # Perturb y coordinate and calculate z values
        z_perturbed_y_plus = z_perturbed_y_minus = z
        z_perturbed_y_plus += epsilon
        z_perturbed_y_minus -= epsilon
        
        # Perturb z coordinate and calculate z values
        z_perturbed_z_plus = z_perturbed_z_minus = z
        z_perturbed_z_plus += epsilon
        z_perturbed_z_minus -= epsilon
        
        # Compute partial derivatives using finite differences
        dz_dx = (z_perturbed_x_plus - z_perturbed_x_minus) / (2 * epsilon)
        dz_dy = (z_perturbed_y_plus - z_perturbed_y_minus) / (2 * epsilon)
        dz_dz = (z_perturbed_z_plus - z_perturbed_z_minus) / (2 * epsilon)
        
        # Store the partial derivatives for the vertex
        partial_derivatives[i] = [dz_dx, dz_dy, dz_dz]
    
    return partial_derivatives

# Example vertices of the mesh (replace this with your vertex data)
# Each row represents (x, y, z) coordinates of a vertex
vertices = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
    # Add more vertices here as needed
])

# Calculate partial derivatives for the given vertices
partial_derivatives = estimate_partial_derivatives(vertices)

# Print the estimated partial derivatives for each vertex
print("Partial derivatives (dz/dx, dz/dy, dz/dz) for each vertex:")
print(partial_derivatives)
