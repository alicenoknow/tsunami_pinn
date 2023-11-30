import meshio
import numpy as np
import torch

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