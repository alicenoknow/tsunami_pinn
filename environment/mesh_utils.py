import meshio
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