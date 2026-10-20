"""6-nearest-neighbor voxel graph construction and PyG Data dataset assembly."""
import torch
from torch_geometric.data import Data


def build_6nn_edge_index(voxel_coords):
    coord_to_idx = {tuple(c): i for i, c in enumerate(voxel_coords)}
    neigh = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    edges = []
    for i, (x, y, z) in enumerate(voxel_coords):
        for dx, dy, dz in neigh:
            j = coord_to_idx.get((x + dx, y + dy, z + dz), None)
            if j is not None:
                edges.append([i, j])

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def make_graph_dataset(X_scaled, y, edge_idx):
    data_list = []
    eidx = edge_idx.clone()
    for i in range(X_scaled.shape[0]):
        x_i = torch.tensor(X_scaled[i].reshape(-1, 1), dtype=torch.float32)
        y_i = torch.tensor(y[i], dtype=torch.long)
        data_list.append(Data(x=x_i, edge_index=eidx, y=y_i))
    return data_list