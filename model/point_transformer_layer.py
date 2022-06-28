'''
Neural network architecture description.
Adapted from:
https://github.com/POSTECH-CVLab/point-transformer/blob/master/point_transformer_lib/point_transformer_ops/point_transformer_modules.py
https://github.com/POSTECH-CVLab/point-transformer/blob/master/point_transformer_lib/point_transformer_ops/point_transformer_utils.py
'''

from __init__ import *

# Library imports.
import open3d as o3d
import torch.nn.functional as F
from torch import nn, einsum


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def kNN(query, dataset, k):
    """
    inputs
        query: (B, N0, D) shaped torch gpu Tensor.
        dataset: (B, N1, D) shaped torch gpu Tensor.
        k: int
    outputs
        neighbors: (B * N0, k) shaped torch Tensor.
                   Each row is the indices of a neighboring points.
                   It is flattened along batch dimension.
    """
    assert query.is_cuda and dataset.is_cuda, "Input tensors should be gpu tensors."
    assert query.dim() == 3 and dataset.dim() == 3, "Input tensors should be 3D."
    assert (
        query.shape[0] == dataset.shape[0]
    ), "Input tensors should have same batch size."
    assert (
        query.shape[2] == dataset.shape[2]
    ), "Input tensors should have same dimension."

    B, N1, _ = dataset.shape

    query_o3d = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(query))
    dataset_o3d = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(dataset))

    indices = []
    for i in range(query_o3d.shape[0]):
        _query = query_o3d[i]
        _dataset = dataset_o3d[i]
        nns = o3d.core.nns.NearestNeighborSearch(_dataset)
        status = nns.knn_index()
        if not status:
            raise Exception("Index failed.")
        neighbors, _ = nns.knn_search(_query, k)
        # calculate prefix sum of indices
        # neighbors += N1 * i
        indices.append(torch.utils.dlpack.from_dlpack(neighbors.to_dlpack()))

    # flatten indices
    indices = torch.stack(indices)
    return indices


def kNN_torch(query, dataset, k):
    """
    inputs
        query: (B, N0, D) shaped torch gpu Tensor.
        dataset: (B, N1, D) shaped torch gpu Tensor.
        k: int
    outputs
        neighbors: (B * N0, k) shaped torch Tensor.
                   Each row is the indices of a neighboring points.
                   It is flattened along batch dimension.
    """
    # assert query.is_cuda and dataset.is_cuda, "Input tensors should be gpu tensors."
    assert query.dim() == 3 and dataset.dim() == 3, "Input tensors should be 3D."
    assert (
        query.shape[0] == dataset.shape[0]
    ), "Input tensors should have same batch size."
    assert (
        query.shape[2] == dataset.shape[2]
    ), "Input tensors should have same dimension."

    dists = square_distance(query, dataset)  # dists: [B, N0, N1]
    neighbors = dists.argsort()[:, :, :k]  # neighbors: [B, N0, k]
    # torch.cuda.empty_cache()
    return neighbors


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class PointTransformerLayer(nn.Module):
    def __init__(self, dim, pos_mlp_hidden_dim=32, attn_mlp_hidden_mult=2,
                 num_neighbors=16, dim2=None):
        super().__init__()

        # self.prev_linear = nn.Linear(dim, dim)

        self.num_neighbors = num_neighbors

        if dim2 is None:
            dim2 = dim

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim2, dim, bias=False)
        self.to_v = nn.Linear(dim2, dim, bias=False)

        # Encoding function theta.
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, dim)
        )

        # Mapping function gamma.
        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim * attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(dim * attn_mlp_hidden_mult, dim)
        )

        # self.final_linear = nn.Linear(dim, dim)

    def forward(self, x, pos, x2=None, pos2=None):
        '''
        If x2 and pos2 exist, then we let x (query) attend to x2 (abstract).
        :param x (B, N, D) tensor.
        :param pos (B, N, 3) tensor.
        :param x2 (B, M, D) tensor.
        :param pos2 (B, M, 3) tensor.
        :return agg (B, N, D) tensor.
        '''
        # queries, keys, values

        # NOTE: Seems like they forgot to apply prev_linear

        # x_pre = x

        if x2 is None:
            x2 = x
            pos2 = pos

        knn_idx = kNN_torch(pos, pos2, self.num_neighbors)  # (B, N, K).
        knn_xyz = index_points(pos2, knn_idx)  # (B, N, K, 3).

        q = self.to_q(x)  # (B, N, D).
        k = index_points(self.to_k(x2), knn_idx)  # (B, N, K, D).
        v = index_points(self.to_v(x2), knn_idx)  # (B, N, K, D).

        pos_enc = self.pos_mlp(pos[:, :, None] - knn_xyz)  # (B, N, K, D).

        attn = self.attn_mlp(q[:, :, None] - k + pos_enc)  # (B, N, K, D).
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # (B, N, K, D).

        agg = einsum('b i j d, b i j d -> b i d', attn, v + pos_enc)  # (B, N, D).

        # agg = self.final_linear(agg) + x_pre

        return agg
