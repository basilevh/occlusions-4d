'''
Neural network architecture description.
Based on:
https://arxiv.org/pdf/2012.09164.pdf
Created by Basile Van Hoorick for Revealing Occlusions with 4D Neural Fields.
'''

from __init__ import *

# Library imports.
import torch_cluster

# Internal imports.
import geometry
import point_transformer_layer


class PointTransformerBlock(torch.nn.Module):
    '''
    Linear + Point Transformer Layer + Linear.
    '''

    def __init__(self, d_in, d_hidden, d_out, num_neighbors=16,
                 d_hidden_abstract=None):
        '''
        :param d_in, d_hidden, d_out (int): Number of input, hidden, and output features per point
            respectively.
        '''
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.num_neighbors = num_neighbors

        self.layer1 = torch.nn.Linear(
            d_in, d_hidden)
        
        self.layer2 = point_transformer_layer.PointTransformerLayer(
            d_hidden, pos_mlp_hidden_dim=32, attn_mlp_hidden_mult=2,
            num_neighbors=num_neighbors, dim2=d_hidden_abstract)
        
        self.layer3 = torch.nn.Linear(
            d_hidden, d_out)

    def forward(self, x, p, x2=None, p2=None):
        '''
        If x2 and p2 exist, then we let x (query) attend to x2 (abstract).
        :param x, x2 (B, N, d_in) tensor: Point features.
        :param p, p2 (B, N, 3) tensor: Point coordinates.
        :return (z, p, scalar_attn?, knn_idx?): Output point features, coordinates,
                and coordinate indices.
            z (B, N, d_out) tensor.
            p (B, N, 3) tensor.
            scalar_attn (B, N, K) tensor.
            knn_idx (B, N, K) tensor.
        '''
        assert x.shape[:2] == p.shape[:2]
        if x2 is not None:
            assert x2.shape[:2] == p2.shape[:2]

        y = self.layer1(x)
        scalar_attn = None
        y = self.layer2(y, p, x2=x2, pos2=p2)
        y = self.layer3(y)
        z = x + y

        return (z, p)


class DownTransition(torch.nn.Module):
    '''
    Farthest Point Sampling + kNN / MLP + Local Max Pooling.
    Typically used in the encoding stage of a point transformer model.
    '''

    def __init__(self, d_in, d_out, factor=2, knn_k=8, norm_type='none', fps_random_start=True):
        '''
        :param d_in, d_out (int): Number of input and output features per point respectively.
        :param factor (int): Downsampling ratio in terms of number of points.
        :param knn_k (int): Number of nearest neighbors to consider for max-pooling of features.
        :param norm_type (str): Normalization layer type.
        :param fps_random_start (bool): Whether farthest point sampling should incorporate random
            choices. Set to False for deterministic inference.
        '''
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.factor = factor
        self.knn_k = knn_k
        self.norm_type = norm_type
        self.fps_random_start = fps_random_start

        if norm_type == 'none':
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(d_in, d_out),
                torch.nn.ReLU())
        
        elif norm_type == 'batch':
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(d_in, d_out),
                torch.nn.BatchNorm1d(d_out, eps=1e-3),  # , momentum=0.005),
                torch.nn.ReLU())
        
        elif norm_type == 'layer':
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(d_in, d_out),
                torch.nn.LayerNorm(d_out),
                torch.nn.ReLU())
        
        else:
            raise ValueError()

    def forward(self, x, p):
        '''
        NOTE: No perfect sub/super-set alignment should be assumed.
        :param x (B, N, d_in) tensor: Point features.
        :param p (B, N, 3) tensor: Point coordinates.
        :return (z, p_sub, scalar_attn?, knn_idx?).
            z (B, N/factor, d_out) tensor.
            p_sub (B, N/factor, 3) tensor.
            scalar_attn (??) tensor.
            knn_idx (??) tensor.
        '''
        assert x.shape[:2] == p.shape[:2]
        (B, N, d_in) = x.shape
        N_new = int(np.ceil(N / self.factor))

        p_flat = p.view(B * N, 3)
        batch = torch.arange(B).repeat_interleave(N)  # (B*N).
        batch = batch.to(x.device)
        # https://github.com/rusty1s/pytorch_cluster/blob/master/torch_cluster/fps.py
        # NOTE / WARNING: This fps call has inherent randomness unless random_start is False!
        inds = torch_cluster.fps(
            src=p_flat, batch=batch, ratio=1.0 / self.factor, random_start=self.fps_random_start)
        inds = torch.sort(inds)[0]  # (B*N/factor).

        p_sub_flat = p_flat[inds]  # (B*N/factor, 3).
        # (B*N/factor).
        batch_sub = torch.arange(B).repeat_interleave(N_new)
        batch_sub = batch_sub.to(x.device)
        # https://github.com/rusty1s/pytorch_cluster/blob/master/torch_cluster/knn.py
        knn_inds = torch_cluster.knn(
            x=p_flat, y=p_sub_flat, k=self.knn_k, batch_x=batch, batch_y=batch_sub)
        # (2, k*B*N/factor).
        # (B*N/factor, k).
        knn_inds = knn_inds[1].view(B * N_new, self.knn_k)
        # knn_inds describes, for every subsampled point, where to find the nearest k points in the
        # original, larger point cloud, as indicated a row of indices.

        x_flat = x.view(B * N, self.d_in)

        y_flat = self.mlp(x_flat)  # (B*N, d_out).

        # Perform local max pooling.
        # Naive: y_flat = y_flat[inds]
        z_flat = y_flat[knn_inds[:, 0]]  # (B*N/factor, d_out).
        for i in range(1, self.knn_k):
            z_flat = torch.maximum(z_flat, y_flat[knn_inds[:, i]])

        z = z_flat.view(B, N_new, self.d_out)
        p_sub = p_sub_flat.view(B, N_new, 3)

        return (z, p_sub)


class UpTransition(torch.nn.Module):
    '''
    Linear + Interpolation + Summation.
    Typically used in the decoding stage of a point transformer model.
    '''

    def __init__(self, d_in, d_out, factor=2, knn_k=3,
                 skip_connections=False, norm_type='none'):
        '''
        :param d_in, d_out (int): Number of input and output features per point respectively.
        :param factor (int): Upsampling ratio in terms of number of points.
        :param knn_k (int): Number of nearest neighbors to consider for trilinear interpolation of
            features.
        :param skip_connections (bool): If True, also accept features and points from the
            corresponding earlier layer at the encoding stage.
        :param norm_type (str): Normalization layer type.
        '''
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.factor = factor
        self.knn_k = knn_k
        self.skip_connections = skip_connections
        self.norm_type = norm_type
        assert skip_connections, 'Cannot upsample coordinates this way.'

        if skip_connections:

            if norm_type == 'none':
                self.mlp1 = torch.nn.Sequential(
                    torch.nn.Linear(d_in, d_out),
                    torch.nn.ReLU())
                self.mlp2 = torch.nn.Sequential(
                    torch.nn.Linear(
                        d_out, d_out),
                    torch.nn.ReLU())

            elif norm_type == 'batch':
                self.mlp1 = torch.nn.Sequential(
                    torch.nn.Linear(d_in, d_out),
                    torch.nn.BatchNorm1d(d_out, eps=1e-3),  # , momentum=0.005),
                    torch.nn.ReLU())
                self.mlp2 = torch.nn.Sequential(
                    torch.nn.Linear(
                        d_out, d_out),
                    torch.nn.BatchNorm1d(d_out, eps=1e-3),  # , momentum=0.005),
                    torch.nn.ReLU())

            elif norm_type == 'layer':
                self.mlp1 = torch.nn.Sequential(
                    torch.nn.Linear(d_in, d_out),
                    torch.nn.LayerNorm(d_out),
                    torch.nn.ReLU())
                self.mlp2 = torch.nn.Sequential(
                    torch.nn.Linear(
                        d_out, d_out),
                    torch.nn.LayerNorm(d_out),
                    torch.nn.ReLU())

            else:
                raise ValueError()

        else:
            # Special case because we have no reference with the supersampled amount of points.
            # This means we have to create more points "out of nowhere". To achieve this, simply
            # partition the output feature channels of this MLP.

            if norm_type == 'none':
                self.mlp1 = torch.nn.Sequential(
                    torch.nn.Linear(
                        d_in, d_out * factor),
                    torch.nn.ReLU())
            
            elif norm_type == 'batch':
                self.mlp1 = torch.nn.Sequential(
                    torch.nn.Linear(
                        d_in, d_out * factor),
                    torch.nn.BatchNorm1d(d_out * factor, eps=1e-3),  # , momentum=0.005),
                    torch.nn.ReLU())
            
            elif norm_type == 'layer':
                self.mlp1 = torch.nn.Sequential(
                    torch.nn.Linear(
                        d_in, d_out * factor),
                    torch.nn.LayerNorm(d_out * factor),
                    torch.nn.ReLU())
            
            else:
                raise ValueError()

    def forward(self, x1, p1, x2=None, p2=None):
        '''
        NOTE: No perfect sub/super-set alignment should be assumed.
        :param x1 (B, N/factor, d_in) tensor: Point features.
        :param p1 (B, N/factor, 3) tensor: Point coordinates.
        :param x2 (B, N, d_out) tensor: Skip connection point features.
        :param p2 (B, N, 3) tensor: Skip connection point coordinates.
        :return (y, p) where y = (B, N, d_out) tensor, and p = (B, N, 3) = p2 or new.
        '''
        assert x1.shape[:2] == p1.shape[:2]
        (B, N_in) = x1.shape[:2]
        N_out = N_in * self.factor

        if self.skip_connections:
            assert x2 is not None and p2 is not None

            y1 = self.mlp1(x1)
            y2 = self.mlp2(x2)
            y1_super = geometry.trilinear_interpolation(
                y1, p1, p2, knn_k=self.knn_k)
            # y1_super = geometry.trilinear_interpolation_old(y1, p1, p2)
            y = y1_super + y2
            return y, p2

        else:
            assert x2 is None and p2 is None

            xp1 = torch.cat([x1, p1], dim=-1)  # (B, N/factor, d_in+3).
            y1 = self.mlp1(xp1)  # (B, N/factor, d_out*factor).
            y = y1.view(B, N_out, self.d_out)  # (B, N, d_out).
            # (B, N, 3).
            p_repeat = torch.repeat_interleave(p1, self.factor, dim=1)
            p = p_repeat + self.points_res(y)  # (B, N, 3).
            return y, p
