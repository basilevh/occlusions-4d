'''
Neural network architecture description.
Some parts are adapted from:
https://github.com/sxyu/pixel-nerf
Created by Basile Van Hoorick for Revealing Occlusions with 4D Neural Fields.
'''

from __init__ import *

# Library imports.
from multiprocessing import Value
import torch.nn.functional as F
import torch.autograd.profiler as profiler

# Internal imports.
import geometry
import modules


def positional_encode(points, base_frequency, num_powers):
    '''
    :param points (..., 4) tensor with (x, y, z, t).
    :param base_frequency (float): First frequency for sin and cos.
        NOTE: Because of periodicity, this value should be such that the extent (diameter) of the
        scene is never larger than 1 / base_frequency.
    :param num_powers (int): F = number of powers of 2 of the base frequency to use.
    :return (..., 4*(F*2+1)) tensor with Fourier encoded coordinates.
    '''
    result = []

    # Calculate and include all F powers of two.
    for p in range(num_powers):
        cur_freq = base_frequency * (2 ** p)
        omega = cur_freq * np.pi * 2.0
        sin = torch.sin(points * omega)  # (..., 4).
        cos = torch.cos(points * omega)  # (..., 4).
        result.append(sin)
        result.append(cos)

    # Include original coordinates as well to ensure no information is lost.
    result = torch.cat([points, *result], dim=-1)

    return result


class Swish(torch.nn.Module):
    '''
    https://arxiv.org/pdf/1710.05941.pdf
    '''

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


def instantiate_activation_fn(activation_str):
    if activation_str == 'relu':
        return torch.nn.ReLU()
    elif activation_str == 'swish':
        return Swish()
    else:
        raise ValueError('Unknown activation: ' + str(activation_str))


# Resnet Blocks
class ResnetBlockFC(torch.nn.Module):

    def __init__(self, d_in=64, d_hidden=256, d_out=64, activation='relu'):
        '''
        Fully connected ResNet Block class. Taken from DVR code.
        :param d_in (int): input dimension.
        :param d_hidden (int): hidden dimension.
        :param d_out (int): output dimension.
        :param activation (str): relu / swish.
        '''
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out

        self.fc_0 = torch.nn.Linear(d_in, d_hidden, bias=True)
        self.fc_1 = torch.nn.Linear(d_hidden, d_out, bias=True)
        self.activation = instantiate_activation_fn(activation)

        if d_in == d_out:
            self.shortcut = None
        else:
            self.shortcut = torch.nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        net = self.fc_0(self.activation(x))
        dx = self.fc_1(self.activation(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetFC(torch.nn.Module):
    '''
    Regular continuous representation (CR) similar to pixelNeRF that also supports positional
        encoding and relu / swish activation.
    '''

    def __init__(self, mixed_precision=False, d_in=4, d_hidden=256, d_out=64, d_latent=256,
                 n_blocks=5, pos_encoding_freqs=0, activation='relu'):
        '''
        :param d_in (int): Input size (0 = disable).
        :param d_hidden (int): H = hidden dimension throughout network.
        :param d_out (int): G = output size.
        :param d_latent (int): D = latent size, added in each resnet block (0 = disable).
        :param n_blocks (int): number of resnet blocks.
        :param pos_encoding_freqs (int): F = number of frequencies to use for Fourier positional
            encoding of all input coordinates (see NeRF & pixelNeRF).
        :param activation (str): relu / swish.
        '''
        super().__init__()
        self.mixed_precision = mixed_precision
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.d_latent = d_latent
        self.n_blocks = n_blocks
        self.pos_encoding_freqs = pos_encoding_freqs

        if pos_encoding_freqs > 0:
            # Pass-through plus sin and cos per frequency.
            self.actual_d_in = d_in * (pos_encoding_freqs * 2 + 1)
        else:
            self.actual_d_in = d_in

        if self.actual_d_in > 0:
            self.lin_in = torch.nn.Linear(self.actual_d_in, d_hidden, bias=True)

        self.lin_out = torch.nn.Linear(d_hidden, d_out, bias=True)

        self.blocks = torch.nn.ModuleList(
            [ResnetBlockFC(d_in=d_hidden, d_hidden=d_hidden, d_out=d_hidden,
                           activation=activation) for _ in range(n_blocks)])

        if d_latent > 0:
            self.lin_z = torch.nn.ModuleList(
                [torch.nn.Linear(d_latent, d_hidden, bias=True) for _ in range(n_blocks)])

        self.activation = instantiate_activation_fn(activation)

    def forward(self, points, features):
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            return self.do_forward(points, features)

    def do_forward(self, points, features):
        '''
        :param points (B, N, 4) tensor with (x, y, z, t).
        :param features (B, D) or (B, N, D) tensor.
            NOTE: If features is (B, D), every example is considered to have one global embedding
            that steers all generated features for points.
        :return (output, penult): (B, N, G) and (B, N, H) tensors.
        '''
        if len(points.shape) == 2:
            points = points.unsqueeze(0)  # (N, 4) => (B, N, 4) with B = 1.
            features = features.unsqueeze(0)
            no_batch = True
        else:
            no_batch = False

        assert points.shape[0] == features.shape[0]
        (B, N, _) = points.shape
        if len(features.shape) == 2:
            (B, D) = features.shape
        else:
            assert points.shape[1] == features.shape[1]
            (B, N, D) = features.shape

        assert points.shape[-1] == self.d_in
        assert features.shape[-1] == self.d_latent

        if self.d_in > 0:
            if self.pos_encoding_freqs > 0:
                base_frequency = 0.1
                points = positional_encode(points, base_frequency, self.pos_encoding_freqs)
            x = self.lin_in(points)  # (B, N, H).
        else:
            x = torch.zeros((B, self.d_hidden), device=features.device)  # (B, H).

        # Loop over all blocks.
        for blkid in range(self.n_blocks):
            if self.d_latent > 0:
                # Add input features to the current representation in a residual manner.
                z = self.lin_z[blkid](features)  # (B, H) or (B, N, H).
                if len(z.shape) == 2:
                    z = z.unsqueeze(1).expand_as(x)  # (B, H) => (B, N, H).
                x = x + z  # (B, N, H).
            x = self.blocks[blkid](x)  # (B, N, H).

        penult = x  # (B, N, H).
        x = self.activation(x)  # (B, N, H).
        output = self.lin_out(x)  # (B, N, G).

        if no_batch:
            output = output.squeeze(0)
            penult = penult.squeeze(0)

        return (output, penult)


class LocalPclResnetFC(ResnetFC):
    '''
    Upgrades ResnetFC with local CR feature conditioning and cross attention functionality
    for 3D point clouds.
    '''

    def __init__(self, num_local_features=0, local_mode='attention', d_latent_local=64,
                 cross_attn_neighbors=12, cross_attn_layers=1,
                 cr_attn_type='cccccccccc', **kwargs):
        '''
        :param num_local_features (int): If > 0, number of spatially nearest neighbors whose
            features to incorporate in addition to the global embedding.
        :param local_mode (str): If feature: Manual interpolation.
            If attention: Use point transformer block for query-to-abstract vector attention.
        :param d_latent_local (int): Local abstract point cloud embedding size.
        :param cross_attn_neighbors (int): Number of nearest neighbors in the attention layer.
        :param cross_attn_layers (int): Number of attention layers to use.
        :param cr_attn_type (str): Type of each attention layer (length = cross_attn_layers).
        '''
        self.num_local_features = num_local_features
        self.local_mode = local_mode
        self.d_latent_local = d_latent_local
        self.cross_attn_neighbors = cross_attn_neighbors
        self.cross_attn_layers = cross_attn_layers
        self.cr_attn_type = cr_attn_type
        super().__init__(**kwargs)

        if local_mode == 'attention':
            self.pt_blocks = []
            self.use_pt_inds = []

            for pt_idx in range(cross_attn_layers):

                if cr_attn_type[pt_idx] == 'c':
                    # CROSS-attention layer.
                    # Use total d_latent here for query, and d_latent_local for abstract.
                    pt_block = modules.PointTransformerBlock(
                        d_in=self.d_latent, d_hidden=self.d_latent, d_out=self.d_latent,
                        num_neighbors=cross_attn_neighbors,
                        d_hidden_abstract=d_latent_local)

                elif cr_attn_type[pt_idx] == 's':
                    raise NotImplementedError()
                    # NOTE: This is obsolete because it doesn't quite work well with a neural field!
                    # SELF-attention layer.
                    # Keys are now also from the CR input.
                    pt_block = modules.PointTransformerBlock(
                        d_in=self.d_latent, d_hidden=self.d_latent, d_out=self.d_latent,
                        num_neighbors=cross_attn_neighbors)

                else:
                    raise ValueError()

                self.pt_blocks.append(pt_block)
                use_at_idx = int((pt_idx + 1) * self.n_blocks / (cross_attn_layers + 1))
                self.use_pt_inds.append(use_at_idx)

            self.pt_blocks = torch.nn.ModuleList(self.pt_blocks)
            self.use_pt_inds = {j: i for i, j in enumerate(self.use_pt_inds)}

    def forward(self, points_query, points_abstract, features_global, features_abstract):
        '''
        NOTE: Currently supports B == 1 only.
        :param points_query (B, N, 4) tensor with (x, y, z, t): Spatiotemporal coordinates to query
            the continuous representation at.
        :param points_abstract (B, M, 3) tensor: Coordinates from the downsampled featurized point
            cloud video corresponding to features_global.
        :param features_global (B, D) tensor: Embedding summarizing the entire scene.
        :param features_abstract (B, M, E) tensor: Per-point features of points_abstract.
        :return (output, penult).
            output (B, N, G) tensor.
            penult (B, N, H) tensor.
        '''
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):

            if points_abstract is not None and features_abstract is None:
                # This means coordinates and features were not yet separated.
                pcl_abstract = points_abstract  # (..., 3 + E).
                points_abstract = pcl_abstract[..., :3]  # (..., 3).
                features_abstract = pcl_abstract[..., 3:]  # (..., E).

            if len(points_query.shape) == 2:
                points_query = points_query.unsqueeze(0)  # (N, 4) => (B, N, 4) with B = 1.
                points_abstract = points_abstract.unsqueeze(0) \
                    if points_abstract is not None else None
                features_global = features_global.unsqueeze(0)
                features_abstract = features_abstract.unsqueeze(0) \
                    if features_abstract is not None else None
                no_batch = True
            else:
                no_batch = False

            if self.num_local_features > 0:

                # Verify batch size and abstract point cloud size consistency.
                assert points_query.shape[0] == points_abstract.shape[0]
                assert points_query.shape[0] == features_global.shape[0]
                assert points_query.shape[0] == features_abstract.shape[0]
                assert points_abstract.shape[1] == features_abstract.shape[1]
                (B, N, _) = points_query.shape
                (B, M, _) = points_abstract.shape
                (B, D) = features_global.shape
                (B, M, E) = features_abstract.shape
                if B != 1:
                    print(points_query.shape, points_abstract.shape,
                          features_global.shape, features_abstract.shape)
                assert B == 1

                if self.local_mode in ['feature', 'attention']:

                    # Remove batch dimension.
                    points_query = points_query.squeeze(0)  # (12288, 4).
                    points_abstract = points_abstract.squeeze(0)  # (96, 3).
                    features_global = features_global.squeeze(0)  # (512).
                    features_abstract = features_abstract.squeeze(0)  # (96, 128).
                    abstract = torch.cat([points_abstract, features_abstract], dim=-1)  # (96, 131).

                    (sel_inds, sel_abstract, sel_dists) = geometry.my_knn_torch(
                        points_query, abstract, self.num_local_features,
                        return_inds=True, return_knn=True, return_dists=True)
                    # sel_inds = (N, K) = (12288, 8).
                    sel_points = sel_abstract[..., :3]  # (N, K, 3) = (12288, 8, 3).
                    sel_features = sel_abstract[..., 3:]  # (N, K, E) = (12288, 8, 128).
                    # sel_dists = (N, K) = (12288, 8).

                    # Weight local features by inverse distance: (N, K) * (N, K, E) -> (N, E).
                    weights = 1.0 / (sel_dists + 1e-4)  # (N, K).
                    norm_weights = F.normalize(weights, p=1, dim=-1)  # (N, K).
                    features_local = torch.einsum('ik,ikf->if', norm_weights, sel_features)
                    features_global_exp = features_global[None, :].expand(N, D)
                    # (N, D + E).
                    features_query = torch.cat([features_global_exp, features_local], dim=-1)

                    if self.local_mode == 'attention':
                        # Reinsert batch dimension.
                        points_query = points_query.unsqueeze(0)
                        points_abstract = points_abstract.unsqueeze(0)
                        features_query = features_query.unsqueeze(0)
                        features_abstract = features_abstract.unsqueeze(0)

                        (output, penult) = self.do_forward_attention(
                            points_query, points_abstract, features_query, features_abstract)

                    else:
                        (output, penult) = super(LocalPclResnetFC, self).do_forward(
                            points_query, features_query)

                        # Reinsert batch dimension.
                        output = output.unsqueeze(0)
                        penult = penult.unsqueeze(0)

                elif self.local_mode == 'function':
                    raise NotImplementedError()

                else:
                    raise ValueError()

            else:

                # No additional functionality exists in this case.
                (output, penult) = super(LocalPclResnetFC, self).do_forward(
                    points_query, features_global)

            if no_batch:
                output = output.squeeze(0)
                penult = penult.squeeze(0)

            return (output, penult)

    def do_forward_attention(self, points_query, points_abstract, features_query, features_abstract):
        '''
        Adaptation of do_forward() for query-to-abstract vector attention.
        :param points_query (B, N, 4) tensor with (x, y, z, t): Spatiotemporal coordinates to query
            the continuous representation at.
        :param points_abstract (B, M, 3) tensor: Coordinates from the downsampled featurized point
            cloud video corresponding to features_global.
        :param features_query (B, N, D) tensor: Per-point features of points_query, which is
            generated by interpolating features_abstract to provide a starting point.
        :param features_abstract (B, M, E) tensor: Per-point features of points_abstract.
        :return (output, penult).
            output (B, N, G) tensor.
            penult (B, N, G) tensor.
        '''
        (B, N, _) = points_query.shape
        (B, M, _) = points_abstract.shape
        (B, N, D) = features_query.shape
        (B, M, E) = features_abstract.shape
        assert B == 1
        assert points_query.shape[-1] == self.d_in
        assert D == self.d_latent  # Actually contains global + local.
        assert E == self.d_latent_local

        if self.d_in > 0:
            if self.pos_encoding_freqs > 0:
                base_frequency = 0.1
                points_query = positional_encode(
                    points_query, base_frequency, self.pos_encoding_freqs)
            x = self.lin_in(points_query)  # (B, N, H).
        else:
            x = torch.zeros((B, self.d_hidden), device=features_query.device)  # (B, H).

        # Loop over all blocks.
        for blkid in range(self.n_blocks):
            if self.d_latent > 0:
                # Add input features to the current representation in a residual manner.
                z = self.lin_z[blkid](features_query)  # (B, N, H).
                x = x + z  # (B, N, H).
            x = self.blocks[blkid](x)  # (B, N, H).

            # Perform vector attention in-between the regular layers.
            if blkid in self.use_pt_inds:
                use_pt_idx = self.use_pt_inds[blkid]
                pt_block = self.pt_blocks[use_pt_idx]
                p = points_query[..., :3]  # Ignore time (this info has already been embedded).

                if self.cr_attn_type[use_pt_idx] == 'c':
                    # CROSS-attention layer.
                    x2 = features_abstract
                    p2 = points_abstract

                    (x_new, _) = pt_block(x, p, x2=x2, p2=p2)

                elif self.cr_attn_type[use_pt_idx] == 's':
                    raise NotImplementedError()
                    # NOTE: This is obsolete because it doesn't quite work well with a neural field!
                    # SELF-attention layer.
                    # (x_new, _) = pt_block(x, p)

                x = x_new

        penult = x  # (B, N, H).
        x = self.activation(x)  # (B, N, H).
        output = self.lin_out(x)  # (B, N, G).

        return (output, penult)
