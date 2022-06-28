'''
Neural network architecture description.
Created by Basile Van Hoorick for Revealing Occlusions with 4D Neural Fields.
'''

from __init__ import *

# Internal imports.
import modules


class PointCompletionNetV3(torch.nn.Module):
    '''
    Deep neural network based on point transformer.
    Maps one decorated 3D point cloud to another.
    '''

    def __init__(self, mixed_precision=False, n_input=4096, n_output=1024, d_in=6, d_out=6,
                 d_feat=32, down_blocks=3, up_blocks=2, transition_factor=4,
                 pt_num_neighbors=16, pt_norm_type='none', down_neighbors=8, abstract_levels=1,
                 skip_connections=False, enable_decoder=False, output_featurized=True,
                 output_global_emb=True, global_dim=512, fps_random_start=True):
        '''
        :param n_input (int): Size of input point cloud.
        :param n_output (int): Size of output point cloud.
            If < n_input, the factor must match with the network architecture and options.
        :param d_in (int): Number of input features per point
            (for example, 3 if just coordinates, +3 if decorated with RGB, +1 if temporal).
        :param d_out (int): Number of output features per point.
        :param d_feat (int): Initial embedding size for the first MLP.
        :param down_blocks (int): Number of down transition blocks.
        :param up_blocks (int): Number of up transition blocks.
        :param transition_factor (int): Resampling factor for all down and up blocks.
        :param pt_num_neighbors (int): Number of nearest neighbors in the point transformer block
            self attention layer.
        :param pt_norm_type (str): Normalization layer type.
        :param down_neighbors (int): Number of nearest neighbors in the down transition block.
        :param abstract_levels (int): How many hierarchies the output consists of, analogous to
            forwarding skip connections to outside this network.
        :param skip_connections (bool): If True, enable long range connections between corresponding
            down and up blocks. Does not affect within-block operation.
        :param enable_decoder (bool): If True, enable the decoder.
        :param output_featurized (bool): If True, return so-called abstract point cloud, which
            consists of separate embeddings per point.
        :param output_global_emb (bool): If True, average all information after the encoder into a
            single output embedding.
        :param global_dim (int): If output_global_emb, size of the global output embedding.
        :param fps_random_start (bool): Whether farthest point sampling in the down transition
            should incorporate random choices. Set to False for deterministic inference.
        '''
        super().__init__()
        self.mixed_precision = mixed_precision
        self.n_input = n_input
        self.n_output = n_output
        self.d_in = d_in
        self.d_out = d_out
        self.d_feat = d_feat
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        self.transition_factor = transition_factor
        self.pt_num_neighbors = pt_num_neighbors
        self.pt_norm_type = pt_norm_type
        self.down_neighbors = down_neighbors
        self.abstract_levels = abstract_levels
        self.skip_connections = skip_connections
        self.enable_decoder = enable_decoder
        self.output_featurized = output_featurized
        self.output_global_emb = output_global_emb
        self.global_dim = global_dim
        self.fps_random_start = fps_random_start

        if enable_decoder:
            assert output_featurized, 'The decoder is useless if we do not output the featurized '\
                'point cloud.'

        # Encoder.
        dim = d_feat
        self.pre_mlp = torch.nn.Sequential(
            torch.nn.Linear(d_in, dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, dim))
        blocks = []

        lpc = n_input
        layer_point_counts = [lpc]
        layer_dims = [dim]
        for _ in range(down_blocks):
            blocks.append(modules.PointTransformerBlock(
                d_in=dim, d_hidden=dim, d_out=dim,
                num_neighbors=pt_num_neighbors))
            blocks.append(modules.DownTransition(
                d_in=dim, d_out=dim * 2, factor=transition_factor, knn_k=down_neighbors,
                norm_type=pt_norm_type, fps_random_start=fps_random_start))
            lpc = lpc // transition_factor
            dim *= 2
            layer_point_counts.append(lpc)
            layer_dims.append(dim)

        # Center.
        blocks.append(modules.PointTransformerBlock(
            d_in=dim, d_hidden=dim, d_out=dim,
            num_neighbors=pt_num_neighbors))
        layer_point_counts.append(layer_point_counts)
        layer_dims.append(dim)
        self.center_block_idx = len(blocks) - 1

        if output_global_emb:
            self.global_mlp = torch.nn.Sequential(
                torch.nn.Linear(dim, global_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(global_dim, global_dim))

        # External skip connections for abstract point cloud.
        if abstract_levels > 1:
            assert not self.skip_connections
            abstract_skip_mlps = []

            for level_idx in range(abstract_levels - 1):
                cur_dim = dim // int(2 ** (abstract_levels - 1 - level_idx))
                abstract_skip_mlps.append(torch.nn.Linear(cur_dim, dim))

            self.abstract_skip_mlps = torch.nn.ModuleList(abstract_skip_mlps)

        # Decoder.
        if enable_decoder:
            for _ in range(up_blocks):
                blocks.append(modules.UpTransition(
                    d_in=dim, d_out=dim // 2, factor=transition_factor, knn_k=3,
                    skip_connections=skip_connections,
                    norm_type=pt_norm_type))
                blocks.append(modules.PointTransformerBlock(
                    d_in=dim // 2, d_hidden=dim // 2, d_out=dim // 2,
                    num_neighbors=pt_num_neighbors))
                lpc *= transition_factor
                dim = dim // 2
                layer_point_counts.append(lpc)
                layer_dims.append(dim)

            assert lpc == n_output

            self.post_mlp = torch.nn.Sequential(
                torch.nn.Linear(dim, dim),
                torch.nn.ReLU(),
                torch.nn.Linear(dim, d_out - 3))

        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, pcl, return_intermediate):
        '''
        :param pcl (B, N, D) tensor.
        :param return_intermediate (bool): Save coordinates at every step, e.g. for debugging.
        :return (pcl_out, x_global, layer_coords).
            pcl_out: (B, N, D) tensor, if output_featurized.
            x_global: (B, F) tensor, if output_global_emb.
            layer_coords: List of (B, N, 3) tensors, if return_intermediate.
        '''
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):

            # NOTE: Coordinates always come first in the input and output point clouds,
            # but this order is inverted during processing by the neural network layers.
            if return_intermediate:
                layer_coords = []
                layer_coords.append(pcl[..., :3])
            else:
                layer_coords = None

            x0 = self.pre_mlp(pcl)
            pos0 = pcl[..., :3]

            if return_intermediate:
                layer_coords.append(pos0)

            x, pos = x0, pos0
            skip_data = []  # List of (x, pos) tuples for UpTransitions blocks.
            x_global = None

            for i, block in enumerate(self.blocks):
                # Execute block, using early layer data if relevant.
                if self.skip_connections and isinstance(block, modules.UpTransition):
                    x, pos = block(x, pos, *skip_data[-1])
                    skip_data.pop(-1)

                else:
                    # In case of PointTransformerBlock or DownTransition.
                    (x, pos) = block(x, pos)

                # Get global embedding if desired.
                if self.output_global_emb and i == self.center_block_idx:
                    x_avg = torch.mean(x, dim=1)  # (B, 128/256).
                    x_global = self.global_mlp(x_avg)

                # Save coordinates for debugging.
                if return_intermediate:
                    layer_coords.append(pos)

                # Retain data just before down transitions for later layers (internal skip).
                if self.skip_connections and isinstance(block, modules.PointTransformerBlock):
                    if len(skip_data) < self.down_blocks and i < self.center_block_idx:
                        skip_data.append((x, pos))

                # Retain features just after down transitions for external skip connections.
                if self.abstract_levels > 1 and isinstance(block, modules.DownTransition):
                    for j, abstract_skip_mlp in enumerate(self.abstract_skip_mlps):
                        if abstract_skip_mlp.in_features == x.shape[-1]:
                            y = abstract_skip_mlp(x)  # (B, N, 128) to (B, N, 256).
                            y[..., -1] = j + 1.0
                            skip_data.append(torch.cat([pos, y], dim=-1))  # (B, N, 259).

            if self.enable_decoder:
                x2 = self.post_mlp(x)
                pos2 = pos0
                pcl_out = torch.cat([pos2, x2], dim=-1)  # (B, N, 6).

                if return_intermediate:
                    layer_coords.append(pcl_out[..., :3])

            elif self.output_featurized:
                # NOTE: We do not use post_mlp in this case, but directly output the latest
                # information.
                pcl_out = torch.cat([pos, x], dim=-1)  # (B, N, ??).

                # Concatenate with external abstract skip information if available. The last
                # feature of every embedding now indicates the hierarchy level.
                if self.abstract_levels > 1:
                    pcl_out[..., -1] = self.abstract_levels
                    assert len(skip_data) == self.abstract_levels - 1
                    skip_data = torch.cat(skip_data, dim=1)  # (B, N, 259).
                    pcl_out = torch.cat([skip_data, pcl_out], dim=1)  # (B, N, 259).

            else:
                pcl_out = None

            return (pcl_out, x_global, layer_coords)
