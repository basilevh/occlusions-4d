'''
Entire training pipeline logic.
Created by Basile Van Hoorick for Revealing Occlusions with 4D Neural Fields.
'''

from __init__ import *

# Library imports.
import torch.cuda.amp
import torch.nn

# Internal imports.
import loss
import utils


_CHECK_NAN_INF = False


class MyTrainPipeline(torch.nn.Module):
    '''
    Wrapper around the entire training iteration such that DataParallel can be leveraged.
    '''

    def __init__(self, networks, point_sampler, device, task, logger, mixed_precision, color_lw,
                 density_lw, segmentation_lw, tracking_lw, color_mode, semantic_classes,
                 past_frames, future_frames, data_kind):
        super().__init__()
        self.pcl_net = networks[0]
        self.implicit_net = networks[1]
        self.point_sampler = point_sampler
        self.device = device
        self.task = task
        self.stage = None
        self.logger = logger
        self.mixed_precision = mixed_precision
        self.color_lw = color_lw
        self.density_lw = density_lw
        self.segmentation_lw = segmentation_lw
        self.tracking_lw = tracking_lw
        self.color_mode = color_mode
        self.semantic_classes = semantic_classes
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.data_kind = data_kind
        self.losses = None  # Instantiated only by set_stage().

    def set_stage(self, stage):
        self.stage = stage
        self.losses = loss.MyLosses(
            stage, self.logger, self.mixed_precision, self.color_lw, self.density_lw,
            self.segmentation_lw, self.tracking_lw, self.color_mode, self.semantic_classes,
            self.past_frames, self.future_frames)

    def forward(self, batch, cur_step):
        '''
        Handles one parallel iteration of the training or validation phase.
        Executes the models and calculates the per-example losses.
        This is all done in a parallelized manner to minimize unnecessary communication.
        :param batch (dict): Element from data loader.
        :param cur_step (int): Current data loader index.
        :return remnant (dict): Combines input and preliminary output information.
        '''
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):

            # Assume data setup: previd (single-view past frames) => last_couple_merged (multi-view future frames).
            rgb = batch['rgb']
            cam_RT = batch['cam_RT']
            cam_K = batch['cam_K']
            pcl_input = batch['pcl_input']
            # (N, 8) with (x, y, z, R, G, B, t, mark_track).
            pcl_target = batch['pcl_target']
            # List of (M, 11) with (x, y, z, cosine_angle, instance_id, semantic_tag, view_idx, R, G, B, mark_track).
            meta_data = batch['meta_data']
            pcl_target_size = meta_data['pcl_target_size']
            valo_ids = meta_data['valo_ids']
            num_valo_ids = meta_data['num_valo_ids']
            live_occl = meta_data['live_occl']  # Unused during training.

            if self.data_kind == 'greater':
                depth = batch['depth']
            elif self.data_kind == 'carla':
                depth = None

            # Move everything to CUDA / GPU.
            pcl_input = pcl_input.to(self.device)
            assert len(pcl_target) == self.past_frames + self.future_frames
            for time_idx in range(self.past_frames + self.future_frames):
                pcl_target[time_idx] = pcl_target[time_idx].to(self.device)

            # Use point transformer model to contextualize / encode / featurize the input.
            return_intermediate = (cur_step % 80 == 0)
            (pcl_abstract, features_global, layer_coords, _) = self.pcl_net(
                pcl_input, return_intermediate, False)

            # pcl_target and pcl_target_size are lists of size past_frames + future_frames.
            # pcl_target describes the visual reconstruction aspects only, i.e. color and geometry.

            points_query = []
            implicit_output = []
            implicit_target = []

            # Loop over all predicted frames / time indices, which encapsulates both past and future.
            for time_idx in range(self.past_frames + self.future_frames):

                (points_query_frame, implicit_output_frame, implicit_target_frame,
                 solid_sbs, air_sbs) = self.handle_frame(
                    time_idx, pcl_target, pcl_target_size,
                    valo_ids, num_valo_ids, live_occl,
                    pcl_abstract, features_global)

                # WARNING: Sometimes we get NaN, appears to be more frequent with mixed precision.
                if _CHECK_NAN_INF:
                    if torch.any(torch.isnan(pcl_abstract)):
                        raise RuntimeError(
                            f'MyTrainPipeline => pcl_abstract {pcl_abstract.shape} has NaN values! '
                            'Skipping this batch...')
                    if torch.any(torch.isinf(pcl_abstract)):
                        raise RuntimeError(
                            f'MyTrainPipeline => pcl_abstract {pcl_abstract.shape} has infinity values! '
                            'Skipping this batch...')
                    if torch.any(torch.isnan(implicit_output_frame)):
                        raise RuntimeError(
                            f'MyTrainPipeline => implicit_output_frame {implicit_output_frame.shape} '
                            'has NaN values! Skipping this batch...')
                    if torch.any(torch.isinf(implicit_output_frame)):
                        raise RuntimeError(
                            f'MyTrainPipeline => implicit_output_frame {implicit_output_frame.shape} '
                            'has infinity values! Skipping this batch...')

                # Predictions and ground truth for this frame.
                points_query.append(points_query_frame)
                implicit_output.append(implicit_output_frame)
                implicit_target.append(implicit_target_frame)

                # Plot point sample bias shares.
                solid_sbs = solid_sbs.mean(dim=-2)
                air_sbs = air_sbs.mean(dim=-2)

                del points_query_frame
                del implicit_output_frame, implicit_target_frame

            # Supervise what can already be supervised, i.e. density and RGB.
            (loss_rgb, loss_dens, loss_segm, loss_track) = \
                self.losses.per_example(
                    pcl_target, pcl_target_size,
                    implicit_output, implicit_target)

            # Add a batch dimension again because we recently averaged per GPU.
            loss_rgb = loss_rgb.unsqueeze(0) if torch.is_tensor(loss_rgb) else None
            loss_dens = loss_dens.unsqueeze(0) if torch.is_tensor(loss_dens) else None
            loss_segm = loss_segm.unsqueeze(0) if torch.is_tensor(loss_segm) else None
            loss_track = loss_track.unsqueeze(0) if torch.is_tensor(loss_track) else None

            # This will be passed on to the 'entire batch' loss method.
            remnant = (loss_rgb, loss_dens, loss_segm, loss_track,
                       rgb, depth, pcl_input, pcl_abstract, pcl_target,
                       meta_data, cam_RT, cam_K, layer_coords,
                       points_query, implicit_output, features_global)

            return remnant

    def handle_frame(self, time_idx, pcl_target, pcl_target_size,
                     valo_ids, num_valo_ids, live_occl,
                     pcl_abstract, features_global):
        '''
        :return (points_query_frame, implicit_output_frame, implicit_target_frame,
                 solid_sbs, air_sbs).
        '''
        pcl_target_frame = pcl_target[time_idx]
        pcl_target_frame_size = pcl_target_size[time_idx]
        is_future_frame = (time_idx >= self.past_frames)

        # Use target to guide ground truth sampling; there is no direct output point cloud.
        # NOTE: The sampler is quite advanced and needs access to context, i.e. all frames.
        (solid_input, air_input, solid_target, air_target, solid_sbs, air_sbs) = self.point_sampler(
            pcl_target, pcl_target_size, valo_ids, num_valo_ids, time_idx)

        # Avoid potential weird batch normalization shortcuts by forwarding all solid and air
        # points for this frame at once.
        points_query_frame = torch.cat([solid_input, air_input], dim=1)
        implicit_target_frame = torch.cat([solid_target, air_target], dim=1)
        # points_query_frame = (B, N, 4) with (x, y, z, t).
        # implicit_target_frame = (B, N, 6) with (density, R, G, B, mark_track, segm).

        # Select relevant neural field.
        cur_imp_net = self.implicit_net
        cur_pcl_abs = pcl_abstract
        cur_feats_global = features_global

        # Use continuous model. pcl_abstract is actually points + local features concatenated, but
        # the separation is handled by implicit_net itself.
        (implicit_output_frame, implicit_penult_frame, _) = cur_imp_net(
            points_query_frame, cur_pcl_abs, cur_feats_global, None, False)
        # implicit_output_frame = (B, N, 5+) with
        # (density, R, G, B, mark_track, segm?).

        # Squash & clamp values where needed. Leave density as logit because we apply BCE later.
        if self.color_mode == 'rgb':
            # Q = 3 with (R, G, B).
            implicit_output_frame[..., 1:4] = torch.sigmoid(implicit_output_frame[..., 1:4])
        elif self.color_mode == 'rgb_nosigmoid':
            # Q = 3 with (R, G, B).
            implicit_output_frame[..., 1:4] = torch.clamp(
                implicit_output_frame[..., 1:4].clone(), min=0.0, max=1.0)
        elif self.color_mode == 'hsv':
            # Q = 14 with (H0, ..., H11, S, V).
            implicit_output_frame[..., 13:15] = torch.clamp(
                implicit_output_frame[..., 13:15].clone(), min=0.0, max=1.0)
        elif self.color_mode == 'bins':
            # Q = 9 with (B0, ..., B8), all logits.
            pass

        del solid_input, air_input, solid_target, air_target

        return (points_query_frame, implicit_output_frame, implicit_target_frame,
                solid_sbs, air_sbs)

    def process_entire_batch(self, cur_step, total_step, *remnant):
        '''
        Finalizes the training step. Calculates all losses.
        '''
        (loss_rgb, loss_dens, loss_segm, loss_track) = remnant[:6]
        (points_query, implicit_output, features_global) = remnant[-3:]
        log_info = remnant[6:]

        (total_loss, loss_rgb, loss_dens, loss_segm, loss_track) = self.losses.entire_batch(
            total_step, loss_rgb, loss_dens, loss_segm, loss_track,
            points_query, implicit_output, features_global)

        log_info += (total_loss.item(),
                     loss_rgb, loss_dens, loss_segm, loss_track)

        return (total_loss, log_info)
