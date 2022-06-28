'''
Objective functions.
Created by Basile Van Hoorick for Revealing Occlusions with 4D Neural Fields.
'''

from __init__ import *

# Internal imports.
import utils


_CHECK_NAN = False


class MyLosses():
    '''
    Wrapper around the loss functionality such that DataParallel can be leveraged.
    '''

    def __init__(self, stage, logger, mixed_precision, color_lw, density_lw, segmentation_lw,
                 tracking_lw, color_mode, semantic_classes, past_frames, future_frames):
        '''
        :param color_lw (float): Lambda (loss term weight) for RGB color distance loss.
        :param density_lw (float).
        :param segmentation_lw (float).
        :param tracking_lw (float).
        :param color_mode (int).
        :param semantic_classes (int).
        :param past_frames (int).
        :param future_frames (int).
        '''
        super().__init__()
        self.stage = stage
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
        self.huber_loss = torch.nn.SmoothL1Loss(reduction='mean', beta=0.5)
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.l1_loss = torch.nn.L1Loss(reduction='mean')
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def implicit_density_loss(self, implicit_output, implicit_target):
        '''
        :param implicit_output (B, N, 5+) tensor with
            (density, R, G, B, mark_track, segm?).
        :param implicit_target (B, N, 6) tensor with (density, R, G, B, mark_track, segm).
        :return loss_dens (tensor).
        '''
        density_output = implicit_output[..., 0]
        density_target = implicit_target[..., 0]
        loss_dens = self.bce_loss(density_output, density_target)  # >= v28.

        if _CHECK_NAN and torch.any(torch.isnan(loss_dens)):
            raise RuntimeError('implicit_density_loss => NaN loss value!')

        return loss_dens

    def implicit_color_loss(self, implicit_output, implicit_target):
        '''
        :param implicit_output (B, N, 5+) tensor with
            (density, R, G, B, mark_track, segm?).
        :param implicit_target (B, N, 6) tensor with (density, R, G, B, mark_track, segm).
        :return loss_clr (tensor).
        '''
        solid_mask = (implicit_target[..., 0] >= 0.1)
        color_available_mask = (implicit_target[..., 1] >= 0.0)
        supervise_mask = torch.logical_and(solid_mask, color_available_mask)
        implicit_output = implicit_output[supervise_mask]
        implicit_target = implicit_target[supervise_mask]

        if self.color_mode in ['rgb', 'rgb_nosigmoid']:
            # Q = 3 with (R, G, B).
            rgb_output = implicit_output[..., 1:4]
            rgb_target = implicit_target[..., 1:4]
            loss_clr = self.l1_loss(rgb_output, rgb_target)  # >= v28.

        elif self.color_mode == 'hsv':
            # Q = 14 with (H0, ..., H11, S, V).
            # Hue is classified into 12 bins, while saturation and value remain regressed.
            num_classes = 12
            hsv_output = implicit_output[..., 1:1 + num_classes + 2]
            rgb_target = implicit_target[..., 1:4]

            hsv_target = utils.rgb_to_hsv(rgb_target)
            hue_target = hsv_target[..., 0] / 360.0 * num_classes
            hue_target = torch.round(hue_target).type(torch.int64)
            hue_target[hue_target == num_classes] = 0  # E.g. 11.9 becomes 12 but is actually 0.
            assert not torch.any(hue_target < 0)
            assert not torch.any(hue_target >= num_classes)
            sat_target = hsv_target[..., 1]
            val_target = hsv_target[..., 2]

            # Don't check hue where it is too bland and/or too dark.
            # https://en.wikipedia.org/wiki/HSL_and_HSV
            supervise_hue_mask = torch.logical_and(sat_target >= 0.2, val_target >= 0.2)

            if supervise_hue_mask.sum() >= 16:
                hue_output = hsv_output[..., :num_classes]
                hue_output = hue_output[supervise_hue_mask]
                hue_target = hue_target[supervise_hue_mask]
                loss_hue = self.ce_loss(hue_output, hue_target) / 2.0
            else:
                loss_hue = 0.0

            loss_sat = self.l1_loss(hsv_output[..., num_classes], sat_target)
            loss_val = self.l1_loss(hsv_output[..., num_classes + 1], val_target)
            loss_clr = (loss_hue + loss_sat + loss_val) / 3.0

        elif self.color_mode == 'bins':
            # Q = 9 with (B0, ..., B8), all logits.
            # Everything is classified into 6 saturated colors + black / gray / white.
            num_satcolor_classes = 6
            num_grayscale_classes = 3
            bins_output = implicit_output[..., 1:1 + num_satcolor_classes + num_grayscale_classes]
            rgb_target = implicit_target[..., 1:4]

            hsv_target = utils.rgb_to_hsv(rgb_target)
            hue_target = hsv_target[..., 0] / 360.0 * num_satcolor_classes
            hue_target = torch.round(hue_target).type(torch.int64)
            hue_target[hue_target == num_satcolor_classes] = 0
            assert not torch.any(hue_target < 0)
            assert not torch.any(hue_target >= num_satcolor_classes)
            sat_target = hsv_target[..., 1]
            val_target = hsv_target[..., 2]

            bland_mask = torch.logical_or(sat_target < 0.3, val_target < 0.3)
            black_mask = (val_target < 0.2)
            black_mask = torch.logical_and(black_mask, bland_mask)
            gray_mask = torch.logical_and(0.2 <= val_target, val_target < 0.6)
            gray_mask = torch.logical_and(gray_mask, bland_mask)
            white_mask = (0.6 <= val_target)
            white_mask = torch.logical_and(white_mask, bland_mask)

            bins_target = hue_target
            bins_target[black_mask] = num_satcolor_classes
            bins_target[gray_mask] = num_satcolor_classes + 1
            bins_target[white_mask] = num_satcolor_classes + 2
            assert not torch.any(bins_target < 0)
            assert not torch.any(bins_target >= num_satcolor_classes + num_grayscale_classes)

            loss_clr = self.ce_loss(bins_output, bins_target) / 3.0

        if _CHECK_NAN and torch.any(torch.isnan(loss_clr)):
            raise RuntimeError('implicit_color_loss => NaN loss value!')

        return loss_clr

    def implicit_segm_loss(self, implicit_output, implicit_target):
        '''
        :param implicit_output (B, N, 5+) tensor with
            (density, R, G, B, mark_track, segm?).
        :param implicit_target (B, N, 6) tensor with (density, R, G, B, mark_track, segm).
        :return loss_segm (tensor).
        '''
        segm_output = implicit_output[..., -self.semantic_classes:]
        segm_target = implicit_target[..., -1].type(torch.int64)
        supervise_mask = (segm_target >= 0)
        segm_output = segm_output[supervise_mask]
        segm_target = segm_target[supervise_mask]
        loss_segm = self.ce_loss(segm_output, segm_target)

        if _CHECK_NAN and torch.any(torch.isnan(loss_segm)):
            raise RuntimeError('implicit_segm_loss => NaN loss value!')

        return loss_segm

    def implicit_track_loss(self, implicit_output, implicit_target):
        '''
        :param implicit_output (B, N, 5+) tensor with
            (density, R, G, B, mark_track, segm?).
        :param implicit_target (B, N, 6) tensor with (density, R, G, B, mark_track, segm).
        :return loss_track (tensor).
        '''
        track_idx = utils.get_track_idx(self.color_mode)
        
        solid_mask = (implicit_target[..., 0] >= 0.1)
        track_available_mask = (implicit_target[..., 4] >= 0.0)
        supervise_mask = torch.logical_and(solid_mask, track_available_mask)
        implicit_output = implicit_output[supervise_mask]
        implicit_target = implicit_target[supervise_mask]
        
        track_output = implicit_output[..., track_idx]
        track_target = implicit_target[..., 4]
        loss_track = self.bce_loss(track_output, track_target)
        
        return loss_track

    def per_example(self, pcl_target, pcl_target_size,
                    implicit_output, implicit_target):
        '''
        Loss calculation that *can* be performed independently for each example within a batch.
        :param pcl_target: List of (B, M, 8-11) tensors, one per frame, with
            (x, y, z, cosine_angle?, instance_id?, semantic_tag?, view_idx, R, G, B, mark_track).
        :param pcl_target_size: List of (B) tensors: int values denoting which target points are
            relevant.
        :param implicit_output: List of (B, N, 5+) tensors, one per frame,
            with (density, R, G, B, mark_track, segm?).
        :param implicit_target: List of (B, N, 6) tensors, one per frame,
            with (density, R, G, B, mark_track, segm).
        :return (loss_rgb, loss_dens, loss_segm, loss_track).
        '''
        (B, M, E) = pcl_target[0].shape
        assert torch.all(torch.tensor(pcl_target_size) <= M)

        # Any of the following loss vectors being None means it is not applicable.
        loss_rgb_all = [] if self.color_lw > 0.0 else None
        loss_dens_all = [] if self.density_lw > 0.0 else None
        loss_segm_all = [] if self.segmentation_lw > 0.0 else None
        loss_track_all = [] if self.tracking_lw > 0.0 else None

        # Per-example losses.
        for i in range(B):

            assert implicit_output is not None

            for time_idx in range(self.past_frames + self.future_frames):

                implicit_output_frame = implicit_output[time_idx][i:i + 1]
                implicit_target_frame = implicit_target[time_idx][i:i + 1]

                # Calculate implicit loss values for this example.
                if loss_dens_all is not None:
                    loss_dens_all.append(self.implicit_density_loss(
                        implicit_output_frame, implicit_target_frame))

                if loss_rgb_all is not None:
                    loss_rgb_all.append(self.implicit_color_loss(
                        implicit_output_frame, implicit_target_frame))

                if loss_segm_all is not None:
                    loss_segm_all.append(self.implicit_segm_loss(
                        implicit_output_frame, implicit_target_frame))

                if loss_track_all is not None:
                    loss_track_all.append(self.implicit_track_loss(
                        implicit_output_frame, implicit_target_frame))

        # Average & return losses + other informative metrics within this GPU.
        loss_rgb = torch.mean(torch.stack(loss_rgb_all)) if loss_rgb_all is not None else None
        loss_dens = torch.mean(torch.stack(loss_dens_all)) if loss_dens_all is not None else None
        loss_segm = torch.mean(torch.stack(loss_segm_all)) if loss_segm_all is not None else None
        loss_track = torch.mean(torch.stack(loss_track_all)) if loss_track_all is not None else None

        result = (loss_rgb, loss_dens, loss_segm, loss_track)
        return result

    def entire_batch(self, total_step, loss_rgb, loss_dens, loss_segm,
                     loss_track, points_query, implicit_output, features_global):
        '''
        Loss calculation that *cannot* be performed independently for each example within a batch.
        :param total_step (int).
        :param loss_rgb (B) tensor.
        :param loss_dens (B) tensor.
        :param loss_segm (B) tensor.
        :param loss_track (B) tensor.
        :param points_query: List of (B, N, 4) tensors, one per frame, with (x, y, z, t).
        :param implicit_output: List of (B, N, 5+) tensors, one per frame,
            with (density, R, G, B, mark_track,  segm?).
        :param features_global: List of (B, D) tensors, one per frame.
        :return (total_loss, loss_rgb, loss_dens, loss_segm, loss_track).
        '''

        # Average & report *all* losses, including per-example.
        loss_rgb = loss_rgb.mean() if torch.is_tensor(loss_rgb) else 0.0
        loss_dens = loss_dens.mean() if torch.is_tensor(loss_dens) else 0.0
        loss_segm = loss_segm.mean() if torch.is_tensor(loss_segm) else 0.0
        loss_track = loss_track.mean() if torch.is_tensor(loss_track) else 0.0

        total_loss = loss_rgb * self.color_lw + loss_dens * self.density_lw + \
            loss_segm * self.segmentation_lw + loss_track * self.tracking_lw
        self.logger.report_scalar(self.stage + '/total_loss', total_loss.item(), remember=True)

        if loss_rgb != 0.0:
            loss_rgb = loss_rgb.item()
            self.logger.report_scalar(self.stage + '/loss_rgb', loss_rgb, remember=True)
        if loss_dens != 0.0:
            loss_dens = loss_dens.item()
            self.logger.report_scalar(self.stage + '/loss_dens', loss_dens, remember=True)
        if loss_segm != 0.0:
            loss_segm = loss_segm.item()
            self.logger.report_scalar(self.stage + '/loss_segm', loss_segm, remember=True)
        if loss_track != 0.0:
            loss_track = loss_track.item()
            self.logger.report_scalar(self.stage + '/loss_track', loss_track, remember=True)

        return (total_loss, loss_rgb, loss_dens, loss_segm, loss_track)
