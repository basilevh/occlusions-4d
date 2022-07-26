'''
Evaluation logic.
Created by Basile Van Hoorick and Purva Tendulkar for Revealing Occlusions with 4D Neural Fields.
'''

from __init__ import *

# Library imports.
import traceback
import pdb

# Internal imports.
import args
import data
import geometry
import implicit
import logvis
import loss
import model
import utils


def load_models(checkpoint_path, device, epoch=-1, logger=None):
    '''
    :param checkpoint_path (str): Path to model checkpoint folder or file.
    :param epoch (int): If >= 0, desired checkpoint epoch to load.
    :return (networks, train_args, dset_args, pcl_args, implicit_args, epoch).
        networks [pcl_net, implicit_net].
            pcl_net (PointCompletionNet module).
            implicit_net (ResnetFC module).
        train_args (dict).
        dset_args (dict).
        pcl_args (dict).
        implicit_args (dict).
        epoch (int).
    '''
    print_fn = logger.info if logger is not None else print
    assert os.path.exists(checkpoint_path)
    if os.path.isdir(checkpoint_path):
        model_fn = f'model_{epoch}.pth' if epoch >= 0 else 'checkpoint.pth'
        checkpoint_path = os.path.join(checkpoint_path, model_fn)

    print_fn('Loading weights from: ' + checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Display all arguments to help verify correctness.
    train_args = checkpoint['args']
    dset_args = checkpoint['dset_args']
    print_fn('Train command args: ' + str(train_args))
    print_fn('Train dataset args: ' + str(dset_args))

    # Get network instance parameters.
    pcl_args = checkpoint['pcl_args']
    implicit_args = checkpoint['implicit_args']
    print_fn('Point transformer model args: ' + str(pcl_args))
    print_fn('Continuous model args: ' + str(implicit_args))

    # Make test-time model deterministic.
    pcl_args['fps_random_start'] = False

    # Fix deprecations.
    checkpoint['implicit_net'] = utils.rename_state_dict_keys(
        checkpoint['implicit_net'], 'pt_block.', 'pt_blocks.0.')

    # Instantiate point transformer network.
    pcl_net = model.PointCompletionNetV3(**pcl_args)
    pcl_net = pcl_net.to(device)
    pcl_net.load_state_dict(checkpoint['pcl_net'])

    # Instantiate continuous network.
    implicit_net = implicit.LocalPclResnetFC(**implicit_args)
    implicit_net = implicit_net.to(device)
    implicit_net.load_state_dict(checkpoint['implicit_net'])

    networks = [pcl_net, implicit_net]

    epoch = checkpoint['epoch']
    print_fn('=> Loaded epoch (1-based): ' + str(epoch + 1))

    return (networks, train_args, dset_args, pcl_args, implicit_args, epoch)


def perform_inference(pcl_input, pcl_input_sem, pcl_target_frame, networks, device, task, min_z,
                      cube_bounds, color_mode, time_idx, logger,
                      sample_implicit=True, num_sample=16384, point_sample_mode='random',
                      batch_size=1024, predict_segmentation=False, track_mode='none',
                      point_occupancy_radius=0.2, semantic_classes=13,
                      density_threshold=0.5, data_kind='', cube_mode=4, compress_air=False):
    '''
    Generate test time predictions corresponding to one point in time of the combined architecture.
    :param pcl_input (N, 8) numpy array or (B, N, 8) tensor
        with (x, y, z, R, G, B, t, mark_track).
    :param pcl_input_sem (N, 1-3) numpy array or (B, N, 1-3) tensor
        with (cosine_angle?, instance_id, semantic_tag?).
    :param pcl_target_frame (M, 9-11) numpy array
        with (x, y, z, cosine_angle?, instance_id, semantic_tag?, view_idx, R, G, B, mark_track).
    :param networks [pcl_net, implicit_net].
        pcl_net (PointCompletionNet module).
        implicit_net (ResnetFC module).
    :param device.
    :param task (str).
    :param min_z (float).
    :param cube_bounds (float).
    :param color_mode (str): How to interpret the predicted color.
    :param time_idx (int): Selected frame to query.
    :param sample_implicit (bool): Whether to sample pcl_output from the continuous representation.
    :param num_sample (int): M for the implicit network.
    :param point_sample_mode (str): How to sample points within CRs (random / grid).
        If random: sample at uniformly random locations within the CR cuboid.
        If grid: sample according to a fixed, equally spaced grid.
    :param batch_size (int): B for the implicit network.
    :param predict_segmentation (bool).
    :param track_mode (str): How many objects to track, if at all (none / one / all).
    :param point_occupancy_radius (float).
    :param semantic_classes (int).
    :param density_threshold (float): Where to distinguish between solid and air output points.
    :param data_kind (str): Guides how to spatially sample points (greater / carla).
    :param cube_mode (int): Which cuboid shape to use for CARLA (1 / 2 / 3 / 4).
    :param compress_air (bool): If True, sampled_air_output is (A, 6) instead of (A, 13+).
    :return: dict with keys (output_solid, output_air, pcl_abstract, features_global,
                             implicit_output, attn_neurons).
        output_solid (S, 13+) numpy array.
        output_air (A, 6/13+) numpy array.
        pcl_abstract (M, E) numpy array.
        features_global (D) numpy array.
        implicit_output (N, 9+) numpy array.
    '''
    assert task == 'if'
    assert sample_implicit

    print_fn = logger.info if logger is not None else print
    gt_available = (pcl_target_frame is not None)

    output_track_idx = utils.get_track_idx(color_mode)
    input_inst_idx = 0 if data_kind == 'greater' else 1
    target_inst_idx = 3 if data_kind == 'greater' else None

    assert len(networks) == 2
    pcl_net = networks[0]
    implicit_net = networks[1]

    # Determine how many times to rerun inference for this clip. Typically this is just once, but
    # if we want to perform dense instance segmentation, then we first need to track every object
    # independently (as this is how the model is trained), and subsequently merge the results by
    # assigning the predicted instance_id to mark_track via argmax calculation.
    if track_mode in ['none', 'one']:
        track_instance_ids = [-1]
        num_tracks = 1

    else:
        # Determine all the valid objects (instance ids) visible from the first input frame
        # with at least 16 points.
        assert data_kind == 'greater'
        assert pcl_input_sem.shape[-1] == 1

        if isinstance(pcl_input_sem, np.ndarray):
            pcl_input_sem_numpy = pcl_input_sem
            pcl_input_sem = torch.from_numpy(pcl_input_sem).unsqueeze(0).to(device)
        else:
            pcl_input_sem_numpy = pcl_input_sem[0].detach().cpu().numpy()
            pcl_input_sem = pcl_input_sem.to(device)

        (exist_instance_ids, counts) = np.unique(pcl_input_sem_numpy, return_counts=True)
        track_instance_ids = []
        for (inst_id, freq) in zip(exist_instance_ids, counts):
            if inst_id >= 0 and freq >= 16:
                track_instance_ids.append(int(inst_id))
        num_tracks = len(track_instance_ids)
        # print_fn(f'track_instance_ids: {track_instance_ids}')

    if isinstance(pcl_input, np.ndarray):
        pcl_input = torch.from_numpy(pcl_input).unsqueeze(0).to(device)

    # Generate fixed CR query points (i.e. used for all reruns).
    points_query = geometry.sample_implicit_points_blind_numpy(
        num_sample, min_z, cube_bounds, time_idx, data_kind,
        cube_mode, point_sample_mode)  # (N, 4) with (x, y, z, t).
    # NOTE: The actual size of points_query may deviate slightly from the requested num_sample.
    # This should only happen if point_sample_mode is grid.
    num_batches = int(np.ceil(points_query.shape[0] / batch_size))

    all_pcl_abstract = []
    all_features_global = []
    all_implicit_output = []

    # Loop over all reruns (one for each track).
    for track_idx in range(num_tracks):

        # Adjust mark_track in input and target.
        mark_inst_id = track_instance_ids[track_idx]
        if mark_inst_id >= 0:
            input_mask = (pcl_input_sem[..., input_inst_idx] == mark_inst_id)
            pcl_input[..., -1] = input_mask

        (pcl_abstract, features_global, layer_coords) = pcl_net(
            pcl_input, False)
        if pcl_abstract is not None:
            pcl_abstract = pcl_abstract.squeeze(0)  # (M, 3 + E) = (32, 259).
        features_global = features_global.squeeze(0)  # (D) = (512).
            
        implicit_output = []

        # Loop over all query mini-batches for this run.
        for b in range(num_batches):
            points_query_batch = points_query[b * batch_size:(b + 1) * batch_size]  # (B, 4).
            points_query_batch = torch.from_numpy(points_query_batch).to(device)
            # points_query_batch is already "squeezed".

            # Use continuous model. pcl_abstract is actually coordinates + local features
            # concatenated, but the separation is handled by implicit_net itself.
            (implicit_output_batch, implicit_penult_batch) = implicit_net(
                points_query_batch, pcl_abstract, features_global, None)
            # implicit_output_frame_batch = (N, 5+) with
            # (density, R, G, B, mark_track, segm?).
            
            # Squash & clamp values where needed.
            # For exampe, convert density values from logits to probits.
            implicit_output_batch[..., 0] = torch.sigmoid(implicit_output_batch[..., 0])

            if color_mode == 'rgb':
                # Q = 3 with (R, G, B).
                implicit_output_batch[..., 1:4] = torch.sigmoid(implicit_output_batch[..., 1:4])
            elif color_mode == 'rgb_nosigmoid':
                # Q = 3 with (R, G, B).
                implicit_output_batch[..., 1:4] = torch.clamp(
                    implicit_output_batch[..., 1:4].clone(), min=0.0, max=1.0)
            elif color_mode == 'hsv':
                # Q = 14 with (H0, ..., H11, S, V).
                implicit_output_batch[..., 1:13] = torch.sigmoid(
                    implicit_output_batch[..., 1:13])
                implicit_output_batch[..., 13:15] = torch.clamp(
                    implicit_output_batch[..., 13:15].clone(), min=0.0, max=1.0)
            elif color_mode == 'bins':
                # Q = 9 with (B0, ..., B8).
                implicit_output_batch[..., 1:10] = torch.sigmoid(
                    implicit_output_batch[..., 1:10])

            if predict_segmentation:
                implicit_output_batch[..., -semantic_classes:] = torch.sigmoid(
                    implicit_output_batch[..., -semantic_classes:])
            if track_mode != 'none':
                implicit_output_batch[..., output_track_idx] = torch.sigmoid(
                    implicit_output_batch[..., output_track_idx])

            implicit_output_batch = implicit_output_batch.detach().cpu().numpy()
            implicit_output.append(implicit_output_batch)

        implicit_output = np.concatenate(implicit_output, axis=0)
        # (N, 5+) with (density, R, G, B, mark_track, segm?).

        if pcl_abstract is not None:
            pcl_abstract = pcl_abstract.detach().cpu().numpy()
        features_global = features_global.detach().cpu().numpy()

        # Save all info for this run.
        all_pcl_abstract.append(pcl_abstract)
        all_features_global.append(features_global)
        all_implicit_output.append(implicit_output)

        # del implicit_output_batch, implicit_penult_batch
        # del pcl_abstract, features_global, implicit_output

    # Now merge all runs, which essentially argmaxes over the predicted track data and averages
    # everything else. This will only have effect if track_mode is all.
    (pcl_abstract, features_global, implicit_output) = utils.multi_track_merge(
        track_instance_ids, all_pcl_abstract, all_features_global, all_implicit_output,
        output_track_idx)

    # Get Nearest Neighbor (NN) info for metric calculation.
    if gt_available:
        pcl_target_xyz = pcl_target_frame[..., :3]  # (M, 3)
        target_labels, nn_indices = geometry.get_1nn_label(
            points_query[:, :3], pcl_target_xyz, point_occupancy_radius)
        query_nn1 = pcl_target_frame[nn_indices][:, 0, :]  # (N, 9)
        points_nngt = np.concatenate([
            target_labels[:, None], query_nn1], axis=-1)
        # (N, 10-12) with (label, x, y, z, cosine_angle?, instance_id, semantic_tag?, view_idx, R, G, B, mark_track).

    # Combine query coordinates with implicit output, and split solid from air by predicted density.
    points_io = np.concatenate([points_query, implicit_output], axis=-1)
    # (N, 9+) with (x, y, z, t, density, R, G, B, mark_track, segm?).

    solid_points = points_io[points_io[..., 4] >= density_threshold]  # (S, 13+).
    air_points = points_io[points_io[..., 4] < density_threshold]  # (A, 13+).
    if gt_available:
        solid_points_nngt = points_nngt[points_io[..., 4] >= density_threshold]  # (S, 10-12).
        air_points_nngt = points_nngt[points_io[..., 4] < density_threshold]  # (A, 10-12).

    # The term point cloud by definition means solid points only.
    # However, we also store the query outputs that map to air for later usage.
    sampled_pcl_output = solid_points
    sampled_air_output = air_points
    if gt_available:
        sampled_pcl_gt = solid_points_nngt
        sampled_air_gt = air_points_nngt

    # To save disk space, ignore time, color, track, and features,
    # but retain location, density, pred_segm, NN-label and NN-instance_id.
    if compress_air:

        air_pred_segm = sampled_air_output[..., -semantic_classes:].argmax(axis=-1)
        sampled_air_output = np.concatenate(
            [sampled_air_output[..., :3], sampled_air_output[..., 4:5],
                air_pred_segm[..., None]], axis=-1)
        # (A, 6) with (x, y, z, density, pred_segm).
        
        if gt_available:
            # NOTE: Assumes that cosine_angle is not present in pcl_target_frame
            sampled_air_gt = np.concatenate(
                [sampled_air_gt[..., :1], sampled_air_gt[..., 4:5]], axis=-1)
            # (A, 2) with (label, instance_id) of Nearest Neighbor GT.

    result = dict()
    result['output_solid'] = sampled_pcl_output
    result['output_air'] = sampled_air_output
    result['pcl_abstract'] = pcl_abstract
    result['features_global'] = features_global
    result['implicit_output'] = implicit_output
    result['points_query'] = points_query

    if gt_available:
        result['gt_solid'] = sampled_pcl_gt
        result['gt_air'] = sampled_air_gt

    return result
