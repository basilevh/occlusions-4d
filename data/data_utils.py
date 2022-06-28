'''
Helper methods for data loading and processing.
Created by Basile Van Hoorick for Revealing Occlusions with 4D Neural Fields.
'''

from __init__ import *

# Internal imports.
import utils


def get_valo_ids(live_occl_mode, scene_idx, scene_dp,
                 filter_vehped, sem_inst_col, sem_cat_col, merged_inst_col,
                 pcl_input_frames, video_length, frame_start, frame_end, frame_skip,
                 sb_occl_frame_shift, src_view, num_views, max_valo_ids, logger,
                 all_pcl, pcl_input_sem, pcl_merged_frames):
    '''
    Calculates approximate per-instance occlusion percentage over time.
    We call this live because it is inferred at runtime for flexibility purposes.
    If mode is unfiltered (recommended), then this result does not change between runs.
    :param filter_vehped (bool): False for GREATER, True for CARLA.
    :param sem_inst_col (int): 0 for GREATER, 1 for CARLA.
    :param sem_cat_col (int): undefined for GREATER, 2 for CARLA.
    :param merged_inst_col (int): 3 for GREATER, 4 for CARLA.
    :param all_pcl: List-V of List-T of:
        (N, 7) with (x, y, z, instance_id, R, G, B) for GREATER;
        (N, 9) with (x, y, z, cosine_angle, instance_id, semantic_tag, R, G, B) for CARLA.
        NOTE: all_pcl is either subsampled or not; depends on caller.
    :return (valo_ids_pad, num_valo_ids, vehped_mask)
    '''
    to_tensor = torchvision.transforms.ToTensor()
    vehped_mask = None

    if 'unfilt' in live_occl_mode:
        # Repeat these time and view merging steps but without subsampling.
        assert pcl_input_frames == video_length
        nss_video_views = utils.accumulate_pcl_time_numpy(all_pcl)
        # List-V of (T*N, 8/10) with
        # (x, y, z, cosine_angle?, instance_id, semantic_tag?, R, G, B, t).
        nss_merged_frames = utils.merge_pcl_views_numpy(all_pcl, insert_view_idx=True)
        # List-T of (V*N, 8/10) with
        # (x, y, z, cosine_angle?, instance_id, semantic_tag?, view_idx, R, G, B).
        nss_input = nss_video_views[src_view]
        nss_input = to_tensor(nss_input).squeeze(0)
        # (T*N, 8/10) with
        # (x, y, z, cosine_angle?, instance_id, semantic_tag?, R, G, B, t).
        nss_input_sem = nss_input[..., 3:-4]
        # (N, 1/3) with (cosine_angle?, instance_id, semantic_tag?).

        # Update variables we were already working with.
        used_input_sem = nss_input_sem  # tensor.
        used_merged_frames = nss_merged_frames  # list of numpy array.
        valo_min_points = 16

    elif 'normal' in live_occl_mode:
        used_input_sem = pcl_input_sem  # tensor.
        used_merged_frames = pcl_merged_frames  # list of numpy array.
        valo_min_points = 8

    else:
        raise ValueError(live_occl_mode)

    # Detect set of VALO (visible at least once) vehped ids and filter by at least 8 points.
    if filter_vehped:
        # 4 = Pedestrian, 10 = Vehicles.
        vehped_mask = torch.logical_or(
            used_input_sem[..., sem_cat_col] == 4, used_input_sem[..., sem_cat_col] == 10)
        pcl_input_vehped_sem = used_input_sem[vehped_mask]
    else:
        vehped_mask = None
        pcl_input_vehped_sem = used_input_sem

    valo_ids = used_input_sem[..., sem_inst_col].type(torch.int32).unique()
    valo_ids = sorted(list(valo_ids.detach().cpu().numpy()))
    valo_ids = [valo_id for valo_id in valo_ids
                if valo_id >= 0 and
                (pcl_input_vehped_sem[..., sem_inst_col] == valo_id).sum() >= valo_min_points]
    num_valo_ids = len(valo_ids)

    # Calculate live occlusion fractions for all instances for all input frames.
    live_occl = np.zeros((pcl_input_frames, max_valo_ids))
    for i, vis_id in enumerate(valo_ids):

        # First, determine the maximum number of points that are ever seen by all views.
        max_id_merged_count = -1
        for t in range(video_length):
            cur_count = (used_merged_frames[t][..., merged_inst_col] == vis_id).sum()
            max_id_merged_count = max(cur_count, max_id_merged_count)

        # Then, count every input point V times to approximate degree of visibility per frame.
        for t in range(pcl_input_frames):
            cur_id_input_count = (all_pcl[src_view][t][..., merged_inst_col] == vis_id).sum()
            cur_occl = max(1.0 - cur_id_input_count * num_views /
                           (max_id_merged_count + 1e-6), 0.0)
            live_occl[t, i] = cur_occl

    valo_ids_pad = -np.ones(max_valo_ids, dtype=np.int32)
    valo_ids_pad[:num_valo_ids] = valo_ids[:max_valo_ids]

    return (live_occl, valo_ids_pad, num_valo_ids, vehped_mask)
