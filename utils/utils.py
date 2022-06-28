'''
Miscellaneous helper methods.
Created by Basile Van Hoorick for Revealing Occlusions with 4D Neural Fields.
'''

from __init__ import *
import numpy as np
import os
import pathlib
import pickle
import tqdm


def accumulate_pcl_time_numpy(pcl):
    '''
    Converts a series of RGB point cloud snapshots into a point cloud video by adding a feature that
        represents time with values {0, 1, ..., T-1}.
    :param pcl (V, T, N, D) numpy array or list-V of list-T of (N, D) numpy arrays.
    :return (V, T*N, D+1) numpy array or list-V of (T*N, D+1) numpy arrays.
    '''
    if isinstance(pcl, np.ndarray):
        # Fully within the numpy array domain.
        (V, T, N, D) = pcl.shape
        time_vals = np.arange(T, dtype=np.float32)[None, :, None, None]  # (1, T, 1, 1).
        time_vals = np.tile(time_vals, (V, 1, N, 1))  # (V, T, N, 1).
        pcl_out = np.concatenate((pcl, time_vals), axis=-1)  # (V, T, N, 7).
        pcl_out = pcl_out.reshape(V, T * N, D + 1)

    else:
        # Mixed list and array domain, which is more complicated but more flexible.
        (V, T) = len(pcl), len(pcl[0])
        pcl_out = []
        for view_idx in range(V):
            pcl_view = []
            for time_idx in range(T):
                pcl_frame = pcl[view_idx][time_idx]  # (N, 6).
                time_vals = np.ones_like(pcl_frame[..., 0:1]) * time_idx  # (N, 1).
                pcl_frame_timed = np.concatenate([pcl_frame, time_vals], axis=-1)  # (N, 7).
                pcl_view.append(pcl_frame_timed)
            pcl_view = np.concatenate(pcl_view, axis=0)  # (T*N, 7).
            pcl_out.append(pcl_view)

    return pcl_out


def accumulate_pcl_layer_torch(pcls):
    '''
    Converts a list of point cloud snapshots into a point cloud video by adding a feature that
        represents time with values {0, 1, ..., T-1}.
    :param pcls Length-L list of (N, 3) tensors.
    :return (L*N, 4) tensor.
    '''
    L = len(pcls)
    for layer in range(L):
        # (B, N, _) = pcls[layer].shape
        (N, _) = pcls[layer].shape
        to_cat = torch.ones_like(pcls[layer][..., 0:1]) * layer  # (B, N, 1).
        pcls[layer] = torch.cat([pcls[layer], to_cat], dim=-1)
    pcl = torch.cat(pcls, dim=0)

    return pcl


def merge_pcl_views_numpy(pcl, insert_view_idx=False):
    '''
    Converts a set of RGB point clouds from different camera viewpoints into one combined point
        cloud.
    :param pcl (V, T, N, D) numpy array or list-V of list-T of (N, D) numpy arrays.
    :return (T, V*N, D) numpy array or list-T of (V*N, D) numpy arrays.
    '''
    if isinstance(pcl, np.ndarray):
        # Fully within the numpy array domain.
        assert not insert_view_idx
        (V, T, N, D) = pcl.shape
        pcl_out = pcl.transpose(1, 0, 2, 3)  # (T, V, N, 6).
        pcl_out = pcl_out.reshape(T, V * N, D)

    else:
        # Mixed list and array domain, which is more complicated but more flexible.
        V, T = len(pcl), len(pcl[0])
        pcl_out = []
        for time_idx in range(T):
            pcl_time = []

            for view_idx in range(V):
                cur_xyz_sem = pcl[view_idx][time_idx][..., :-3]
                cur_rgb = pcl[view_idx][time_idx][..., -3:]

                if insert_view_idx:
                    cur_idx = np.ones_like(cur_xyz_sem[..., 0:1]) * view_idx
                    pcl_time_view = np.concatenate([cur_xyz_sem, cur_idx, cur_rgb], axis=-1)

                else:
                    pcl_time_view = np.concatenate([cur_xyz_sem, cur_rgb], axis=-1)

                pcl_time.append(pcl_time_view)

            pcl_time = np.concatenate(pcl_time, axis=0)  # (V*N, 6).
            pcl_out.append(pcl_time)

    return pcl_out


def find_mask_ranges(mask):
    '''
    :param mask (B, N) tensor with booleans.
    :return (B, 2) tensor with [start, end) integer values per batch index.
        start represents the first True value, and end represents the first False value after start.
    '''
    mask = mask.type(torch.int32)
    mask_delta = mask[..., 1:] - mask[..., :-1]

    # Prepend +0.5 at the start, which will be the argmax if the mask already starts at True.
    # Append -0.5 at the end, which will be the argmin if the mask never turns False again.
    mask_delta = torch.cat([torch.ones_like(mask_delta[..., 0:1]) * (0.5),
                            mask_delta,
                            torch.ones_like(mask_delta[..., 0:1]) * (-0.5)], dim=-1)

    start = mask_delta.argmax(dim=-1)
    end = mask_delta.argmin(dim=-1)

    result = torch.stack([start, end], dim=-1)

    return result


def rename_state_dict_keys(state_dict, old_str, new_str):
    new_dict = collections.OrderedDict()
    for key in state_dict:
        if key.startswith(old_str):
            new_key = new_str + key[len(old_str):]
            new_dict[new_key] = state_dict[key]
        else:
            new_dict[key] = state_dict[key]
    return new_dict


def write_video(file_path, frames, fps):
    frames = [frame[..., ::-1] for frame in frames]  # BGR -> RGB.
    if frames[0].dtype == np.float32:
        frames = [(frame * 255.0).astype(np.uint8) for frame in frames]
    frames = [frame for frame in frames if frame.shape[0] != 0]
    if len(frames) > 0:
        imageio.mimwrite(file_path, frames, fps=fps, quality=10)


def read_video(file_path):
    reader = cv2.VideoCapture(file_path)
    frames = []
    while reader.isOpened():
        ret, frame = reader.read()
        if not ret:
            break
        frames.append(frame)
    reader.release()
    return frames


def get_data_kind(dset_root):
    if 'gr_' in dset_root.lower() or 'greater' in dset_root.lower():
        data_kind = 'greater'
    elif 'carla' in dset_root.lower():
        data_kind = 'carla'
    else:
        raise ValueError()
    return data_kind


def rgb_to_hsv(input, epsilon=1e-10):
    '''
    https://www.linuxtut.com/en/20819a90872275811439/
    :param input (N, 3) tensor.
    :return (N, 3) tensor.
    '''
    assert(input.shape[1] == 3)

    r, g, b = input[:, 0], input[:, 1], input[:, 2]
    max_rgb, argmax_rgb = input.max(1)
    min_rgb, argmin_rgb = input.min(1)

    max_min = max_rgb - min_rgb + epsilon

    h1 = 60.0 * (g - r) / max_min + 60.0
    h2 = 60.0 * (b - g) / max_min + 180.0
    h3 = 60.0 * (r - b) / max_min + 300.0

    h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
    s = max_min / (max_rgb + epsilon)
    v = max_rgb

    return torch.stack((h, s, v), dim=1)


def rgb_to_hsv_mpl(rgb_torch):
    '''
    NOTE: I verified that this gives the same result as rgb_to_hsv().
    '''
    rgb_numpy = rgb_torch.detach().cpu().numpy()
    hsv_numpy = matplotlib.colors.rgb_to_hsv(rgb_numpy)
    hsv_torch = torch.tensor(hsv_numpy, device=rgb_torch.device, dtype=rgb_torch.dtype)
    return hsv_torch


def get_track_idx(color_mode):
    '''
    Finds the relevant dimension in the implicit output.
    Assumes location and time (x, y, z, t) are not included, so the first value is density, and the
        next Q values encode color.
    '''
    if color_mode == 'rgb':
        # Q = 3 with (R, G, B).
        idx = 4
    elif color_mode == 'rgb_nosigmoid':
        # Q = 3 with (R, G, B).
        idx = 4
    elif color_mode == 'hsv':
        # Q = 14 with (H0, ..., H11, S, V).
        idx = 15
    elif color_mode == 'bins':
        # Q = 9 with (B0, ..., B8), all logits.
        idx = 10
    else:
        raise ValueError()
    return idx


def model_hsv_to_rgb(model_hsv):
    '''
    Interprets CR model output for HSV color mode and converts to RGB values for visualization.
    :param model_hsv (N, 14) tensor with (H0, ..., H11, S, V).
    :return (N, 3) tensor with (R, G, B).
    '''
    num_classes = 12
    hue_scores = model_hsv[..., :num_classes]  # (N, 12).
    hue_preds = hue_scores.argmax(axis=-1).astype(model_hsv.dtype) / num_classes  # (N).
    sat_preds = model_hsv[..., -2]  # (N).

    # Make more vivid.
    sat_preds = np.sqrt(sat_preds)

    val_preds = model_hsv[..., -1]  # (N).
    hsv_preds = np.stack([hue_preds, sat_preds, val_preds], axis=-1)  # (N, 3).
    rgb_preds = matplotlib.colors.hsv_to_rgb(hsv_preds)  # (N, 3).
    return rgb_preds


def model_bins_to_rgb(model_bins):
    '''
    Interprets CR model output for bins color mode and converts to RGB values for visualization.
    :param model_bins (N, 9) tensor with (B0, ..., B8).
    :return (N, 3) tensor with (R, G, B).
    '''
    rgb_colors = np.array([(255, 0, 0), (255, 255, 0), (0, 255, 0),
                           (0, 255, 255), (0, 0, 255), (255, 0, 255),
                           (26, 26, 26), (102, 102, 102), (204, 204, 204)])
    num_classes = 9
    bins_scores = model_bins[..., :num_classes]  # (N, 9).
    bins_preds = bins_scores.argmax(axis=-1).astype(np.int32)  # (N).
    rgb_preds = rgb_colors[bins_preds] / 255.0
    return rgb_preds


def shuffle_together(x, y):
    '''
    Shuffles numpy arrays x and y together
    '''
    assert(x.shape[0] == y.shape[0])
    p = np.random.permutation(x.shape[0])
    return x[p], y[p]


def elitist_shuffle(items, inequality):
    """
    https://github.com/rragundez/elitist-shuffle
    Shuffle array with bias over initial ranks
    A higher ranked content has a higher probability to end up higher
    ranked after the shuffle than an initially lower ranked one.
    Args:
        items (numpy.array): Items to be shuffled
        inequality (int/float): how biased you want the shuffle to be.
            A higher value will yield a lower probabilty of a higher initially
            ranked item to end up in a lower ranked position in the
            sequence.
    """
    weights = np.power(
        np.linspace(1, 0, num=len(items), endpoint=False),
        inequality
    )
    weights = weights / np.linalg.norm(weights, ord=1)
    return np.random.choice(items, size=len(items), replace=False, p=weights)


def find_testres_pcl_fp_list(input_path, dp_filter=None, step_idx=None):
    '''
    Constructs a list of test result file paths (i.e. pcl_io_sX.p).
    :param dp_filter (str): Filter log subdirectories by this contained substring.
    :param input_path (str): Prefix for one or multiple test log directories.
    '''
    src_fp_list = []

    input_path = pathlib.Path(input_path)
    parent = str(input_path.parent)
    prefix = str(input_path.name)
    run_dns = os.listdir(parent)
    run_dns = [dn for dn in run_dns if dn.startswith(prefix)]
    print()

    for run_dn in run_dns:
        run_dp = os.path.join(parent, run_dn)
        test_dns = os.listdir(run_dp)
        test_dns = [dn for dn in test_dns if dn.startswith('test_')]
        test_dps = [os.path.join(run_dp, dn) for dn in test_dns]
        test_dps.append(run_dp)  # In case we point directly to a test results folder.

        for test_dp in test_dps:
            if os.path.isdir(test_dp) and not('_povvid' in test_dp or '_open3d' in test_dp) and \
                    (dp_filter is None or dp_filter in test_dp):
                # print('Found:', run_dn, '/', test_dp)

                try:
                    pcl_fns = os.listdir(test_dp)
                    pcl_fns = [fn for fn in pcl_fns if fn.startswith(
                        'pcl_io_') and fn.endswith('.p')]

                    # Filter for a specific test result if desired.
                    if step_idx is not None:
                        pcl_fns = [fn for fn in pcl_fns if f'_s{step_idx}.' in fn]

                    for pcl_fn in pcl_fns:
                        src_fp = os.path.join(test_dp, pcl_fn)
                        src_fp_list.append(src_fp)

                except:
                    print('Failed')
                    pass

    print()
    src_fp_list.sort()

    return src_fp_list


def multi_track_merge(
        track_instance_ids, pcl_abstract, features_global, implicit_output, output_track_idx):
    '''
    Combines all reruns by merging detected tracks but averaging everything else.
    This should be used only at test time.
    :param track_instance_ids: List of int.
    :param pcl_abstract: List of (M, E) numpy arrays, one per instance.
    :param features_global: List of (D) numpy arrays, one per instance.
    :param implicit_output: List of (N, 5+) numpy arrays, one per instance.
    :param output_track_idx (int): Dimension of mark_track in implicit_output.
        Add 4 for solid_points or air_points.
    :return (merged_pcl_abstract, merged_features_global, merged_implicit_output).
    '''
    assert len(pcl_abstract) == len(features_global)
    assert len(pcl_abstract) == len(implicit_output)
    num_tracks = len(pcl_abstract)

    # start_time = time.time()

    # NOTE: fps_random_start is disabled by inference.load_models(), so all pcl_abstract coordinates
    # should be the the same across runs given the same pcl_input coordinates.
    if num_tracks >= 3 and pcl_abstract[0] is not None:
        np.testing.assert_array_almost_equal(pcl_abstract[0][..., :3], pcl_abstract[1][..., :3])
        np.testing.assert_array_almost_equal(pcl_abstract[0][..., :3], pcl_abstract[-1][..., :3])

    # Check for default operation (nothing to be merged then).
    if num_tracks == 1 and track_instance_ids[0] == -1:
        return (pcl_abstract[0], features_global[0], implicit_output[0])

    # First, calculate mean of all features.
    if pcl_abstract[0] is not None:
        merged_pcl_abstract = np.mean(pcl_abstract, axis=0)
    else:
        merged_pcl_abstract = None
    merged_features_global = np.mean(features_global, axis=0)
    merged_implicit_output = np.mean(implicit_output, axis=0)

    # Then, overwrite the mark_track column with instance ids according to who has the highest
    # predicted score there. If all scores (= probits) for a particular point are < 0.5 (which
    # indicates lack of confidence, such as for most air points), then the id remains -1.
    merged_mark_track = -np.ones_like(merged_implicit_output[..., 0])
    confidence_so_far = np.zeros_like(merged_implicit_output[..., 0])

    for track_idx in range(num_tracks):
        mark_inst_id = track_instance_ids[track_idx]
        detect_score = implicit_output[track_idx][..., output_track_idx]
        detect_mask = np.logical_and(detect_score >= 0.5, detect_score >= confidence_so_far)
        merged_mark_track[detect_mask] = mark_inst_id
        confidence_so_far = np.maximum(detect_score, confidence_so_far)

    merged_implicit_output[..., output_track_idx] = merged_mark_track

    # print(f'multi_track_merge() took {time.time() - start_time:.3f}s')

    return (merged_pcl_abstract, merged_features_global, merged_implicit_output)


def merge_pcl_all_steps_into_long(pcl_all_list, last_minus=0):
    '''
    From a list of clips exported at test time, merge results into one long video.
    :param pcl_all_list: List-To of List-Ti of (input, abstract, output_solid, target, output_air)
            tuples; one outer list per test step, one inner list per predicted frame within a clip.
        input (N, 8) numpy array with
            (x, y, z, R, G, B, t, mark_track).
        abstract (M, 3+E) numpy array with
            (x, y, z, features).
        output_solid (N, 9+) numpy array with
            (x, y, z, t, density, R, G, B, mark_track, segm?).
        target (N, 9-11) numpy array with
            (x, y, z, cosine_angle?, instance_id, semantic_tag?, view_idx, R, G, B, mark_track).
        output_air (A, 5) numpy array with
            (x, y, z, density, pred_segm).
    :param last_minus (int): 0 for the last, 1 for one before, etc.
    :return pcl_all_long: List-To of (input, abstract, output_solid, target, output_air).
    '''

    # Combine all last frames of all clips into this long variant of pcl_all.
    pcl_all_long = []
    print('pcl_all_list:', len(pcl_all_list))

    for i, step_pcl_all in tqdm.tqdm(enumerate(pcl_all_list)):

        pcl_input = step_pcl_all[0][0]
        pcl_abstract = step_pcl_all[0][1]
        input_frames = len(np.unique(pcl_input[..., -2]))

        sel_input = pcl_input[pcl_input[..., -2] == input_frames - 1 - last_minus].copy()
        sel_abstract = pcl_abstract
        sel_output_solid = step_pcl_all[-1 - last_minus][2].copy()
        sel_target = step_pcl_all[-1 - last_minus][3]
        sel_output_air = step_pcl_all[-1 - last_minus][4]

        if i == 0:
            print('sel_input:', sel_input.shape)
            print('sel_abstract:', sel_abstract.shape)
            print('sel_output_solid:', sel_output_solid.shape)
            print('sel_target:', sel_target.shape)
            print('sel_output_air:', sel_output_air.shape)

        # Substitute time indices in input and output arrays.
        sel_input[..., -2] = i
        sel_output_solid[..., 3] = i

        cur_pcl_all = [sel_input, sel_abstract, sel_output_solid, sel_target, sel_output_air]
        pcl_all_long.append(cur_pcl_all)

    # Replace first input point cloud to include all input frames (with varying time index).
    # This allows MyPlot and display_input_points_output_mesh_attn to continue working.
    pcl_all_long[0][0] = np.concatenate([pcl[0] for pcl in pcl_all_long], axis=0)

    return pcl_all_long


def load_pcl_all_list(input_path, dp_filter=None, step_inds=None):
    pcl_all_list = []
    
    src_fp_list = find_testres_pcl_fp_list(input_path, dp_filter=dp_filter)
    step_idx = 0
    while True:
        if step_inds is not None and step_idx not in step_inds:
            break
        
        matches = [fp for fp in src_fp_list if f'_s{step_idx}.p' in fp]
        if len(matches) == 0 and (step_inds is None or step_idx > max(step_inds)):
            break
        
        pcl_src_fp = matches[0]
        if np.log2(max(step_idx, 1)) % 1 == 0:
            print(step_idx, pcl_src_fp)
        
        with open(pcl_src_fp, 'rb') as f:
            pcl_all = pickle.load(f)
        pcl_all_list.append(pcl_all)
        
        step_idx += 1
    
    return pcl_all_list
