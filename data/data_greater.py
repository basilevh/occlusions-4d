'''
Data loading and processing logic.
Created by Basile Van Hoorick for Revealing Occlusions with 4D Neural Fields.
'''

from __init__ import *
import torch

# Library imports.
import glob
import imageio
import json
from threading import RLock

# Internal imports.
import data_utils
import geometry
import utils

_MAX_DEPTH_CLIP = 32.0

# Use preflat instead of flat because of Blender. Values are ints in the range [0, 360] degrees.
_PREFLAT_HUE_CLUSTERS = [0, 35, 47, 65, 90, 160, 180, 188, 219, 284, 302, 324]

_MAX_VALO_IDS = 32


def get_occlusion_rate(scene_dp, src_view):
    snitch_occl_fp = os.path.join(scene_dp, 'occl.txt')
    snitch_occl = np.loadtxt(snitch_occl_fp)  # (T).
    snitch_occl = snitch_occl[src_view]

    frame_step = 3
    occlusion_rate = np.zeros_like(snitch_occl)
    occlusion_rate[frame_step:] = snitch_occl[frame_step:] - snitch_occl[:-frame_step]
    occlusion_rate = np.clip(occlusion_rate, 0.0, 1.0)

    return occlusion_rate


class GREATERDataset(torch.utils.data.Dataset):
    '''
    Assumes directory & file structure:
    dataset_root\train\GREATER_000012\images_view2\0123.png + 0123_depth.png + 0123_preflat.png.
    '''

    @staticmethod
    def max_depth_clip():
        return _MAX_DEPTH_CLIP

    @staticmethod
    def preflat_hue_clusters():
        return _PREFLAT_HUE_CLUSTERS

    def __init__(self, dataset_root, logger, stage='train',
                 ss_frame_step=2, video_length=4, frame_skip=4, convert_to_pcl=True,
                 n_points_rnd=8192, n_fps_input=1024, n_fps_target=1024,
                 pcl_input_frames=3, pcl_target_frames=1,
                 sample_bias='none', sb_occl_frame_shift=2,
                 min_z=-1.0, other_bounds=5.0, return_segm=True, track_mode='none',
                 use_data_frac=1.0, use_json=True, verbose=False,
                 live_occl_mode='normal', force_view_idx=-1):
        '''
        :param dataset_root (str): Path to dataset or single scene.
        :param stage (str): Subfolder (dataset split) to use; may remain empty.
        :param ss_frame_step (int): If a loop over an entire single example video is desired,
            set >= 0 to select which one. This typically takes a while.
        :param n_points_rnd (int): Number of points to retain after initial random subsampling.
            This is applied almost directly after converting the RGB-D data to a point cloud.
        :param n_fps_input, n_fps_target (int): If > 0, number of points to retain after farthest
            point sampling of the input and target point clouds respectively after processing for
            time and/or view aggregation.
        :param pcl_input_frames (int): If previd, number of input frames to show, counting from the
            beginning.
        :param pcl_target_frames (int): If last_couple_merged, number of target frames to provide,
            counting from the end. Typically, video_length <= pcl_input_frames + pcl_target_frames.
        :param sample_bias (str): Mode for sampling clips within scene videos (none / occl).
        :param sb_occl_frame_shift (int): If sample_bias is occl, then the peak occlusion rate
            occurs at precisely this many frames before the present (i.e. end of input).
        :param track_mode (str): Which object to mark for tracking (none / snitch / random).
        :param force_view_idx (int): If >= 0, always pick this camera view index.
        '''
        self.dataset_root = dataset_root
        self.logger = logger
        self.stage = stage
        self.ss_frame_step = ss_frame_step
        self.video_length = video_length
        self.frame_skip = frame_skip
        self.convert_to_pcl = convert_to_pcl
        self.n_points_rnd = n_points_rnd
        self.n_fps_input = n_fps_input
        self.n_fps_target = n_fps_target
        self.pcl_input_frames = pcl_input_frames
        self.pcl_target_frames = pcl_target_frames
        self.sample_bias = sample_bias
        self.sb_occl_frame_shift = sb_occl_frame_shift
        self.min_z = min_z
        self.other_bounds = other_bounds
        self.return_segm = return_segm
        self.track_mode = track_mode
        self.use_data_frac = use_data_frac
        self.use_json = use_json
        self.verbose = verbose
        self.allow_random_frames = True
        self.live_occl_mode = live_occl_mode
        self.force_view_idx = force_view_idx

        self.stage_dir = os.path.join(dataset_root, stage)
        if not os.path.exists(self.stage_dir):
            self.stage_dir = dataset_root  # We may already be pointing to the stage directory.
            self.dataset_root = str(pathlib.Path(dataset_root).parent)

        self.is_single_scene = ('images_view1' in os.listdir(self.stage_dir))

        if self.is_single_scene:
            logger.warning(f'({stage}) Pointing to single example! Ignoring parameters: '
                           f'sample_bias, sb_occl_frame_shift, use_json.')
            self.num_scenes = 1
            self.all_scenes = [self.stage_dir]

            image_dp = os.path.join(self.stage_dir, 'images_view1')
            rgb_fns = [fn for fn in os.listdir(image_dp) if fn[-4:] == '.png' and len(fn) <= 8]
            num_total_frames = len(rgb_fns)

            # Incorporate absolute dataset size if specified.
            # NOTE: use_data_frac has a different meaning here, compared to the usual case.
            if use_data_frac < 0.0:
                self.use_data_frac = 1.0
                self.multiplier = use_data_frac
            else:
                self.use_data_frac = use_data_frac
                self.multiplier = num_total_frames / self.ss_frame_step - \
                    self.video_length * self.frame_skip

            self.dset_size = int(self.multiplier * self.use_data_frac)

        else:
            all_scenes = os.listdir(self.stage_dir)
            all_scenes = [dn for dn in all_scenes if '_' in dn
                          and os.path.isdir(os.path.join(self.stage_dir, dn))]
            all_scenes.sort()
            self.all_scenes = all_scenes
            self.num_scenes = len(all_scenes)

            # Incorporate absolute dataset size if specified.
            if use_data_frac < 0.0:
                self.num_scenes = int(-use_data_frac)
                self.all_scenes = self.all_scenes[:self.num_scenes]
                logger.warning(f'({stage}) Using absolute dataset size: {self.num_scenes}')
                logger.info(f'({stage}) Dataset examples: {self.all_scenes}')
                logger.warning(f'({stage}) No random frame starts (always pick middle)!')
                self.use_data_frac = 1.0
                self.allow_random_frames = False

            # Ensure non-trivial epoch size even with tiny datasets.
            # This is justified because we select a small clip within every relatively large video.
            target_size = 960 if 'train' in stage else 120
            self.multiplier = max(int(np.ceil(target_size / self.num_scenes)), 1)
            self.dset_size = int(self.num_scenes * self.multiplier * self.use_data_frac)
            if self.multiplier > 1:
                logger.warning(f'({stage}) Using dataset size multiplier: {self.multiplier}')

            if self.sample_bias != 'none':
                logger.warning(f'({stage}) Sample bias is {self.sample_bias}, '
                               f'so some options will be ignored.')
                # Number of returned clips per scene + lock for synchrony.
                self.max_frames_ever = 10101
                self.scene_counter = multiprocessing.Array(
                    'i', self.num_scenes * self.max_frames_ever)
                self.counter_lock = RLock()

            # Load indices of starting frames (if pre-computed).
            self.starting_frames = None
            if 'test' in self.stage and self.use_json:

                if 1:
                    # Ensure different ablations have exact same last frame as regular by hardcoding
                    # reference to 12. Also remember manual shift to apply.
                    test_frames_fn = f'test_start_frames_shift{sb_occl_frame_shift}_inputframes12_skip{frame_skip}.json'
                    self.json_shift = (12 - pcl_input_frames) * frame_skip

                else:
                    test_frames_fn = f'test_start_frames_shift{sb_occl_frame_shift}_inputframes{pcl_input_frames}_skip{frame_skip}.json'
                    self.json_shift = 0

                test_frames_fp = os.path.join(self.dataset_root, test_frames_fn)
                if os.path.exists(test_frames_fp):
                    logger.warning(
                        f'({stage}) Using {test_frames_fn}, so some options will be ignored.')
                    logger.warning(
                        f'({stage}) Manual additional backward frame shift: {self.json_shift}')
                    with open(test_frames_fp, 'r') as f:
                        self.starting_frames = json.load(f)

                else:
                    logger.warning(f'({stage}) {test_frames_fp} not found.')

        self.to_tensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return self.dset_size

    def _get_frame_start(self, index, scene_dp, src_view):
        '''
        :return (frame_start, src_view, num_frames, occl_frame_idx, found_occl_rate,
                 proceed_sample_bias).
        '''
        image_dp = os.path.join(scene_dp, 'images_view1')
        rgb_fns = [fn for fn in os.listdir(image_dp) if fn[-4:] == '.png' and len(fn) <= 8]
        num_frames = len(rgb_fns)
        occl_frame_idx = -1
        found_occl_rate = -1.0
        proceed_sample_bias = False

        if self.is_single_scene:
            # index now refers to frame start instead of scene index.
            scene_idx = -1
            frame_start = index * self.ss_frame_step

        else:
            scene_idx = index // self.multiplier
            frame_low = 0
            frame_high = num_frames
            frame_start_high = frame_high - self.video_length * self.frame_skip

            # Make initial random clip selection, to be refined later.
            frame_start = np.random.randint(0, frame_start_high)

            proceed_sample_bias = True
            if self.starting_frames is not None:
                (frame_start, src_view) = self.starting_frames[str(scene_idx)]
                frame_start += self.json_shift
                proceed_sample_bias = False
            elif 'test' not in self.stage:
                # During train & val, only perform biased clip sampling 30% of the time.
                proceed_sample_bias = (np.random.rand() < 0.30)

            elif self.sample_bias != 'none' and proceed_sample_bias:

                if 'occl' in self.sample_bias:
                    # Assume that the occlusion "happens at" the index where occlusion_rate is the
                    # largest. First, look for the frame indices with maximal occlusion rates, and
                    # then take care to assign the clip bounds correctly.
                    occlusion_rate = get_occlusion_rate(scene_dp, src_view)
                    select_top = 40
                    top_frame_inds = np.argpartition(occlusion_rate, -select_top)[-select_top:]
                    top_frame_inds = top_frame_inds[np.argsort(occlusion_rate[top_frame_inds])]
                    top_frame_inds = top_frame_inds[::-1]  # Rank highest to lowest.

                    # During train & val, only follow the ranking very roughly and with extra
                    # randomness. https://github.com/rragundez/elitist-shuffle
                    if 'test' not in self.stage:
                        top_frame_inds = utils.elitist_shuffle(top_frame_inds, inequality=4)

                    # The occlusion must happen near the end of the input.
                    time_shift = int((self.pcl_input_frames - self.sb_occl_frame_shift) *
                                     self.frame_skip)
                    found_occl_rate = -1.0
                    range_blocks = 0
                    counter_blocks = 0

                    for occl_frame_idx in top_frame_inds:
                        try_frame_start = occl_frame_idx - time_shift

                        if try_frame_start < frame_low or frame_start_high <= try_frame_start:
                            range_blocks += 1
                            continue

                        with self.counter_lock:
                            if self.scene_counter[scene_idx * self.max_frames_ever + try_frame_start]:
                                counter_blocks += 1
                                continue

                            # Commit to this clip so that it cannot be reselected. Next time
                            # this scene index comes up, it will map to a lower occlusion rate.
                            frame_start = try_frame_start
                            self.scene_counter[scene_idx * self.max_frames_ever + frame_start] = 1
                            found_occl_rate = occlusion_rate[occl_frame_idx]
                            break

                    if found_occl_rate < 0.0:
                        self.logger.warning(f'No clip with high occlusion rate found! '
                                            f'Time range blocks: {range_blocks}. '
                                            f'Counter blocks: {counter_blocks}. '
                                            f'Keeping random frame_start: {frame_start}...')

            elif not self.allow_random_frames:
                frame_start = num_frames // 2

        return (frame_start, src_view, num_frames, occl_frame_idx, found_occl_rate, proceed_sample_bias)

    def __getitem__(self, index):
        '''
        :return Dictionary with all information for a single example.
        '''
        initial_index = index
        keep_nss = ('unfilt' in self.live_occl_mode)

        if self.is_single_scene:
            # index now refers to frame start instead of scene index.
            scene_idx = -1
            scene_dp = self.all_scenes[0]

        else:
            scene_idx = index // self.multiplier
            scene_dp = os.path.join(self.stage_dir, self.all_scenes[scene_idx])

        image_dns = [dn for dn in os.listdir(scene_dp) if 'images' in dn]
        image_dns.sort()
        image_dps = [os.path.join(scene_dp, dn) for dn in image_dns]
        pose_dns = [dn for dn in os.listdir(scene_dp) if 'poses' in dn]
        pose_dns.sort()
        pose_dps = [os.path.join(scene_dp, dn) for dn in pose_dns]
        assert len(image_dns) == len(pose_dns)

        num_views = len(image_dns)
        # Already make a random selection of input view index for future use.
        if self.force_view_idx >= 0:
            src_view = self.force_view_idx
        else:
            src_view = np.random.randint(num_views)

        (frame_start, src_view, num_frames, occl_frame_idx, found_occl_rate,
         proceed_sample_bias) = self._get_frame_start(index, scene_dp, src_view)

        frame_end = frame_start + self.video_length * self.frame_skip
        frame_inds = np.arange(frame_start, frame_end, self.frame_skip)

        all_rgb = []
        all_depth = []
        all_flat = []
        all_snitch = []
        all_RT = []
        all_K = []
        all_pcl = []
        all_pcl_nss = []  # Not subsampled for accurate live_occl.
        cuboid_filter_ratios = []
        sample_input_ratios = []
        sample_target_ratios = []

        for v in range(num_views):
            view_rgb = []
            view_depth = []
            view_flat = []
            view_snitch = []
            view_RT = []
            view_K = []
            view_pcl = []
            view_pcl_nss = []

            src_RT_fp = os.path.join(pose_dps[v], 'camera_RT.npy')
            src_K_fp = os.path.join(pose_dps[v], 'camera_K.npy')
            src_RT = np.load(src_RT_fp)
            src_K = np.load(src_K_fp)

            for f in frame_inds:
                rgb_fp = os.path.join(image_dps[v], f'{f:04d}.png')
                flat_fp = os.path.join(image_dps[v], f'{f:04d}_preflat.png')
                depth_fp = os.path.join(image_dps[v], f'{f:04d}_depth.png')

                rgb = plt.imread(rgb_fp)[..., :3].astype(np.float32)
                flat = plt.imread(flat_fp)[..., :3].astype(np.float32)
                depth = plt.imread(depth_fp).astype(np.float32) * _MAX_DEPTH_CLIP
                cam_RT = src_RT[f].astype(np.float32)
                cam_K = src_K[f].astype(np.float32)
                cam_K[1, 1] = cam_K[0, 0]

                view_rgb.append(rgb)
                view_depth.append(depth)
                view_flat.append(flat)
                view_RT.append(cam_RT)
                view_K.append(cam_K)

                if self.return_segm:
                    snitch_fp = os.path.join(image_dps[v], f'{f:04d}_preflat_snitch.png')
                    snitch = plt.imread(snitch_fp)
                    view_snitch.append(snitch)

            view_rgb = np.stack(view_rgb)  # (T, H, W, 3).
            view_depth = np.stack(view_depth)  # (T, H, W).
            view_flat = np.stack(view_flat)  # (T, H, W, 3).
            view_snitch = np.stack(view_snitch) if self.return_segm else []  # (T, H, W, 3).
            view_RT = np.stack(view_RT)  # (T, 3, 4).
            view_K = np.stack(view_K)  # (T, 3, 3).

            # Extract point clouds.
            for f in range(len(frame_inds)):
                rgb = view_rgb[f]
                flat = view_flat[f]
                depth = view_depth[f]
                cam_RT = view_RT[f]
                cam_K = view_K[f]

                # Extract instance_id from flat by rounding to the nearest known hue cluster.
                flat_hsv = matplotlib.colors.rgb_to_hsv(flat)
                inst_ids = np.round(flat_hsv[..., 0:1] * 360.0)  # (H, W, 1).
                inst_ids = np.abs(inst_ids[..., None] - _PREFLAT_HUE_CLUSTERS)  # (H, W, 12).
                inst_ids = inst_ids.argmin(axis=-1)  # (H, W, 1).
                inst_ids[flat_hsv[..., 1] < 0.9] = -1.0  # Background or floor is irrelevant.

                # OLD: we falsely assumed that hue is always a multiple of 0.05 (18 deg).
                # inst_ids = np.round(flat_hsv[..., 0:1] * 20.0).astype(np.int32)
                # inst_ids[inst_ids >= 20] = 0

                # Incorporate both color and instance segmentation info in the point cloud.
                rgb_inst = np.concatenate([inst_ids, rgb], axis=-1)  # (H, W, 4).
                pcl_full = geometry.point_cloud_from_rgbd(rgb_inst, depth, cam_RT, cam_K)
                pcl_full = pcl_full.astype(np.float32)
                # (N, 7) with (x, y, z, instance_id, R, G, B).

                # Restrict to cuboid of interest.
                # NOTE: min_z >= 0.1 means discard the entire floor.
                pre_filter_size = pcl_full.shape[0]
                pcl_full = geometry.filter_pcl_bounds_numpy(
                    pcl_full, x_min=-self.other_bounds, x_max=self.other_bounds,
                    y_min=-self.other_bounds, y_max=self.other_bounds,
                    z_min=self.min_z, z_max=self.other_bounds, greater_floor_fix=True)
                post_filter_size = pcl_full.shape[0]
                cuboid_filter_ratios.append(post_filter_size / max(pre_filter_size, 1))

                # NOTE: This step has no effect if there are insufficient points.
                if keep_nss:
                    pcl_full_nss = pcl_full
                else:
                    pcl_full_nss = None
                if self.n_points_rnd > 0:
                    pcl_full = geometry.subsample_pad_pcl_numpy(
                        pcl_full, self.n_points_rnd, subsample_only=False)
                # NOTE: At this stage, we have only used primitive subsampling techniques, and
                # the point cloud is still generally oversized. Later, we use FPS and match
                # sizes taking into account how frames and/or views are aggregated.

                view_pcl.append(pcl_full)
                view_pcl_nss.append(pcl_full_nss)

            all_rgb.append(view_rgb)
            all_depth.append(view_depth)
            all_flat.append(view_flat)
            all_snitch.append(view_snitch)
            all_RT.append(view_RT)
            all_K.append(view_K)
            all_pcl.append(view_pcl)
            all_pcl_nss.append(view_pcl_nss)

        all_rgb = np.stack(all_rgb)  # (V, T, H, W, 3).
        all_depth = np.stack(all_depth)  # (V, T, H, W).
        all_flat = np.stack(all_flat)  # (V, T, H, W, 3).
        all_snitch = np.stack(all_snitch) if self.return_segm else []  # (V, T, H, W, 3).
        all_RT = np.stack(all_RT)  # (V, T, 3, 4).
        all_K = np.stack(all_K)  # (V, T, 3, 3).

        # Generate appropriate versions of the point cloud data.
        (V, T) = (num_views, self.video_length)
        all_pcl_sizes = np.array([[all_pcl[v][t].shape[0] for t in range(T)] for v in range(V)])
        # List-V of List-T of (N, 7) with (x, y, z, instance_id, R, G, B).
        pcl_video_views = utils.accumulate_pcl_time_numpy(all_pcl)
        # List-V of (T*N, 8) with (x, y, z, instance_id, R, G, B, t).
        pcl_merged_frames = utils.merge_pcl_views_numpy(all_pcl, insert_view_idx=True)
        # List-T of (V*N, 8) with (x, y, z, instance_id, view_idx, R, G, B).

        # Limit input to the desired time range.
        if self.pcl_input_frames < self.video_length:
            show_frame_size_sum = 0
            for t in range(self.pcl_input_frames):
                show_frame_size_sum += all_pcl[src_view][t].shape[0]
            pcl_input = pcl_video_views[src_view][:show_frame_size_sum]
        else:
            pcl_input = pcl_video_views[src_view]
        # (x, y, z, instance_id, R, G, B, t).

        # Always shuffle point cloud data just before converting to tensor.
        np.random.shuffle(pcl_input)
        pcl_input = self.to_tensor(pcl_input).squeeze(0)  # (T*N, 8).

        # Subsample random input video and merged target here for efficiency.
        pre_sample_size = pcl_input.shape[0]
        pcl_input = geometry.subsample_pad_pcl_torch(
            pcl_input, self.n_fps_input,
            sample_mode='farthest_point', subsample_only=False)
        post_sample_size = pcl_input.shape[0]
        sample_input_ratios.append(post_sample_size / max(pre_sample_size, 1))
        pcl_input_size = min(pre_sample_size, post_sample_size)

        pcl_target = []  # List-T of (V*N, 7).
        pcl_target_size = []
        for t in range(self.pcl_target_frames):
            pcl_target_frame = pcl_merged_frames[-self.pcl_target_frames + t]
            np.random.shuffle(pcl_target_frame)
            pcl_target_frame = self.to_tensor(pcl_target_frame).squeeze(0)  # (V*N, 8).
            # (x, y, z, instance_id, view_idx, R, G, B).
            pcl_target.append(pcl_target_frame)
            pcl_target_size.append(pcl_target_frame.shape[0])

        if self.n_fps_target != 0:
            # NOTE: farthest_point is relatively expensive, while random is fast but less spatially
            # balanced.
            sample_mode = 'farthest_point' if self.n_fps_target > 0 else 'random'

            for i in range(self.pcl_target_frames):
                pre_sample_size = pcl_target[i].shape[0]
                pcl_target[i] = geometry.subsample_pad_pcl_torch(
                    pcl_target[i], abs(self.n_fps_target),
                    sample_mode=sample_mode, subsample_only=False)
                post_sample_size = pcl_target[i].shape[0]
                sample_target_ratios.append(post_sample_size / max(pre_sample_size, 1))
                pcl_target_size[i] = min(pre_sample_size, post_sample_size)

        else:
            # Do not further subsample target point cloud.
            for i in range(self.pcl_target_frames):
                assert pcl_target[i].shape[0] == pcl_target_size[i]

        # Ensure instance id is kept separate in input view.
        pcl_input_sem = pcl_input[..., 3:-4]
        # (N, 1) with (instance_id).
        pcl_input = torch.cat([pcl_input[..., :3], pcl_input[..., -4:]], dim=-1)
        # (N, 7) with (x, y, z, R, G, B, t).

        # Get approximate per-instance occlusion percentage over time.
        all_pcl_for_occl = (all_pcl_nss if keep_nss else all_pcl)
        (live_occl, valo_ids_pad, num_valo_ids, _) = data_utils.get_valo_ids(
            self.live_occl_mode, scene_idx, scene_dp,
            False, 0, None, 3,
            self.pcl_input_frames, self.video_length, frame_start, frame_end, self.frame_skip,
            self.sb_occl_frame_shift, src_view, num_views, _MAX_VALO_IDS, self.logger,
            all_pcl_for_occl, pcl_input_sem, pcl_merged_frames)

        # Mark object of interest to track in input and target.
        # NOTE: At test time, this is typically taken care of in inference.py!
        track_id = -1
        pcl_input_track = torch.zeros_like(pcl_input[..., 0:1])
        pcl_target_track = [torch.zeros_like(ptf[..., 0:1]) for ptf in pcl_target]

        if self.track_mode != 'none':
            # We first check for IDs visible in the initial frame.
            pcl_input_first_sem = pcl_input_sem[pcl_input[..., -1] == 0]
            vis_ids = pcl_input_first_sem[..., 0].type(torch.int32).unique()
            vis_ids = list(vis_ids.detach().cpu().numpy())

            # Now filter by at least 16 points.
            vis_ids = [vis_id for vis_id in vis_ids
                       if vis_id >= 0 and (pcl_input_first_sem[..., 0] == vis_id).sum() >= 16]

            # If no appropriate id is available, we will not be tracking anything in this
            # example!
            if len(vis_ids) > 0:
                if self.track_mode == 'snitch':
                    track_id = 0
                elif self.track_mode == 'random':
                    track_id = np.random.choice(vis_ids)
                else:
                    raise ValueError()

                # Mark object in *first* input frame in time, but *all* target frames.
                mark_input_mask = torch.logical_and(
                    pcl_input_sem[..., 0] == track_id, pcl_input[..., -1] == 0)
                pcl_input_track[mark_input_mask] = 1.0
                for i in range(self.pcl_target_frames):
                    mark_target_mask = (pcl_target[i][..., 3] == track_id)
                    pcl_target_track[i][mark_target_mask] = 1.0

        # Append mark_track as extra (last) feature to input & target point clouds.
        pcl_input = torch.cat([pcl_input, pcl_input_track], dim=-1)
        # (N, 8) with (x, y, z, R, G, B, t, mark_track).
        for i in range(self.pcl_target_frames):
            pcl_target[i] = torch.cat([pcl_target[i], pcl_target_track[i]], dim=-1)
            # (M, 9) with (x, y, z, instance_id, view_idx, R, G, B, mark_track).

        # Metadata is all lightweight stuff (so no big arrays or tensors).
        meta_data = dict()
        meta_data['data_kind'] = 1001  # Cannot be string.
        meta_data['num_views'] = num_views
        meta_data['num_frames'] = num_frames  # Total in this scene video.
        meta_data['scene_idx'] = scene_idx
        meta_data['frame_inds'] = frame_inds  # Clip subselection, e.g. [88, 92, 96, 100].
        meta_data['src_view'] = src_view
        meta_data['n_fps_input'] = self.n_fps_input
        meta_data['n_fps_target'] = self.n_fps_target
        meta_data['pcl_sizes'] = all_pcl_sizes  # Per view and per frame.
        meta_data['pcl_input_size'] = pcl_input_size
        meta_data['pcl_target_size'] = pcl_target_size
        meta_data['cuboid_filter_ratios'] = cuboid_filter_ratios
        meta_data['sample_input_ratios'] = sample_input_ratios
        meta_data['sample_target_ratios'] = sample_target_ratios
        meta_data['occl_frame_idx'] = occl_frame_idx
        meta_data['found_occl_rate'] = found_occl_rate
        meta_data['proceed_sample_bias'] = proceed_sample_bias
        meta_data['valo_ids'] = valo_ids_pad
        meta_data['num_valo_ids'] = num_valo_ids
        meta_data['track_id'] = track_id

        # Make all information easily accessible.
        to_return = dict()
        to_return['rgb'] = all_rgb
        to_return['depth'] = all_depth
        to_return['flat'] = all_flat
        to_return['snitch'] = all_snitch
        to_return['cam_RT'] = all_RT
        to_return['cam_K'] = all_K
        to_return['pcl_input'] = pcl_input
        # (N, 8) with (x, y, z, R, G, B, t, mark_track).
        to_return['pcl_input_sem'] = pcl_input_sem
        # (N, 1) with (instance_id).
        to_return['pcl_target'] = pcl_target
        # List of (M, 9) with (x, y, z, instance_id, view_idx, R, G, B, mark_track).
        to_return['meta_data'] = meta_data

        if self.verbose and initial_index < 32:
            self.logger.info(f'scene_idx: {scene_idx}  frame_inds: {frame_inds}')
            if 'occl' in self.sample_bias and proceed_sample_bias:
                self.logger.info(f'occl_frame_idx: {occl_frame_idx}  '
                                 f'found_occl_rate: {found_occl_rate:.3f}')

        return to_return
