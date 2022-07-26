'''
Handling of parameters that can be passed to training and testing scripts.
Created by Basile Van Hoorick for Revealing Occlusions with 4D Neural Fields.
'''

from __init__ import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _fix_resume(resume, checkpoint_root):
    '''
    If resume == 'v6', then finds latest epoch
    'checkpoints/v6_abc/model_24.pt'.
    Does not actually modify args.
    '''
    search_dir = checkpoint_root
    dns = os.listdir(search_dir)
    dps = [os.path.join(search_dir, dn) for dn in dns]
    dps = [dp for dp in dps if os.path.isdir(dp)]
    dps = [dp for dp in dps if resume + '_' in dp]
    assert len(dps) == 1, \
        'Exactly one matching checkpoint folder is expected, but found: ' + \
        str(dps)
    checkpoint_fp = os.path.join(dps[0], 'checkpoint.pth')
    print('Searched for resume', resume, 'and found:', checkpoint_fp)
    return checkpoint_fp


def _arg2str(arg_value):
    if isinstance(arg_value, bool):
        return '1' if arg_value else '0'
    else:
        return str(arg_value)


def shared_args(parser):

    # Misc options.
    parser.add_argument('--device', default='cuda', type=str,
                        help='cuda or cpu.')
    parser.add_argument('--num_workers', default=-1, type=int,
                        help='Number of data loading workers; -1 means automatic.')
    parser.add_argument('--seed', default=1830, type=int,
                        help='Random number generator seed.')
    parser.add_argument('--mixed_precision', default=False, type=str2bool,
                        help='Use 16-bit float to save GPU memory.')

    # Logging & checkpointing options.
    parser.add_argument('--data_path', default='/local/vondrick/GR_vbsnitch_4/', type=str,
                        help='Path to dataset root folder, or single scene for sequential '
                        'iteration.')
    parser.add_argument('--name', default='', type=str,
                        help='Tag of this experiment for bookkeeping.')
    parser.add_argument('--log_root', default='logs/', type=str,
                        help='Path to parent collection of logs, visualizations, and results.')
    parser.add_argument('--resume', '--checkpoint_path', default='', type=str,
                        help='Checkpoint to resume from.')
    parser.add_argument('--checkpoint_root', default='checkpoints/', type=str,
                        help='Path to parent collection of checkpoint folders.')

    # Data options (all phases).
    parser.add_argument('--use_data_frac', default=1.0, type=float,
                        help='If < 1.0, use a smaller dataset. If negative, interpret as absolute '
                        'size (number of elements).')
    parser.add_argument('--sample_bias', default='none', type=str,
                        help='Mode for sampling clips within scene videos (none / occl / move). '
                        'If occl: Use occlusion_rate.npy or occl.txt. '
                        'If move: Only return example if car (sensor extrinsics) is in motion.')
    parser.add_argument('--sb_occl_frame_shift', default=2, type=int,
                        help='If sample_bias is occl, then the peak occlusion rate occurs at '
                        'precisely this many frames before the present (i.e. end of input). '
                        'For example, if this value is 2, then the input sees the object occluded '
                        'roughly the last 2 frames. Set equal to past_frames if you want to start '
                        'predicting right when the occlusion happens.')


def verify_args(args, is_train=False):
    assert args.device in ['cuda', 'cpu']
    assert args.sample_bias in ['none', 'move', 'occl', 'move_occl', 'occl_move']

    if args.num_workers < 0:
        if is_train:
            args.num_workers = int(multiprocessing.cpu_count() * 0.9) - 12
        else:
            args.num_workers = multiprocessing.cpu_count() // 4 - 6

    if is_train:
        if args.cr_cube_bounds <= 0.0:
            args.cr_cube_bounds = args.pt_cube_bounds

        while len(args.cr_attn_type) < args.cross_attn_layers:
            # Repeat attention layer type until all instances are specified.
            assert len(args.cr_attn_type) != 0
            args.cr_attn_type = args.cr_attn_type + args.cr_attn_type

        assert 256 <= args.n_points <= 65536
        assert args.pt_norm_type in ['none', 'batch', 'layer']
        assert args.past_frames + args.future_frames <= args.video_len
        assert args.future_frames < args.video_len
        assert args.local_implicit_mode in ['none', 'feature', 'attention']
        assert args.color_mode in ['rgb', 'rgb_nosigmoid', 'hsv', 'bins']

    else:
        assert args.point_sample_mode in ['random', 'grid']


def train_args():
    parser = argparse.ArgumentParser()

    shared_args(parser)

    # Misc options.
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size.')
    parser.add_argument('--output_path', default='auto', type=str,
                        help='Path to checkpoint folder for this run.')

    # Point transformer model / architecture options.
    parser.add_argument('--up_down_blocks', default=3, type=int,
                        help='Number of down blocks and up blocks in the point transformer model.')
    parser.add_argument('--transition_factor', default=3, type=int,
                        help='Point cloud set abstraction (downsampling and upsampling) ratio.')
    parser.add_argument('--pt_feat_dim', default=32, type=int,
                        help='Initial embedding size for the first MLP in the network.')
    parser.add_argument('--pt_num_neighbors', default=14, type=int,
                        help='Number of nearest neighbors in the point transformer block self '
                        'attention layers, and the CR cross-attention layers.')
    parser.add_argument('--pt_norm_type', default='none', type=str,
                        help='none / batch / layer. Normalization layer in the point transformer '
                        'model (the CR itself never has any normalization). Note that this '
                        'historically has stability issues. The point transformer paper uses '
                        'BatchNorm, while 3DETR uses LayerNorm.')
    parser.add_argument('--down_neighbors', default=8, type=int,
                        help='Number of nearest neighbors in the down transition block.')
    parser.add_argument('--global_size', default=128, type=int,
                        help='Global embedding size after the encoder, if active.')
    parser.add_argument('--num_cr_local_feats', default=8, type=int,
                        help='Number of local features to interpolate for the CR.')

    # Data options (all phases).
    parser.add_argument('--n_points', default=8192, type=int,
                        help='Input number of points for the point transformer model. '
                        'A combination of random and farthest point subsampling (FPS) is used to '
                        'meet this condition.')
    parser.add_argument('--n_data_rnd', default=16384, type=int,
                        help='Initial random subsampling per frame per view for preprocessing and '
                        'faster dataloading. Set to -1 to disable this step and use FPS only.')
    parser.add_argument('--video_len', default=6, type=int,
                        help='Total video length, i.e. number of frames, including past, present, '
                        'and future.')
    parser.add_argument('--frame_skip', default=4, type=int,
                        help='Frame interval for input video. FPS = 24 / frame_skip.')
    parser.add_argument('--min_z', default=-1.0, type=float,
                        help='Vertical minimum point of the data cube to consider; discard '
                        'everything outside this range.')
    parser.add_argument('--pt_cube_bounds', default=5.0, type=float,
                        help='All other dimensions (x, y, z) in meters of the input data cube to '
                        'consider for the point transformer; discard everything outside this '
                        'range. Recommended between 5.0 for GREATER, 16.0 for CARLA.')
    parser.add_argument('--cr_cube_bounds', default=-1.0, type=float,
                        help='Output data cube bounds for the CR; typically smaller than PT. '
                        'Recommended between 5.0 for GREATER, 12.0 for CARLA.')
    parser.add_argument('--cube_mode', default=4, type=int,
                        help='Which cuboid shape to use for CARLA (1 / 2 / 3 / 4).')
    parser.add_argument('--correct_ego_motion', default=True, type=str2bool,
                        help='CARLA only; transform all lidar measurements to the coordinate '
                        'system of the present frame. Otherwise, transforms happen across views '
                        'only, i.e. the sensors are static and the world is moving.')
    parser.add_argument('--correct_origin_ground', default=True, type=str2bool,
                        help='CARLA only; translate all lidar measurements in the Z direction such '
                        'that the origin equals the ground, instead of existing relative to the '
                        'car height.')

    # Continuous representation (CR) model / architecture options.
    parser.add_argument('--positional_encoding', default=True, type=str2bool,
                        help='Use NeRF-inspired Fourier positional encoding for xyzt coordinates.')
    parser.add_argument('--activation', default='relu', type=str,
                        help='Scene function MLP activation function (relu / swish). '
                        'Swish is x * sigmoid(x).')
    parser.add_argument('--implicit_mlp_blocks', default=6, type=int,
                        help='Number of ResNet FC blocks, each of which contain two linear layers.')
    parser.add_argument('--use_global_embedding', default=True, type=str2bool,
                        help='Use the global feature of the entire input video.')
    parser.add_argument('--local_implicit_mode', default='attention', type=str,
                        help='none / feature / attention. If feature: Condition CR on manually '
                        'weighted average of K = 8 nearest abstract points. If attention: '
                        'Incorporate vector cross attention for more expressive correspondence.')
    parser.add_argument('--cross_attn_layers', default=1, type=int,
                        help='Number of query-to-abstract point transformer vector attention '
                        'layers, placed uniformly inbetween the residual MLP blocks.')
    parser.add_argument('--cross_attn_neighbors', default=12, type=int,
                        help='Number of nearest neighbors in the point transformer CR '
                        'cross-attention layers.')
    parser.add_argument('--cr_attn_type', default='c', type=str,
                        help='Type of vector attention (cross-attention or self-attention) layers '
                        ' to use. For example, if --cross_attn_layers 3, give csc for cross, then '
                        'self, then cross among the residual MLP blocks.')
    parser.add_argument('--abstract_levels', default=1, type=int,
                        help='How many abstract point cloud feature sets to use. If > 1, we use '
                        'hierarchical features similar to skip connections, but the lowest levels '
                        'will always be the most used because of nearest neighbor attention.')

    # CR output options.
    parser.add_argument('--color_mode', default='rgb', type=str,
                        help='How to predict color (rgb / rgb_nosigmoid / hsv / bins). '
                        'If rgb: Regress 3 output values with L1 loss. '
                        'If rgb_nosigmoid: Like rgb, but without sigmoid. '
                        'If hsv: Classify hue into Q bins, regress saturation and value. '
                        'If bins: Classify all colors into Q bins.')
    parser.add_argument('--semantic_classes', default=13, type=int,
                        help='If segmentation_lw > 0, how many segmentation categories to use. '
                        'CARLA has 23, but only the first 13 seem more important. '
                        'If < 23, all overflow classes get mapped to tag index 3 (Other).')

    # Training options.
    parser.add_argument('--parallel_pipeline', default=True, type=str2bool,
                        help='If True, use maximal parallelization for examples within batches '
                        'across model inference and loss calculation.')
    parser.add_argument('--learn_rate', default=1e-3, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_decay', default=0.4, type=float,
                        help='Multiplicative learning rate (gamma) factor per epoch for step '
                        'scheduler.')
    parser.add_argument('--num_epochs', default=20, type=int,
                        help='Number of epochs.')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='Starting epoch index (0-based).')
    parser.add_argument('--gradient_clip', default=0.2, type=float,
                        help='If > 0, clip gradient L2 norm to this value for stability.')

    # Loss options.
    parser.add_argument('--density_lw', default=1.0, type=float,
                        help='CR density (occupancy) loss term weight.')
    parser.add_argument('--color_lw', default=0.0, type=float,
                        help='CR output RGB regression / HSV mixed / bins classification loss '
                        'weight.')
    parser.add_argument('--segmentation_lw', default=0.0, type=float,
                        help='CR semantic category output classification loss term weight.')
    parser.add_argument('--tracking_lw', default=0.0, type=float,
                        help='Semi-supervised tracking by instance ids loss term weight.')
    parser.add_argument('--point_occupancy_radius', default=0.2, type=float,
                        help='Occupancy should be 1 when this close to a ground truth point, '
                        'and 0 otherwise.')
    parser.add_argument('--num_cr_solid', default=7168, type=int,
                        help='How many solid query points to sample and supervise per frame. '
                        'This is heavily correlated with GPU memory usage.')
    parser.add_argument('--air_sampling_ratio', default=1.5, type=float,
                        help='Sample this many free space points outside of the ground truth '
                        'point cloud, relative to the number of points.')
    parser.add_argument('--point_sample_bias', default='none', type=str,
                        help='How loss weighting works within every scene (none / low / moving / '
                        'vehped / ivalo / sembal). Most of these are for CARLA only. '
                        'If none: Baseline where air points are sampled uniformly. '
                        'If low: More air points between z = 0 and 2. '
                        'If moving: More solid + air points near dynamic regions. '
                        'If vehped: More solid points near vehicles and pedestrians using semantic LiDAR data. '
                        'If ivalo (invisible but visible at least once): Compare input to target for solvable cars and people. '
                        'If sembal: Balance point counts by semantic category. '
                        'Changing this option may improve the output quality thanks to more focused supervision.')
    parser.add_argument('--oversample_vehped_target', default=False, type=str2bool,
                        help='Retain all cars & people in the CARLA data loader when subsampling target.')
    parser.add_argument('--past_frames', default=2, type=int,
                        help='How many past / present frames (t < 0) to predict and supervise.')
    parser.add_argument('--future_frames', default=0, type=int,
                        help='How many future frames (t >= 0) to predict and supervise.')

    args = parser.parse_args()
    verify_args(args, is_train=True)

    if args.output_path == 'auto':
        keys = {
            'mixed_precision': 'mp',
            'up_down_blocks': 'ud',
            'n_points': 'np',
            'video_len': 'vl',
            'frame_skip': 'fs',
            'density_lw': 'dl',
            'color_lw': 'cl',
            'segmentation_lw': 'sl',
            'tracking_lw': 'tl',
        }
        tag = args.name + '_'
        tag += '_'.join(keys[k] + _arg2str(getattr(args, k)) for k in keys)

        if args.use_data_frac < 1.0:
            tag += f'_df{_arg2str(args.use_data_frac)}'
        tag += f'_gs{_arg2str(args.global_size) if args.use_global_embedding else 0}'
        tag += f'_a{_arg2str(args.activation[:2])}'
        tag += f'_im{_arg2str(args.local_implicit_mode[:2])}'
        tag += f'_pt{_arg2str(args.past_frames)}_{_arg2str(args.future_frames)}'

        args.tag = tag
        args.output_path = os.path.join(args.checkpoint_root, args.tag)

    if args.resume and not(os.path.exists(args.resume) and os.path.isfile(args.resume)):
        args.resume = _fix_resume(args.resume, args.checkpoint_root)

    return args


def test_args():
    parser = argparse.ArgumentParser()

    shared_args(parser)

    parser.add_argument('--ss_frame_step', default=3, type=int,
                        help='If a loop over an entire single example video is desired, select '
                        'video frame skip interval here.')
    parser.add_argument('--force_view_idx', default=-1, type=int,
                        help='If >= 0, for GREATER, always pick this input camera view index.')
    parser.add_argument('--log_path', default='auto', type=str,
                        help='Path to test results folder for this run.')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='GPU index.')
    parser.add_argument('--epoch', default=-1, type=int,
                        help='If >= 0, desired model epoch to use and test (0-based), otherwise '
                        'latest.')
    parser.add_argument('--implicit_batch_size', default=65536, type=int,
                        help='Maximum number of random points to feed into the CR at once. '
                        'This is typically correlated with GPU memory usage.')
    parser.add_argument('--sample_implicit', default=True, type=str2bool,
                        help='Whether to sample CRs in order to obtain and '
                        'be able to visualize output point clouds.')
    parser.add_argument('--num_sample', default=262144, type=int,
                        help='Number of random points to forward through the continuous '
                        'representation in order to obtain an output point cloud of solids.')
    parser.add_argument('--point_sample_mode', default='random', type=str,
                        help='How to sample points within CRs (random / grid).')
    parser.add_argument('--store_pcl', default=True, type=str2bool,
                        help='Whether to store point clouds for later 3D visualization.')
    parser.add_argument('--density_threshold', default=0.5, type=float,
                        help='Within the sampled implicit output, how to distinguish between solid '
                        'and air points in terms of predicted density / occupancy. The output '
                        'point cloud is defined as retaining all solid points only.')
    parser.add_argument('--store_activations', default=False, type=str2bool,
                        help='Store internal network data (neuron activations) for later '
                        'attention-related and emergence of tracking insights.')
    parser.add_argument('--save_metrics', default=False, type=str2bool,
                        help='Whether to save metric data for visualization.')
    parser.add_argument('--track_mode', default='none', type=str,
                        help='How many objects to track, if at all (none / one / all). '
                        'If none: Assume mark_track is not supervised. '
                        'If one: Use dataset instance default (snitch / random). '
                        'If all: Cycle through all instance IDs in the input frame and merge '
                        'predictions by averaging everything, while argmaxing over mark_track '
                        '(point_sample_mode must be grid).')
    parser.add_argument('--use_json', default=False, type=str2bool,
                        help='Use predetermined test clips (frame start + source view).')
    parser.add_argument('--live_occl_mode', default='normal', type=str,
                        help='Which points to use for estimating occlusion rates and visible at '
                        'least once objects (normal / unfilt).')

    args = parser.parse_args()
    verify_args(args, is_train=False)

    # When we point to dataset root and test subfolder exists, assume that is meant.
    if os.path.exists(os.path.join(args.data_path, 'test')):
        args.data_path = os.path.join(args.data_path, 'test')

    if args.resume and not(os.path.exists(args.resume) and os.path.isfile(args.resume)):
        args.resume = _fix_resume(args.resume, args.checkpoint_root)
        # Allow load_models to pick desired epoch.
        args.resume = str(pathlib.Path(args.resume).parent)

    if args.log_path == 'auto':
        # Generate test results folder within logs, given checkpoint path.
        args.log_path = str(pathlib.Path(
            args.resume.replace('checkpoints', 'logs')))
        keys = {
            'use_data_frac': 'df',
            'sample_bias': 'sb',
            'num_sample': 'ns',
            'point_sample_mode': 'sm',
            'density_threshold': 'dt',
            'store_activations': 'sa',
            'save_metrics': 'sm',
            'track_mode': 'tm',
            'use_json': 'uj',
        }
        if len(args.name) != 0:
            test_tag = args.name + '_'
        else:
            test_tag = ''
        test_tag += '_'.join(keys[k] + _arg2str(getattr(args, k))
                             for k in keys)
        args.test_tag = test_tag  # Epoch gets appended later.

    else:
        # Take the parent folder because the specified subfolder is treated as the tag.
        args.log_path = str(pathlib.Path(args.log_path).parent)
        assert os.path.exists(args.log_path) and os.path.isdir(args.log_path)
        # Epoch gets appended later.
        args.test_tag = str(pathlib.Path(args.log_path).name)

    # Update these args because MyLogger uses them.
    args.log_root = str(pathlib.Path(args.log_path).parent)
    args.train_tag = str(pathlib.Path(args.log_path).name)
    args.tag = args.train_tag

    return args
