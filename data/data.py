'''
Data loading and processing logic.
Created by Basile Van Hoorick for Revealing Occlusions with 4D Neural Fields.
'''

from __init__ import *

# Internal imports.
import utils
from data_carla import CARLADataset
from data_greater import GREATERDataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_train_val_data_loaders(args, logger):
    '''
    return (data_kind, train_loader, val_aug_loader).
        data_kind (str): greater / carla.
    '''

    if 'carla' in args.data_path.lower():

        data_kind = 'carla'

        n_data_rnd = args.n_data_rnd
        n_model_input = args.n_points
        n_model_output = args.n_points
        # Negative means random instead of farthest_point. NOTE: This multiplier will typically
        # cause zero padding, but at least we avoid losing information.
        n_model_target = -int(max(abs(args.n_points), abs(args.n_data_rnd)) * 2)

        if args.correct_ego_motion:
            reference_frame = args.video_len - args.future_frames - 1
        else:
            reference_frame = None

        logger.info(f'n_model_input: {n_model_input}  n_model_output: {n_model_output}  '
                    f'n_model_target: {n_model_target}  n_data_rnd: {n_data_rnd}  '
                    f'reference_frame: {reference_frame}')

        dset_args = dict(
            video_length=args.video_len, frame_skip=args.frame_skip,
            n_points_rnd=n_data_rnd, n_fps_input=n_model_input, n_fps_target=n_model_target,
            pcl_input_frames=args.video_len - args.future_frames,
            pcl_target_frames=args.past_frames + args.future_frames,
            reference_frame=reference_frame, correct_origin_ground=args.correct_origin_ground,
            sample_bias=args.sample_bias, sb_occl_frame_shift=args.sb_occl_frame_shift,
            min_z=args.min_z, other_bounds=args.pt_cube_bounds, target_bounds=args.cr_cube_bounds,
            cube_mode=args.cube_mode, oversample_vehped_target=args.oversample_vehped_target,
            use_data_frac=args.use_data_frac, verbose='dbg' in args.name)

        train_dataset = CARLADataset(
            args.data_path, logger, stage='train', **dset_args)
        val_aug_dataset = CARLADataset(
            args.data_path, logger, stage='val', **dset_args)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=True, worker_init_fn=seed_worker, drop_last=True, pin_memory=False)
        val_aug_loader = torch.utils.data.DataLoader(
            val_aug_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=True, worker_init_fn=seed_worker, drop_last=True, pin_memory=False)

    else:

        data_kind = 'greater'

        assert args.sample_bias in ['none', 'occl']  # Move is not supported.

        # Dataset internal sizes / model input & output.
        # First step is just random subsampling, but sometimes the final step also is.
        # NOTE: Should ensure probability of fewer than n_data_rnd points in any frame is low.
        # NOTE: Could drop as low as ~1000 if we ignore floor and there are many occlusions!
        n_data_rnd = args.n_data_rnd
        n_model_input = args.n_points
        n_model_output = args.n_points
        # Negative means random instead of farthest_point.
        n_model_target = -int(max(abs(args.n_points), abs(args.n_data_rnd)) * 2)

        logger.info(f'n_model_input: {n_model_input}  n_model_output: {n_model_output}  '
                    f'n_model_target: {n_model_target}  n_data_rnd: {n_data_rnd}')

        dset_args = dict(
            video_length=args.video_len, frame_skip=args.frame_skip, convert_to_pcl=True,
            n_points_rnd=n_data_rnd, n_fps_input=n_model_input, n_fps_target=n_model_target,
            pcl_input_frames=args.video_len - args.future_frames,
            pcl_target_frames=args.past_frames + args.future_frames,
            sample_bias=args.sample_bias, sb_occl_frame_shift=args.sb_occl_frame_shift,
            min_z=args.min_z, other_bounds=args.pt_cube_bounds, return_segm=True,
            track_mode='random' if args.tracking_lw > 0.0 else 'none',
            use_data_frac=args.use_data_frac,
            verbose='dbg' in args.name)

        train_dataset = GREATERDataset(
            args.data_path, logger, stage='train', **dset_args)
        val_aug_dataset = GREATERDataset(
            args.data_path, logger, stage='val', **dset_args)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=True, worker_init_fn=seed_worker, drop_last=True, pin_memory=False)
        val_aug_loader = torch.utils.data.DataLoader(
            val_aug_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=True, worker_init_fn=seed_worker, drop_last=True, pin_memory=False)

    return (data_kind, train_loader, val_aug_loader, dset_args)


def create_test_data_loader(args, dset_args, logger):
    
    dset_args['ss_frame_step'] = args.ss_frame_step
    dset_args['n_fps_target'] = 0
    dset_args['use_data_frac'] = args.use_data_frac
    dset_args['sample_bias'] = args.sample_bias
    dset_args['sb_occl_frame_shift'] = args.sb_occl_frame_shift
    dset_args['verbose'] = ('dbg' in args.name)
    dset_args['use_json'] = args.use_json

    if 'carla' in args.data_path.lower():

        data_kind = 'carla'

        dset_args['oversample_vehped_target'] = False

        test_dataset = CARLADataset(
            args.data_path, logger, stage='test', **dset_args)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False,
            worker_init_fn=seed_worker)

    else:

        data_kind = 'greater'

        assert args.sample_bias in ['none', 'occl']  # Move is not supported.

        dset_args['force_view_idx'] = args.force_view_idx

        if args.track_mode in ['none', 'all']:
            dset_args['track_mode'] = 'none'  # If all, inference.py takes care of it.
        elif args.track_mode == 'one':
            pass  # Leave default (none / snitch / random).
        else:
            raise ValueError(args.track_mode)

        test_dataset = GREATERDataset(
            args.data_path, logger, stage='test', **dset_args)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False,
            worker_init_fn=seed_worker)

    return (data_kind, test_loader)

