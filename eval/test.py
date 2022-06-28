'''
Evaluation logic.
Created by Basile Van Hoorick and Purva Tendulkar for Revealing Occlusions with 4D Neural Fields.
'''

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'eval/'))

from __init__ import *

# Library imports.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# Internal imports.
import args
import data
import inference
import logvis


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def test(args, networks, epoch, data_kind, data_loader, device, logger):
    for net in networks:
        net.eval()
    torch.set_grad_enabled(False)

    num_steps = len(data_loader)
    log_folder = 'test_' + args.test_tag

    start_time = time.time()

    for cur_step, batch in enumerate(tqdm.tqdm(data_loader)):

        if cur_step == 0:
            logger.info(f'Enter first data loader iteration took {time.time() - start_time:.3f}s')

        cam_RT = batch['cam_RT']
        cam_K = batch['cam_K']
        meta_data = batch['meta_data']
        pcl_target_size = meta_data['pcl_target_size']

        pcl_input = batch['pcl_input']
        # (N, 8) with (x, y, z, R, G, B, t, mark_track).
        pcl_input_sem = batch['pcl_input_sem']
        # (N, 1-3) with (cosine_angle?, instance_id, semantic_tag?).
        pcl_target = batch['pcl_target']
        # List of (M, 9-11) with (x, y, z, cosine_angle?, instance_id, semantic_tag?, view_idx, R, G, B, mark_track).

        pcl_input_numpy = pcl_input[0].detach().cpu().numpy()
        pcl_input = pcl_input.to(device)
        pcl_input_sem_numpy = pcl_input_sem[0].detach().cpu().numpy()
        pcl_input_sem_inference = pcl_input_sem_numpy if args.track_mode != 'none' else None

        # Get predictions per frame.
        num_frames = len(pcl_target)
        pcl_all = []

        for time_idx in range(num_frames):
            pcl_target_frame = pcl_target[time_idx]
            pcl_target_frame = pcl_target_frame.to(device)
            pcl_target_frame = pcl_target_frame[0].detach().cpu().numpy()
            pcl_target_frame_size = pcl_target_size[time_idx].item()
            pcl_target_frame = pcl_target_frame[:pcl_target_frame_size]
            pcl_target_frame_inference = pcl_target_frame if args.save_gt else None

            inf_res = inference.perform_inference(
                pcl_input, pcl_input_sem_inference, pcl_target_frame_inference, networks, device,
                'if', args.min_z, args.cr_cube_bounds,
                args.color_mode, time_idx, logger, sample_implicit=args.sample_implicit,
                num_sample=args.num_sample, point_sample_mode=args.point_sample_mode,
                batch_size=args.implicit_batch_size,
                predict_segmentation=args.segmentation_lw > 0.0,
                track_mode=args.track_mode,
                point_occupancy_radius=args.point_occupancy_radius,
                semantic_classes=args.semantic_classes,
                density_threshold=args.density_threshold, data_kind=data_kind,
                cube_mode=args.cube_mode, compress_air=True)

            (pcl_output_frame, air_output_frame, pcl_abstract,
             features_global, implicit_output_frame) = \
                (inf_res['output_solid'], inf_res['output_air'], inf_res['pcl_abstract'],
                 inf_res['features_global'], inf_res['implicit_output'])
            if args.save_gt:
                points_query = inf_res['points_query']

            # pcl_output_frame is:
            # (S, 9+) with (x, y, z, t, density, R, G, B, mark_track, segm?).
            # air_output_frame is:
            # (A, 5) with (x, y, z, density, pred_segm).

            # All arrays should be numpy (instead of tensor) at this point.

            # Plot some data distributions for spotting anomalies.
            if cur_step % 4 == 0:
                # First, summarize raw implicit output (this contains both solid and air).
                # NOTE: implicit_output_frame is usually huge, but is abandoned after this.
                logger.report_implicit_histograms(
                    'test', implicit_output_frame, args.color_mode, time_idx,
                    args.segmentation_lw > 0.0, args.semantic_classes,
                    args.tracking_lw > 0.0, cur_step)

            # Save example input-output point cloud pair for later visualization.
            if args.save_gt:
                pcl_all.append((pcl_input_numpy, pcl_abstract, pcl_output_frame,
                                pcl_target_frame, air_output_frame,
                                pcl_input_sem_numpy, points_query))
            else:
                pcl_all.append((pcl_input_numpy, pcl_abstract, pcl_output_frame,
                                pcl_target_frame, air_output_frame))

        if args.store_pcl:
            # pcl_all is a list of (input, abstract, output_solid, target, output_air)
            # tuples of numpy arrays.
            pcl_dst_fn = f'pcl_io_s{cur_step}.p'
            logger.save_pickle(pcl_all, pcl_dst_fn, folder=log_folder)

        # Save some useful info.
        logger.report_scalar('test/pcl_input_size', pcl_input_numpy.shape[0], step=cur_step)
        logger.report_scalar('test/pcl_output_size', pcl_output_frame.shape[0], step=cur_step)
        logger.report_scalar('test/pcl_target_size', pcl_target_frame.shape[0], step=cur_step)
        logger.report_scalar('test/air_output_size', air_output_frame.shape[0], step=cur_step)
        logger.report_histogram(f'test/features_global', features_global, step=cur_step)

        # All tensors in meta_data, cam_RT, and cam_K are on CPU.
        md_dst_fn = f'metadata_s{cur_step}.p'
        logger.save_pickle((meta_data, cam_RT, cam_K), md_dst_fn, folder=log_folder)


def main(args, logger):

    logger.info()
    logger.info('Args: ' + str(args))
    logger.info('torch version: ' + str(torch.__version__))
    logger.info('torchvision version: ' + str(torchvision.__version__))
    logger.save_args(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    logger.info('Initializing model...')
    start_time = time.time()

    # Instantiate networks and load weights.
    if args.device == 'cuda':
        device = torch.device('cuda:' + str(args.gpu_id))
    else:
        device = torch.device(args.device)
    (networks, train_args, dset_args, pcl_args, implicit_args, epoch) = \
        inference.load_models(args.resume, device, epoch=args.epoch, logger=logger)
    args.test_tag += f'_e{epoch}'

    # Copy & correct arguments for later use.
    # NOTE: Certain parameters, such as sample_bias, have to stay up to date with the current
    # command since they are meant to be potentially different across train vs test.
    args.min_z = train_args.min_z
    if 'cube_bounds' in train_args:
        args.pt_cube_bounds = train_args.cube_bounds
        args.cr_cube_bounds = train_args.cube_bounds
    else:
        args.pt_cube_bounds = train_args.pt_cube_bounds
        args.cr_cube_bounds = train_args.cr_cube_bounds
    if 'cube_mode' in train_args:
        args.cube_mode = train_args.cube_mode
    else:
        args.cube_mode = 4
    if 'color_mode' in train_args:
        args.color_mode = train_args.color_mode
    else:
        args.color_mode = 'rgb'
    args.segmentation_lw = train_args.segmentation_lw
    if 'tracking_lw' in train_args:
        args.tracking_lw = train_args.tracking_lw
    else:
        args.tracking_lw = 0.0
    args.point_occupancy_radius = train_args.point_occupancy_radius
    if 'semantic_classes' in train_args:
        args.semantic_classes = train_args.semantic_classes
    else:
        args.semantic_classes = 13

    # Verify option consistency with respect to tracking.
    # if args.tracking_lw == 0.0:
    #     assert args.track_mode == 'none'

    logger.info(f'Took {time.time() - start_time:.3f}s')
    logger.info('Initializing data loader...')
    start_time = time.time()

    # Instantiate dataset.
    (data_kind, test_loader) = data.create_test_data_loader(args, dset_args, logger)

    logger.info(f'Took {time.time() - start_time:.3f}s')

    # if 'dbg' not in args.name:
    if 1:
        logger.init_wandb(PROJECT_NAME + '_test', args, networks, name=args.test_tag)

    # Print test arguments.
    logger.info('Final test command args: ' + str(args))
    logger.info('Final test dataset args: ' + str(dset_args))

    # Run actual test loop.
    test(args, networks, epoch, data_kind, test_loader, device, logger)


if __name__ == '__main__':

    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.empty_cache()

    args = args.test_args()

    logger = logvis.MyLogger(args, context='test_' + args.test_tag)

    try:

        main(args, logger)

    except Exception as e:

        logger.exception(e)

        logger.warning('Shutting down due to exception...')
