'''
Manages training & validation.
Created by Basile Van Hoorick for Revealing Occlusions with 4D Neural Fields.
'''

from __init__ import *

# Library imports.
import traceback
import torch.cuda.amp
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# Internal imports.
import args
import data
import geometry
import implicit
import logvis
import loss
import model
import pipeline
import utils


def get_learn_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_one_epoch(args, train_pipeline, networks, all_parameters, stage, epoch, optimizer,
                    lr_scheduler, scaler, data_loader, device, logger):
    # NOTE: all_parameters is (should be) exactly the same as train_pipeline.parameters(),
    # because the latter is also a torch.nn.Module instance containing all networks.

    assert stage in ['train', 'val', 'val_aug']
    logger.info(f'Epoch (1-based): {epoch + 1} / {args.num_epochs}')
    num_steps_per_epoch = len(data_loader)
    total_step_base = num_steps_per_epoch * epoch  # This has already happened so far.
    (train_pipeline, train_pipeline_nodp) = train_pipeline

    if stage == 'train':
        train_pipeline.train()
        for net in networks:
            if net is not None:
                net.train()
        torch.set_grad_enabled(True)
        logger.info(f'===> Train ({stage})')
        logger.report_scalar(stage + '/learn_rate', get_learn_rate(optimizer), step=epoch)

    else:
        train_pipeline.eval()
        for net in networks:
            if net is not None:
                net.eval()
        torch.set_grad_enabled(False)
        logger.info(f'===> Validation ({stage})')

    train_pipeline_nodp.set_stage(stage)

    start_time = time.time()
    num_exceptions = 0

    for cur_step, batch in enumerate(tqdm.tqdm(data_loader)):

        if cur_step == 0:
            logger.info(f'Enter first data loader iteration took {time.time() - start_time:.3f}s')

        total_step = cur_step + total_step_base  # For continuity in wandb.

        if stage == 'train':
            optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.mixed_precision):

            # First, address every example independently.
            # This part has zero interaction between any pair of GPUs.
            try:
                remnant = train_pipeline(batch, cur_step)

            except Exception as e:
                num_exceptions += 1
                if num_exceptions >= 12:
                    raise e
                else:
                    logger.exception(e)
                    continue

            # Second, process accumulated information.
            # This part typically happens on the first GPU, so it should be kept minimal in memory.
            (total_loss, log_info) = train_pipeline_nodp.process_entire_batch(
                cur_step, total_step, *remnant)

        # Perform backpropagation to update model parameters.
        if stage == 'train':
            scaler.scale(total_loss).backward()

            # Apply gradient clipping if desired, but this requires unscaled gradients.
            # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
            if args.gradient_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_parameters, args.gradient_clip)

            # For debugging:
            if torch.any(torch.stack([p.grad.isnan().any() for p in all_parameters])):
                logger.error('NaN gradient detected!')
                # logger.error('Manually skipping optimizer step...')
                # continue

            scaler.step(optimizer)
            scaler.update()

            # For debugging:
            if torch.any(torch.stack([p.isnan().any() for p in all_parameters])):
                raise RuntimeError('NaN model parameter detected!')

        # Print and visualize stuff.
        logger.handle_step(
            epoch, stage, cur_step, total_step, num_steps_per_epoch, *log_info)

        del batch
        del remnant
        del total_loss

    if stage == 'train':
        lr_scheduler.step()


def train_all_epochs(args, train_pipeline, networks, all_parameters, optimizer,
                     lr_scheduler, scaler, train_loader, val_aug_loader, device,
                     logger, checkpoint_fn):
    logger.info('Start training loop...')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.num_epochs):

        # Training.
        train_one_epoch(
            args, train_pipeline, networks, all_parameters, 'train', epoch, optimizer,
            lr_scheduler, scaler, train_loader, device, logger)

        # Save model weights.
        checkpoint_fn(epoch)

        # Validation.
        train_one_epoch(
            args, train_pipeline, networks, all_parameters, 'val_aug', epoch, optimizer,
            lr_scheduler, scaler, val_aug_loader, device, logger)

        logger.epoch_finished(epoch)

    total_time = time.time() - start_time
    logger.info(f'Total time: {total_time / 3600.0:.3f} hours')


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
    device = torch.device(args.device)

    logger.info('Output path: ' + args.output_path)
    os.makedirs(args.output_path, exist_ok=True)

    # Instantiate datasets.
    logger.info('Initializing data loaders...')
    start_time = time.time()
    (data_kind, train_loader, val_aug_loader, dset_args) = \
        data.create_train_val_data_loaders(args, logger)
    logger.info(f'Took {time.time() - start_time:.3f}s')

    logger.info('Initializing model...')
    start_time = time.time()

    # Instantiate networks.
    assert args.use_global_embedding
    assert args.local_implicit_mode in ['none', 'feature', 'attention']

    # Point transformer (PT).
    if data_kind == 'greater':
        d_in = 8  # (x, y, z, R, G, B, t, mark_track).
        d_out = 1  # Obsolete.

    elif data_kind == 'carla':
        d_in = 8  # (x, y, z, R, G, B, t, mark_track).
        d_out = 1  # Obsolete.

    else:
        raise ValueError()
        
    # if args.tracking_lw > 0.0:
    #     d_in += 1  # (mark_track).

    n_model_input = args.n_points
    n_model_output = args.n_points
    down_blocks = args.up_down_blocks
    up_blocks = args.up_down_blocks
    output_featurized = (args.local_implicit_mode != 'none')
    global_dim = args.global_size

    pcl_args = dict(
        mixed_precision=args.mixed_precision,
        n_input=n_model_input, n_output=n_model_output, d_in=d_in, d_out=d_out,
        d_feat=args.pt_feat_dim, down_blocks=down_blocks, up_blocks=up_blocks,
        transition_factor=args.transition_factor,
        pt_num_neighbors=args.pt_num_neighbors, pt_norm_type=args.pt_norm_type,
        down_neighbors=args.down_neighbors, abstract_levels=args.abstract_levels,
        skip_connections=False, enable_decoder=False, output_featurized=output_featurized,
        output_global_emb=True, global_dim=global_dim, fps_random_start=True)
    pcl_net = model.PointCompletionNetV3(**pcl_args)

    # Continuous representation (CR).
    d_out = 1  # Density (sigma) is always present.
    predict_tracking = args.tracking_lw > 0.0
    predict_segmentation = args.segmentation_lw > 0.0
    if args.color_mode in ['rgb', 'rgb_nosigmoid']:
        d_out += 3  # Adds (R, G, B).
    elif args.color_mode == 'hsv':
        d_out += 14  # Adds (H0, ..., H11, S, V).
    elif args.color_mode == 'bins':
        d_out += 9  # Adds (B0, ..., B8) with 6 colors + black / gray / white.
    else:
        raise ValueError()
    
    d_out += 1  # Adds track mark (m), always present.
    
    if predict_segmentation:
        d_out += args.semantic_classes
    pos_encoding_freqs = 8 if args.positional_encoding else 0
    activation = args.activation
    local_mode = args.local_implicit_mode
    if local_mode == 'none':
        num_local_features = 0
        d_latent_local = 0
        d_hidden = args.global_size
        d_latent = args.global_size
    else:
        num_local_features = args.num_cr_local_feats
        d_latent_local = int(args.pt_feat_dim * (2 ** down_blocks))
        d_hidden = args.global_size + d_latent_local
        d_latent = args.global_size + d_latent_local
    implicit_args = dict(
        mixed_precision=args.mixed_precision,
        d_in=4, d_hidden=d_hidden, d_out=d_out, d_latent=d_latent,
        n_blocks=args.implicit_mlp_blocks, pos_encoding_freqs=pos_encoding_freqs,
        activation=activation, num_local_features=num_local_features,
        local_mode=local_mode, d_latent_local=d_latent_local,
        cross_attn_neighbors=args.cross_attn_neighbors,
        cross_attn_layers=args.cross_attn_layers, cr_attn_type=args.cr_attn_type)
    implicit_net = implicit.LocalPclResnetFC(**implicit_args)

    networks = [pcl_net, implicit_net]

    # Smart point sampler for train-time loss weighting in CARLA.
    num_solid = args.num_cr_solid  # >= bc41 / bg11.
    sampler_args = dict(
        min_z=args.min_z, cube_bounds=args.cr_cube_bounds,
        point_occupancy_radius=args.point_occupancy_radius, num_solid=num_solid,
        num_air=int(num_solid * args.air_sampling_ratio),
        predict_segmentation=predict_segmentation,
        semantic_classes=args.semantic_classes,
        predict_tracking=predict_tracking, data_kind=data_kind,
        point_sample_bias=args.point_sample_bias, cube_mode=args.cube_mode)
    point_sampler = geometry.GuidedImplicitPointSampler(logger, **sampler_args)

    # Configure device logistics in case of inactive parallel pipeline.
    for i in range(len(networks)):
        networks[i] = networks[i].to(device)
    networks_nodp = [net for net in networks]
    if not args.parallel_pipeline:
        point_sampler = point_sampler.to(device)
        if args.device == 'cuda':
            for i in range(len(networks)):
                networks[i] = torch.nn.DataParallel(networks[i])
            point_sampler = torch.nn.DataParallel(point_sampler)
            networks_nodp = [net.module for net in networks]

    # Instantiate encompassing pipeline for more efficient parallelization.
    train_pipeline = pipeline.MyTrainPipeline(
        networks, point_sampler, device, 'if', logger, args.mixed_precision,
        args.color_lw, args.density_lw, args.segmentation_lw, args.tracking_lw, args.color_mode,
        args.semantic_classes, args.past_frames, args.future_frames, data_kind)
    train_pipeline_nodp = train_pipeline

    # Configure device logistics in case of active parallel pipeline.
    train_pipeline = train_pipeline.to(device)
    train_pipeline_nodp = train_pipeline
    if args.parallel_pipeline and args.device == 'cuda':
        # NOTE: The variables pcl_net and implicit_net remain unwrapped.
        train_pipeline = torch.nn.DataParallel(train_pipeline)
        train_pipeline_nodp = train_pipeline.module

    # Instantiate optimizer & learning rate scheduler.
    all_parameters = []
    for net in networks:
        all_parameters += net.parameters()
    # optimizer = torch.optim.Adam(all_parameters, lr=args.learn_rate)
    optimizer = torch.optim.AdamW(all_parameters, lr=args.learn_rate, weight_decay=1e-2,
                                  eps=1e-4 if args.mixed_precision else 1e-8)
    milestones = [(args.num_epochs * 2) // 5,
                  (args.num_epochs * 3) // 5,
                  (args.num_epochs * 4) // 5]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones, gamma=args.lr_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    # Load weights from checkpoint if specified.
    if args.resume:
        logger.info('Loading weights from: ' + args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        networks_nodp[0].load_state_dict(checkpoint['pcl_net'])
        networks_nodp[1].load_state_dict(checkpoint['implicit_net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        args.start_epoch = checkpoint['epoch'] + 1

    logger.info(f'Took {time.time() - start_time:.3f}s')

    # Define logic for how to store checkpoints.
    def save_model_checkpoint(epoch):
        if args.output_path:
            logger.info(f'Saving model checkpoint to {args.output_path}...')
            checkpoint = {
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'args': args,
                'pcl_args': pcl_args,
                'dset_args': dset_args,
                'implicit_args': implicit_args,
            }
            checkpoint['pcl_net'] = networks_nodp[0].state_dict()
            checkpoint['implicit_net'] = networks_nodp[1].state_dict()
            torch.save(
                checkpoint,
                os.path.join(args.output_path, 'model_{}.pth'.format(epoch)))
            torch.save(
                checkpoint,
                os.path.join(args.output_path, 'checkpoint.pth'))
            logger.info()

    if 1:
        logger.init_wandb(PROJECT_NAME, args, networks, name=args.name + '-if')

    train_all_epochs(
        args, [train_pipeline, train_pipeline_nodp], networks, all_parameters, optimizer,
        lr_scheduler, scaler, train_loader, val_aug_loader, device, logger,
        save_model_checkpoint)


if __name__ == '__main__':

    # For debugging. This makes things slow, but we can detect NaNs etc. this way:
    # torch.autograd.set_detect_anomaly(True)

    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.empty_cache()

    args = args.train_args()

    logger = logvis.MyLogger(args, context='train')

    try:

        main(args, logger)

    except Exception as e:

        logger.exception(e)
        # tb = traceback.format_exc()
        # logger.error(tb)

        logger.warning('Shutting down due to exception...')
