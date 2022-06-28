'''
Logging and visualization logic.
Created by Basile Van Hoorick for Revealing Occlusions with 4D Neural Fields.
'''

from __init__ import *

# Library imports.
import wandb
from collections import defaultdict
from collections.abc import Iterable

# Internal imports.
import data
import utils


class Logger:
    '''
    Provides generic logging and visualization functionality.
    '''

    def __init__(self, log_dir, context):
        '''
        :param log_dir (str): Path to logging folder for this run.
        :param context (str): Name of this particular logger instance, for example train / test.
        '''
        self.log_dir = log_dir
        self.context = context
        self.log_path = os.path.join(self.log_dir, context + '.log')
        self.vis_dir = os.path.join(self.log_dir, 'visuals')
        self.npy_dir = os.path.join(self.log_dir, 'numpy')
        self.pkl_dir = os.path.join(self.log_dir, 'pickle')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.npy_dir, exist_ok=True)
        os.makedirs(self.pkl_dir, exist_ok=True)

        # Instantiate logger.
        # logger = logging.getLogger(context)
        # fh = logging.FileHandler(self.log_path)
        # fh.setLevel(logging.INFO)
        # logger.addHandler(fh)
        # self.logger = logger

        logging.root.handlers = []
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler()
            ]
        )

        # Overwrite the default print method reference.
        # print = self.logger.info
        self.scalar_memory = defaultdict(list)
        self.scalar_memory_hist = dict()
        self.initialized = False

    def save_args(self, args):
        args_path = os.path.join(self.log_dir, self.context + '_args.txt')
        with open(args_path, 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    def init_wandb(self, project, args, models, group='debug', name=None):
        if name is None:
            name = args.name
        wandb.init(project=project, group=group, config=args, name=name)
        if not isinstance(models, Iterable):
            models = [models]
        for model in models:
            if model is not None:
                wandb.watch(model)
        self.initialized = True

        # Redo config because some package must have overwritten us??
        logging.root.handlers = []
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler()
            ]
        )

    def debug(self, *args):
        if args == ():
            args = ['']
        logging.debug(*args)

    def info(self, *args):
        if args == ():
            args = ['']
        logging.info(*args)

    def warning(self, *args):
        if args == ():
            args = ['']
        logging.warning(*args)

    def error(self, *args):
        if args == ():
            args = ['']
        logging.error(*args)

    def critical(self, *args):
        if args == ():
            args = ['']
        logging.critical(*args)

    def exception(self, *args):
        if args == ():
            args = ['']
        logging.exception(*args)

    def report_scalar(self, key, value, step=None, remember=False, commit_histogram=False):
        '''
        X
        '''
        if not remember:
            if self.initialized:
                wandb.log({key: value}, step=step)
            else:
                self.debug(str(key) + ': ' + str(value))
        else:
            self.scalar_memory[key].append(value)
            self.scalar_memory_hist[key] = commit_histogram

    def commit_scalars(self, keys=None, step=None):
        '''
        X
        '''
        if keys is None:
            keys = list(self.scalar_memory.keys())
        for key in keys:
            if len(self.scalar_memory[key]) == 0:
                continue

            value = np.mean(self.scalar_memory[key])
            if self.initialized:
                if self.scalar_memory_hist[key]:
                    wandb.log({key: wandb.Histogram(np.array(self.scalar_memory[key]))}, step=step)
                else:
                    wandb.log({key: value}, step=step)

            else:
                self.debug(str(key) + ': ' + str(value))
            self.scalar_memory[key].clear()

    def report_histogram(self, key, value, step=None):
        '''
        X
        '''
        if self.initialized:
            wandb.log({key: wandb.Histogram(value)}, step=step)

    def save_image(self, image, step=None, file_name=None, online_name=None):
        '''
        X
        '''
        if image.dtype == np.float32:
            image = (image * 255.0).astype(np.uint8)
        if file_name is not None:
            plt.imsave(os.path.join(self.vis_dir, file_name), image)
        if online_name is not None and self.initialized:
            wandb.log({online_name: wandb.Image(image)}, step=step)

    def save_video(self, frames, step=None, file_name=None, online_name=None, fps=6):
        '''
        X
        '''
        # Duplicate last frame for better visibility.
        last_frame = frames[len(frames) - 1:len(frames)]
        frames = np.concatenate([frames, last_frame], axis=0)
        if frames.dtype == np.float32:
            frames = (frames * 255.0).astype(np.uint8)
        if file_name is not None:
            file_path = os.path.join(self.vis_dir, file_name)
            imageio.mimwrite(file_path, frames, fps=fps)
        if online_name is not None and self.initialized:
            # This seems to be bugged in wandb?
            # wandb.log({online_name: wandb.Video(frames, fps=fps, format='gif')}, step=step)
            assert file_name is not None
            wandb.log({online_name: wandb.Video(file_path, fps=fps, format='gif')}, step=step)

    def save_gallery(self, frames, step=None, file_name=None, online_name=None):
        '''
        X
        '''
        if frames.shape[-1] > 3:  # Grayscale: (..., H, W).
            arrangement = frames.shape[:-2]
        else:  # RGB: (..., H, W, 1/3).
            arrangement = frames.shape[:-3]
        if len(arrangement) == 1:  # (A, H, W, 1/3?).
            gallery = np.concatenate(frames, axis=1)  # (H, A*W, 1/3?).
        elif len(arrangement) == 2:  # (A, B, H, W, 1/3?).
            gallery = np.concatenate(frames, axis=1)  # (B, A*H, W, 1/3?).
            gallery = np.concatenate(gallery, axis=1)  # (A*H, B*W, 1/3?).
        else:
            raise ValueError('Too many dimensions to create a gallery.')
        if gallery.dtype == np.float32:
            gallery = (gallery * 255.0).astype(np.uint8)
        if file_name is not None:
            plt.imsave(os.path.join(self.vis_dir, file_name), gallery)
        if online_name is not None and self.initialized:
            wandb.log({online_name: wandb.Image(gallery)}, step=step)

    def save_numpy(self, array, file_name, step=None, folder=None):
        '''
        X
        '''
        if folder is None:
            dst_dp = self.npy_dir
        else:
            dst_dp = os.path.join(self.log_dir, folder)
            os.makedirs(dst_dp, exist_ok=True)
        np.save(os.path.join(dst_dp, file_name), array)

    def save_pickle(self, obj, file_name, step=None, folder=None):
        '''
        X
        '''
        if folder is None:
            dst_dp = self.pkl_dir
        else:
            dst_dp = os.path.join(self.log_dir, folder)
            os.makedirs(dst_dp, exist_ok=True)
        dst_fp = os.path.join(dst_dp, file_name)
        # Regular:
        with open(dst_fp, 'wb') as f:
            pickle.dump(obj, f)
        # Compressed (SLIGHTLY smaller but MUCH slower):
        # with bz2.BZ2File(dst_fp, 'wb') as f:
        #     cPickle.dump(obj, f)


class MyLogger(Logger):
    '''
    Adapts the generic logger to this specific point transformer framework.
    '''

    def __init__(self, args, context='train'):
        if 'color_mode' in args:
            self.color_mode = args.color_mode
            self.semantic_classes = args.semantic_classes
        else:
            self.color_mode = 'rgb'
            self.semantic_classes = 13
        if 'batch_size' in args:
            self.step_interval = 160 // args.batch_size
        else:
            self.step_interval = 40
        log_dir = os.path.join(args.log_root, args.tag)
        super().__init__(log_dir, context)

    def handle_step(self, epoch, stage, cur_step, total_step, num_steps_per_epoch,
                    rgb, depth, pcl_input, pcl_abstract, pcl_target,
                    meta_data, cam_RT, cam_K, layer_coords,
                    points_query, implicit_output, features_global,
                    total_loss, loss_rgb, loss_dens, loss_segm, loss_track):
        data_kind = meta_data['data_kind'][0].item()

        if data_kind == 1001:  # GREATER.
            src_view = meta_data['src_view'][0].item()
            has_depth = True
            max_depth_clip = data.GREATERDataset.max_depth_clip()

        elif data_kind == 1002:  # CARLA.
            src_view = 0
            has_depth = False
            max_depth_clip = data.CARLADataset.max_depth_clip()
            cuboid_filter_ratios = torch.stack(meta_data['cuboid_filter_ratios']).cpu().numpy()
            sample_input_ratios = torch.stack(meta_data['sample_input_ratios']).cpu().numpy()
            sample_target_ratios = torch.stack(meta_data['sample_target_ratios']).cpu().numpy()
            # (24, B), (1, B), (1, B).

        else:
            raise RuntimeError()

        T = meta_data['frame_inds'].shape[-1]
        stage_abbrev = {'train': 't', 'val': 'v', 'val_aug': 'va'}
        stage_abbrev = stage_abbrev[stage]

        if cur_step % self.step_interval == 0:
            # Print metrics in console.
            self.info(f'[Step {cur_step} / {num_steps_per_epoch}]  '
                      f'total_loss: {total_loss:.3f}  '
                      f'loss_dens: {loss_dens:.3f}  '
                      + (f'loss_rgb: {loss_rgb:.3f}  ' if loss_rgb != 0.0 else '')
                      + (f'loss_segm: {loss_segm:.3f} ' if loss_segm != 0.0 else '')
                      + (f'loss_track: {loss_track:.3f} ' if loss_track != 0.0 else ''))

            # Save example 2D inputs for debugging.
            # This is only done at the beginning of training to save space.
            if epoch <= 5:
                rgb = rgb[0, src_view].detach().cpu().numpy()  # (T, H, W, 3).
                if has_depth:
                    depth = depth[0, src_view].detach().cpu().numpy()  # (T, H, W).
                    depth /= max_depth_clip
                    depth = np.tile(np.expand_dims(depth, -1), (1, 1, 1, 3))  # (T, H, W, 3).

                if has_depth:
                    frames = np.stack([rgb, depth])  # (2, T, H, W, 3).
                else:
                    frames = np.stack([rgb])  # (1, T, H, W, 3).
                online_name = f'rgbd_gal_v{src_view}'
                file_name = online_name + f'_e{epoch}_p{stage_abbrev}_s{cur_step}.png'
                self.save_gallery(frames, step=epoch,
                                  file_name=file_name, online_name=online_name)

                if 1:
                    (_, H, W, _) = rgb.shape
                    if has_depth:
                        frames = frames.transpose(1, 0, 2, 3, 4).reshape(T, 2 * H, W, 3)
                    else:
                        frames = frames.reshape(T, H, W, 3)
                    online_name = f'rgbd_video_v{src_view}'
                    file_name = online_name + f'_e{epoch}_p{stage_abbrev}_s{cur_step}.mp4'
                    self.save_video(frames, step=epoch,
                                    file_name=file_name, online_name=online_name, fps=4)

            # Save example point clouds to numpy files for later visualization.
            num_frames = len(pcl_target)
            if epoch % 5 == 0 and np.random.rand() < 0.2:
                # Input.
                pcl_input = pcl_input[0].detach().cpu()
                pcl_input = pcl_input.numpy().astype(np.float32)
                npy_dst_fn = f'pcl_input_e{epoch}_p{stage_abbrev}_s{cur_step}.npy'
                self.save_numpy(pcl_input, npy_dst_fn, step=epoch)

                # Abstract.
                if pcl_abstract is not None:
                    pcl_abstract = pcl_abstract[0].detach().cpu()
                    pcl_abstract = pcl_abstract.numpy().astype(np.float32)
                    npy_dst_fn = f'pcl_abstract_e{epoch}_p{stage_abbrev}_s{cur_step}.npy'
                    self.save_numpy(pcl_abstract, npy_dst_fn, step=epoch)

                for time_idx in range(num_frames):

                    # NOTE: Some frames are occasionally invalid due to insufficient target points.
                    implicit_output_frame = implicit_output[time_idx]
                    if implicit_output_frame is not None:
                        implicit_output_frame = implicit_output_frame[0].detach().cpu()
                        implicit_output_frame = implicit_output_frame.numpy().astype(np.float32)
                        npy_dst_fn = f'imp_output_e{epoch}_p{stage_abbrev}_s{cur_step}_t{time_idx}.npy'
                        self.save_numpy(implicit_output_frame, npy_dst_fn, step=epoch)

                    # Target.
                    pcl_target_frame = pcl_target[time_idx]
                    pcl_target_frame = pcl_target_frame[0].detach().cpu()
                    pcl_target_frame = pcl_target_frame.numpy().astype(np.float32)
                    npy_dst_fn = f'pcl_target_e{epoch}_p{stage_abbrev}_s{cur_step}_t{time_idx}.npy'
                    self.save_numpy(pcl_target_frame, npy_dst_fn, step=epoch)

                # Occasionally save evolution of internal coordinates over all layers.
                if layer_coords is not None:
                    layer_coords = [x[0].detach().cpu() for x in layer_coords]
                    pcl_layers = utils.accumulate_pcl_layer_torch(layer_coords)
                    pcl_layers[..., 2] += pcl_layers[..., -1] * 3.0

                    pcl_layers = pcl_layers.numpy()
                    npy_dst_fn = f'pcl_layers_e{epoch}_p{stage_abbrev}_s{cur_step}.npy'
                    self.info(f'Saving layer-wise point clouds to {npy_dst_fn}...')
                    self.save_numpy(pcl_layers, npy_dst_fn, step=epoch)

                # Save metadata for later; all tensors are already on CPU.
                pkl_dst_fn = f'npy_e{epoch}_p{stage_abbrev}_s{cur_step}.p'
                self.save_pickle((meta_data, cam_RT, cam_K), pkl_dst_fn, step=epoch)

            # Plot data distributions to spot anomalies and visualize signal evolution over time.
            if 'val' in stage and epoch % 2 == 0 and np.random.rand() < 0.5:
                features_global = features_global[0].detach().cpu()
                features_global = features_global.type(torch.float32)  # Undo autocast.

                for time_idx in range(num_frames):

                    # NOTE: Some frames are occasionally invalid due to insufficient target points.
                    implicit_output_frame = implicit_output[time_idx]
                    if implicit_output_frame is not None:
                        implicit_output_frame = implicit_output_frame[0].detach().cpu()
                        implicit_output_frame = implicit_output_frame.type(torch.float32)
                        implicit_output_frame = implicit_output_frame.numpy()

                        self.report_implicit_histograms(
                            stage, implicit_output_frame, self.color_mode, time_idx,
                            loss_segm != 0.0, self.semantic_classes, loss_track != 0.0, epoch)

                        pcl_output_frame = implicit_output_frame[implicit_output_frame[..., 0] >= 0.0]
                        air_output_frame = implicit_output_frame[implicit_output_frame[..., 0] < 0.0]

                        self.report_pcl_air_histograms(
                            stage, pcl_output_frame, air_output_frame, self.color_mode, time_idx,
                            loss_segm != 0.0, self.semantic_classes, loss_track != 0.0, False, epoch)

                    self.report_histogram(stage + f'/features_global', features_global, step=epoch)

            if data_kind == 1002:
                # Plot dataset point filtering ratios.
                # Ignore outliers to avoid low resolution.
                for ratio in cuboid_filter_ratios.flatten():
                    if ratio <= 10.0:
                        self.report_scalar(stage + '/cuboid_filter_ratio',
                                            ratio, step=epoch,
                                            remember=True, commit_histogram=True)
                for ratio in sample_input_ratios.flatten():
                    if ratio <= 10.0:
                        self.report_scalar(stage + '/sample_input_ratio',
                                            ratio, step=epoch,
                                            remember=True, commit_histogram=True)
                for ratio in sample_target_ratios.flatten():
                    if ratio <= 10.0:
                        self.report_scalar(stage + '/sample_target_ratio',
                                            ratio, step=epoch,
                                            remember=True, commit_histogram=True)

    def report_implicit_histograms(
            self, stage, implicit_output, color_mode, time_idx,
            predict_segmentation, semantic_classes, predict_tracking, step):
        '''
        :param implicit_output (N, 5+) tensor with
            (density, R, G, B, mark_track, segm?).
        '''
        track_idx = utils.get_track_idx(color_mode)

        self.report_histogram(
            f'{stage}/imp_dens', implicit_output[..., 0], step=step)

        if color_mode in ['rgb', 'rgb_nosigmoid']:
            self.report_histogram(
                f'{stage}/imp_red', implicit_output[..., 1], step=step)
            self.report_histogram(
                f'{stage}/imp_green', implicit_output[..., 2], step=step)
            self.report_histogram(
                f'{stage}/imp_blue', implicit_output[..., 3], step=step)

        elif color_mode == 'hsv':
            num_classes = 12
            self.report_histogram(
                f'{stage}/imp_clr_hue',
                implicit_output[..., 1:1 + num_classes].argmax(axis=-1), step=step)
            self.report_histogram(
                f'{stage}/imp_clr_sat',
                implicit_output[..., 1 + num_classes], step=step)
            self.report_histogram(
                f'{stage}/imp_clr_val',
                implicit_output[..., 2 + num_classes], step=step)

        elif color_mode == 'bins':
            num_classes = 9
            self.report_histogram(
                f'{stage}/imp_clr_bin',
                implicit_output[..., 1:1 + num_classes].argmax(axis=-1), step=step)

        if predict_tracking:
            self.report_histogram(
                f'{stage}/imp_mark_track',
                implicit_output[..., track_idx], step=step)

        if predict_segmentation:
            self.report_histogram(
                f'{stage}/imp_segm',
                implicit_output[..., -semantic_classes:].argmax(axis=-1), step=step)

    def report_pcl_air_histograms(
            self, stage, pcl_output, air_output, color_mode, time_idx,
            predict_segmentation, semantic_classes, predict_tracking, has_xyzt, step):
        '''
        :param pcl_output (S, 5+) numpy array with
            (x?, y?, z?, t?, density, R, G, B, mark_track, segm?).
        :param air_output (A, 5+) numpy array with
            (x?, y?, z?, t?, density, R, G, B, mark_track, segm?)
            or (A, 1-4) numpy array with (x?, y?, z?, density).
        :param has_xyzt (bool): Whether or not both arrays contain location and time data, or just
            the output features.
        '''
        track_idx = utils.get_track_idx(color_mode)

        if has_xyzt:
            self.report_histogram(
                f'{stage}/pcl_xyz', pcl_output[..., :3], step=step)
            # Remove (x, y, z, t) for the remainder of this method.
            pcl_output = pcl_output[..., 4:]
            # pcl_output = (density, R, G, B, mark_track, segm?).

            if air_output is not None:
                self.report_histogram(
                    f'{stage}/air_xyz', air_output[..., :3], step=step)
                # Remove (x, y, z) for the remainder of this method.
                air_output = air_output[..., 3:]
                # air_output = (density, R, G, B, mark_track, segm?)
                # or (density).

        self.report_histogram(
            f'{stage}/pcl_dens', pcl_output[..., 0], step=step)
        if air_output is not None:
            self.report_histogram(
                f'{stage}/air_dens', air_output[..., 0], step=step)

        if color_mode in ['rgb', 'rgb_nosigmoid']:
            self.report_histogram(
                f'{stage}/pcl_red', pcl_output[..., 1], step=step)
            self.report_histogram(
                f'{stage}/pcl_green', pcl_output[..., 2], step=step)
            self.report_histogram(
                f'{stage}/pcl_blue', pcl_output[..., 3], step=step)

        elif color_mode == 'hsv':
            num_classes = 12
            self.report_histogram(
                f'{stage}/pcl_clr_hue',
                pcl_output[..., 1:1 + num_classes].argmax(axis=-1), step=step)
            self.report_histogram(
                f'{stage}/pcl_clr_sat',
                pcl_output[..., 1 + num_classes], step=step)
            self.report_histogram(
                f'{stage}/pcl_clr_val',
                pcl_output[..., 2 + num_classes], step=step)

        elif color_mode == 'bins':
            num_classes = 9
            self.report_histogram(
                f'{stage}/pcl_clr_bin',
                pcl_output[..., 1:1 + num_classes].argmax(axis=-1), step=step)

        if predict_tracking:
            self.report_histogram(
                f'{stage}/pcl_mark_track', pcl_output[..., track_idx], step=step)

        if predict_segmentation:
            self.report_histogram(
                f'{stage}/pcl_segm',
                pcl_output[..., -semantic_classes:].argmax(axis=-1), step=step)

    def epoch_finished(self, epoch):
        self.commit_scalars(step=epoch)
