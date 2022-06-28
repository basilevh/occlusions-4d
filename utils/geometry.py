'''
3D / 4D related helper methods.
Created by Basile Van Hoorick for Revealing Occlusions with 4D Neural Fields.
'''

from __init__ import *
import torch

# Library imports.
from multiprocessing import Value
import sklearn.neighbors
import torch.nn.functional as F
try:
    import torch_cluster
except:
    pass


def point_cloud_from_pixel_coords(x, y, z, cam_RT, cam_K):
    '''
    Converts a set of source pixel coordinates and depth values to 3D world coordinates.
    NOTE: Source coordinates must always be a 1D list or numpy array.

    Args:
        x: List of horizontal integer source positions in [0, width - 1].
        y: List of vertical integer source positions in [0, height - 1].
        z: List of depth values in meters (represents Z axis offset, not Euclidean distance).
        cam_RT: 3x4 camera extrinsics matrix for source view (2D points given).
        cam_K: 3x3 camera intrinsics matrix for source view.

    Returns:
        (N, 3) numpy array consisting of 3D world coordinates.
    '''
    assert len(x) == len(y)
    assert len(x) == len(z)
    N_points = len(x)
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    z = np.array(z, dtype=np.float32)

    # Expand all matrices into 4x4 for consistency.
    cam_RT_4x4 = np.eye(4, dtype=np.float32)
    cam_RT_4x4[:3] = cam_RT
    cam_K_4x4 = np.eye(4, dtype=np.float32)
    cam_K_4x4[:3, :3] = cam_K

    # Get 2D pixels in image space: 4 x N.
    coords_src = np.ones((4, N_points), dtype=np.float32)
    coords_src[0, :] = x
    coords_src[1, :] = y

    # Get 3D points in source camera space: (4 x 4) x (4 x N) = (4 x N).
    points_src = np.matmul(np.linalg.inv(cam_K_4x4), coords_src)

    # Scale 3D points by depth value in all dimensions: 4 x N.
    points_src[:3, :] *= z[np.newaxis, :]

    # Transform 3D points to world space: (4 x 4) x (4 x N) = (4 x N).
    points_world = np.matmul(np.linalg.inv(cam_RT_4x4), points_src)

    # Reshape to N x 3.
    points_world = points_world.transpose()[:, :3]

    return points_world


def pixel_coords_from_point_cloud(pcl, cam_RT, cam_K, flip_xy=False):
    '''
    Converts a set of 3D world coordinates to pixel coordinates and depth values.
    NOTE: Source coordinates must always be a 1D list or numpy array.

    Args:
        pcl: (N, D) numpy array with world coordinates (x, y, z) + features.
        cam_RT: 3x4 camera extrinsics matrix for destination view (3D points given).
        cam_K: 3x3 camera intrinsics matrix for destination view.
        flip_xy: If True, return (y, x, ...) instead of (x, y, ...).

    Returns:
        (N, D) numpy array with pixel coordinates (x, y, z) + features.
    '''
    (N, D) = pcl.shape
    pcl = np.array(pcl, dtype=np.float32)
    points_world = pcl[..., :3]  # (N, 3).

    # Expand all matrices into 4x4 for consistency.
    cam_RT_4x4 = np.eye(4, dtype=np.float32)
    cam_RT_4x4[:3] = cam_RT
    cam_K_4x4 = np.eye(4, dtype=np.float32)
    cam_K_4x4[:3, :3] = cam_K

    # Reshape to 4 x N.
    points_world = np.ones((4, N), dtype=np.float32)
    points_world[:3] = pcl[:, :3].transpose()

    # Transform 3D points to destination camera space: (4 x 4) x (4 x N) = (4 x N).
    points_dst = np.matmul(cam_RT_4x4, points_world)

    # Scale 3D points by depth value back to z=1.
    dst_z = points_dst[2, :].copy()
    points_dst[:2, :] /= dst_z[None, :]
    points_dst[2, :] = 1.0

    # Get 2D pixels in image space: (4 x 4) x (4 x N) = (4 x N).
    coords_dst = np.matmul(cam_K_4x4, points_dst)

    # Append depth and reshape to N x 3.
    coords_dst = coords_dst.transpose()[:, :2]
    if flip_xy:
        coords_dst = np.flip(coords_dst, axis=-1)
    coords_dst = np.concatenate([coords_dst, dst_z[:, None]], axis=-1)

    # Append features.
    pcl_dst = np.concatenate([coords_dst, pcl[..., 3:]], axis=-1)

    return pcl_dst


def point_cloud_from_rgbd(rgb, depth, cam_RT, cam_K):
    '''
    Converts an image with depth information to a colorized point cloud.

    Args:
        rgb: (H, W, 3) numpy array.
        depth: (H, W) numpy array (represents Z axis offset, not Euclidean distance).
        cam_RT: 3x4 camera extrinsics matrix for source view.
        cam_K: 3x3 camera intrinsics matrix for source view.

    Returns:
        (N, 6) numpy array consisting of 3D world coordinates + RGB values.
    '''
    # First, obtain 3D world coordinates.
    H, W = rgb.shape[:2]
    valid_y, valid_x = np.where(depth > 0.0)  # (N, 3) int each.
    # (H, W) int each.
    all_y, all_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    y = all_y[valid_y, valid_x]  # (N) int.
    x = all_x[valid_y, valid_x]  # (N) int.
    z = depth[valid_y, valid_x]  # (N) float32.
    # (N, 3) float32.
    points = point_cloud_from_pixel_coords(x, y, z, cam_RT, cam_K)

    # Then, attach attributes.
    colors = rgb[valid_y, valid_x]  # (N, 3) float32.
    pcl = np.concatenate((points, colors), axis=1)  # (N, 6) float32.

    return pcl


def filter_pcl_bounds_numpy(pcl, x_min=-10.0, x_max=10.0, y_min=-10.0, y_max=10.0,
                            z_min=-10.0, z_max=10.0, greater_floor_fix=False):
    '''
    Restricts a point cloud to exclude coordinates outside a certain cube.
    This method is tailored to the GREATER dataset.
    :param pcl (N, D) numpy array: Point cloud with first 3 elements per row = (x, y, z).
    :param greater_floor_fix (bool): If True, remove the weird curving floor in GREATER.
    :return (N, D) numpy array: Filtered point cloud.
    '''
    mask_x = np.logical_and(x_min <= pcl[..., 0], pcl[..., 0] <= x_max)
    mask_y = np.logical_and(y_min <= pcl[..., 1], pcl[..., 1] <= y_max)
    mask_z = np.logical_and(z_min <= pcl[..., 2], pcl[..., 2] <= z_max)
    mask_xy = np.logical_and(mask_x, mask_y)
    mask_xyz = np.logical_and(mask_xy, mask_z)

    if greater_floor_fix:
        inv_pyramid = np.maximum(np.abs(pcl[..., 0]), np.abs(pcl[..., 1]))
        mask_gf = (pcl[..., 2] > (inv_pyramid - 4.5) / 3.5)
        mask = np.logical_and(mask_gf, mask_xyz)
    else:
        mask = mask_xyz

    result = pcl[mask]
    return result


def filter_pcl_bounds_torch(pcl, x_min=-10.0, x_max=10.0, y_min=-10.0, y_max=10.0,
                            z_min=-10.0, z_max=10.0):
    '''
    Restricts a point cloud to exclude coordinates outside a certain cube.
    :param pcl (B, N, D) tensor: Point cloud with first 3 elements per row = (x, y, z).
    :return (B, N, D) tensor: Filtered point cloud.
    '''
    mask_x = torch.logical_and(x_min <= pcl[..., 0], pcl[..., 0] <= x_max)
    mask_y = torch.logical_and(y_min <= pcl[..., 1], pcl[..., 1] <= y_max)
    mask_z = torch.logical_and(z_min <= pcl[..., 2], pcl[..., 2] <= z_max)
    mask_xy = torch.logical_and(mask_x, mask_y)
    mask_xyz = torch.logical_and(mask_xy, mask_z)
    result = pcl[mask_xyz]
    return result


def filter_pcl_bounds_carla_input_numpy(pcl, min_z=-0.5, other_bounds=20.0, cube_mode=4):
    '''
    Restricts a point cloud to exclude coordinates outside a certain cube.
    This method is tailored to the CARLA dataset.
    :param pcl (N, D) numpy array: Point cloud with first 3 elements per row = (x, y, z).
    :param min_z (float): Discard everything spatially below this value.
    :param other_bounds (float): Input data cube bounds for the point transformer.
    :return (N, D) numpy array: Filtered point cloud.
    '''
    if cube_mode == 1:
        # NOTE: x > -8 to allow for context.
        pcl = filter_pcl_bounds_numpy(
            pcl, x_min=-other_bounds * 0.5, x_max=other_bounds * 2.0, y_min=-other_bounds * 1.0,
            y_max=other_bounds * 1.0, z_min=min_z, z_max=other_bounds * 0.5)

    elif cube_mode == 2:
        pcl = filter_pcl_bounds_numpy(
            pcl, x_min=-other_bounds * 0.6, x_max=other_bounds * 2.4, y_min=-other_bounds * 0.8,
            y_max=other_bounds * 0.8, z_min=min_z, z_max=other_bounds * 0.6)

    elif cube_mode == 3:
        pcl = filter_pcl_bounds_numpy(
            pcl, x_min=-other_bounds * 0.7, x_max=other_bounds * 2.2, y_min=-other_bounds * 1.0,
            y_max=other_bounds * 1.0, z_min=min_z, z_max=other_bounds * 0.5)

    elif cube_mode == 4:
        pcl = filter_pcl_bounds_numpy(
            pcl, x_min=-other_bounds * 0.7, x_max=other_bounds * 2.5, y_min=-other_bounds * 1.0,
            y_max=other_bounds * 1.0, z_min=min_z, z_max=other_bounds * 0.5)

    return pcl


def filter_pcl_bounds_carla_output_torch(pcl, min_z=-0.5, other_bounds=16.0,
                                         padding=0.0, cube_mode=4):
    '''
    Restricts a point cloud to exclude coordinates outside a certain cube.
    This method is tailored to the CARLA dataset.
    :param pcl (B, N, D) tensor: Point cloud with first 3 elements per row = (x, y, z).
    :param min_z (float): Discard everything spatially below this value.
    :param other_bounds (float): Output data cube bounds for the point transformer.
    :param padding (float): Still include this buffer in 5 directions for context.
    :return (B, N, D) tensor: Filtered point cloud.
    '''
    # NOTE: x > 0.0 because this is the output cube!
    if cube_mode == 1:
        pcl = filter_pcl_bounds_torch(
            pcl, x_min=0.0 - padding, x_max=other_bounds * 2.0 + padding,
            y_min=-other_bounds * 1.0 - padding, y_max=other_bounds * 1.0 + padding,
            z_min=min_z, z_max=other_bounds * 0.5)

    elif cube_mode == 2:
        pcl = filter_pcl_bounds_torch(
            pcl, x_min=0.0 - padding, x_max=other_bounds * 2.4 + padding,
            y_min=-other_bounds * 0.8 - padding, y_max=other_bounds * 0.8 + padding,
            z_min=min_z, z_max=other_bounds * 0.4)

    elif cube_mode == 3:
        pcl = filter_pcl_bounds_torch(
            pcl, x_min=0.0 - padding, x_max=other_bounds * 2.2 + padding,
            y_min=-other_bounds * 1.0 - padding, y_max=other_bounds * 1.0 + padding,
            z_min=min_z, z_max=other_bounds * 0.4)

    elif cube_mode == 4:
        pcl = filter_pcl_bounds_torch(
            pcl, x_min=0.0 - padding, x_max=other_bounds * 2.5 + padding,
            y_min=-other_bounds * 1.0 - padding, y_max=other_bounds * 1.0 + padding,
            z_min=min_z, z_max=other_bounds * 0.4)

    return pcl


def subsample_pad_pcl_numpy(pcl, n_desired, subsample_only=False):
    '''
    If the point cloud is too small, leave as is (nothing changes).
    If the point cloud is too large, subsample uniformly randomly.
    :param pcl (N, D) numpy array.
    :param n_desired (int).
    :param sample_mode (str): random or farthest_point.
    :param subsample_only (bool): If True, do not allow padding.
    :return (n_desired, D) numpy array.
    '''
    N = pcl.shape[0]

    if N < n_desired:
        if subsample_only:
            raise RuntimeError('Too few input points: ' +
                               str(N) + ' vs ' + str(n_desired) + '.')

        return pcl
        # zeros = np.zeros((n_desired - N, pcl.shape[1]), dtype=pcl.dtype)
        # result = np.concatenate((pcl, zeros), axis=0)
        # return result

    elif N > n_desired:
        inds = np.random.choice(N, n_desired, replace=False)
        inds.sort()
        result = pcl[inds]
        return result

    else:
        return pcl


def subsample_pad_pcl_torch(pcl, n_desired, sample_mode='random', subsample_only=False,
                            retain_vehped=False, segm_idx=None):
    '''
    If the point cloud is too small, apply zero padding.
    If the point cloud is too large, subsample either uniformly randomly or by farthest point
        sampling (FPS) per batch item.
    :param pcl (B, N, D) tensor.
    :param n_desired (int).
    :param sample_mode (str): random or farthest_point.
    :param subsample_only (bool): If True, do not allow padding.
    :param retain_vehped (bool): Do not subsample cars & people (semantic tags 4 and 10).
    :param segm_idx (int): Semantic tag index.
    :return (B, n_desired, D) tensor.
    '''
    assert sample_mode in ['random', 'farthest_point']
    no_batch = (len(pcl.shape) == 2)
    if no_batch:
        pcl = pcl.unsqueeze(0)
    (B, N, D) = pcl.shape

    if N < n_desired:
        if subsample_only:
            raise RuntimeError('Too few input points: ' +
                               str(N) + ' vs ' + str(n_desired) + '.')

        zeros = torch.zeros((B, n_desired - N, D), dtype=pcl.dtype)
        zeros = zeros.to(pcl.device)
        result = torch.cat((pcl, zeros), axis=1)
        if no_batch:
            result = result.squeeze(0)
        return result

    elif N > n_desired:
        n_remain = n_desired

        if retain_vehped:
            # 4 = Pedestrian, 10 = Vehicles.
            assert B == 1
            retain_mask = np.logical_or(pcl[..., segm_idx] == 4, pcl[..., segm_idx] == 10)[0]
            retain_inds = np.where(retain_mask)[0]  # (F).

            # We can now continue randomly sampling everything except vehicles.
            remain_mask = (pcl[..., segm_idx] != 10)[0]
            remain_inds = np.where(remain_mask)[0]  # (N - F).
            n_remain -= retain_inds.shape[0]

        else:
            assert B == 1
            remain_inds = np.arange(N)  # (N).

        result = torch.zeros((B, n_remain, D), dtype=pcl.dtype)

        if sample_mode == 'random':
            for i in range(B):
                inds = np.random.choice(remain_inds, n_remain, replace=False)
                inds.sort()
                result[i] = pcl[i, inds]

        else:  # farthest_point.
            assert not retain_vehped
            pcl_flat = pcl.view(B * N, D)
            coords_flat = pcl_flat[..., :3]
            batch = torch.arange(B).repeat_interleave(N)  # (B*N).
            batch = batch.to(pcl.device)
            # NOTE: This fps call has inherent randomness!
            inds = torch_cluster.fps(
                src=coords_flat, batch=batch, ratio=n_remain / N - 1e-7)
            inds = torch.sort(inds)[0]
            pcl_sub_flat = pcl_flat[inds]
            result = pcl_sub_flat.view(B, n_remain, D)

        if no_batch:
            result = result.squeeze(0)

        if retain_vehped:
            assert B == 1
            retain_pcl = pcl[0][retain_inds]
            result = torch.cat([retain_pcl, result], dim=0)

        assert result.shape[0] == n_desired
        return result

    else:
        if no_batch:
            pcl = pcl.squeeze(0)
        return pcl


def my_knn_numpy(pcl_query, pcl_key, k, bidirectional=False,
                 return_inds=False, return_knn=True, return_dists=False):
    '''
    Returns, for each query point, the k nearest key points by 3D Euclidean distance.
        This method considers all pairs, which appears to be surprisingly fast.
    :param pcl_query (N, D) numpy array with (x, y, z, *).
    :param pcl_key (M, E) numpy array with (x, y, z, *).
    :param k (int): The number of neighbors to return.
    :param bidirectional (bool): Also calculate the knn result when the roles of both input point
        clouds are swapped.
    :param return_knn (bool): Whether to return the k nearest neighbors themselves.
    :param return_dists (bool): Whether to return the calculated distances to the k nearest
        neighbors.
    :return
        pcl_knn if not bidirectional and not return_dists;
        (pcl_knn, pcl_knn_rev) if bidirectional and not return_dists;
        (pcl_knn, dists_qtk) if not bidirectional and return_dists;
        (pcl_knn, dists_qtk, pcl_knn_rev, dists_ktq) if bidirectional and return_dists.
            pcl_knn (N, K, E) numpy array.
            dists_qtk (N, K) numpy array.
            pcl_knn_rev (M, K, D) numpy array.
            dists_ktq (M, K) numpy array.
    '''
    (N, D) = pcl_query.shape
    (M, E) = pcl_key.shape
    assert return_inds or return_knn or return_dists

    # Calculate Euclidean distances between all pairs of points.
    diffs = pcl_query[None, :, :3] - pcl_key[:, None, :3]
    # By broadcasting, (1, N, 3) - (M, 1, 3) = (M, N, 3).
    dists = np.linalg.norm(diffs, axis=-1, ord=2)  # (M, N).

    inds_qtk = np.argpartition(dists, k, axis=0)[:k]  # (K, N).
    inds_qtk = np.transpose(inds_qtk, (1, 0))  # (N, K).
    dists_qtk = np.take_along_axis(dists.T, inds_qtk, axis=1)  # (N, K).

    # Not yet sorted by argpartition along K-dimension, so apply argsort.
    reorder_key = np.argsort(dists_qtk, axis=1)  # (N, K).
    inds_qtk = np.take_along_axis(inds_qtk, reorder_key, axis=1)  # (N, K).
    dists_qtk = np.take_along_axis(dists_qtk, reorder_key, axis=1)  # (N, K).

    result = tuple()

    if return_inds:
        result += (inds_qtk, )

    if return_knn:
        # Select (N, K) indices (values in [0, M - 1]) within (M, E) tensor => (N, K, E) tensor.
        pcl_qtk = pcl_key[inds_qtk]
        result += (pcl_qtk, )

    if return_dists:
        result += (dists_qtk, )

    if bidirectional:
        # inds_ktq = np.argpartition(dists, k, dim=1)[:k]  # (M, K).
        raise NotImplementedError()

    return result


def get_1nn_label(points_query, pcl_target_xyz, thresh=1.0):
    '''
    Method to obtain pseudo label for query points based on Nearest Neighbor (NN) search.
    :param points_query (N, 3) numpy array.
    :param pcl_target_xyz (M, 3) numpy array.
    :return target_labels (N)
            nn_indices (N)
    '''
    kdt = sklearn.neighbors.KDTree(pcl_target_xyz, leaf_size=30, metric='euclidean')
    nn_distances, nn_indices = kdt.query(points_query, k=1, return_distance=True)
    target_labels = (nn_distances[:, 0] < thresh) * 1
    return target_labels, nn_indices


def my_knn_torch(pcl_query, pcl_key, num_neighbors, bidirectional=False,
                 return_inds=False, return_knn=True, return_dists=False):
    '''
    Returns, for each query point, the k nearest key points by 3D Euclidean distance.
        This method considers all pairs, which appears to be surprisingly fast.
        NOTE: This is also GPU memory intensive, so using DataParallel is recommended.
    :param pcl_query (N, D) tensor with (x, y, z, *).
    :param pcl_key (M, E) tensor with (x, y, z, *).
    :param num_neighbors (int): K = the number of nearest neighbors to return.
    :param bidirectional (bool): Also calculate the knn result when the roles of both input point
        clouds are swapped.
    :param return_knn (bool): Whether to return the k nearest neighbors themselves.
    :param return_dists (bool): Whether to return the calculated distances to the k nearest
        neighbors.
    :return (inds_qtk?, pcl_qtk?, dists_qtk?, inds_ktq?, pcl_ktq?, dists_ktq?).
    '''
    (N, D) = pcl_query.shape
    (M, E) = pcl_key.shape
    assert return_inds or return_knn or return_dists

    # Calculate Euclidean distances between all pairs of points.
    diffs = pcl_query[None, :, :3] - pcl_key[:, None, :3]
    # By broadcasting, (1, N, 3) - (M, 1, 3) = (M, N, 3).
    dists = torch.linalg.norm(diffs, axis=-1, ord=2)  # (M, N).

    # (K, N), (K, N).
    (dists_qtk, inds_qtk) = dists.topk(num_neighbors, dim=0, largest=False)
    inds_qtk = inds_qtk.permute(1, 0)  # (N, K).
    result = tuple()

    if return_inds:
        result += (inds_qtk, )

    if return_knn:
        # Select (N, K) indices (values in [0, M - 1]) within (M, E) tensor => (N, K, E) tensor.
        pcl_qtk = pcl_key[inds_qtk]
        result += (pcl_qtk, )

    if return_dists:
        dists_qtk = dists_qtk.permute(1, 0)  # (N, K).
        result += (dists_qtk, )

    if bidirectional:
        raise NotImplementedError()

    return result


def trilinear_interpolation(features, points, points_super, knn_k=4):
    '''
    Upsample features by looking at how a larger point cloud compares to its subset.
    :param features (B, N, D) tensor.
    :param points (B, N, 3) tensor.
    :param points_super (B, M, 3) tensor, where M > N.
    :param knn_k (int): Number of points of the source point cloud that features_super should
        consider when reconstructing features for the larger point cloud.
    :return features_super (B, M, D) tensor.
    '''
    (B, N, D) = features.shape
    M = points_super.shape[1]
    assert N >= knn_k, 'Not enough source points to perform nearest neighbors with.'
    assert points.shape == (B, N, 3)
    assert points_super.shape == (B, M, 3)

    # Gathering indices is easier with flattened tensors.
    features_flat = features.view(B * N, D)
    points_flat = points.view(B * N, 3)
    points_super_flat = points_super.view(B * M, 3)
    batch = torch.arange(B).repeat_interleave(N).to(features.device)  # (B*N).
    # (B*M).
    batch_super = torch.arange(B).repeat_interleave(M).to(features.device)

    # This knn() method finds for each element in y the k nearest neighbors in x.
    knn_inds = torch_cluster.knn(x=points_flat, y=points_super_flat,
                                 k=knn_k, batch_x=batch, batch_y=batch_super)  # (2, B*M*k).
    knn_inds = knn_inds[1].view(B * M, knn_k)  # (B*M, k), values in B*N.

    points_nn_flat = torch.stack([points_flat[knn_inds[:, j]]
                                  for j in range(knn_k)], dim=1)  # (B*M, knn_k, 3).
    points_nn = points_nn_flat.view(B, M, knn_k, 3)

    # Calculate Euclidean distances between destination points and K nearest source points.
    # By broadcasting, (B, M, 1, 3) - (B, M, K, 3) = (B, M, K, 3).
    diffs = points_super[:, :, None, :] - points_nn

    # Normalize along the source dimension and calculate the weighting per destination point.
    dists = torch.linalg.norm(diffs, ord=2, dim=-1)  # (B, M, K).
    weights = 1.0 / (dists + 1e-7)  # (B, M, K).
    norm_weights = F.normalize(weights, p=1, dim=-1)  # (B, M, K).

    # Interpret normalized weights as attention vectors to upsample features.
    # NOTE: This is not really the "mathematically correct" way to interpolate, but rather
    # kind of an approximation. The true way would be to use barycentric coordinates relative
    # to an encompassing tetrahedron, in place of inverse distance metrics, even though this
    # also assumes that the source features themselves are consistent in that manner.
    features_nn_flat = torch.stack([features_flat[knn_inds[:, j]]
                                    for j in range(knn_k)], dim=1)  # (B*M, knn_k, D).
    features_nn = features_nn_flat.view(B, M, knn_k, D)
    features_super = torch.einsum('bik,bikd->bid', norm_weights, features_nn)

    assert features_super.shape == (B, M, D)
    return features_super


def sample_random_uniform_3ball(num_points, max_radius, min_radius=0.0):
    '''
    Generate random points that are uniformly distributed within the interior of a 3D sphere.
    http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    :param num_points (int) = N.
    :param max_radius (float) = R.
    :return (N, 3) tensor filled with vectors where L2 norm <= radius R.
    '''
    uvw = torch.randn(num_points, 3, dtype=torch.float32)
    uvw = F.normalize(uvw, p=2, dim=-1)
    radius = torch.tensor(np.cbrt(np.random.rand(num_points).astype(np.float32)))
    radius = radius * (max_radius - min_radius) + min_radius
    result = uvw * radius[:, None]
    return result


class GuidedImplicitPointSampler(torch.nn.Module):
    '''
    Wrapper around the training time point sampler such that DataParallel can be leveraged.
    NOTE: This object does not contain any learnable parameters.
    '''

    def __init__(self, logger, min_z=-1.0, cube_bounds=10.0, point_occupancy_radius=0.25,
                num_solid=1024, num_air=1024, predict_segmentation=False, semantic_classes=13,
                predict_tracking=False, data_kind='', point_sample_bias='none', cube_mode=4):
        '''
        :param min_z, cube_bounds (float): Output data cube to consider, do not sample any points
            below min_z or otherwise outside cube_bounds along any dimension.
        :param point_occupancy_radius (float): Solid points are sampled at least this close to at
            least one target point, and air points are at least this far away from all target
            points.
        :param num_solid, num_air (int): S and A respectively.
        :param predict_segmentation (bool).
        :param semantic_classes (int).
        :param predict_tracking (bool).
        :param data_kind (str): Guides how to spatially sample points (greater / carla).
        :param point_sample_bias (str): none / low / moving / vehped / ivalo / sembal or a combination.
        :param cube_mode (int): Which cuboid shape to use for CARLA (1 / 2 / 3 / 4).
        '''
        super().__init__()
        self.logger = logger
        self.min_z = min_z
        self.cube_bounds = cube_bounds
        self.point_occupancy_radius = point_occupancy_radius
        self.num_solid = num_solid
        self.num_air = num_air
        self.predict_segmentation = predict_segmentation
        self.semantic_classes = semantic_classes
        self.predict_tracking = predict_tracking
        self.data_kind = data_kind
        self.point_sample_bias = point_sample_bias
        self.cube_mode = cube_mode
        self.low_prefer_min_z = 0.0
        self.low_prefer_max_z = 2.0

    def forward(self, pcl_target, pcl_target_size, valo_ids, num_valo_ids, time_idx):
        '''
        Sample 4D points within the given spatio-temporal cube, with a controlled mixture of solid
            and free space points. Call this method multiple times, once for each frame.
        :param pcl_target: List-T of (B, M, E) tensors: Ground truth point cloud that the continuous
            representation should encompass. Typically,
            E = 9 with (x, y, z, instance_id, view_idx, R, G, B, mark_track), or
            E = 11 with (x, y, z, cosine_angle, instance_id, semantic_tag, view_idx, R, G, B, mark_track).
        :param pcl_target_size: List-T of (B) tensors: int values denoting which target points are
            relevant.
        :param valo_ids (B, R) tensor: Vehped instance ids that were seen at least once in the
            entire input.
        :param num_valo_ids (B) tensor: int values denoting how many entries of valo_ids are valid.
        :param time_idx (int): Frame time value.
        :return (solid_input, air_input, solid_target, air_target).
            solid_input (B, S, 4) tensor with (x, y, z, t).
            air_input (B, A, 4) tensor with (x, y, z, t).
            solid_target (B, S, 5+) tensor with (density, R, G, B, mark_track, segm?).
            air_target (B, A, 5+) tensor with (density, R, G, B, mark_track, segm?).
            NOTE: The two returned inputs later become concatenated into pcl_query (per frame),
            while the two returned outputs become concatenated into implicit_target (per frame).
        '''
        pcl_target_frame = pcl_target[time_idx]
        pcl_target_frame_size = pcl_target_size[time_idx]

        (B, M, E) = pcl_target_frame.shape
        assert torch.all(pcl_target_frame_size <= M)
        if self.data_kind == 'greater':
            assert E == 9
        elif self.data_kind == 'carla':
            assert E == 11

        # Select one other frame from which to generate dynamic air,
        # in order to eventually discourage false positives.
        if len(pcl_target) > 1:
            other_time = np.random.randint(len(pcl_target) - 1)
            if other_time == time_idx:
                other_time += 1
            pcl_other_frame = pcl_target[other_time]
            pcl_other_frame_size = pcl_target_size[other_time]
        else:
            pcl_other_frame = None
            pcl_other_frame_size = None

        solid_input = []
        solid_target = []
        solid_sbs = []
        air_input = []
        air_target = []
        air_sbs = []

        for i in range(B):

            # Select valid target points for this example.
            cur_tgt_pcl_count = pcl_target_frame_size[i].item()
            cur_tgt_pcl = pcl_target_frame[i, :cur_tgt_pcl_count]  # (M, 9-11).

            # Select valid metadata for this example.
            cur_num_valo_ids = num_valo_ids[i].item()
            cur_valo_ids = sorted(list(valo_ids[i, :cur_num_valo_ids].detach().cpu().numpy()))
            # (R).

            # We are only interested in the *output* cube for all (including solid) points!
            if self.data_kind == 'carla':
                cur_tgt_pcl = filter_pcl_bounds_carla_output_torch(
                    cur_tgt_pcl, min_z=self.min_z, other_bounds=self.cube_bounds,
                    cube_mode=self.cube_mode)
                cur_tgt_pcl_count = cur_tgt_pcl.shape[0]

            # NOTE: The target is sometimes empty in CARLA after filtering, even if it passed a
            # similar check in the dataloader. Safeguard against this to avoid faulty supervision.
            if cur_tgt_pcl_count < 256:
                raise RuntimeError(f'Invalid due to cur_tgt_pcl_count: {cur_tgt_pcl_count}')

            # Determine appropriate target slice size.
            max_slice_size = int((2 ** 27) // self.num_air)
            num_slices = int(np.ceil(cur_tgt_pcl_count / max_slice_size))
            used_slice_size = cur_tgt_pcl_count // num_slices + 1

            # Repeat target preprocessing but for the random other frame instead.
            if 'moving' in self.point_sample_bias:
                cur_other_pcl_count = pcl_other_frame_size[i].item()
                cur_other_pcl = pcl_other_frame[i, :cur_other_pcl_count]  # (M, 9-11).
                if self.data_kind == 'carla':
                    cur_other_pcl = filter_pcl_bounds_carla_output_torch(
                        cur_other_pcl, min_z=self.min_z, other_bounds=self.cube_bounds,
                        cube_mode=self.cube_mode)
                    cur_other_pcl_count = cur_tgt_pcl.shape[0]
                if cur_other_pcl_count < 256:
                    raise RuntimeError(f'Invalid due to cur_other_pcl_count: {cur_other_pcl_count}')

                # We use subsampled versions of both frames such that this step goes faster.
                # NOTE: This requires shuffled point clouds (in the data loader) to work correctly!
                # start_time = time.time()

                cur_tgt_sub = cur_tgt_pcl[:used_slice_size]
                cur_other_sub = cur_other_pcl[:used_slice_size]
                (cur_tgt_unique, _, _) = filter_air_solid_gap(
                    cur_tgt_sub, cur_other_sub[..., :3], used_slice_size,
                    self.point_occupancy_radius * 2.0)
                (cur_other_unique, _, _) = filter_air_solid_gap(
                    cur_other_sub, cur_tgt_sub[..., :3], used_slice_size,
                    self.point_occupancy_radius * 2.0)

                # (cur_tgt_unique, tgt_other_dists) = filter_air_solid_gap(
                #     cur_tgt_pcl, cur_other_pcl[..., :3], used_slice_size,
                #     self.point_occupancy_radius * 2.0)
                # (cur_other_unique, other_tgt_dists) = filter_air_solid_gap(
                #     cur_other_pcl, cur_tgt_pcl[..., :3], used_slice_size,
                #     self.point_occupancy_radius * 2.0)

                # print(f'took {time.time() - start_time:.3f}s')

            else:
                cur_tgt_unique = None
                cur_other_unique = None

            # Sample solid points by using the target point cloud, then add small random offsets.
            # used_num_solid = self.num_solid // 2 if time_idx < len(pcl_target) else self.num_solid
            (cur_solid_input, cur_solid_target, cur_solid_sbs) = self.construct_solid_input_target(
                cur_tgt_pcl, cur_tgt_unique, cur_valo_ids, time_idx)

            # Sample air points by oversampling and then filtering, such that we stay away from the
            # target point cloud.
            # used_num_air = self.num_air // 2 if time_idx < len(pcl_target) else self.num_air
            (cur_air_input, cur_air_target, cur_air_sbs) = \
                self.construct_air_input_target(
                cur_tgt_pcl, cur_other_unique, cur_solid_input, cur_valo_ids,
                time_idx)

            solid_input.append(cur_solid_input)
            solid_target.append(cur_solid_target)
            solid_sbs.append(cur_solid_sbs)
            air_input.append(cur_air_input)
            air_target.append(cur_air_target)
            air_sbs.append(cur_air_sbs)

        # This microbatch is done.
        solid_input = torch.stack(solid_input)
        solid_target = torch.stack(solid_target)
        solid_sbs = torch.stack(solid_sbs)
        air_input = torch.stack(air_input)
        air_target = torch.stack(air_target)
        air_sbs = torch.stack(air_sbs)

        return (solid_input, air_input, solid_target, air_target, solid_sbs, air_sbs)

    def construct_solid_input_target(
            self, cur_tgt_pcl, cur_tgt_unique, cur_valo_ids, time_idx):
        '''
        For this frame, generates input + target arrays representing query points and the associated
            solid ground truth data. This supervision (specifically, the point sampling algorithm)
            may optionally be steered / biased in one or more complementary ways.
        :param cur_tgt_pcl: Entire target point cloud.
            If greater: (M, 9) tensor with (x, y, z, instance_id, view_idx, R, G, B, mark_track).
            If carla: (M, 11) tensor with (x, y, z, cosine_angle, instance_id,
                                        semantic_tag, view_idx, R, G, B, mark_track).
        :param cur_tgt_unique: Dynamic regions only; shape similar to cur_tgt_pcl.
        :param time_idx (int).
        :return (cur_solid_input, cur_solid_target).
            cur_solid_input: (S, 4) tensor with (x, y, z, t).
            cur_solid_target: (S, 5+) tensor with (density, R, G, B, mark_track, segm?).
        '''
        inst_idx = 4 if self.data_kind == 'carla' else 3
        segm_idx = 5 if self.data_kind == 'carla' else 3
        view_idx = 6 if self.data_kind == 'carla' else 4
        copy_count = 4  # Non-special features = last few colums of target = (R, G, B, mark_track).

        # Construct set of target points to extract solid implicit input + target pairs from.
        # NOTE: We typically sample indices with replacement.
        sel_target_points_pool = []  # List of (S, 6-10) tensors.
        sample_bias_shares = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # (regular, low, moving, vehped, ivalo, sembal).
        # Max sum = 2.8.

        if 'none' in self.point_sample_bias:
            pass  # This is default so no need to manually specify.

        if 'low' in self.point_sample_bias:
            # This bias is conditional, because it depends on the actual presence of low points.
            prefer_mask = torch.logical_and(self.low_prefer_min_z <= cur_tgt_pcl[..., 2],
                                            cur_tgt_pcl[..., 2] <= self.low_prefer_max_z)
            cur_tgt_low = cur_tgt_pcl[prefer_mask]
            if cur_tgt_low.shape[0] >= 256:
                sample_bias_shares[1] += 1.0

        if 'moving' in self.point_sample_bias:
            # This bias is conditional, because it depends on the degree of dynamic action.
            if cur_tgt_unique.shape[0] >= 256:
                sample_bias_shares[2] += 0.4
            elif cur_tgt_unique.shape[0] >= 16:
                sample_bias_shares[2] += cur_tgt_unique.shape[0] * 0.4 / 256.0  # Max 0.4.

        if 'vehped' in self.point_sample_bias:
            # This bias is conditional, because it depends on the actual presence of the objects of
            # interest (4 = Pedestrian, 10 = Vehicles).
            # https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
            assert self.data_kind == 'carla'
            tgt_vehped = get_vehped_points(cur_tgt_pcl, segm_idx)
            if tgt_vehped.shape[0] >= 256:
                sample_bias_shares[3] += 0.2
            elif tgt_vehped.shape[0] >= 16:
                sample_bias_shares[3] += tgt_vehped.shape[0] * 0.2 / 256.0  # Max 0.2.

        if 'ivalo' in self.point_sample_bias:
            # This bias is conditional, because it depends on the presence of solvable cars and
            # people.
            assert self.data_kind == 'carla'

            if len(cur_valo_ids) > 0:
                # Get currently visible ids.
                cur_vis_pcl = cur_tgt_pcl[cur_tgt_pcl[..., view_idx] == 0]
                vis_vehped = get_vehped_points(cur_vis_pcl, segm_idx)
                vis_ids = vis_vehped[..., inst_idx].type(torch.int32).unique()
                vis_ids = sorted(list(vis_ids.detach().cpu().numpy()))

                # Get all non-input points of valo ids.
                cur_invis_pcl = cur_tgt_pcl[cur_tgt_pcl[..., view_idx] != 0]
                invis_vehped = get_vehped_points(cur_invis_pcl, segm_idx)
                ivalo_vehped = []
                for valo_id in cur_valo_ids:
                    inst_pcl = invis_vehped[invis_vehped[..., inst_idx] == valo_id]
                    ivalo_vehped.append(inst_pcl)
                    if valo_id not in vis_ids:
                        # Total occlusion, so oversample by adding twice.
                        ivalo_vehped.append(inst_pcl)
                ivalo_vehped = torch.cat(ivalo_vehped, dim=0)

                if ivalo_vehped.shape[0] >= 256:
                    sample_bias_shares[4] += 0.2
                elif ivalo_vehped.shape[0] >= 16:
                    # Max 0.2.
                    sample_bias_shares[4] += min(ivalo_vehped.shape[0] * 0.2 / 256.0, 0.2)

        if 'sembal' in self.point_sample_bias:
            assert self.data_kind == 'carla'
            sample_bias_shares[5] += 0.4

        # Renormalize such that we always end up with a count of exactly num_solid points.
        sample_bias_shares /= sample_bias_shares.sum()

        # Similar to regular, but more specifically at the height of vehicles and pedestrians.
        num_low = int(sample_bias_shares[1] * self.num_solid)
        if num_low > 0:
            low_inds = torch.randint(0, cur_tgt_low.shape[0], (num_low, ))
            sel_target_points_pool.append(cur_tgt_low[low_inds])

        # Dynamic regions only.
        num_moving = int(sample_bias_shares[2] * self.num_solid)
        if num_moving > 0:
            moving_inds = torch.randint(0, cur_tgt_unique.shape[0], (num_moving, ))
            sel_target_points_pool.append(cur_tgt_unique[moving_inds])

        # Cars (vehicles) and people (pedestrians) only.
        num_vehped = int(sample_bias_shares[3] * self.num_solid)
        if num_vehped > 0:
            vehped_inds = torch.randint(0, tgt_vehped.shape[0], (num_vehped, ))
            sel_target_points_pool.append(tgt_vehped[vehped_inds])

        # Invisible (occluded) cars and people but visible at least once (in the input).
        num_ivalo = int(sample_bias_shares[4] * self.num_solid)
        if num_ivalo > 0:
            ivalo_inds = torch.randint(0, ivalo_vehped.shape[0], (num_ivalo, ))
            sel_target_points_pool.append(ivalo_vehped[ivalo_inds])

        # Extra points to balance all semantic classes.
        num_sembal = int(sample_bias_shares[5] * self.num_solid)
        if num_sembal > 0:
            # We first check for at least 1 occurence, but later filter by at least 16 points.
            exist_segm_ids = cur_tgt_pcl[..., segm_idx].type(torch.int32).unique()
            exist_segm_ids = list(exist_segm_ids.detach().cpu().numpy())
            num_cats = len(exist_segm_ids)
            actual_num_sembal = 0

            for exist_id in exist_segm_ids:
                tgt_cat = cur_tgt_pcl[cur_tgt_pcl[..., segm_idx] == exist_id]

                if tgt_cat.shape[0] >= 16:
                    num_cat = num_sembal // num_cats
                    cat_inds = torch.randint(0, tgt_cat.shape[0], (num_cat, ))
                    sel_target_points_pool.append(tgt_cat[cat_inds])
                    actual_num_sembal += num_cat

            num_sembal = actual_num_sembal  # Some categories were discarded.

        # Regular, unbiased, uniformly random sampling of the solid target point cloud frame.
        num_regular = self.num_solid - num_low - num_moving - num_vehped - num_ivalo - num_sembal
        if num_regular > 0:
            regular_inds = torch.randint(0, cur_tgt_pcl.shape[0], (num_regular, ))
            sel_target_points_pool.append(cur_tgt_pcl[regular_inds])

        # Merge all selected target points and construct solid implicit query + target arrays.
        sel_target_points = torch.cat(sel_target_points_pool, dim=0)
        cur_solid_input = sel_target_points[..., :3]  # (S, 3) with (x, y, z).
        cur_solid_target = sel_target_points[..., -copy_count:]
        # (S, 4) with (R, G, B, mark_track).
        assert cur_solid_input.shape[0] == self.num_solid

        # Add random small spatial offset of max half the point radius in an L2 sense.
        offset = sample_random_uniform_3ball(self.num_solid, self.point_occupancy_radius / 2.0)
        offset = offset.to(cur_solid_input.device)
        cur_solid_input += offset

        # Append constant time index value to input => (S, 4) with (x, y, z, t).
        cur_solid_input = torch.cat(
            [cur_solid_input, torch.ones_like(cur_solid_input[..., 0:1]) * time_idx], dim=-1)

        # Prepend density = 1 to target => (S, 5) with (density, R, G, B, mark_track).
        cur_solid_target = torch.cat(
            [torch.ones_like(cur_solid_target[..., 0:1]), cur_solid_target], dim=-1)

        # Append segmentation to target if desired => (S, 6) with
        # (density, R, G, B, mark_track, segm).
        if self.predict_segmentation:
            cur_solid_segm = sel_target_points[..., segm_idx:segm_idx + 1]  # (S, 1).
            cur_solid_segm[cur_solid_segm >= self.semantic_classes] = 3  # = Other.
            cur_solid_target = torch.cat([cur_solid_target, cur_solid_segm], dim=-1)
        else:
            cur_solid_target = torch.cat(
                [cur_solid_target, -torch.ones_like(cur_solid_target[..., 0:1])], dim=-1)

        return (cur_solid_input, cur_solid_target, sample_bias_shares)

    def construct_air_input_target(
            self, cur_tgt_pcl, cur_other_unique, cur_solid_input, cur_valo_ids,
            time_idx):
        '''
        Same as construct_solid_input_target(), but for air points (empty space) instead.
        :return (cur_air_input, cur_air_target).
        '''

        cur_tgt_pcl_xyz = cur_tgt_pcl[..., :3]  # (M, 3).
        # cur_tgt_pcl_count = cur_tgt_pcl.shape[0]
        # if cur_other_unique is not None:
        #     cur_other_unique_count = cur_other_unique.shape[0]
        #     cur_other_unique_xyz = cur_other_unique[..., :3]  # (U, 3).

        # Air filtering requires knn, so chop target for iterative filtering.
        # For example, if num_air is 16384, then this is 8192.
        max_target_slice_size = int((2 ** 27) // self.num_air)
        num_slices = int(np.ceil(cur_tgt_pcl.shape[0] / max_target_slice_size))
        used_target_slice_size = cur_tgt_pcl.shape[0] // num_slices + 1
        assert used_target_slice_size * num_slices >= cur_tgt_pcl.shape[0]

        # Construct set of target points to extract air implicit input + target pairs from.
        sel_input_points_pool = []  # List of (A, 3) tensors with (x, y, z) only.
        sel_air_solid_dists_pool = []  # List of (A) tensors with distance values.
        sample_bias_shares = torch.tensor([0.5, 0.0, 0.3, 0.2])
        # (regular, moving, hard_solid_query, hard_target).
        # NOTE: We ignore low and vehped because they follow from hard_solid_query.

        if 'moving' in self.point_sample_bias:
            # This bias is conditional, because it depends on the degree of dynamic action.
            if cur_other_unique.shape[0] >= 256:
                sample_bias_shares[1] += 0.4
            elif cur_other_unique.shape[0] >= 16:
                sample_bias_shares[1] += cur_other_unique.shape[0] * 0.4 / 256.0  # Max 0.4.

        # Renormalize such that we always end up with a count of exactly num_air points.
        sample_bias_shares /= sample_bias_shares.sum()

        # Next, we sample for every type of bias such that the number of valid points after
        # filtering almost always exceeds the desired count for that bias type. Next, for each bias
        # type, exclude some air points by proximity to solids. Perform this in a sequential manner
        # with slices of the target point cloud to minimize GPU memory due to knn. We aim for slices
        # with approximately equal size, while remaining below the maximum.

        # Dynamic regions only, by looking at a random other target frame (= at a different time!).
        # This should discourage false positives.
        num_moving = int(sample_bias_shares[1] * self.num_air)
        num_moving_sample = int(num_moving * 1.6)
        if num_moving > 0:
            moving_inds = torch.randint(0, cur_other_unique.shape[0], (num_moving_sample, ))
            moving_input = cur_other_unique[moving_inds][..., :3]
            offset = sample_random_uniform_3ball(num_moving_sample,
                                                self.point_occupancy_radius * 2.0)
            offset = offset.to(moving_input.device)
            moving_input += offset

            # Proximity filtering is a little more drastic here to ensure we're supervising
            # distinctly dynamic regions only.
            (moving_input, moving_dists, moving_ratio) = filter_air_solid_gap(
                moving_input, cur_tgt_pcl_xyz, used_target_slice_size,
                self.point_occupancy_radius)  # * 2.0)
            moving_input = self.select_safely(moving_input, num_moving, warn_insufficient=False)
            moving_dists = self.select_safely(moving_dists, num_moving, warn_insufficient=False)

            sel_input_points_pool.append(moving_input)
            sel_air_solid_dists_pool.append(moving_dists)

        # Hard cases by staying close to the sampled solid input points. This should improve the
        # visual quality of objects of interest.
        num_hsq = int(sample_bias_shares[2] * self.num_air)
        num_hsq_sample = int(num_hsq * 2.0)
        if num_hsq > 0:
            hsq_inds = torch.randint(0, cur_solid_input.shape[0], (num_hsq_sample, ))
            hsq_input = cur_solid_input[hsq_inds][..., :3]
            offset = sample_random_uniform_3ball(num_hsq_sample,
                                                max_radius=self.point_occupancy_radius * 3.0,
                                                min_radius=self.point_occupancy_radius)
            offset = offset.to(hsq_input.device)
            hsq_input += offset

            # The original points (from which the offsets were calculated) won't be too close thanks
            # to min_radius, but neighboring points might be.
            (hsq_input, hsq_dists, hsq_ratio) = filter_air_solid_gap(
                hsq_input, cur_tgt_pcl_xyz, used_target_slice_size, self.point_occupancy_radius)
            hsq_input = self.select_safely(hsq_input, num_hsq)
            hsq_dists = self.select_safely(hsq_dists, num_hsq)

            sel_input_points_pool.append(hsq_input)
            sel_air_solid_dists_pool.append(hsq_dists)

        # Hard cases by staying close to the target point cloud.
        num_ht = int(sample_bias_shares[3] * self.num_air)
        num_ht_sample = int(num_ht * 2.0)
        if num_ht > 0:
            ht_inds = torch.randint(0, cur_tgt_pcl.shape[0], (num_ht_sample, ))
            ht_input = cur_tgt_pcl[ht_inds][..., :3]
            offset = sample_random_uniform_3ball(num_ht_sample,
                                                max_radius=self.point_occupancy_radius * 3.0,
                                                min_radius=self.point_occupancy_radius)
            offset = offset.to(ht_input.device)
            ht_input += offset

            # The original points (from which the offsets were calculated) won't be too close thanks
            # to min_radius, but neighboring points might be.
            (ht_input, ht_dists, ht_ratio) = filter_air_solid_gap(
                ht_input, cur_tgt_pcl_xyz, used_target_slice_size, self.point_occupancy_radius)
            ht_input = self.select_safely(ht_input, num_ht)
            ht_dists = self.select_safely(ht_dists, num_ht)

            sel_input_points_pool.append(ht_input)
            sel_air_solid_dists_pool.append(ht_dists)

        # Regular, unbiased, uniformly random sampling of empty space.
        num_regular = self.num_air - num_moving - num_hsq - num_ht
        if self.data_kind == 'greater':
            num_regular_sample = int(num_regular * 1.3)
        elif self.data_kind == 'carla':
            num_regular_sample = int(num_regular * 1.1)
        if num_regular > 0:
            # NOTE: This cuboid is different from filter_pcl_bounds_carla_input(), because we
            # cut off at x > 0 instead of > -6 (the latter is just for context).
            regular_input = sample_implicit_points_blind_torch(
                self.data_kind, num_regular_sample, self.cube_mode, self.cube_bounds, self.min_z,
                cur_tgt_pcl.device)

            (regular_input, regular_dists, regular_ratio) = filter_air_solid_gap(
                regular_input, cur_tgt_pcl_xyz, used_target_slice_size, self.point_occupancy_radius)
            regular_input = self.select_safely(regular_input, num_regular)
            regular_dists = self.select_safely(regular_dists, num_regular)

            sel_input_points_pool.append(regular_input)
            sel_air_solid_dists_pool.append(regular_dists)

        # Merge all selected air points and construct implicit query + target arrays.
        cur_air_input = torch.cat(sel_input_points_pool, dim=0)  # (A, 3) with (x, y, z).
        air_solid_dists = torch.cat(sel_air_solid_dists_pool, dim=0)  # (A) with tsdf.
        assert cur_air_input.shape[0] == self.num_air

        # Append constant time index value => (A, 4) with (x, y, z, t).
        cur_air_input = torch.cat(
            [cur_air_input, torch.ones_like(cur_air_input[..., 0:1]) * time_idx], dim=-1)

        # Target for density, color, mark_track is just 0, -1, -1 respectively => (A, 5) with
        # (density, R, G, B, mark_track).
        cur_air_target = -torch.ones((self.num_air, 5), device=cur_tgt_pcl.device,
                                    dtype=cur_tgt_pcl.dtype)
        cur_air_target[..., 0] = 0.0

        # Append segmentation = -1 to target => (A, 6) with
        # (density, R, G, B, mark_track, segm).
        cur_air_target = torch.cat(
            [cur_air_target, -torch.ones_like(cur_air_target[..., 0:1])], dim=-1)

        return (cur_air_input, cur_air_target, sample_bias_shares)

    def select_safely(self, pcl, num_select, warn_insufficient=True):
        '''
        Selects the first N elements of a tensor of unknown size.
        Sometimes, there are insufficient points, so as a fallback, we duplicate existing content.
        '''
        while pcl.shape[0] < num_select:
            if warn_insufficient:
                self.logger.warning(f'Size {pcl.shape[0]} is insufficient for {num_select}!')
            pcl = torch.cat([pcl, pcl], dim=0)
        pcl = pcl[:num_select].clone()
        return pcl


def sample_implicit_points_blind_torch(
        data_kind, num_sample, cube_mode, cube_bounds, min_z, device):

    if data_kind == 'greater':

        rnd_air_xy = torch.rand((num_sample, 2), device=device)
        rnd_air_xy = rnd_air_xy * cube_bounds * 2.0 - cube_bounds
        rnd_air_z = torch.rand((num_sample, 1), device=device)
        rnd_air_z = rnd_air_z * (cube_bounds - min_z) + min_z
        rnd_air_input = torch.cat([rnd_air_xy, rnd_air_z], dim=-1)  # (A, 3).

    elif data_kind == 'carla':

        if cube_mode == 1:
            rnd_air_x = torch.rand((num_sample, 1), device=device)
            rnd_air_x = rnd_air_x * cube_bounds * 2.0
            rnd_air_y = torch.rand((num_sample, 1), device=device)
            rnd_air_y = rnd_air_y * cube_bounds * 2.0 - cube_bounds * 1.0
            rnd_air_z = torch.rand((num_sample, 1), device=device)
            rnd_air_z = rnd_air_z * (cube_bounds * 0.5 - min_z) + min_z

        elif cube_mode == 2:
            rnd_air_x = torch.rand((num_sample, 1), device=device)
            rnd_air_x = rnd_air_x * cube_bounds * 2.4
            rnd_air_y = torch.rand((num_sample, 1), device=device)
            rnd_air_y = rnd_air_y * cube_bounds * 1.6 - cube_bounds * 0.8
            rnd_air_z = torch.rand((num_sample, 1), device=device)
            rnd_air_z = rnd_air_z * (cube_bounds * 0.4 - min_z) + min_z

        elif cube_mode == 3:
            rnd_air_x = torch.rand((num_sample, 1), device=device)
            rnd_air_x = rnd_air_x * cube_bounds * 2.2
            rnd_air_y = torch.rand((num_sample, 1), device=device)
            rnd_air_y = rnd_air_y * cube_bounds * 2.0 - cube_bounds * 1.0
            rnd_air_z = torch.rand((num_sample, 1), device=device)
            rnd_air_z = rnd_air_z * (cube_bounds * 0.4 - min_z) + min_z

        elif cube_mode == 4:
            rnd_air_x = torch.rand((num_sample, 1), device=device)
            rnd_air_x = rnd_air_x * cube_bounds * 2.5
            rnd_air_y = torch.rand((num_sample, 1), device=device)
            rnd_air_y = rnd_air_y * cube_bounds * 2.0 - cube_bounds * 1.0
            rnd_air_z = torch.rand((num_sample, 1), device=device)
            rnd_air_z = rnd_air_z * (cube_bounds * 0.4 - min_z) + min_z

        else:
            raise ValueError()

        rnd_air_input = torch.cat([rnd_air_x, rnd_air_y, rnd_air_z], dim=-1)  # (A, 3).

    else:
        raise ValueError()

    return rnd_air_input


def filter_air_solid_gap(to_filter, target_coords, target_slice_size, point_occupancy_radius):
    '''
    Removes all points from an input point cloud that are too close to the ground truth, i.e. a
        target point cloud consisting of solid points.
    :param to_filter (N, D) tensor.
    :param target_coords (M, 3) tensor.
    :param target_slice_size (int): Number of target points to consider at once for memory
        considerations.
    :param point_occupancy_radius (float): The Euclidean distance threshold to use to decide
        proximity.
    :return (to_filter, air_solid_dists, good_ratio).
        to_filter: (N', D) tensor.
        air_solid_dists: (N') tensor.
    '''
    tgt_slices = torch.split(target_coords, target_slice_size)
    air_solid_dists = None

    for tgt_slice in tgt_slices:
        cur_air_solid_dists = my_knn_torch(
            to_filter, tgt_slice, 1, return_inds=False,
            return_knn=False, return_dists=True)[0].squeeze(-1)
        if air_solid_dists is None:
            air_solid_dists = cur_air_solid_dists
        else:
            air_solid_dists = torch.minimum(cur_air_solid_dists, air_solid_dists)

    # air_solid_dists is now complete, i.e. as correct as feasible.
    good_mask = air_solid_dists > point_occupancy_radius
    good_ratio = good_mask.sum() / air_solid_dists.shape[0]
    to_filter = to_filter[good_mask]
    air_solid_dists = air_solid_dists[good_mask]

    return (to_filter, air_solid_dists, good_ratio)


def sample_implicit_points_blind_numpy(num_sample, min_z, cube_bounds, time_idx, data_kind,
                                    cube_mode, point_sample_mode):
    '''
    Sample 4D points within the given spatio-temporal cube, where the ground truth is unknown.
    :param num_sample (int): N.
    :param min_z, cube_bounds (float): Output data cube to consider, do not sample any points below
        min_z or otherwise outside cube_bounds along any dimension.
    :param time_idx (int): Query frame time value.
    :param data_kind (str): Guides how to spatially sample points (greater / carla).
    :param cube_mode (int): Which cuboid shape to use for CARLA.
    :param point_sample_mode (str): How to sample points within CRs (random / grid).
        If random: sample at uniformly random locations within the CR cuboid.
        If grid: sample according to a fixed, equally spaced grid.
    :return points_input (N, 4) numpy array with (x, y, z, t).
    '''

    if data_kind == 'greater':
        (x_min, x_max) = (-cube_bounds, cube_bounds)
        (y_min, y_max) = (-cube_bounds, cube_bounds)
        (z_min, z_max) = (min_z, cube_bounds)

    elif data_kind == 'carla':
        # NOTE: This is slightly different from filter_pcl_bounds_carla_input(), because we cut
        # off at x > 0 instead of > -6 (the latter is just for context).
        if cube_mode == 1:
            (x_min, x_max) = (0.0, cube_bounds * 2.0)
            (y_min, y_max) = (-cube_bounds, cube_bounds)
            (z_min, z_max) = (min_z, cube_bounds * 0.5)

        elif cube_mode == 2:
            (x_min, x_max) = (0.0, cube_bounds * 2.4)
            (y_min, y_max) = (-cube_bounds * 0.8, cube_bounds * 0.8)
            (z_min, z_max) = (min_z, cube_bounds * 0.4)

        elif cube_mode == 3:
            (x_min, x_max) = (0.0, cube_bounds * 2.2)
            (y_min, y_max) = (-cube_bounds, cube_bounds)
            (z_min, z_max) = (min_z, cube_bounds * 0.4)

        elif cube_mode == 4:
            (x_min, x_max) = (0.0, cube_bounds * 2.5)
            (y_min, y_max) = (-cube_bounds, cube_bounds)
            (z_min, z_max) = (min_z, cube_bounds * 0.4)

    else:
        raise ValueError(data_kind)

    # Construct points within specified cuboid.
    if point_sample_mode == 'random':
        used_num_sample = num_sample
        points_x = np.random.rand(num_sample).astype(np.float32)
        points_y = np.random.rand(num_sample).astype(np.float32)
        points_z = np.random.rand(num_sample).astype(np.float32)
        points_x = points_x * (x_max - x_min) + x_min
        points_y = points_y * (y_max - y_min) + y_min
        points_z = points_z * (z_max - z_min) + z_min
        points_xyz = np.stack([points_x, points_y, points_z], axis=-1)  # (N, 3).

    elif point_sample_mode == 'grid':
        volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        requested_points_per_cubic = num_sample / volume
        requested_points_per_unit = np.cbrt(requested_points_per_cubic)
        used_points_x = int(np.ceil(requested_points_per_unit * (x_max - x_min)))
        used_points_y = int(np.ceil(requested_points_per_unit * (y_max - y_min)))
        used_points_z = int(np.ceil(requested_points_per_unit * (z_max - z_min)))
        spacing_x = (x_max - x_min) / used_points_x
        spacing_y = (y_max - y_min) / used_points_y
        spacing_z = (z_max - z_min) / used_points_z
        used_num_sample = used_points_x * used_points_y * used_points_z
        points_x = (np.arange(used_points_x, dtype=np.float32) + 0.5) * spacing_x + x_min
        points_y = (np.arange(used_points_y, dtype=np.float32) + 0.5) * spacing_y + y_min
        points_z = (np.arange(used_points_z, dtype=np.float32) + 0.5) * spacing_z + z_min
        points_x = np.repeat(points_x, used_points_y * used_points_z)
        points_y = np.repeat(points_y, used_points_z)
        points_y = np.tile(points_y, used_points_x)
        points_z = np.tile(points_z, used_points_x * used_points_y)
        points_xyz = np.stack([points_x, points_y, points_z], axis=-1)  # (N, 3).

    else:
        raise ValueError(point_sample_mode)

    # Append considered time index to input.
    points_t = np.ones((used_num_sample, 1), dtype=np.float32) * time_idx
    points_input = np.concatenate([points_xyz, points_t], axis=-1)  # (N, 4).
    return points_input


def transform_lidar_frame(lidar_pcl, source_matrix, target_matrix):
    '''
    Converts the coordinates of the measured point cloud data from one coordinate frame to another.
    :param lidar_pcl (N, D) numpy array with rows (x, y, z, *).
    :param source_matrix (4, 4) numpy array.
    :param target_matrix (4, 4) numpy array.
    :return transformed_pcl (N, D) numpy array with rows (x, y, z, *).
    '''
    (N, D) = lidar_pcl.shape
    inv_target_matrix = np.linalg.inv(target_matrix)

    pcl_xyz = lidar_pcl[..., :3].T  # (3, N).
    points_source = np.concatenate([pcl_xyz, np.ones_like(pcl_xyz[:1])], axis=0)  # (4, N).
    points_world = np.dot(source_matrix, points_source)  # (4, N).
    points_target = np.dot(inv_target_matrix, points_world)  # (4, N).

    pcl_xyz = points_target[:3].T  # (N, 3).
    transformed_pcl = lidar_pcl.copy()
    transformed_pcl[..., :3] = pcl_xyz

    return transformed_pcl


def get_pcl_transience(pcl_first, pcl_last, n_fps=1024):
    '''
    Constructs a point cloud with associated measured values of transience, which indicates the
        per-region dynamic-ness / motion-ness between a pair of temporally separated point clouds.
        Specifically, the delta-NN distance values are calculated for a set of points of interest.
        The coordinates are subsampled using farthest point sampling from both point clouds in the
        input pair. This implies that for every output point, it has an NN distance of 0 for either
        pcl_first or pcl_last, so the delta-NN actually becomes the distance to the other input.
    :param pcl_first (N, D) numpy array.
    :param pcl_last (N, D) numpy array.
    '''
    pass


def get_vehped_points(pcl, segm_idx):
    '''
    :param pcl (N, D) tensor.
    :param segm_idx (int).
    :return (M, D) tensor.
    '''
    pcl_ped = pcl[pcl[..., segm_idx] == 4]  # (P, 9).
    pcl_veh = pcl[pcl[..., segm_idx] == 10]  # (V, 9).
    pcl_vehped = torch.cat([pcl_ped, pcl_veh], dim=0)  # (P+V, 9).
    return pcl_vehped
