import time

import numba
import numpy as np
import math

@numba.jit(nopython=True)
def _spherical_points_to_voxel_reverse_kernel(spherical_points, 
                                    points,
                                    grid_size,
                                    coors_range,
                                    cartesian_coors_range,
                                    num_points_per_voxel,
                                    coor_to_voxelidx,
                                    voxels,
                                    coors,
                                    max_points=35,
                                    max_voxels=20000,
                                    RGB_embedding=None):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = spherical_points.shape[0]
    # ndim = spherical_points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1

    # theta_min, phi_min = coors_range[0], coors_range[1]
    # theta_range, phi_range = coors_range[3], coors_range[4]
    phi_min, theta_min = coors_range[0], coors_range[1]
    phi_range, theta_range = coors_range[3], coors_range[4]
    distance_max = coors_range[5]

    x_min, y_min, z_min = cartesian_coors_range[0], cartesian_coors_range[1], cartesian_coors_range[2]
    x_max, y_max, z_max = cartesian_coors_range[3], cartesian_coors_range[4], cartesian_coors_range[5]

    # normalize
    points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
    points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)
    points[:, 2] = (points[:, 2] - z_min) / (z_max - z_min)

    #
    # phi_max, phi_min = spherical_points[:, 0].max(), spherical_points[:, 0].min()
    # theta_max, theta_min = spherical_points[:, 1].max(), spherical_points[:, 1].min()
    # phi_range = phi_max - phi_min
    # theta_range = theta_max - theta_min

    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            if j == 0:
                c = np.floor((spherical_points[i, j] - phi_min) / phi_range * grid_size[0])
                # c = np.floor((spherical_points[i, j] - theta_min) / theta_range * grid_size[0])
            elif j == 1:
                c = np.floor((spherical_points[i, j] - theta_min) / theta_range * grid_size[1])
                # c = np.floor((spherical_points[i, j] - phi_min) / phi_range * grid_size[1])
            else:
                c = 0

            if c < 0 or c >= grid_size[j]:
                failed = True
                break

            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            spherical_points[i, 2] = spherical_points[i, 2] / distance_max
            if RGB_embedding is not None:
                voxels[voxelidx, num] = np.concatenate((points[i, :3], spherical_points[i, 2:], RGB_embedding[i, :]/255.0), axis=0)
            else:
                voxels[voxelidx, num] = np.concatenate((points[i, :3], spherical_points[i, 2:]), axis=0)
            num_points_per_voxel[voxelidx] += 1
    # sum = num_points_per_voxel.sum()
    return voxel_num

# @numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(points,
                                    voxel_size,
                                    coors_range,
                                    num_points_per_voxel,
                                    coor_to_voxelidx,
                                    voxels,
                                    coors,
                                    max_points=35,
                                    max_voxels=20000):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num

@numba.jit(nopython=True)
def _points_to_voxel_kernel(points,
                            voxel_size,
                            coors_range,
                            num_points_per_voxel,
                            coor_to_voxelidx,
                            voxels,
                            coors,
                            max_points=35,
                            max_voxels=20000):
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    lower_bound = coors_range[:3]
    upper_bound = coors_range[3:]
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num

def convert_to_spherical_coor(points):
    x = np.expand_dims(points[:, 0], axis=1)
    y = np.expand_dims(points[:, 1], axis=1)
    z = np.expand_dims(points[:, 2], axis=1)

    distance = np.sqrt(x * x + y * y + z * z)
    phi = np.arctan(y / x)
    theta = np.arccos(z / distance)

    if points.shape[1] == 4:
        etc = np.expand_dims(points[:, 3], axis=1)
        return np.concatenate((phi, theta, distance, etc), axis=1)
    elif points.shape[1] > 4:
        etc = points[:, 3:]
        return np.concatenate((phi, theta, distance, etc), axis=1)
    else:
        return np.concatenate((phi, theta, distance), axis=1)



def show_fv_map(spherical_points, image_size, coors_range):
    import cv2
    fvmap = np.zeros((image_size[0], image_size[1], 1))

    phi_max, phi_min = spherical_points[:, 0].max(), spherical_points[:, 0].min()
    theta_max, theta_min = spherical_points[:, 1].max(), spherical_points[:, 1].min()
    phi_range = phi_max - phi_min
    theta_range = theta_max - theta_min

    phi = np.floor((spherical_points[:, 0] - phi_min) / phi_range * image_size[1])
    theta = np.floor((spherical_points[:, 1] - theta_min) / theta_range * image_size[0])

    distance = spherical_points[:, 2]
    reflectance = spherical_points[:, 3]

    for i in range(fvmap.shape[0]):
        for j in range(fvmap.shape[1]):
            idx = (phi == j) & (theta == i)
            if np.count_nonzero(idx) > 0:
                fvmap[i, j, 0] = distance[idx].mean() / coors_range[-1]

    fvmap[fvmap != 0] = 1 - fvmap[fvmap != 0]

    cv2.imshow('fvmap', fvmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def points_to_voxel(points,
                    grid_size,
                     voxel_size,
                     coors_range,
                     cartesian_coors_range,
                     max_points=35,
                     reverse_index=True,
                     max_voxels=20000,
                     spherical_coor=False,
                     RGB_embedding=None):
    """convert kitti points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud) 
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        reverse_index: boolean. indicate whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output 
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor.
        num_points_per_voxel: [M] int32 tensor.
    """
    if spherical_coor:
        spherical_points = convert_to_spherical_coor(points)
        phi_min = spherical_points[:, 0].min()
        theta_min = spherical_points[:, 1].min()

        spherical_points[:, 0] = spherical_points[:, 0] - phi_min
        spherical_points[:, 1] = spherical_points[:, 1] - theta_min

        phi_max = spherical_points[:, 0].max()
        theta_max = spherical_points[:, 1].max()

        coors_range[3] = phi_max
        coors_range[4] = theta_max
        voxel_size = coors_range[3:] / grid_size


    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    if spherical_coor:
        voxelmap_shape = tuple(grid_size)
    else:
        voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
        voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())

    # if spherical_coor:
    #     show_fv_map(spherical_points, voxel_size, coors_range)

    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxel_num_channel = 5
    if RGB_embedding is not None:
        voxel_num_channel += 3
    voxels = np.zeros(
        shape=(max_voxels, max_points, voxel_num_channel), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    voxel_num = _spherical_points_to_voxel_reverse_kernel(
        spherical_points, points, grid_size, coors_range, cartesian_coors_range, num_points_per_voxel,
        coor_to_voxelidx, voxels, coors, max_points, max_voxels, RGB_embedding=RGB_embedding)

    # if reverse_index:
    #     if spherical_coor:
    #         voxel_num = _spherical_points_to_voxel_reverse_kernel(
    #             spherical_points, points, grid_size, coors_range, num_points_per_voxel,
    #             coor_to_voxelidx, voxels, coors, max_points, max_voxels)
    #     else:
    #         voxel_num = _points_to_voxel_reverse_kernel(
    #             points, voxel_size, coors_range, num_points_per_voxel,
    #             coor_to_voxelidx, voxels, coors, max_points, max_voxels)
    #
    # else:
    #     if spherical_coor:
    #         voxel_num = _spherical_points_to_voxel_reverse_kernel(
    #             spherical_points, grid_size, coors_range, num_points_per_voxel,
    #             coor_to_voxelidx, voxels, coors, max_points, max_voxels)
    #     else:
    #         voxel_num = _points_to_voxel_kernel(
    #             points, voxel_size, coors_range, num_points_per_voxel,
    #             coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    # voxels[:, :, -3:] = voxels[:, :, :3] - \
    #     voxels[:, :, :3].sum(axis=1, keepdims=True)/num_points_per_voxel.reshape(-1, 1, 1)
    return voxels, coors, num_points_per_voxel, phi_min, theta_min, voxel_size, coors_range


@numba.jit(nopython=True)
def bound_points_jit(points, upper_bound, lower_bound):
    # to use nopython=True, np.bool is not supported. so you need
    # convert result to np.bool after this function.
    N = points.shape[0]
    ndim = points.shape[1]
    keep_indices = np.zeros((N, ), dtype=np.int32)
    success = 0
    for i in range(N):
        success = 1
        for j in range(ndim):
            if points[i, j] < lower_bound[j] or points[i, j] >= upper_bound[j]:
                success = 0
                break
        keep_indices[i] = success
    return keep_indices
