import pathlib
import pickle
import time
from collections import defaultdict

import numpy as np
from skimage import io as imgio

from second.core import box_np_ops
from second.core import preprocess as prep
from second.core.geometry import points_in_convex_polygon_3d_jit
from second.core.point_cloud.bev_ops import points_to_bev
from second.data import kitti_common as kitti
from second.core.point_cloud.point_cloud_ops import convert_to_spherical_coor
from second.data.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian

import copy
import cv2
import math

def merge_second_batch(batch_list, _unused=False):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    # example_merged.pop("num_voxels")
    for key, elems in example_merged.items():
        if key in [
                'voxels', 'num_points', 'num_gt', 'gt_boxes', 'voxel_labels',
                'match_indices'
        ]:
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'match_indices_num':
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)),
                    mode='constant',
                    constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        else:
            ret[key] = np.stack(elems, axis=0)
    return ret


def filter_outside_range(spherical_gt_boxes, num_objs, fv_dim):
    mask1 = np.logical_or(spherical_gt_boxes[:, 0] >= fv_dim[0], spherical_gt_boxes[:, 1] >= fv_dim[1])
    mask2 = np.logical_or(spherical_gt_boxes[:, 0] < 0, spherical_gt_boxes[:, 1] < 0)
    mask = np.logical_or(mask1, mask2)
    delete_cnt = mask.astype(int).sum()
    if delete_cnt > 0:
        spherical_gt_boxes = spherical_gt_boxes[np.logical_not(mask),:]
        spherical_gt_boxes = np.append(spherical_gt_boxes, np.zeros((delete_cnt, 7), dtype=float), axis=0)

    return spherical_gt_boxes, num_objs-delete_cnt

def filter_outside_range2(points, fv_dim):
    mask1 = np.logical_or(points[:, 0] >= fv_dim[0], points[:, 1] >= fv_dim[1])
    mask2 = np.logical_or(points[:, 0] < 0, points[:, 1] < 0)
    mask = np.logical_or(mask1, mask2)
    return np.logical_not(mask)


def prep_pointcloud(input_dict,
                    root_path,
                    # voxel_generator,
                    fv_generator,
                    target_assigner,
                    db_sampler=None,
                    max_voxels=20000,
                    class_names=['Car'],
                    remove_outside_points=False,
                    training=True,
                    create_targets=True,
                    shuffle_points=False,
                    reduce_valid_area=False,
                    remove_unknown=False,
                    gt_rotation_noise=[-np.pi / 3, np.pi / 3],
                    gt_loc_noise_std=[1.0, 1.0, 1.0],
                    global_rotation_noise=[-np.pi / 4, np.pi / 4],
                    global_scaling_noise=[0.95, 1.05],
                    global_loc_noise_std=(0.2, 0.2, 0.2),
                    global_random_rot_range=[0.78, 2.35],
                    generate_bev=False,
                    without_reflectivity=False,
                    num_point_features=4,
                    anchor_area_threshold=1,
                    gt_points_drop=0.0,
                    gt_drop_max_keep=10,
                    remove_points_after_sample=False,
                    anchor_cache=None,
                    remove_environment=False,
                    random_crop=False,
                    reference_detections=None,
                    add_rgb_to_points=False,
                    lidar_input=False,
                    unlabeled_db_sampler=None,
                    out_size_factor=2,
                    min_gt_point_dict=None,
                    bev_only=False,
                    use_group_id=False,
                    out_dtype=np.float32,
                    num_classes=1,
                    RGB_embedding=False):
    """convert point cloud to voxels, create targets if ground truths 
    exists.
    """
    # prep_pointcloud_start = time.time()
    points = input_dict["points"]
    # if training:
    gt_boxes = input_dict["gt_boxes"]
    gt_names = input_dict["gt_names"]
    difficulty = input_dict["difficulty"]
    group_ids = None
    if use_group_id and "group_ids" in input_dict:
        group_ids = input_dict["group_ids"]

    rect = input_dict["rect"]
    Trv2c = input_dict["Trv2c"]
    P2 = input_dict["P2"]
    unlabeled_training = unlabeled_db_sampler is not None
    image_idx = input_dict["image_idx"]

    # t1 = time.time() - prep_pointcloud_start
    if shuffle_points:
        # shuffle is a little slow.
        np.random.shuffle(points)
    # t2 = time.time() - prep_pointcloud_start
    # print("t2-t1: ", t2-t1)     # 0.035

    if reference_detections is not None:
        C, R, T = box_np_ops.projection_matrix_to_CRT_kitti(P2)
        frustums = box_np_ops.get_frustum_v2(reference_detections, C)
        frustums -= T
        # frustums = np.linalg.inv(R) @ frustums.T
        frustums = np.einsum('ij, akj->aki', np.linalg.inv(R), frustums)
        frustums = box_np_ops.camera_to_lidar(frustums, rect, Trv2c)
        surfaces = box_np_ops.corner_to_surfaces_3d_jit(frustums)
        masks = points_in_convex_polygon_3d_jit(points, surfaces)
        points = points[masks.any(-1)]

    if remove_outside_points:# and not lidar_input:
        image_shape = input_dict["image_shape"]
        points = box_np_ops.remove_outside_points(points, rect, Trv2c, P2,
                                                  image_shape)
    if remove_environment is True:# and training:
        selected = kitti.keep_arrays_by_name(gt_names, class_names)
        gt_boxes = gt_boxes[selected]
        gt_names = gt_names[selected]
        difficulty = difficulty[selected]
        if group_ids is not None:
            group_ids = group_ids[selected]
        points = prep.remove_points_outside_boxes(points, gt_boxes)

    # if training:
        # print(gt_names)
    selected = kitti.drop_arrays_by_name(gt_names, ["DontCare"])
    gt_boxes = gt_boxes[selected]
    gt_names = gt_names[selected]
    difficulty = difficulty[selected]
    if group_ids is not None:
        group_ids = group_ids[selected]
    # t3 = time.time() - prep_pointcloud_start
    # print("t3-t2: ", t3 - t2)   # 0.0002
    gt_boxes = box_np_ops.box_camera_to_lidar(gt_boxes, rect, Trv2c)

    if remove_unknown:
        remove_mask = difficulty == -1
        """
        gt_boxes_remove = gt_boxes[remove_mask]
        gt_boxes_remove[:, 3:6] += 0.25
        points = prep.remove_points_in_boxes(points, gt_boxes_remove)
        """
        keep_mask = np.logical_not(remove_mask)
        gt_boxes = gt_boxes[keep_mask]
        gt_names = gt_names[keep_mask]
        difficulty = difficulty[keep_mask]
        if group_ids is not None:
            group_ids = group_ids[keep_mask]
    gt_boxes_mask = np.array(
        [n in class_names for n in gt_names], dtype=np.bool_)
    # t4 = time.time() - prep_pointcloud_start
    # print("t4-t3: ", t4 - t3)   # 0.001
    if RGB_embedding:
        RGB_image = cv2.imread(input_dict['image_path'])
        points_camera = box_np_ops.box_lidar_to_camera(points[:, :3], rect, Trv2c)
        points_to_image_idx = box_np_ops.project_to_image(points_camera, P2)
        points_to_image_idx = points_to_image_idx.astype(int)
        mask = box_np_ops.remove_points_outside_image(RGB_image, points_to_image_idx)
        points = points[mask]
        points_to_image_idx = points_to_image_idx[mask]
        BGR = RGB_image[points_to_image_idx[:,1], points_to_image_idx[:,0]]

        points = np.concatenate((points, BGR), axis=1)

        # test_mask = points_camera[mask][:, 0] < 0
        # test_image_idx = points_to_image_idx[test_mask]
        # RGB_image[test_image_idx[:, 1], test_image_idx[:, 0]] = [255, 0, 0]
        # test_mask = points_camera[mask][:, 0] >= 0
        # test_image_idx = points_to_image_idx[test_mask]
        # RGB_image[test_image_idx[:, 1], test_image_idx[:, 0]] = [0, 0, 255]
        # print()
    # t5 = time.time() - prep_pointcloud_start
    # print("t5-t4: ", t5 - t4)   # 0.019
    # TODO
    if db_sampler is not None and training:# and not RGB_embedding:
        if RGB_embedding:
            num_point_features += 3

        fg_points_mask = box_np_ops.points_in_rbbox(points, gt_boxes)
        fg_points_list = []
        bg_points_mask = np.zeros((points.shape[0]), dtype=bool)
        for i in range(fg_points_mask.shape[1]):
            fg_points_list.append(points[fg_points_mask[:, i]])
            bg_points_mask = np.logical_or(bg_points_mask, fg_points_mask[:, i])
        bg_points_mask = np.logical_not(bg_points_mask)
        sampled_dict = db_sampler.sample_all(
            root_path,
            points[bg_points_mask],
            gt_boxes,
            gt_names,
            fg_points_list,
            num_point_features,
            random_crop,
            gt_group_ids=group_ids,
            rect=rect,
            Trv2c=Trv2c,
            P2=P2)

        # sampled_dict = db_sampler.sample_all(
        #     root_path,
        #     gt_boxes,
        #     gt_names,
        #     num_point_features,
        #     random_crop,
        #     gt_group_ids=group_ids,
        #     rect=rect,
        #     Trv2c=Trv2c,
        #     P2=P2)
        # t_sample_all = time.time() - prep_pointcloud_start
        # print("t_sample_all - t5: ", t_sample_all - t5)     # 3.83

        if sampled_dict is not None:
            sampled_gt_names = sampled_dict["gt_names"]
            sampled_gt_boxes = sampled_dict["gt_boxes"]
            points = sampled_dict["points"]
            sampled_gt_masks = sampled_dict["gt_masks"]
            remained_boxes_idx = sampled_dict["remained_boxes_idx"]
            # gt_names = gt_names[gt_boxes_mask].tolist()
            gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
            # gt_names += [s["name"] for s in sampled]
            gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes])
            gt_boxes_mask = np.concatenate([gt_boxes_mask, sampled_gt_masks], axis=0)

            gt_names = gt_names[remained_boxes_idx]
            gt_boxes = gt_boxes[remained_boxes_idx]
            gt_boxes_mask = gt_boxes_mask[remained_boxes_idx]

            if group_ids is not None:
                sampled_group_ids = sampled_dict["group_ids"]
                group_ids = np.concatenate([group_ids, sampled_group_ids])
                group_ids = group_ids[remained_boxes_idx]

            # if remove_points_after_sample:
            #     # points = prep.remove_points_in_boxes(
            #     #     points, sampled_gt_boxes)
            #     locs = sampled_gt_boxes[:, 0:3]
            #     dims = sampled_gt_boxes[:, 3:6]
            #     angles = sampled_gt_boxes[:, 6]
            #     camera_box_origin = [0.5, 0.5, 0]
            #
            #     box_corners = box_np_ops.center_to_corner_box3d(
            #         locs, dims, angles, camera_box_origin, axis=2)
            #     box_corners_in_image = box_np_ops.project_to_fv_image(
            #         box_corners, example['grid_size'][i], example['meta'][i])
            #     box_centers_in_image = box_np_ops.project_to_fv_image(
            #         locs, example['grid_size'][i], example['meta'][i])

            # t_sample_all2 = time.time() - prep_pointcloud_start
            # print("t_sample_all2 - t_sample_all: ", t_sample_all2 - t_sample_all)   # 0.0002


    # unlabeled_mask = np.zeros((gt_boxes.shape[0], ), dtype=np.bool_)
    # if without_reflectivity and training:
    #     used_point_axes = list(range(num_point_features))
    #     used_point_axes.pop(3)
    #     points = points[:, used_point_axes]
    # pc_range = voxel_generator.point_cloud_range
    # bev_only = False
    # if bev_only:  # set z and h to limits
    #     gt_boxes[:, 2] = pc_range[2]
    #     gt_boxes[:, 5] = pc_range[5] - pc_range[2]
    if training:
        gt_loc_noise_std = [0.0, 0.0, 0.0]
        prep.noise_per_object_v3_(
            gt_boxes,
            points,
            gt_boxes_mask,
            rotation_perturb=gt_rotation_noise,
            center_noise_std=gt_loc_noise_std,
            global_random_rot_range=global_random_rot_range,
            group_ids=group_ids,
            num_try=100)
        # t_noise = time.time() - prep_pointcloud_start
        # print("t_noise - t_sample_all2: ", t_noise - t_sample_all2)     # 12.01
    # should remove unrelated objects after noise per object
    gt_boxes = gt_boxes[gt_boxes_mask]
    gt_names = gt_names[gt_boxes_mask]
    if group_ids is not None:
        group_ids = group_ids[gt_boxes_mask]
    gt_classes = np.array(
        [class_names.index(n) + 1 for n in gt_names], dtype=np.int32)

    # t6 = time.time() - prep_pointcloud_start
    # print("t6-t5: ", t6 - t5)   # 16.0

    if training:
        gt_boxes, points = prep.random_flip(gt_boxes, points)
        # gt_boxes, points = prep.global_rotation(
        #     gt_boxes, points, rotation=global_rotation_noise)
        gt_boxes, points = prep.global_scaling_v2(gt_boxes, points,
                                                  *global_scaling_noise)
        # Global translation
        # gt_boxes, points = prep.global_translate(gt_boxes, points, global_loc_noise_std)

    # bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    bv_range = [0, -40, 70.4, 40]
    mask = prep.filter_gt_box_outside_range(gt_boxes, bv_range)
    gt_boxes = gt_boxes[mask]
    gt_classes = gt_classes[mask]
    if group_ids is not None:
        group_ids = group_ids[mask]

    # limit rad to [-pi, pi]
    gt_boxes[:, 6] = box_np_ops.limit_period(
        gt_boxes[:, 6], offset=0.5, period=2 * np.pi)

    # TODO
    # if shuffle_points:
    #     # shuffle is a little slow.
    #     np.random.shuffle(points)

    # # t7 = time.time() - prep_pointcloud_start
    # # print("t7-t6: ", t7 - t6)   # 1.95
    # voxels, coordinates, num_points = voxel_generator.generate(
    #     points, max_voxels, RGB_embedding=RGB_embedding)
    # # t8 = time.time() - prep_pointcloud_start
    # # print("t8-t7: ", t8 - t7)   # 2.0
    # voxel_size = voxel_generator.voxel_size
    # grid_size = voxel_generator.grid_size
    # pc_range = copy.deepcopy(voxel_generator.point_cloud_range)
    # grid_size = voxel_generator.grid_size
    # phi_min = voxel_generator.phi_min
    # theta_min = voxel_generator.theta_min
    # image_h, image_w = grid_size[1], grid_size[0]
    # c = np.array([image_w / 2., image_h / 2.])
    # s = np.array([image_w, image_h], dtype=np.int32)
    # meta = {'c': c, 's': s, 'calib': P2, 'phi_min': phi_min, 'theta_min': theta_min}

    # t7 = time.time() - prep_pointcloud_start
    # print("t7-t6: ", t7 - t6)   # 1.95
    fv_image, points_mask = fv_generator.generate(points, RGB_embedding=RGB_embedding, occupancy_embedding=False)

    # t8 = time.time() - prep_pointcloud_start
    # print("t8-t7: ", t8 - t7)   # 2.0

    fv_dim = fv_generator.fv_dim
    pc_range = copy.deepcopy(fv_generator.spherical_coord_range)
    grid_size = fv_generator.grid_size
    phi_min = fv_generator.phi_min
    theta_min = fv_generator.theta_min
    image_h, image_w = fv_dim[1], fv_dim[0]
    c = np.array([image_w / 2., image_h / 2.])
    s = np.array([image_w, image_h], dtype=np.int32)
    meta = {'c': c, 's': s, 'calib': P2, 'phi_min': phi_min, 'theta_min': theta_min}

    fv_image = np.transpose(fv_image, [2, 1, 0])
    max_objs = 50
    num_objs = min(gt_boxes.shape[0], max_objs)

    box_np_ops.change_box3d_center_(gt_boxes, src=[0.5, 0.5, 0], dst=[0.5, 0.5, 0.5])
    spherical_gt_boxes = np.zeros((max_objs, gt_boxes.shape[1]))
    spherical_gt_boxes[:num_objs, :] = gt_boxes[:num_objs, :]
    spherical_gt_boxes[:num_objs, :] = convert_to_spherical_coor(gt_boxes[:num_objs, :])
    spherical_gt_boxes[:num_objs, 0] -= phi_min
    spherical_gt_boxes[:num_objs, 1] -= theta_min
    spherical_gt_boxes[:num_objs, 0] /= grid_size[0]
    spherical_gt_boxes[:num_objs, 1] /= grid_size[1]

    spherical_gt_boxes, num_objs = filter_outside_range(spherical_gt_boxes, num_objs, fv_dim)

    # t9 = time.time() - prep_pointcloud_start
    # print("t9-t8: ", t9 - t8)   # 0.0005
    example = {
        'fv_image': fv_image,
        'grid_size': grid_size,
        'pc_range': pc_range,
        'meta': meta,
        'spherical_gt_boxes': spherical_gt_boxes,
        'resized_image_shape': grid_size
    }

    example.update({
        'rect': rect,
        'Trv2c': Trv2c,
        'P2': P2
    })
    if RGB_embedding:
        RGB_image = cv2.resize(RGB_image, (image_w, image_h))
        example.update({
            'RGB_image': RGB_image
        })

    if training:
        hm = np.zeros(
            (num_classes, image_h, image_w), dtype=np.float32)
        reg = np.zeros((max_objs, 2), dtype=np.float32)
        dep = np.zeros((max_objs, 1), dtype=np.float32)
        rotbin = np.zeros((max_objs, 2), dtype=np.int64)
        rotres = np.zeros((max_objs, 2), dtype=np.float32)
        dim = np.zeros((max_objs, 3), dtype=np.float32)
        ind = np.zeros((max_objs), dtype=np.int64)
        reg_mask = np.zeros((max_objs), dtype=np.uint8)
        rot_mask = np.zeros((max_objs), dtype=np.uint8)
        #
        # hm = np.zeros((num_classes, image_h, image_w), dtype=np.float32)
        # reg = np.zeros((image_w, image_h, 2), dtype=np.float32)
        # dep = np.zeros((image_w, image_h, 1), dtype=np.float32)
        # rotbin = np.zeros((image_w, image_h, 2), dtype=np.int64)
        # rotres = np.zeros((image_w, image_h, 2), dtype=np.float32)
        # dim = np.zeros((image_w, image_h, 3), dtype=np.float32)
        # # ind = np.zeros((max_objs), dtype=np.int64)
        # fg_mask = np.zeros((image_w, image_h), dtype=np.uint8)
        # # rot_mask = np.zeros((max_objs), dtype=np.uint8)

        draw_gaussian = draw_umich_gaussian
        # center heatmap
        radius = int(image_h / 30)
        # radius = int(image_h / 25)

        for k in range(num_objs):
            gt_3d_box = spherical_gt_boxes[k]
            cls_id = 0

            # print('heatmap gaussian radius: ' + str(radius))
            ct = np.array([gt_3d_box[0], gt_3d_box[1]], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_gaussian(hm[cls_id], ct, radius)

            # depth(distance), wlh
            dep[k] = gt_3d_box[2]
            dim[k] = gt_3d_box[3:6]
            # dep[ct_int[0], ct_int[1], 0] = gt_3d_box[2]
            # dim[ct_int[0], ct_int[1], :] = gt_3d_box[3:6]


            # reg, ind, mask
            reg[k] = ct - ct_int
            ind[k] = ct_int[1] * image_w + ct_int[0]
            reg_mask[k] = rot_mask[k] = 1
            # fg_mask[ct_int[0], ct_int[1]] = 1


            # ry
            ry = gt_3d_box[6]
            if ry < np.pi / 6. or ry > 5 * np.pi / 6.:
                rotbin[k, 0] = 1
                rotres[k, 0] = ry - (-0.5 * np.pi)
                # rotbin[ct_int[0], ct_int[1], 0] = 1
                # rotres[ct_int[0], ct_int[1], 0] = ry - (-0.5 * np.pi)
            if ry > -np.pi / 6. or ry < -5 * np.pi / 6.:
                rotbin[k, 1] = 1
                rotres[k, 1] = ry - (0.5 * np.pi)
                # rotbin[ct_int[0], ct_int[1], 1] = 1
                # rotres[ct_int[0], ct_int[1], 1] = ry - (0.5 * np.pi)

        example.update({
            'hm': hm, 'dep': dep, 'dim': dim, 'ind': ind,
            'rotbin': rotbin, 'rotres': rotres, 'reg_mask': reg_mask,
            'rot_mask': rot_mask, 'reg': reg
        })
        # example.update({
        #     'hm': hm, 'dep': dep, 'dim': dim,
        #     'rotbin': rotbin, 'rotres': rotres,
        #     'fg_mask': fg_mask, 'reg': reg
        # })

    # t10 = time.time() - prep_pointcloud_start
    # print("total: ", t10)       # 19.58
    return example


def _read_and_prep_v9(info, root_path, num_point_features, prep_func):
    """read data from KITTI-format infos, then call prep function.
    """
    # velodyne_path = str(pathlib.Path(root_path) / info['velodyne_path'])
    # velodyne_path += '_reduced'
    v_path = pathlib.Path(root_path) / info['velodyne_path']
    v_path = v_path.parent.parent / (
        v_path.parent.stem + "_reduced") / v_path.name

    points = np.fromfile(
        str(v_path), dtype=np.float32,
        count=-1).reshape([-1, num_point_features])
    image_idx = info['image_idx']
    rect = info['calib/R0_rect'].astype(np.float32)
    Trv2c = info['calib/Tr_velo_to_cam'].astype(np.float32)
    P2 = info['calib/P2'].astype(np.float32)

    input_dict = {
        'points': points,
        'rect': rect,
        'Trv2c': Trv2c,
        'P2': P2,
        'image_shape': np.array(info["img_shape"], dtype=np.int32),
        'image_idx': image_idx,
        'image_path': root_path + '/' + info['img_path'],
        # 'pointcloud_num_features': num_point_features,
    }

    if 'annos' in info:
        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = kitti.remove_dontcare(annos)
        loc = annos["location"]
        dims = annos["dimensions"]
        rots = annos["rotation_y"]
        # alpha = annos["alpha"]
        gt_names = annos["name"]
        # print(gt_names, len(loc))
        gt_boxes = np.concatenate(
            [loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
        # gt_boxes = np.concatenate(
        #     [loc, dims, alpha[..., np.newaxis]], axis=1).astype(np.float32)
        # gt_boxes = box_np_ops.box_camera_to_lidar(gt_boxes, rect, Trv2c)
        difficulty = annos["difficulty"]
        input_dict.update({
            'gt_boxes': gt_boxes,
            'gt_names': gt_names,
            'difficulty': difficulty,
        })
        if 'group_ids' in annos:
            input_dict['group_ids'] = annos["group_ids"]
    example = prep_func(input_dict=input_dict)
    example["image_idx"] = image_idx
    example["image_shape"] = input_dict["image_shape"]
    if "anchors_mask" in example:
        example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
    return example

