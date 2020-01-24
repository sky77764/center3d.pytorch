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

def merge_second_batch(batch_list, _unused=False):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    example_merged.pop("num_voxels")
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


def filter_outside_range(spherical_gt_boxes, num_objs, grid_size):
    mask = np.logical_or(spherical_gt_boxes[:, 0] > grid_size[0], spherical_gt_boxes[:, 1] > grid_size[1])
    delete_cnt = mask.astype(int).sum()
    if delete_cnt > 0:
        spherical_gt_boxes = spherical_gt_boxes[np.logical_not(mask),:]
        spherical_gt_boxes = np.append(spherical_gt_boxes, np.zeros((delete_cnt, 7), dtype=float), axis=0)

    return spherical_gt_boxes, num_objs-delete_cnt



def prep_pointcloud(input_dict,
                    root_path,
                    voxel_generator,
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
                    remove_points_after_sample=True,
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
                    num_classes=1):
    """convert point cloud to voxels, create targets if ground truths 
    exists.
    """
    points = input_dict["points"]
    if training:
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

    if remove_outside_points and not lidar_input:
        image_shape = input_dict["image_shape"]
        points = box_np_ops.remove_outside_points(points, rect, Trv2c, P2,
                                                  image_shape)
    if remove_environment is True and training:
        selected = kitti.keep_arrays_by_name(gt_names, class_names)
        gt_boxes = gt_boxes[selected]
        gt_names = gt_names[selected]
        difficulty = difficulty[selected]
        if group_ids is not None:
            group_ids = group_ids[selected]
        points = prep.remove_points_outside_boxes(points, gt_boxes)
    if training:
        # print(gt_names)
        selected = kitti.drop_arrays_by_name(gt_names, ["DontCare"])
        gt_boxes = gt_boxes[selected]
        gt_names = gt_names[selected]
        difficulty = difficulty[selected]
        if group_ids is not None:
            group_ids = group_ids[selected]

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
        if db_sampler is not None:
            sampled_dict = db_sampler.sample_all(
                root_path,
                gt_boxes,
                gt_names,
                num_point_features,
                random_crop,
                gt_group_ids=group_ids,
                rect=rect,
                Trv2c=Trv2c,
                P2=P2)

            if sampled_dict is not None:
                sampled_gt_names = sampled_dict["gt_names"]
                sampled_gt_boxes = sampled_dict["gt_boxes"]
                sampled_points = sampled_dict["points"]
                sampled_gt_masks = sampled_dict["gt_masks"]
                # gt_names = gt_names[gt_boxes_mask].tolist()
                gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
                # gt_names += [s["name"] for s in sampled]
                gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes])
                gt_boxes_mask = np.concatenate(
                    [gt_boxes_mask, sampled_gt_masks], axis=0)
                if group_ids is not None:
                    sampled_group_ids = sampled_dict["group_ids"]
                    group_ids = np.concatenate([group_ids, sampled_group_ids])

                if remove_points_after_sample:
                    points = prep.remove_points_in_boxes(
                        points, sampled_gt_boxes)

                points = np.concatenate([sampled_points, points], axis=0)
        # unlabeled_mask = np.zeros((gt_boxes.shape[0], ), dtype=np.bool_)
        if without_reflectivity:
            used_point_axes = list(range(num_point_features))
            used_point_axes.pop(3)
            points = points[:, used_point_axes]
        # pc_range = voxel_generator.point_cloud_range
        # bev_only = False
        # if bev_only:  # set z and h to limits
        #     gt_boxes[:, 2] = pc_range[2]
        #     gt_boxes[:, 5] = pc_range[5] - pc_range[2]
        prep.noise_per_object_v3_(
            gt_boxes,
            points,
            gt_boxes_mask,
            rotation_perturb=gt_rotation_noise,
            center_noise_std=gt_loc_noise_std,
            global_random_rot_range=global_random_rot_range,
            group_ids=group_ids,
            num_try=100)
        # should remove unrelated objects after noise per object
        gt_boxes = gt_boxes[gt_boxes_mask]
        gt_names = gt_names[gt_boxes_mask]
        if group_ids is not None:
            group_ids = group_ids[gt_boxes_mask]
        gt_classes = np.array(
            [class_names.index(n) + 1 for n in gt_names], dtype=np.int32)

        gt_boxes, points = prep.random_flip(gt_boxes, points)
        # gt_boxes, points = prep.global_rotation(
        #     gt_boxes, points, rotation=global_rotation_noise)
        # gt_boxes, points = prep.global_scaling_v2(gt_boxes, points,
        #                                           *global_scaling_noise)

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

    if shuffle_points:
        # shuffle is a little slow.
        np.random.shuffle(points)

    voxels, coordinates, num_points = voxel_generator.generate(
        points, max_voxels)

    voxel_size = voxel_generator.voxel_size
    pc_range = copy.deepcopy(voxel_generator.point_cloud_range)
    grid_size = voxel_generator.grid_size
    image_h, image_w = grid_size[1], grid_size[0]

    max_objs = 200
    num_objs = min(gt_boxes.shape[0], max_objs)

    box_np_ops.change_box3d_center_(gt_boxes, src=[0.5, 0.5, 0], dst=[0.5, 0.5, 0.5])
    spherical_gt_boxes = np.zeros((max_objs, gt_boxes.shape[1]))
    spherical_gt_boxes[:num_objs, :] = gt_boxes[:num_objs, :]
    spherical_gt_boxes[:num_objs, :] = convert_to_spherical_coor(gt_boxes[:num_objs, :])
    spherical_gt_boxes[:num_objs, 0] -= voxel_generator.phi_min
    spherical_gt_boxes[:num_objs, 1] -= voxel_generator.theta_min
    spherical_gt_boxes[:num_objs, 0] /= voxel_size[0]
    spherical_gt_boxes[:num_objs, 1] /= voxel_size[1]

    spherical_gt_boxes, num_objs = filter_outside_range(spherical_gt_boxes, num_objs, grid_size)

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


    draw_gaussian = draw_umich_gaussian

    for k in range(num_objs):
        gt_3d_box = spherical_gt_boxes[k]
        cls_id = 0

        # center heatmap
        radius = int(image_h / 30)
        ct = np.array([gt_3d_box[0], gt_3d_box[1]], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[cls_id], ct, radius)

        # depth(distance), wlh
        dep[k] = gt_3d_box[2]
        dim[k] = gt_3d_box[3:6]

        # reg, ind, mask
        reg[k] = ct - ct_int
        ind[k] = ct_int[1] * image_w + ct_int[0]
        reg_mask[k] = rot_mask[k] = 1

        # TODO: check alpha ry
        alpha = gt_3d_box[6]
        if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
            rotbin[k, 0] = 1
            rotres[k, 0] = alpha - (-0.5 * np.pi)
        if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
            rotbin[k, 1] = 1
            rotres[k, 1] = alpha - (0.5 * np.pi)

    example = {
        'voxels': voxels,
        'num_points': num_points,
        'coordinates': coordinates,
        "num_voxels": np.array([voxels.shape[0]], dtype=np.int64),
        'voxel_size': voxel_size,
        'pc_range': pc_range,
        'hm': hm, 'dep': dep, 'dim': dim, 'ind': ind,
        'rotbin': rotbin, 'rotres': rotres, 'reg_mask': reg_mask,
        'rot_mask': rot_mask, 'reg': reg
    }
    example.update({
        'rect': rect,
        'Trv2c': Trv2c,
        'P2': P2
    })




    # if not lidar_input:
    # feature_map_size = grid_size[:2] // out_size_factor
    # feature_map_size = [*feature_map_size, 1][::-1]
    # if anchor_cache is not None:
    #     anchors = anchor_cache["anchors"]
    #     anchors_bv = anchor_cache["anchors_bv"]
    #     matched_thresholds = anchor_cache["matched_thresholds"]
    #     unmatched_thresholds = anchor_cache["unmatched_thresholds"]
    # else:
    #     ret = target_assigner.generate_anchors(feature_map_size)
    #     anchors = ret["anchors"]
    #     anchors = anchors.reshape([-1, 7])
    #     matched_thresholds = ret["matched_thresholds"]
    #     unmatched_thresholds = ret["unmatched_thresholds"]
    #     anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
    #         anchors[:, [0, 1, 3, 4, 6]])
    # example["anchors"] = anchors
    # print("debug", anchors.shape, matched_thresholds.shape)
    # anchors_bv = anchors_bv.reshape([-1, 4])
    # anchors_mask = None
    # if anchor_area_threshold >= 0:
    #     coors = coordinates
    #     dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
    #         coors, tuple(grid_size[::-1][1:]))
    #     dense_voxel_map = dense_voxel_map.cumsum(0)
    #     dense_voxel_map = dense_voxel_map.cumsum(1)
    #     anchors_area = box_np_ops.fused_get_anchors_area(
    #         dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size)
    #     anchors_mask = anchors_area > anchor_area_threshold
    #     # example['anchors_mask'] = anchors_mask.astype(np.uint8)
    #     example['anchors_mask'] = anchors_mask
    # if generate_bev:
    #     bev_vxsize = voxel_size.copy()
    #     bev_vxsize[:2] /= 2
    #     bev_vxsize[2] *= 2
    #     bev_map = points_to_bev(points, bev_vxsize, pc_range,
    #                             without_reflectivity)
    #     example["bev_map"] = bev_map
    # if not training:
    #     return example
    # if create_targets:
    #     targets_dict = target_assigner.assign(
    #         anchors,
    #         gt_boxes,
    #         anchors_mask,
    #         gt_classes=gt_classes,
    #         matched_thresholds=matched_thresholds,
    #         unmatched_thresholds=unmatched_thresholds)
    #     example.update({
    #         'labels': targets_dict['labels'],
    #         'reg_targets': targets_dict['bbox_targets'],
    #         'reg_weights': targets_dict['bbox_outside_weights'],
    #     })
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
        'image_path': info['img_path'],
        # 'pointcloud_num_features': num_point_features,
    }

    if 'annos' in info:
        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = kitti.remove_dontcare(annos)
        loc = annos["location"]
        dims = annos["dimensions"]
        rots = annos["rotation_y"]
        gt_names = annos["name"]
        # print(gt_names, len(loc))
        gt_boxes = np.concatenate(
            [loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
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

