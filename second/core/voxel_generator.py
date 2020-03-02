import numpy as np
from second.core.point_cloud.point_cloud_ops import points_to_voxel


class VoxelGenerator:
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        # voxel_size = np.array(voxel_size, dtype=np.float32)
        # grid_size = (
        #     point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        # grid_size = np.round(grid_size).astype(np.int64)
        grid_size = np.array(voxel_size, dtype=np.int64)
        spherical_coord_range = np.array([0, 0, 0, 1.5708, 0.37, 79.756], dtype=np.float32)
        voxel_size = spherical_coord_range[3:] / grid_size

        self._voxel_size = voxel_size
        self._point_cloud_range = spherical_coord_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size
        self._spherical_coor = True
        self._phi_min = 0
        self._theta_min = 0
        self._cartesian_coord_range = point_cloud_range

    def generate(self, points, max_voxels, RGB_embedding=False):
        voxels, coors, num_points_per_voxel, self._phi_min, self._theta_min, self._voxel_size, self._point_cloud_range = points_to_voxel(
            points, self._grid_size, self._voxel_size, self._point_cloud_range, self._cartesian_coord_range,
            self._max_num_points, True, max_voxels, self._spherical_coor, RGB_embedding=RGB_embedding)

        return voxels, coors, num_points_per_voxel

    @property
    def phi_min(self):
        return self._phi_min

    @property
    def theta_min(self):
        return self._theta_min

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points


    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size