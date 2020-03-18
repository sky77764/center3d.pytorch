import numpy as np
from second.core.point_cloud.point_cloud_ops import points_to_voxel


def convert_to_spherical_coord(points):
    x = np.expand_dims(points[:, 0], axis=1)
    y = np.expand_dims(points[:, 1], axis=1)
    z = np.expand_dims(points[:, 2], axis=1)

    distance = np.sqrt(x * x + y * y + z * z)
    phi = np.arctan(y / x)
    theta = np.arccos(z / distance)

    return np.concatenate((phi, theta, distance), axis=1)

def remove_outside_points(fv_image_idx, fv_dim):
    mask1 = np.logical_and(fv_image_idx[:, 0] < fv_dim[0], fv_image_idx[:, 1] < fv_dim[1])
    mask2 = np.logical_and(fv_image_idx[:, 0] >= 0, fv_image_idx[:, 1] >= 0)
    mask = np.logical_and(mask1, mask2)
    return mask


class FrontviewGenerator:
    def __init__(self,
                 fv_dim,
                 cartesian_coord_range,
                 input_normalization=True):

        self._spherical_coord_range = np.array([0, 0, 0, 1.5708, 0.37, 79.756], dtype=np.float32)
        self._fv_dim = np.array(fv_dim, dtype=np.int32)     # H, W, C
        self._grid_size = np.array([0, 0], dtype=np.float32)    # phi, theta
        self._phi_min = 0
        self._theta_min = 0
        self._cartesian_coord_range = np.array(cartesian_coord_range, dtype=np.float32)
        self._input_normalization = input_normalization
        self._mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3)
        self._std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3)

    def generate(self, points, RGB_embedding=False, occupancy_embedding=False):
        # num_points = points.shape[0]
        spherical_points = convert_to_spherical_coord(points)
        self._phi_min = spherical_points[:, 0].min()
        self._theta_min = spherical_points[:, 1].min()

        spherical_points[:, 0] = spherical_points[:, 0] - self._phi_min
        spherical_points[:, 1] = spherical_points[:, 1] - self._theta_min

        phi_max = spherical_points[:, 0].max()
        theta_max = spherical_points[:, 1].max()

        self._spherical_coord_range[3] = phi_max
        self._spherical_coord_range[4] = theta_max
        self._grid_size[0] = self._spherical_coord_range[3] / self._fv_dim[0]
        self._grid_size[1] = self._spherical_coord_range[4] / self._fv_dim[1]

        fv_image_idx = np.zeros((spherical_points.shape[0], 2), dtype=np.int32)
        fv_image_idx[:, 0] = np.floor(spherical_points[:, 0] / phi_max * self._fv_dim[0])  # w index
        fv_image_idx[:, 1] = np.floor(spherical_points[:, 1] / theta_max * self._fv_dim[1])  # h index

        mask = remove_outside_points(fv_image_idx, self._fv_dim)
        points = points[mask]
        fv_image_idx = fv_image_idx[mask]

        if self._input_normalization:
            # xyz normalize
            points[:, 0:3] = (points[:, 0:3] - self._cartesian_coord_range[0:3]) / (self._cartesian_coord_range[3:6] - self._cartesian_coord_range[0:3])
            # RGB normalize
            if RGB_embedding:
                points[:, 4:] = (points[:, 4:] / 255 - self._mean) / self._std

        if occupancy_embedding:
            channel_embedding = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1) # xyzr(bgr)o
        else:
            channel_embedding = points  # xyzr(bgr)

        fv_image = np.zeros(self._fv_dim, dtype=np.float32)
        # num_occupancy = 0
        for i, idx in enumerate(fv_image_idx):
            # if not any(fv_image[idx[0], idx[1]]):
            #     num_occupancy += 1
            fv_image[idx[0], idx[1]] = channel_embedding[i]
        # print("num_points: {}, num_occupancy: {} ({}%), num_pixel: {} ({}%)".format(num_points, num_occupancy, num_occupancy/num_points*100,
        #       fv_image.shape[0]*fv_image.shape[1], num_occupancy/(fv_image.shape[0]*fv_image.shape[1])*100))
        return fv_image, mask

    @property
    def phi_min(self):
        return self._phi_min

    @property
    def theta_min(self):
        return self._theta_min

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def grid_size(self):
        return self._grid_size

    @property
    def spherical_coord_range(self):
        return self._spherical_coord_range

    @property
    def cartesian_coord_range(self):
        return self._cartesian_coord_range

    @property
    def fv_dim(self):
        return self._fv_dim