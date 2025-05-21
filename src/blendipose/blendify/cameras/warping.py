import copy
import functools
import warnings

import cv2
import numba
import numpy as np


class Camera:
    def __init__(
            self, optical_center=None, rot_world_to_cam=None, intrinsic_matrix=np.eye(3),
            distortion_coeffs=None, world_up=(0, 0, 1), extrinsic_matrix=None,
            trans_after_rot=None, dtype=np.float32):
        """Pinhole camera with extrinsic and intrinsic calibration with optional distortions.

        The camera coordinate system has the following axes:
          x points to the right
          y points down
          z points forwards

        The world z direction is assumed to point up by default, but `world_up` can also be
         specified differently.

        Args:
            optical_center: position of the camera in world coordinates (eye point)
            rot_world_to_cam: 3x3 rotation matrix for transforming column vectors
                from being expressed in world reference frame to being expressed in camera
                reference frame as follows:
                column_point_cam = rot_matrix_world_to_cam @ (column_point_world - optical_center)
            intrinsic_matrix: 3x3 matrix that maps 3D points in camera space to homogeneous
                coordinates in image (pixel) space. Its last row must be (0,0,1).
            distortion_coeffs: parameters describing radial and tangential lens distortions,
                following OpenCV's model and order: k1, k2, p1, p2, k3 or None,
                if the camera has no distortion.
            world_up: a world vector that is designated as "pointing up".
            extrinsic_matrix: 4x4 extrinsic transformation matrix as an alternative to
                providing `optical_center` and `rot_world_to_cam`.
            trans_after_rot: translation vector to apply after the rotation
                (alternative to optical_center, which is a negative translation before the rotation)
        """

        if optical_center is not None and extrinsic_matrix is not None:
            raise Exception(
                'Provide only one of `optical_center`, `trans_after_rot` or `extrinsic_matrix`!')
        if extrinsic_matrix is not None and rot_world_to_cam is not None:
            raise Exception('Provide only one of `rot_world_to_cam` or `extrinsic_matrix`!')

        if (optical_center is None) and (trans_after_rot is None) and (extrinsic_matrix is None):
            optical_center = np.zeros(3, dtype=dtype)

        if (rot_world_to_cam is None) and (extrinsic_matrix is None):
            rot_world_to_cam = np.eye(3, dtype=dtype)

        if extrinsic_matrix is not None:
            self.R = np.asarray(extrinsic_matrix[:3, :3], dtype=dtype)
            self.t = -self.R.T @ extrinsic_matrix[:3, 3].astype(dtype)
        else:
            self.R = np.asarray(rot_world_to_cam, dtype=dtype)
            if optical_center is not None:
                self.t = np.asarray(optical_center, dtype=dtype)
            else:
                self.t = -self.R.T @ np.asarray(trans_after_rot, dtype=dtype)

        self.intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=dtype)
        if distortion_coeffs is None:
            self.distortion_coeffs = None
        else:
            self.distortion_coeffs = np.asarray(distortion_coeffs, dtype=dtype)
            if np.all(self.distortion_coeffs == 0):
                self.distortion_coeffs = None

        self.world_up = np.asarray(world_up, dtype=dtype)

        if not np.allclose(self.intrinsic_matrix[2, :], [0, 0, 1]):
            raise Exception(f'Bottom row of intrinsic matrix must be (0,0,1), '
                            f'got {self.intrinsic_matrix[2, :]}.')

    # Methods to transform between coordinate systems (world, camera, image)
    def camera_to_image(self, points):
        """Transform points from 3D camera coordinate space to image space."""

        if self.has_distortion():
            with warnings.catch_warnings():
                warnings.simplefilter(
                    'ignore', category=numba.errors.NumbaPerformanceWarning)
                return project_points(points, self.distortion_coeffs, self.intrinsic_matrix)
        else:
            projected = points[..., :2] / points[..., 2:]
            return projected @ self.intrinsic_matrix[:2, :2].T + self.intrinsic_matrix[:2, 2]

    def world_to_camera(self, points):
        return (points - self.t) @ self.R.T

    def camera_to_world(self, points):
        return points @ self.R + self.t

    def world_to_image(self, points):
        return self.camera_to_image(self.world_to_camera(points))

    def image_to_camera(self, points, depth=1):
        if not self.has_distortion():
            normalized_points = (
                ((points - self.intrinsic_matrix[:2, 2]) @
                 np.linalg.inv(self.intrinsic_matrix[:2, :2])))
            return cv2.convertPointsToHomogeneous(normalized_points)[:, 0, :] * depth

        points = np.expand_dims(np.asarray(points, np.float32), 0)

        if len(self.distortion_coeffs) == 4:
            new_image_points = cv2.fisheye.undistortPoints(
                points, self.intrinsic_matrix, self.distortion_coeffs, None, None, None)
            # To match the behavior of the 5-parameter undistortPoints, we transpose the result
            new_image_points = np.transpose(new_image_points, [1, 0, 2])
        else:
            new_image_points = cv2.undistortPoints(
                points, self.intrinsic_matrix, self.distortion_coeffs, None, None, None)
        return cv2.convertPointsToHomogeneous(new_image_points)[:, 0, :] * depth

    def image_to_world(self, points, camera_depth=1):
        return self.camera_to_world(self.image_to_camera(points, camera_depth))

    def scale_output(self, factor):
        """Adjusts the camera such that the images become scaled by `factor`. It's a scaling with
        the origin as anchor point.
        The difference with `self.zoom` is that this method also moves the principal point,
        multiplying its coordinates by `factor`."""
        self.intrinsic_matrix[:2] *= np.expand_dims(np.float32(factor), -1)

    def undistort(self, alpha_balance=None, new_imshape=None):
        if alpha_balance is not None and self.has_distortion():
            imsize = tuple(imshape[:2][::-1])
            new_imsize = imsize if new_imshape is None else tuple(new_imshape[:2][::-1])
            if len(self.distortion_coeffs) == 4:
                self.intrinsic_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    self.intrinsic_matrix, self.distortion_coeffs, imsize, np.eye(3),
                    new_size=new_imsize, balance=alpha_balance)
            else:
                self.intrinsic_matrix = cv2.getOptimalNewCameraMatrix(
                    self.intrinsic_matrix, self.distortion_coeffs, imsize, alpha_balance,
                    new_imsize, centerPrincipalPoint=False)[0]

        self.distortion_coeffs = None

    def square_pixels(self):
        """Adjusts the intrinsic matrix such that the pixels correspond to squares on the
        image plane."""
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        fmean = 0.5 * (fx + fy)
        multiplier = np.array([[fmean / fx, 0, 0], [0, fmean / fy, 0], [0, 0, 1]], np.float32)
        self.intrinsic_matrix = multiplier @ self.intrinsic_matrix

    def center_principal_point(self, imshape):
        """Adjusts the intrinsic matrix so that the principal point becomes located at the center
        of an image sized imshape (height, width)"""

        self.intrinsic_matrix[:2, 2] = np.float32([(imshape[1] - 1) / 2, (imshape[0] - 1) / 2])

    def get_distortion_coeffs(self):
        if self.distortion_coeffs is None:
            return np.zeros(shape=(5,), dtype=np.float32)
        return self.distortion_coeffs

    def has_distortion(self):
        return self.distortion_coeffs is not None and not np.all(self.distortion_coeffs == 0)

    def copy(self):
        return copy.deepcopy(self)

    @property
    def optical_center(self):
        return self.t


def reproject_image(
        image, old_camera, new_camera, output_imshape, border_mode=cv2.BORDER_CONSTANT,
        border_value=0, interp=None, antialias_factor=1, dst=None):
    """Transform an `image` captured with `old_camera` to look like it was captured by
    `new_camera`. The optical center (3D world position) of the cameras must be the same, otherwise
    we'd have parallax effects and no unambiguous way to construct the output.
    Ignores the issue of aliasing altogether.

    Args:
        image: the input image
        old_camera: the camera that captured `image`
        new_camera: the camera that should capture the newly returned image
        output_imshape: (height, width) for the output image
        border_mode: OpenCV border mode for treating pixels outside `image`
        border_value: OpenCV border value for treating pixels outside `image`
        interp: OpenCV interpolation to be used for resampling.
        antialias_factor: If larger than 1, first render a higher resolution output image
            that is `antialias_factor` times larger than `output_imshape` and subsequently resize
            it by 'area' interpolation to the desired size.
        dst: destination array (optional)

    Returns:
        The new image.
    """
    if antialias_factor == 1:
        return reproject_image_aliased(
            image, old_camera, new_camera, output_imshape, border_mode, border_value, interp,
            dst=dst)

    new_camera = new_camera.copy()
    a = antialias_factor
    new_camera.scale_output(a)
    new_camera.intrinsic_matrix[:2, 2] += (a - 1) / 2
    intermediate_imshape = (a * output_imshape[0], a * output_imshape[1])
    result = reproject_image_aliased(
        image, old_camera, new_camera, intermediate_imshape, border_mode, border_value, interp)
    return cv2.resize(
        result, dsize=(output_imshape[1], output_imshape[0]),
        interpolation=cv2.INTER_AREA, dst=dst)


def reproject_image_aliased(
        image, old_camera, new_camera, output_imshape, border_mode=cv2.BORDER_CONSTANT,
        border_value=0, interp=None, dst=None):
    """Transform an `image` captured with `old_camera` to look like it was captured by
    `new_camera`. The optical center (3D world position) of the cameras must be the same, otherwise
    we'd have parallax effects and no unambiguous way to construct the output.
    Aliasing issues are ignored.
    """

    if interp is None:
        interp = cv2.INTER_LINEAR

    map1, map2 = get_maps_for_remap(old_camera, new_camera, output_imshape)
    remapped = cv2.remap(
        image, map1, map2, interp, borderMode=border_mode, borderValue=border_value, dst=dst)
    if remapped.ndim < image.ndim:
        return np.expand_dims(remapped, -1)
    return remapped


@functools.lru_cache(maxsize=10)
def get_maps_for_remap(old_camera, new_camera, output_imshape):
    new_maps = get_grid_coords((output_imshape[0], output_imshape[1]))
    newim_coords = new_maps.reshape([-1, 2])
    if not new_camera.has_distortion():
        partial_homography = (
                old_camera.R @ np.linalg.inv(new_camera.R) @
                np.linalg.inv(new_camera.intrinsic_matrix))
        new_im_homogeneous = np.squeeze(cv2.convertPointsToHomogeneous(newim_coords), axis=1)
        old_camera_coords = new_im_homogeneous @ partial_homography.T
        oldim_coords = old_camera.camera_to_image(old_camera_coords)
    else:
        world_coords = new_camera.image_to_world(newim_coords)
        oldim_coords = old_camera.world_to_image(world_coords)
    old_maps = oldim_coords.reshape(new_maps.shape).astype(np.float32)
    map1 = old_maps[..., 0]
    map2 = old_maps[..., 1]
    return map1, map2


def allclose_or_nones(a, b):
    """Check if all corresponding values in arrays a and b are close to each other in the sense of
    np.allclose, or both a and b are None, or one is None and the other is filled with zeros.
    """

    if a is None and b is None:
        return True

    if a is None:
        return cv2.countNonZero(b) == 0

    if b is None:
        return cv2.countNonZero(a) == 0

    return np.allclose(a, b)


@numba.njit()
def project_points(points, dist_coeff, intrinsic_matrix):
    intrinsic_matrix = intrinsic_matrix.astype(np.float32)
    points = points.astype(np.float32)
    proj = points[..., :2] / points[..., 2:]

    if dist_coeff.shape[0] == 4:
        # Fisheye distortion
        # https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html
        # coefficients: k1, k2, k3, k4
        r = np.sqrt(np.sum(np.square(proj), axis=1))  # np.linalg.norm(proj, axis=1)
        t = np.arctan(r)
        t2 = np.square(t)
        t_d = (
                ((((dist_coeff[3] * t2 + dist_coeff[2]) * t2 + dist_coeff[1]) * t2 +
                  dist_coeff[0]) * t2 + np.float32(1.0)) * t)
        # with np.errstate(invalid='ignore'):
        proj *= np.nan_to_num(np.expand_dims(t_d / r, 1))
    else:
        # Standard distortion (radial, tangential, thin prism)
        # coefficient order: k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4
        r2 = np.sum(np.square(proj), axis=1)
        # dist_coeff = np.pad(dist_coeff, (0, 12 - len(dist_coeff)))
        dist_coeff = np.concatenate(
            (dist_coeff, np.zeros((12 - dist_coeff.shape[0],), dtype=np.float32)),
            axis=0)
        distorter = (
                ((((dist_coeff[4] * r2 + dist_coeff[1]) * r2 + dist_coeff[0]) * r2 + np.float32(
                    1.0)) /
                 (((dist_coeff[7] * r2 + dist_coeff[6]) * r2 + dist_coeff[5]) * r2 + np.float32(
                     1.0))) +
                + np.float32(2.0) * np.sum(proj * dist_coeff[3:1:-1], axis=1))

        proj[:] = (
                proj * np.expand_dims(distorter, 1) +
                (dist_coeff[9:12:2] * np.expand_dims(r2, 1) +
                 dist_coeff[8:11:2] + dist_coeff[3:1:-1]) * np.expand_dims(r2, 1))

    # Scheimpflug extension from opencv is not used
    # For that, one would need to implement tilt distortion
    # proj[:] = tilt_distort(proj, dist_coeff[12:14])
    return (proj @ intrinsic_matrix[:2, :2].T + intrinsic_matrix[:2, 2]).astype(np.float32)


@functools.lru_cache(5)
def get_grid_coords(output_imshape):
    """Return a meshgrid of coordinates for the image shape `output_imshape` (height, width).

    Returns
        Meshgrid of shape [height, width, 2], with the x and y coordinates (in this order)
            along the last dimension. DType float32.
    """
    y, x = np.mgrid[:output_imshape[0], :output_imshape[1]].astype(np.float32)
    return np.stack([x, y], axis=-1)
