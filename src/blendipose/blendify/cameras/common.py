from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import Camera
from ..internal.types import RotationMode

if TYPE_CHECKING:
    from ..internal.types import Vector2d, Vector2di, Vector3d, RotationParams


class PerspectiveCamera(Camera):
    def __init__(
        self,
        resolution: Vector2di = (1920, 1080),
        focal_dist: float = None,
        fov_x: float = None,
        fov_y: float = None,
        center: Vector2d = None,
        near: float = 0.1,
        far: float = 100,
        tag: str = 'camera',
        rotation_mode: RotationMode = "quaternionWXYZ",
        rotation: RotationParams = None,
        translation: Vector3d = (0, 0, 0),
    ):
        num_given = sum(val is not None for val in (focal_dist, fov_x, fov_y))
        if num_given == 0:
            raise ValueError("One of focal_dist, fov_x or fov_y is required")
        if num_given > 1:
            raise ValueError("Only one of focal_dist, fov_x or fov_y may be given")
        super().__init__(
            resolution=resolution, near=near, far=far, tag=tag,
            rotation_mode=rotation_mode, rotation=rotation, translation=translation,
        )
        camera_object = self.blender_camera
        camera_object.data.type = 'PERSP'
        # camera.data.lens_unit = "FOV"
        if focal_dist is not None:
            self.focal_dist = focal_dist
        if center is not None:
            self.center = center
        if fov_x is not None:
            self.fov_x = fov_x
        if fov_y is not None:
            self.fov_y = fov_y

    @property
    def focal_dist(self):
        return self.blender_camera.data.lens

    @focal_dist.setter
    def focal_dist(self, focal: float):
        self.blender_camera.data.lens = focal

    @property
    def fov_x(self):
        return self.blender_camera.data.angle_x

    @fov_x.setter
    def fov_x(self, val: float):
        self.blender_camera.data.angle_x = val

    @property
    def fov_y(self):
        return self.blender_camera.data.angle_y

    @fov_y.setter
    def fov_y(self, val: float):
        self.blender_camera.data.angle_y = val

    @property
    def center(self) -> np.ndarray:
        """Principal point in pixel coordinates (cx, cy)."""
        camera = self.blender_camera
        W, H = self.resolution
        cx = (0.5 - camera.data.shift_x) * W
        cy = (0.5 + camera.data.shift_y * W / H) * H
        return np.array([cx, cy])

    @center.setter
    def center(self, center: Vector2d):
        """Set principal point in pixel coordinates (cx, cy), OpenCV convention.

        Blender shift convention (with sensor_fit='HORIZONTAL'):
        - shift_x > 0 moves camera RIGHT, objects shift LEFT, so cx decreases
        - shift_y > 0 moves camera UP, objects shift DOWN (y-down), so cy increases
        - Both shifts are in units of sensor_width (= resolution width with our setup)
        """
        camera = self.blender_camera
        cx, cy = np.asarray(center, dtype=np.float64)
        W, H = self.resolution
        camera.data.shift_x = 0.5 - cx / W
        camera.data.shift_y = (cy / H - 0.5) * (H / W)

    def distance2depth(self, distmap: np.ndarray) -> np.ndarray:
        """Convert map of camera ray lengths (distmap) to map of distances to image plane (depthmap)

        Args:
            distmap (np.ndarray): Distance map

        Returns:
            np.ndarray: Depth map
        """
        img_width, img_height = self.resolution
        cx, cy = self.center
        offsets_x = np.arange(img_width) + 0.5 - cx
        offsets_y = np.arange(img_height) + 0.5 - cy
        grid_offsets_x, grid_offsets_y = np.meshgrid(offsets_x, offsets_y)
        depthmap = np.sqrt(distmap ** 2 / ((grid_offsets_x ** 2 + grid_offsets_y ** 2) / (self.focal_dist ** 2) + 1))
        return depthmap



class OrthographicCamera(Camera):
    def __init__(
        self,
        resolution: Vector2di = (1920, 1080),
        ortho_scale: float = 1.,
        near: float = 0.1,
        far: float = 100,
        tag: str = 'camera',
        rotation_mode: RotationMode = "quaternionWXYZ",
        rotation: RotationParams = None,
        translation: Vector3d = (0, 0, 0),
    ):
        super().__init__(
            resolution=resolution, near=near, far=far, tag=tag,
            rotation_mode=rotation_mode, rotation=rotation, translation=translation,
        )
        camera_object = self.blender_camera
        camera_object.data.type = 'ORTHO'
        self.ortho_scale = ortho_scale

    @property
    def ortho_scale(self) -> float:
        return self.blender_camera.data.ortho_scale

    @ortho_scale.setter
    def ortho_scale(self, val: float):
        self.blender_camera.data.ortho_scale = val

    def distance2depth(self, distmap: np.ndarray) -> np.ndarray:
        """Convert map of camera ray lengths (distmap) to map of distances to image plane (depthmap)

        Args:
            distmap (np.ndarray): Distance map

        Returns:
            np.ndarray: Depth map
        """
        # In orthogonal camera rays are orthogonal to the image plane => distmap = depthmap
        return distmap
