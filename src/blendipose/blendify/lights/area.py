from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from .base import Light
from ..internal.types import RotationMode

if TYPE_CHECKING:
    from ..internal.types import Vector3d, Vector2d, RotationParams


class AreaLight(Light):
    """Base class for different AreaLights varying in shape.
    """

    @abstractmethod
    def __init__(
            self,
            color: Vector3d,
            strength: float,
            tag: str,
            cast_shadows: bool = True,
            rotation_mode: RotationMode = "quaternionWXYZ",
            rotation: RotationParams = None,
            translation: Vector3d = (0, 0, 0),
    ):
        blender_light = self._blender_create_light(tag, "AREA")
        super().__init__(
            tag=tag, blender_object=blender_light,
            rotation_mode=rotation_mode, rotation=rotation, translation=translation,
        )
        self.color = color
        self.strength = strength
        self.cast_shadows = cast_shadows


class SquareAreaLight(AreaLight):
    def __init__(
            self,
            size: float,
            color: Vector3d = (1.0, 1.0, 1.0),
            strength: float = 100,
            tag: str = "Area_000",
            cast_shadows: bool = True,
            rotation_mode: RotationMode = "quaternionWXYZ",
            rotation: RotationParams = None,
            translation: Vector3d = (0, 0, 0),
    ):
        super().__init__(
            color=color, strength=strength, tag=tag, cast_shadows=cast_shadows,
            rotation_mode=rotation_mode, rotation=rotation, translation=translation,
        )
        self.blender_light.data.shape = "SQUARE"
        self.size = size

    @property
    def size(self) -> float:
        return self.blender_light.data.size

    @size.setter
    def size(self, val: float):
        self.blender_light.data.size = val


class CircleAreaLight(AreaLight):
    def __init__(
            self,
            size: float,
            color: Vector3d = (1.0, 1.0, 1.0),
            strength: float = 100,
            tag: str = "Area_000",
            cast_shadows: bool = True,
            rotation_mode: RotationMode = "quaternionWXYZ",
            rotation: RotationParams = None,
            translation: Vector3d = (0, 0, 0),
    ):
        super().__init__(
            color=color, strength=strength, tag=tag, cast_shadows=cast_shadows,
            rotation_mode=rotation_mode, rotation=rotation, translation=translation,
        )
        self.blender_light.data.shape = "DISK"
        self.size = size

    @property
    def size(self) -> float:
        return self.blender_light.data.size

    @size.setter
    def size(self, val: float):
        self.blender_light.data.size = val


class RectangleAreaLight(AreaLight):
    def __init__(
            self,
            size: Vector2d,
            color: Vector3d = (1.0, 1.0, 1.0),
            strength: float = 100,
            tag: str = "Area_000",
            cast_shadows: bool = True,
            rotation_mode: RotationMode = "quaternionWXYZ",
            rotation: RotationParams = None,
            translation: Vector3d = (0, 0, 0),
    ):
        super().__init__(
            color=color, strength=strength, tag=tag, cast_shadows=cast_shadows,
            rotation_mode=rotation_mode, rotation=rotation, translation=translation,
        )
        self.blender_light.data.shape = "RECTANGLE"
        self.size = size

    @property
    def size(self) -> np.ndarray:
        return np.array([self.blender_light.data.size, self.blender_light.data.size_y])

    @size.setter
    def size(self, val: Vector2d):
        self.blender_light.data.size = val[0]
        self.blender_light.data.size_y = val[1]


class EllipseAreaLight(AreaLight):
    def __init__(
            self,
            size: Vector2d,
            color: Vector3d = (1.0, 1.0, 1.0),
            strength: float = 100,
            tag: str = "Area_000",
            cast_shadows: bool = True,
            rotation_mode: RotationMode = "quaternionWXYZ",
            rotation: RotationParams = None,
            translation: Vector3d = (0, 0, 0),
    ):
        super().__init__(
            color=color, strength=strength, tag=tag, cast_shadows=cast_shadows,
            rotation_mode=rotation_mode, rotation=rotation, translation=translation,
        )
        self.blender_light.data.shape = "ELLIPSE"
        self.size = size

    @property
    def size(self) -> np.ndarray:
        return np.array([self.blender_light.data.size, self.blender_light.data.size_y])

    @size.setter
    def size(self, val: Vector2d):
        self.blender_light.data.size = val[0]
        self.blender_light.data.size_y = val[1]
