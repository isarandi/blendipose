from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Light
from ..internal.types import RotationMode

if TYPE_CHECKING:
    from ..internal.types import Vector3d, RotationParams


class PointLight(Light):
    def __init__(
        self,
        strength: float,
        shadow_soft_size: float,
        color: Vector3d,
        tag: str,
        cast_shadows: bool = True,
        rotation_mode: RotationMode = "quaternionWXYZ",
        rotation: RotationParams = None,
        translation: Vector3d = (0, 0, 0),
    ):
        blender_light = self._blender_create_light(tag, "POINT")
        super().__init__(
            tag=tag, blender_object=blender_light,
            rotation_mode=rotation_mode, rotation=rotation, translation=translation,
        )
        self.color = color
        self.strength = strength
        self.cast_shadows = cast_shadows
        self.shadow_soft_size = shadow_soft_size

    @property
    def shadow_soft_size(self) -> float:
        return self.blender_light.data.shadow_soft_size

    @shadow_soft_size.setter
    def shadow_soft_size(self, val: float):
        self.blender_light.data.shadow_soft_size = val


class DirectionalLight(Light):
    def __init__(
        self,
        strength: float,
        angular_diameter: float,
        color: Vector3d,
        tag: str,
        cast_shadows: bool = True,
        rotation_mode: RotationMode = "quaternionWXYZ",
        rotation: RotationParams = None,
        translation: Vector3d = (0, 0, 0),
    ):
        blender_light = self._blender_create_light(tag, "SUN")
        super().__init__(
            tag=tag, blender_object=blender_light,
            rotation_mode=rotation_mode, rotation=rotation, translation=translation,
        )
        self.color = color
        self.strength = strength
        self.cast_shadows = cast_shadows
        self.angular_diameter = angular_diameter

    @property
    def angular_diameter(self) -> float:
        return self.blender_light.data.angle

    @angular_diameter.setter
    def angular_diameter(self, val: float):
        self.blender_light.data.angle = val


class SpotLight(Light):
    def __init__(
        self,
        strength: float,
        spot_size: float,
        spot_blend: float,
        color: Vector3d,
        shadow_soft_size: float,
        tag: str,
        cast_shadows: bool = True,
        rotation_mode: RotationMode = "quaternionWXYZ",
        rotation: RotationParams = None,
        translation: Vector3d = (0, 0, 0),
    ):
        blender_light = self._blender_create_light(tag, "SPOT")
        super().__init__(
            tag=tag, blender_object=blender_light,
            rotation_mode=rotation_mode, rotation=rotation, translation=translation,
        )
        self.color = color
        self.strength = strength
        self.spot_size = spot_size
        self.spot_blend = spot_blend
        self.cast_shadows = cast_shadows
        self.shadow_soft_size = shadow_soft_size

    @property
    def spot_size(self) -> float:
        return self.blender_light.data.spot_size

    @spot_size.setter
    def spot_size(self, val: float):
        self.blender_light.data.spot_size = val

    @property
    def spot_blend(self) -> float:
        return self.blender_light.data.spot_blend

    @spot_blend.setter
    def spot_blend(self, val: float):
        self.blender_light.data.spot_blend = val

    @property
    def shadow_soft_size(self) -> float:
        return self.blender_light.data.shadow_soft_size

    @shadow_soft_size.setter
    def shadow_soft_size(self, val: float):
        self.blender_light.data.shadow_soft_size = val
