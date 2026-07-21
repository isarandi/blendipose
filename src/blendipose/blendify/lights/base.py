from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import bpy
import numpy as np

from ..compat import get_light_cast_shadow, set_light_cast_shadow
from ..internal.positionable import Positionable
from ..internal.types import RotationMode

if TYPE_CHECKING:
    from ..internal.types import Vector3d, RotationParams


class Light(Positionable):
    """Abstract base class for all the light sources.
    """
    @abstractmethod
    def __init__(
        self,
        tag: str,
        blender_object: bpy.types.Object,
        rotation_mode: RotationMode = "quaternionWXYZ",
        rotation: RotationParams = None,
        translation: Vector3d = (0, 0, 0),
    ):
        super().__init__(
            tag=tag, blender_object=blender_object,
            rotation_mode=rotation_mode, rotation=rotation, translation=translation,
        )

    def _blender_create_light(self, tag: str, light_type: str) -> bpy.types.Object:
        light_obj = bpy.data.lights.new(name=tag, type=light_type)
        obj = bpy.data.objects.new(name=tag, object_data=light_obj)

        bpy.context.collection.objects.link(obj)
        return obj

    @property
    def blender_light(self) -> bpy.types.Object:
        return self._blender_object

    @property
    def color(self) -> np.ndarray:
        return np.array(self.blender_light.data.color[:3])

    @color.setter
    def color(self, val: Vector3d):
        val = np.array(val)
        self.blender_light.data.color = val.tolist()

    @property
    def cast_shadows(self) -> bool:
        return get_light_cast_shadow(self.blender_light.data)

    @cast_shadows.setter
    def cast_shadows(self, val: bool):
        set_light_cast_shadow(self.blender_light.data, val)

    @property
    def strength(self) -> float:
        return self.blender_light.data.energy

    @strength.setter
    def strength(self, val: float):
        self.blender_light.data.energy = val

    @property
    def max_bounces(self) -> int:
        return self.blender_light.data.cycles.max_bounces

    @max_bounces.setter
    def max_bounces(self, val: int):
        self.blender_light.data.cycles.max_bounces = val
