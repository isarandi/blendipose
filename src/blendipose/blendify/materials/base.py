from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence, Union, Dict, Optional
from dataclasses import dataclass

import bpy

if TYPE_CHECKING:
    MaterialList = Sequence[Material]


@dataclass
class MaterialInstance:
    blender_material: bpy.types.Material
    inputs: Dict[str, bpy.types.NodeSocket]
    colors_node: Optional[bpy.types.ShaderNode] = None


class Material(ABC):
    def __init__(self, use_backface_culling=False):
        self._use_backface_culling = use_backface_culling

    @abstractmethod
    def create_material(self, name: str = "object_material") -> MaterialInstance:
        pass


