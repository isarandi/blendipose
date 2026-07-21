from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NamedTuple, Optional, Union, Sequence

import bpy

if TYPE_CHECKING:
    from ..internal.types import Vector3d
    ColorsList = Sequence[Colors]


class ColorsMetadata(NamedTuple):
    type: type
    color: Optional[Vector3d]
    has_alpha: bool
    texture: Optional[bpy.types.Image]
    interpolation: Optional[str] = None
    frame_start: Optional[int] = None

    def __del__(self):
        if self.texture is not None:
            try:
                if not self.texture.users:
                    bpy.data.images.remove(self.texture)
            except (ReferenceError, AttributeError, RuntimeError):
                pass


class Colors(ABC):
    """An abstract container template for storing the object coloring information
    """

    @abstractmethod
    def __init__(self):
        self._metadata: Optional[ColorsMetadata] = None

    @property
    def metadata(self) -> ColorsMetadata:
        return self._metadata
