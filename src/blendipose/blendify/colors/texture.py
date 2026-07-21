from __future__ import annotations

from abc import ABC, abstractmethod

import bpy
import numpy as np

from .base import Colors, ColorsMetadata
from ..internal.texture import _copy_values_to_image
from memory_tempfile import MemoryTempfile
import cv2

class UVMap(ABC):
    """An abstract container template for storing a UV coordinate map
    """

    @abstractmethod
    def __init__(self, data: np.ndarray):
        self._data = data

    @property
    def data(self) -> np.ndarray:
        """Get the stored UV map data

        Returns:
            np.ndarray: UV map data
        """
        return self._data


class VertexUV(UVMap):
    """A container which stores a UV coordinate for every vertex
    In the form of (N,2) array (N vertices, 2 UV coordinates for each)
    """

    def __init__(self, data: np.ndarray):
        super().__init__(data)


class FacesUV(UVMap):
    """A container which stores a UV coordinate for every vertex in every triangle face
    In the form of (M,3,2) array (M faces, 3 vertices in each face, 2 UV coordinates for each
    vertex in triangle)
    """

    def __init__(self, data: np.ndarray):
        super().__init__(data)


class UVColors(Colors):
    """An abstract container for storing color information bound to UV coordinate space
    """

    @abstractmethod
    def __init__(self, uv_map: UVMap):
        super().__init__()
        self._uv_map = uv_map

    @property
    def uv_map(self) -> UVMap:
        """Get the stored UV map

        Returns:
            UVMap: stored UV map
        """
        return self._uv_map


class TextureColors(UVColors):
    """A container which stores texture in form of pixels array and the corresponding UV mapping
    """

    def __init__(self, texture: np.ndarray, uv_map: UVMap, interpolation: str = 'Linear'):
        """Create the texture container and initialize a Blender texture with the pixels data

        Args:
            texture (np.ndarray): pixels data
            uv_map (UVMap): corresponding UV map
            interpolation (str, optional): interpolation method for the texture, values can be
            'Linear', 'Closest', 'Cubic' or 'Smart'. Defaults to 'Linear'.
        """
        assert texture.ndim == 3 and texture.shape[2] in (
        3, 4), "Texture must be a 3D array of shape (H, W, 3) or (H, W, 4)"
        assert interpolation in ('Linear', 'Closest', 'Cubic',
                                 'Smart'), ("Interpolation must be one of 'Linear', 'Closest', "
                                            "'Cubic' or 'Smart'")
        super().__init__(uv_map)
        if texture.dtype == np.uint8:
            texture = texture.astype(np.float32) / 255.
        blender_image = bpy.data.images.new(name="tex_image", width=texture.shape[1],
                                            height=texture.shape[0], alpha=False)
        last_dim = texture.shape[2]
        # Flip the data vertically before copying
        texture = texture[::-1]
        if last_dim == 3:
            texture = np.concatenate([texture, np.ones_like(texture[..., :1])], axis=-1)
        blender_image.pixels.foreach_set(texture.ravel())
        blender_image.pack()
        self._texture = blender_image
        self._metadata = ColorsMetadata(
            type=self.__class__,
            has_alpha=last_dim == 4,
            color=None,
            texture=self._texture,
            interpolation=interpolation
        )

    @property
    def blender_texture(self) -> bpy.types.Image:
        """Get the current Blender texture created from the pixels array

        Returns:
            bpy.types.Image: current Blender texture
        """
        return self._texture

    def update_pixels(self, texture: np.ndarray):
        """Update the texture with new pixel data

        Args:
            texture (np.ndarray): new pixel data
        """
        assert texture.ndim == 3 and texture.shape[2] in (3, 4), "Texture must be a 3D array of shape (H, W, 3) or (H, W, 4)"
        assert (texture.shape[0] == self._texture.size[1] and
                texture.shape[1] == self._texture.size[0]), "Texture shape must match the original texture shape"
        expected_channels = 4 if self._metadata.has_alpha else 3
        if texture.shape[2] != expected_channels:
            raise ValueError(
                f"Texture channel count mismatch: expected {expected_channels}, "
                f"got {texture.shape[2]}"
            )
        if texture.dtype == np.uint8:
            texture = texture.astype(np.float32) / 255.
        last_dim = texture.shape[2]
        # Flip the data vertically before copying
        texture = texture[::-1]
        if last_dim == 3:
            texture = np.concatenate([texture, np.ones_like(texture[..., :1])], axis=-1)
        self._texture.pixels.foreach_set(texture.ravel())
        #self._texture.pixels = texture.ravel()
        self._texture.update()


class FileTextureColors(UVColors):
    """A container which stores path to the texture file and the corresponding UV mapping
    """

    def __init__(self, texture_path: str, uv_map: UVMap, interpolation: str = 'Linear', has_alpha: bool=True):
        """Create the texture container and load the texture from the path as a Blender texture

        Args:
            texture_path (str): path to the texture
            uv_map: corresponding UV map
            interpolation (str, optional): interpolation method for the texture, values can be
            'Linear', 'Closest', 'Cubic' or 'Smart'. Defaults to 'Linear'.
            has_alpha (bool, optional): whether the texture's alpha should be used. Defaults to True.
        """
        super().__init__(uv_map)
        self._texture = bpy.data.images.load(texture_path)
        self._metadata = ColorsMetadata(
            type=self.__class__,
            has_alpha=has_alpha,
            color=None,
            texture=self._texture,
            interpolation=interpolation
        )

    @property
    def blender_texture(self) -> bpy.types.Image:
        """Get the current Blender texture created from the pixels array

        Returns:
            bpy.types.Image: current Blender texture
        """
        return self._texture



class VideoFileTextureColors(UVColors):
    """A container which stores path to the texture file and the corresponding UV mapping
    """

    def __init__(self, texture_path: str, uv_map: UVMap, interpolation: str = 'Linear', frame_start: int = 0):
        """Create the texture container and load the texture from the path as a Blender texture

        Args:
            texture_path (str): path to the texture
            uv_map: corresponding UV map
            interpolation (str, optional): interpolation method for the texture, values can be
            'Linear', 'Closest', 'Cubic' or 'Smart'. Defaults to 'Linear'.
        """
        super().__init__(uv_map)
        self._texture = bpy.data.images.load(texture_path)
        self._texture.source = 'MOVIE'
        self._metadata = ColorsMetadata(
            type=self.__class__,
            has_alpha=False,
            color=None,
            texture=self._texture,
            interpolation=interpolation,
            frame_start=frame_start
        )


    @property
    def blender_texture(self) -> bpy.types.Image:
        """Get the current Blender texture created from the pixels array

        Returns:
            bpy.types.Image: current Blender texture
        """
        return self._texture


class TextureColorsViaTempFile(FileTextureColors):
    def __init__(
            self, texture: np.ndarray, uv_map: UVMap, interpolation: str = 'Linear', has_alpha: bool = True):
        self.tempfile_maker = MemoryTempfile()
        self.tempfile = self.tempfile_maker.NamedTemporaryFile(mode='wb', suffix='.png')
        self.update_pixels(texture)
        super().__init__(self.tempfile.name, uv_map, interpolation, has_alpha)


    def update_pixels(self, texture: np.ndarray, reload: bool = True):
        if texture.dtype != np.uint8:
            texture = np.round(texture * 255.0).astype(np.uint8)
        if texture.ndim == 3 and texture.shape[2] == 4:
            texture_bgr = cv2.cvtColor(texture, cv2.COLOR_RGBA2BGRA)
        else:
            texture_bgr = cv2.cvtColor(texture, cv2.COLOR_RGB2BGR)
        _, imbuf = cv2.imencode('.png', texture_bgr)
        self.tempfile.seek(0)
        self.tempfile.write(memoryview(imbuf))
        self.tempfile.truncate()
        self.tempfile.flush()
        # During __init__ the file is written before the Blender image is loaded
        if reload and getattr(self, '_texture', None) is not None:
            self._texture.reload()
