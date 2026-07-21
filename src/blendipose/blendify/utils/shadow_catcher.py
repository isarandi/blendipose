from typing import Union

import bpy

from ..renderables.base import Renderable


def set_shadow_catcher(obj: Union[str, Renderable], state=False):
    """Set object is_shadow_catcher and various visibility_* properties to make
    object act as a shadow catcher or revert the changes.

    Args:
        obj (Union[str, Renderable]): tag of the Blender object or instance of Renderable
        state (bool): if True make obj a shadow catcher, otherwise make it a regular object
    """
    if not isinstance(obj, str):
        obj = obj.tag

    shadow_catcher = bpy.data.objects.get(obj)
    if shadow_catcher is None:
        raise ValueError(f"No Blender object named '{obj}' exists in the scene")

    # set / unset shadow catcher properties
    shadow_catcher.is_shadow_catcher = state
    shadow_catcher.visible_glossy = not state
    shadow_catcher.visible_diffuse = not state
    shadow_catcher.visible_transmission = not state
    shadow_catcher.visible_volume_scatter = not state
