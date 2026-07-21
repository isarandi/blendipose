from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Sequence, Union

import bmesh
import bpy
import numpy as np

from .base import RenderableObject
from ..colors import UniformColors
from ..colors.base import Colors
from ..internal.types import FillType, RotationMode
from ..materials.base import Material

if TYPE_CHECKING:
    from typing import Union
    from ..colors.base import ColorsList
    from ..internal.types import Vector3d, RotationParams
    from ..materials.base import MaterialList


# =================================================== Mesh Primitives ==================================================
class MeshPrimitive(RenderableObject):
    """Base class for mesh primitives. Used to throw Exceptions for non-implemented Colors
    subclasses (only UniformColors is supported) and add shared method for setting smooth shading.

    Properties:
        emit_shadow (bool, optional): control whether the object will emit shadow from any light source in the scene.
    """
    @abstractmethod
    def __init__(
        self,
        material: Union[Material, MaterialList],
        colors: Union[Colors, ColorsList],
        tag: str,
        blender_object: bpy.types.Object,
        faces_material: Sequence[int] = None,
        rotation_mode: RotationMode = "quaternionWXYZ",
        rotation: RotationParams = None,
        translation: Vector3d = (0, 0, 0),
    ):
        self._faces_material = faces_material
        super().__init__(
            material=material, colors=colors, tag=tag, blender_object=blender_object,
            rotation_mode=rotation_mode, rotation=rotation, translation=translation,
        )


    def _blender_set_colors(self, colors: Union[Colors, ColorsList]):
        colors_list = [colors] if isinstance(colors, Colors) else colors
        if not all(isinstance(x, UniformColors) for x in colors_list):
            raise NotImplementedError("Non-uniform colors or textures are not supported in primitives, "
                                      "consider creating a primitive through Mesh for that")
        super()._blender_set_colors(colors_list)

    def set_smooth(self, smooth: bool = True):
        """Enables or disables the smooth surface imitation for the object

        Args:
            smooth (bool): Whether to turn the smooth surface on or off
        """
        for polygon in self._blender_mesh.polygons:
            polygon.use_smooth = smooth
        self._blender_mesh.update()

    def _blender_assign_materials(self):
        super()._blender_assign_materials()
        if not (len(self._material_instances) == 1 or self._faces_material is None):
            bpy.context.view_layer.objects.active = self._blender_object
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_mode(type="FACE")
            bpy.ops.mesh.select_all(action='DESELECT')
            bm = bmesh.from_edit_mesh(self._blender_mesh)
            assert self._faces_material is None or (len(self._faces_material) == len(bm.faces)), \
                f"Number of material faces should be equal to the number of faces ({len(bm.faces)})"
            for mat_ind in range(self._materials_count):
                for face in bm.faces:
                    perface_mat_ind = self._faces_material[face.index]
                    if perface_mat_ind == mat_ind:
                        face.select = True
                self._blender_object.active_material_index = mat_ind
                bpy.ops.object.material_slot_assign()
                bpy.ops.mesh.select_all(action='DESELECT')
            # bmesh.update_edit_mesh(self._blender_object.data)
            bpy.ops.object.mode_set(mode='OBJECT')


class CubeMesh(MeshPrimitive):
    """Cube mesh primitive, supports only uniform coloring (UniformColors)

    Properties:
        emit_shadow (bool, optional): control whether the object will emit shadow from any light source in the scene.

    Methods:
        set_smooth(bool): turns smooth shading on and off based on the bool argument.
    """
    def __init__(
        self,
        size: float,
        tag: str,
        material: Union[Material, MaterialList] = None,
        colors: Union[Colors, ColorsList] = None,
        faces_material: Sequence[int] = None,
        rotation_mode: RotationMode = "quaternionWXYZ",
        rotation: RotationParams = None,
        translation: Vector3d = (0, 0, 0),
    ):
        obj = self._blender_create_object(size, tag)
        self._blender_mesh = obj.data
        super().__init__(
            material=material, colors=colors, tag=tag, blender_object=obj,
            faces_material=faces_material,
            rotation_mode=rotation_mode, rotation=rotation, translation=translation,
        )

    def _blender_create_object(
        self,
        size: float,
        tag: str
    ):
        bpy.ops.mesh.primitive_cube_add(size=size)
        obj = bpy.context.object
        obj.name = tag
        return obj

    def _blender_set_colors(
        self,
        colors_list: ColorsList
    ):
        """Remembers current color properties, builds a color node for material, sets color information to mesh

        Args:
            colors_list (ColorsList): list of target colors
        """
        for polygon in self._blender_mesh.polygons:
            polygon.use_smooth = True
        if not any(mod.type == 'EDGE_SPLIT' for mod in self._blender_object.modifiers):
            self._blender_object.modifiers.new(name='EdgeSplit', type='EDGE_SPLIT')
        super()._blender_set_colors(colors_list)


class CircleMesh(MeshPrimitive):
    """Circle mesh primitive, supports only uniform coloring (UniformColors)

    Properties:
        emit_shadow (bool, optional): control whether the object will emit shadow from any light source in the scene

    Methods:
        set_smooth(bool): turns smooth shading on and off based on the bool argument.
    """
    def __init__(
        self,
        radius: float,
        tag: str,
        material: Union[Material, MaterialList] = None,
        colors: Union[Colors, ColorsList] = None,
        num_vertices: int = 32,
        fill_type: FillType = "NGON",
        faces_material: Sequence[int] = None,
        rotation_mode: RotationMode = "quaternionWXYZ",
        rotation: RotationParams = None,
        translation: Vector3d = (0, 0, 0),
    ):
        obj = self._blender_create_object(num_vertices, radius, fill_type, tag)
        self._blender_mesh = obj.data
        super().__init__(
            material=material, colors=colors, tag=tag, blender_object=obj,
            faces_material=faces_material,
            rotation_mode=rotation_mode, rotation=rotation, translation=translation,
        )

    def _blender_create_object(
        self,
        num_vertices: int,
        radius: float,
        fill_type: str,
        tag: str
    ):
        bpy.ops.mesh.primitive_circle_add(vertices=num_vertices, radius=radius, fill_type=fill_type)
        obj = bpy.context.object
        obj.name = tag
        return obj

    def _blender_set_colors(
        self,
        colors_list: Colors
    ):
        """Remembers current color properties, builds a color node for material, sets color information to mesh

        Args:
            colors_list (ColorsList): list of target colors
        """
        for polygon in self._blender_mesh.polygons:
            polygon.use_smooth = True
        if not any(mod.type == 'EDGE_SPLIT' for mod in self._blender_object.modifiers):
            self._blender_object.modifiers.new(name='EdgeSplit', type='EDGE_SPLIT')
        super()._blender_set_colors(colors_list)


class CylinderMesh(MeshPrimitive):
    """Cylinder mesh primitive, supports only uniform coloring (UniformColors)

    Properties:
        emit_shadow (bool, optional): control whether the object will emit shadow from any light source in the scene.

    Methods:
        set_smooth(bool): turns smooth shading on and off based on the bool argument.
    """
    def __init__(
        self,
        radius: float,
        height: float,
        tag: str,
        material: Union[Material, MaterialList] = None,
        colors: Union[Colors, ColorsList] = None,
        num_vertices: int = 32,
        fill_type: FillType = "NGON",
        faces_material: Sequence[int] = None,
        rotation_mode: RotationMode = "quaternionWXYZ",
        rotation: RotationParams = None,
        translation: Vector3d = (0, 0, 0),
    ):
        obj = self._blender_create_object(num_vertices, radius, height, fill_type, tag)
        self._blender_mesh = obj.data
        super().__init__(
            material=material, colors=colors, tag=tag, blender_object=obj,
            faces_material=faces_material,
            rotation_mode=rotation_mode, rotation=rotation, translation=translation,
        )

    def _blender_create_object(
        self,
        num_vertices: int,
        radius: float,
        height: float,
        fill_type: str,
        tag: str
    ):
        bpy.ops.mesh.primitive_cylinder_add(vertices=num_vertices, radius=radius, depth=height, end_fill_type=fill_type)
        obj = bpy.context.object
        obj.name = tag
        return obj

    def _blender_set_colors(
            self,
            colors_list: ColorsList
    ):
        """Remembers current color properties, builds a color node for material, sets color information to mesh

        Args:
            colors_list (ColorsList): list of target colors
        """
        for polygon in self._blender_mesh.polygons:
            polygon.use_smooth = True
        if not any(mod.type == 'EDGE_SPLIT' for mod in self._blender_object.modifiers):
            self._blender_object.modifiers.new(name='EdgeSplit', type='EDGE_SPLIT')
        super()._blender_set_colors(colors_list)


class PlaneMesh(MeshPrimitive):
    """Plane mesh primitive, supports only uniform coloring (UniformColors)

    Properties:
        emit_shadow (bool, optional): control whether the object will emit shadow from any light source in the scene.
        shadow_catcher (bool, optional): control whether the object will act as a shadow catcher (i.e. object
        geometry is hidden from the render, only casted shadows are rendered).

    Methods:
        set_smooth(bool): turns smooth shading on and off based on the bool argument.
    """

    def __init__(
            self,
            size: float,
            tag: str,
            shadow_catcher: bool = False,
            material: Union[Material, MaterialList] = None,
            colors: Union[Colors, ColorsList] = None,
            faces_material: Sequence[int] = None,
            rotation_mode: RotationMode = "quaternionWXYZ",
            rotation: RotationParams = None,
            translation: Vector3d = (0, 0, 0),
    ):
        obj = self._blender_create_object(size, tag, shadow_catcher)
        self._blender_mesh = obj.data
        super().__init__(
            material=material, colors=colors, tag=tag, blender_object=obj,
            faces_material=faces_material,
            rotation_mode=rotation_mode, rotation=rotation, translation=translation,
        )

    def _blender_create_object(
            self,
            size: float,
            tag: str,
            shadow_catcher: bool
    ):
        bpy.ops.mesh.primitive_plane_add(size=size)
        obj = bpy.context.object
        obj.name = tag

        if shadow_catcher:
            obj.is_shadow_catcher = True
            obj.visible_glossy = False
            obj.visible_diffuse = False
            obj.visible_transmission = False
            obj.visible_volume_scatter = False

        return obj

    def _blender_set_colors(
            self,
            colors_list: ColorsList
    ):
        """Remembers current color properties, builds a color node for material, sets color information to mesh

        Args:
            colors_list (ColorsList): list of target colors
        """
        for polygon in self._blender_mesh.polygons:
            polygon.use_smooth = True
        if not any(mod.type == 'EDGE_SPLIT' for mod in self._blender_object.modifiers):
            self._blender_object.modifiers.new(name='EdgeSplit', type='EDGE_SPLIT')
        super()._blender_set_colors(colors_list)
# =============================================== End of Mesh Primitives ===============================================


# ================================================ Parametric Primitives ===============================================
class ParametricPrimitive(RenderableObject):
    """Base class for parametric primitives. Used to throw Exceptions for non-implemented Colors
    subclasses (only UniformColors is supported) and non-implemented Materials (only one Material is supported).

    Properties:
        emit_shadow (bool, optional): control whether the object will emit shadow from any light source in the scene.
    """

    @abstractmethod
    def __init__(
        self,
        material: Union[Material, MaterialList],
        colors: Union[Colors, ColorsList],
        tag: str,
        blender_object: bpy.types.Object,
        rotation_mode: RotationMode = "quaternionWXYZ",
        rotation: RotationParams = None,
        translation: Vector3d = (0, 0, 0),
    ):
        super().__init__(
            material=material, colors=colors, tag=tag, blender_object=blender_object,
            rotation_mode=rotation_mode, rotation=rotation, translation=translation,
        )

    def _blender_set_materials(self, material_list: MaterialList):
        if not len(material_list) == 1:
            raise NotImplementedError("Multiple materials are not supported in parametric primitives, "
                                      "consider creating a primitive through Mesh for that")
        super()._blender_set_materials(material_list)

    def _blender_set_colors(self, colors_list: ColorsList):
        if  not len(colors_list) == 1 or not isinstance(colors_list[0], UniformColors):
            raise NotImplementedError("Multiple materials, non-uniform colors or textures are not supported "
                                      "in parametric primitives, consider creating a primitive through Mesh for that")
        super()._blender_set_colors(colors_list)


class EllipsoidNURBS(ParametricPrimitive):
    """NURBS Ellipsoid, implemented as NURBS Sphere that is rescaled along axes,
     supports only uniform coloring (UniformColors)

    Properties:
        emit_shadow (bool, optional): control whether the object will emit shadow from any light source in the scene
    """
    def __init__(
        self,
        radius: Vector3d,
        tag: str,
        material: Union[Material, MaterialList] = None,
        colors: Union[Colors, ColorsList] = None,
        rotation_mode: RotationMode = "quaternionWXYZ",
        rotation: RotationParams = None,
        translation: Vector3d = (0, 0, 0),
    ):
        obj = self._blender_create_object(radius, tag)
        super().__init__(
            material=material, colors=colors, tag=tag, blender_object=obj,
            rotation_mode=rotation_mode, rotation=rotation, translation=translation,
        )

    def _blender_create_object(
            self,
            radius: Vector3d,
            tag: str
    ):
        bpy.ops.surface.primitive_nurbs_surface_sphere_add(radius=1)
        obj = bpy.context.object
        obj.scale = np.asarray(radius, dtype=np.float64).tolist()
        obj.name = tag
        return obj


class SphereNURBS(EllipsoidNURBS):
    """NURBS Sphere, supports only uniform coloring (UniformColors)

    Properties:
        emit_shadow (bool, optional): control whether the object will emit shadow from any light source in the scene
    """
    def __init__(
        self,
        radius: float,
        tag: str,
        material: Union[Material, MaterialList] = None,
        colors: Union[Colors, ColorsList] = None,
        rotation_mode: RotationMode = "quaternionWXYZ",
        rotation: RotationParams = None,
        translation: Vector3d = (0, 0, 0),
    ):
        super().__init__(
            radius=(radius, radius, radius), tag=tag,
            material=material, colors=colors,
            rotation_mode=rotation_mode, rotation=rotation, translation=translation,
        )


class CurveBezier(ParametricPrimitive):
    """Bezier Curve, supports only uniform coloring (UniformColors)

    Properties:
        emit_shadow (bool, optional): control whether the object will emit shadow from any light source in the scene
    """
    def __init__(
        self,
        keypoints: np.ndarray,
        radius: float,
        tag: str,
        material: Union[Material, MaterialList] = None,
        colors: Union[Colors, ColorsList] = None,
        rotation_mode: RotationMode = "quaternionWXYZ",
        rotation: RotationParams = None,
        translation: Vector3d = (0, 0, 0),
    ):
        if len(keypoints) < 2:
            raise ValueError("CurveBezier requires at least 2 keypoints")
        obj = self._blender_create_object(tag)
        obj.data.bevel_depth = radius
        obj.data.bevel_resolution = 4
        if len(keypoints) > 2:
            obj.data.splines[0].bezier_points.add(count=len(keypoints) - 2)
        for ind, coords in enumerate(keypoints):
            obj.data.splines[0].bezier_points[ind].co = coords
            obj.data.splines[0].bezier_points[ind].handle_left_type = 'VECTOR'
            obj.data.splines[0].bezier_points[ind].handle_right_type = 'VECTOR'
        super().__init__(
            material=material, colors=colors, tag=tag, blender_object=obj,
            rotation_mode=rotation_mode, rotation=rotation, translation=translation,
        )

    def _blender_create_object(
        self,
        tag: str
    ):
        bpy.ops.curve.primitive_bezier_curve_add()
        obj = bpy.context.object
        obj.name = tag
        obj.data.dimensions = '3D'
        obj.data.fill_mode = 'FULL'
        return obj
# ============================================ End of Parametric Primitives ============================================
