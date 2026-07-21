import bpy

from .base import Material, MaterialInstance
from ..compat import bsdf_socket_name


class EmissionMaterial(Material):
    """A semi-transparent emissive material based on the Principled BSDF shader.
    The base color is black, alpha is 0.5 and emission strength is 0.5;
    the "Color" input of the material instance drives the emission color.
    """

    def __init__(self, use_backface_culling=True):
        super().__init__(use_backface_culling=use_backface_culling)

    def create_material(self, name: str = "object_material") -> MaterialInstance:
        """Create the Blender material with the parameters stored in the current object

        Args:
            name (str): a unique material name for Blender

        Returns:
            Tuple[bpy.types.Material, bpy.types.ShaderNodeBsdfGlossy]: Blender material and the
                shader node which uses the created material
        """

        object_material = bpy.data.materials.new(name=name)
        object_material.use_nodes = True
        object_material.use_backface_culling = self._use_backface_culling
        material_nodes = object_material.node_tree.nodes

        principled_bsdf = material_nodes['Principled BSDF']
        principled_bsdf.inputs['Base Color'].default_value = (0.0, 0.0, 0.0, 1)
        principled_bsdf.inputs['Alpha'].default_value = 0.5
        principled_bsdf.inputs[bsdf_socket_name('Specular')].default_value = 0.0
        principled_bsdf.inputs['Emission Strength'].default_value = 0.5

        material_output = material_nodes['Material Output']
        links = object_material.node_tree.links
        links.new(principled_bsdf.outputs['BSDF'], material_output.inputs['Surface'])

        material_instance = MaterialInstance(
            blender_material=object_material,
            inputs={"Color": principled_bsdf.inputs[bsdf_socket_name("Emission")]},
        )
        return material_instance

