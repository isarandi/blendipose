from typing import Tuple

import bpy

from .base import Material, MaterialInstance


class EmissionMaterial(Material):
    """A class which manages the parameters of GlossyBSDF Blender material.
    Full docs: https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/glossy.html
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
        principled_bsdf.inputs['Specular'].default_value = 0.0
        principled_bsdf.inputs['Emission Strength'].default_value = 0.5


        emission_node = material_nodes.new("ShaderNodeEmission")
        emission_node.inputs['Strength'].default_value = 10.0

        mix_node = material_nodes.new(type='ShaderNodeMixShader')
        mix_node.inputs['Fac'].default_value = 0

        material_output = material_nodes['Material Output']

        links = object_material.node_tree.links
        links.new(principled_bsdf.outputs['BSDF'], mix_node.inputs[1])
        links.new(emission_node.outputs['Emission'], mix_node.inputs[2])
        links.new(mix_node.outputs['Shader'], material_output.inputs['Surface'])

        material_instance = MaterialInstance(blender_material=object_material,
                                      #       inputs={"Color": emission_node.inputs["Color"]})
                                             inputs={"Color": principled_bsdf.inputs["Emission"]})
        return material_instance

    @property
    def distribution(self):
        return self._distribution
