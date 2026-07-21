import bpy

from .base import Material, MaterialInstance
from .wireframe import WireframeMaterial
from ..compat import bsdf_socket_name, bsdf_socket_value, bsdf_property_mapping


class PrincipledBSDFMaterial(Material):
    """A class which manages the parameters of PrincipledBSDF Blender material.
    Full docs: https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/principled.html
    """

    def __init__(
            self, metallic=0.0, specular=0.3, specular_tint=0.0, roughness=0.4, anisotropic=0.0,
            anisotropic_rotation=0.0, sheen=0.0, sheen_tint=0.5, clearcoat=0.0, clearcoat_roughness=0.0,
            ior=1.45, transmission=0.0, transmission_roughness=0.0, emission=(0, 0, 0, 0),
            emission_strength=0.0, alpha=1.0, base_color=(0,0,0,1), use_backface_culling=True, use_colors_for_emission=False
    ):
        super().__init__(use_backface_culling=use_backface_culling)
        self._property2blender_mapping = bsdf_property_mapping()
        for argname, argvalue in locals().items():
            if argname in self._property2blender_mapping.keys():
                self.__setattr__("_" + argname, bsdf_socket_value(argname, argvalue))

        self._use_colors_for_emission = use_colors_for_emission

    def create_material(self, name: str = "object_material") -> MaterialInstance:
        """Create the Blender material with the parameters stored in the current object

        Args:
            name (str): a unique material name for Blender

        Returns:
            Tuple[bpy.types.Material, bpy.types.ShaderNodeBsdfPrincipled]: Blender material and the
                shader node which uses the created material
        """

        object_material = bpy.data.materials.new(name=name)
        object_material.use_nodes = True
        object_material.use_backface_culling = self._use_backface_culling
        bsdf_node = object_material.node_tree.nodes["Principled BSDF"]

        emission_key = bsdf_socket_name("Emission")
        color_inp_key = emission_key if self._use_colors_for_emission else "Base Color"
        material_instance = MaterialInstance(blender_material=object_material,
                                             inputs={"Color": bsdf_node.inputs[color_inp_key], "Alpha": bsdf_node.inputs["Alpha"],
                                                     "Emission": bsdf_node.inputs[emission_key],
                                                     "Emission Strength": bsdf_node.inputs["Emission Strength"]})

        # Set material properties
        for property_name, blender_name in self._property2blender_mapping.items():
            bsdf_node.inputs[blender_name].default_value = self.__getattribute__("_" + property_name)

        return material_instance


class GlossyBSDFMaterial(Material):
    """A class which manages the parameters of GlossyBSDF Blender material.
    Full docs: https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/glossy.html
    """

    def __init__(self, roughness=0.4, distribution="GGX", use_backface_culling=True):
        super().__init__(use_backface_culling=use_backface_culling)
        self._roughness = roughness
        self._distribution = distribution

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

        bsdf_node = material_nodes.new("ShaderNodeBsdfGlossy")
        material_nodes.remove(material_nodes['Principled BSDF'])
        object_material.node_tree.links.new(material_nodes["Material Output"].inputs["Surface"],
                                            bsdf_node.outputs[0])
        # Set material properties
        bsdf_node.inputs["Roughness"].default_value = self._roughness
        bsdf_node.distribution = self._distribution

        material_instance = MaterialInstance(blender_material=object_material,
                                             inputs={"Color": bsdf_node.inputs["Color"]})

        return material_instance

    @property
    def distribution(self):
        return self._distribution


class PrincipledBSDFWireframeMaterial(WireframeMaterial, PrincipledBSDFMaterial):
    def __init__(
            self, wireframe_thickness=0.01, wireframe_color=(0., 0., 0., 1.),
            metallic=0.0, specular=0.3, specular_tint=0.0, roughness=0.4, anisotropic=0.0,
            anisotropic_rotation=0.0, sheen=0.0, sheen_tint=0.5, clearcoat=0.0, clearcoat_roughness=0.0,
            ior=1.45, transmission=0.0, transmission_roughness=0.0, emission=(0, 0, 0, 0),
            emission_strength=0.0, alpha=1.0, base_color=(0, 0, 0, 1), use_backface_culling=True,
            use_colors_for_emission=False,
    ):
        PrincipledBSDFMaterial.__init__(
            self, metallic=metallic, specular=specular, specular_tint=specular_tint,
            roughness=roughness, anisotropic=anisotropic, anisotropic_rotation=anisotropic_rotation,
            sheen=sheen, sheen_tint=sheen_tint, clearcoat=clearcoat,
            clearcoat_roughness=clearcoat_roughness, ior=ior, transmission=transmission,
            transmission_roughness=transmission_roughness, emission=emission,
            emission_strength=emission_strength, alpha=alpha, base_color=base_color,
            use_backface_culling=use_backface_culling, use_colors_for_emission=use_colors_for_emission,
        )
        self._wireframe_thickness = wireframe_thickness
        self._wireframe_color = wireframe_color

    def create_material(self, name: str = "object_material") -> MaterialInstance:
        object_material = bpy.data.materials.new(name=name)
        object_material.use_nodes = True
        object_material.use_backface_culling = self._use_backface_culling
        material_nodes = object_material.node_tree.nodes

        # Create BSDF
        bsdf_node = object_material.node_tree.nodes["Principled BSDF"]

        # Set BSDF properties
        for property_name, blender_name in self._property2blender_mapping.items():
            bsdf_node.inputs[blender_name].default_value = self.__getattribute__("_" + property_name)

        # Create and link wireframe
        self.overlay_wireframe(object_material, bsdf_node)

        # Create internal representation
        emission_key = bsdf_socket_name("Emission")
        color_inp_key = emission_key if self._use_colors_for_emission else "Base Color"
        material_instance = MaterialInstance(
            blender_material=object_material,
            inputs={
                "Color": bsdf_node.inputs[color_inp_key],
                "Alpha": bsdf_node.inputs["Alpha"],
                "Emission": bsdf_node.inputs[emission_key],
                "Emission Strength": bsdf_node.inputs["Emission Strength"]
            }
        )

        return material_instance
