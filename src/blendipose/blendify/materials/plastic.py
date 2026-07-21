from .bsdf import PrincipledBSDFMaterial, PrincipledBSDFWireframeMaterial


class PlasticMaterial(PrincipledBSDFMaterial):
    def __init__(
            self, roughness=0.3, ior=1.45, clearcoat=0.0, specular=0.5,
            clearcoat_roughness=0.0, metallic=0.0, specular_tint=0.0, anisotropic=0.0,
            anisotropic_rotation=0.0, sheen=0.0, sheen_tint=0.5, transmission=0.0,
            transmission_roughness=0.0, emission=(0, 0, 0, 0), emission_strength=0.0,
            alpha=1.0, base_color=(0, 0, 0, 1), use_backface_culling=True,
            use_colors_for_emission=False,
    ):
        super().__init__(
            roughness=roughness, ior=ior, clearcoat=clearcoat, specular=specular,
            clearcoat_roughness=clearcoat_roughness, metallic=metallic,
            specular_tint=specular_tint, anisotropic=anisotropic,
            anisotropic_rotation=anisotropic_rotation, sheen=sheen, sheen_tint=sheen_tint,
            transmission=transmission, transmission_roughness=transmission_roughness,
            emission=emission, emission_strength=emission_strength, alpha=alpha,
            base_color=base_color, use_backface_culling=use_backface_culling,
            use_colors_for_emission=use_colors_for_emission,
        )


class PlasticWireframeMaterial(PrincipledBSDFWireframeMaterial):
    def __init__(
            self, roughness=0.3, ior=1.45, clearcoat=0.0, specular=0.5,
            clearcoat_roughness=0.0, wireframe_thickness=0.01, wireframe_color=(0., 0., 0., 1.),
            metallic=0.0, specular_tint=0.0, anisotropic=0.0, anisotropic_rotation=0.0,
            sheen=0.0, sheen_tint=0.5, transmission=0.0, transmission_roughness=0.0,
            emission=(0, 0, 0, 0), emission_strength=0.0, alpha=1.0, base_color=(0, 0, 0, 1),
            use_backface_culling=True, use_colors_for_emission=False,
    ):
        super().__init__(
            roughness=roughness, ior=ior, clearcoat=clearcoat, specular=specular,
            clearcoat_roughness=clearcoat_roughness, wireframe_thickness=wireframe_thickness,
            wireframe_color=wireframe_color, metallic=metallic, specular_tint=specular_tint,
            anisotropic=anisotropic, anisotropic_rotation=anisotropic_rotation,
            sheen=sheen, sheen_tint=sheen_tint, transmission=transmission,
            transmission_roughness=transmission_roughness, emission=emission,
            emission_strength=emission_strength, alpha=alpha, base_color=base_color,
            use_backface_culling=use_backface_culling, use_colors_for_emission=use_colors_for_emission,
        )
