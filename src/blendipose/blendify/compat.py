"""Compatibility layer for bpy 3.6 / 4.x / 5.0+."""

import bpy

V = bpy.app.version  # e.g. (4, 2, 0)

# === Principled BSDF socket name/value mapping ===

_BSDF_SOCKET_RENAMES_4 = {
    "Specular": "Specular IOR Level",
    "Sheen": "Sheen Weight",
    "Clearcoat": "Coat Weight",
    "Clearcoat Roughness": "Coat Roughness",
    "Transmission": "Transmission Weight",
    "Emission": "Emission Color",
    "Transmission Roughness": None,  # removed entirely
}

# Properties whose socket type changed from float to RGBA color in 4.0+
_BSDF_FLOAT_TO_COLOR_4 = {"specular_tint", "sheen_tint"}


def bsdf_socket_name(name_36):
    """Map a 3.6-era Principled BSDF socket name to the current version.
    Returns None if the socket was removed (caller should skip)."""
    if V >= (4, 0, 0) and name_36 in _BSDF_SOCKET_RENAMES_4:
        return _BSDF_SOCKET_RENAMES_4[name_36]
    return name_36


def bsdf_socket_value(name_36, value):
    """Adapt a socket default_value for type changes across versions.
    In 4.0+ Specular Tint and Sheen Tint changed from float to RGBA."""
    if V >= (4, 0, 0) and name_36 in _BSDF_FLOAT_TO_COLOR_4:
        if isinstance(value, (int, float)):
            return (value, value, value, 1.0)
    return value


def bsdf_property_mapping():
    """Build the property-name-to-Blender-socket-name dict, version-correct."""
    mapping = {
        "metallic": "Metallic",
        "specular": bsdf_socket_name("Specular"),
        "specular_tint": "Specular Tint",
        "roughness": "Roughness",
        "anisotropic": "Anisotropic",
        "anisotropic_rotation": "Anisotropic Rotation",
        "sheen": bsdf_socket_name("Sheen"),
        "sheen_tint": "Sheen Tint",
        "clearcoat": bsdf_socket_name("Clearcoat"),
        "clearcoat_roughness": bsdf_socket_name("Clearcoat Roughness"),
        "ior": "IOR",
        "transmission": bsdf_socket_name("Transmission"),
        "transmission_roughness": bsdf_socket_name("Transmission Roughness"),
        "emission": bsdf_socket_name("Emission"),
        "emission_strength": "Emission Strength",
        "alpha": "Alpha",
        "base_color": "Base Color",
    }
    # Remove entries where the socket was removed (value is None)
    return {k: v for k, v in mapping.items() if v is not None}


# === Compositor node tree ===


def get_compositor_node_tree(scene):
    """Get or create the compositor node tree for the scene."""
    if V >= (5, 0, 0):
        nt = scene.compositing_node_group
        if nt is None:
            nt = bpy.data.node_groups.new("Compositing", "CompositorNodeTree")
            scene.compositing_node_group = nt
        return nt
    else:
        return scene.node_tree


def enable_compositor(scene):
    """Enable compositor nodes (no-op on 5.0+ where it's always on)."""
    if V < (5, 0, 0):
        scene.use_nodes = True


# === Compositor output ===

# In bpy 5.0+ CompositorNodeOutputFile lost file_slots/base_path and always writes a single
# multilayer EXR (node.format.file_format only offers OPEN_EXR_MULTILAYER; per-item format
# overrides are accepted but ignored), which OpenCV cannot read. So on 5.0+ the render result
# is saved via the regular render output (write_still) driven by the compositor group output,
# and file output nodes are only used on older versions.
HAS_COMPOSITOR_FILE_OUTPUT = V < (5, 0, 0)


def set_composite_output(node_tree, socket):
    """Route the final image socket to the render output.

    On 5.0+ the compositor is a node group whose group output drives the render result;
    on older versions this is the Composite node."""
    if V >= (5, 0, 0):
        has_output = any(
            item.item_type == "SOCKET" and item.in_out == "OUTPUT"
            for item in node_tree.interface.items_tree
        )
        if not has_output:
            node_tree.interface.new_socket(
                name="Image", in_out="OUTPUT", socket_type="NodeSocketColor"
            )
        out_node = node_tree.nodes.new("NodeGroupOutput")
        node_tree.links.new(socket, out_node.inputs[0])
    else:
        comp = node_tree.nodes.new("CompositorNodeComposite")
        node_tree.links.new(socket, comp.inputs["Image"])


def add_data_file_output_node(node_tree, layers):
    """Create a file output node (bpy 5+) writing the given data passes as raw
    (non-color-managed) layers of a single multilayer EXR.

    Args:
        layers: dict mapping layer name to (socket, socket_type), where socket_type is a
            file output item type such as 'FLOAT' or 'RGBA'. FLOAT layers are stored as
            a '<name>.V' channel, RGBA layers as a '<name>' 4-channel image.

    Set ``node.directory`` and ``node.file_name`` before rendering; the file is written
    to ``directory/file_name.exr`` with no frame number appended."""
    node = node_tree.nodes.new(type="CompositorNodeOutputFile")
    for name, (socket, socket_type) in layers.items():
        item = node.file_output_items.new(socket_type=socket_type, name=name)
        item.save_as_render = False
        node_tree.links.new(socket, node.inputs[name])
    return node


def frame_suffix(frame_number):
    """Frame-number part of filenames written for a render.

    bpy 3/4 file output slots append the 4-digit frame number (the trailing dot of the
    slot path supplies the separator); on 5.0+ the file is written via write_still with
    an exact filepath and no frame number. Returns the dot-prefixed suffix or ''."""
    if V >= (5, 0, 0):
        return ""
    else:
        return f".{frame_number:04d}"


# === Render pass names ===


def diffcol_output_name():
    return "Diffuse Color" if V >= (5, 0, 0) else "DiffCol"


# === EEVEE engine name ===


def eevee_engine_name():
    if (4, 2, 0) <= V < (5, 0, 0):
        return "BLENDER_EEVEE_NEXT"
    return "BLENDER_EEVEE"


# === Light shadow ===


def get_light_cast_shadow(light_data):
    if V >= (4, 2, 0):
        return light_data.use_shadow
    return light_data.cycles.cast_shadow


def set_light_cast_shadow(light_data, value):
    if V >= (4, 2, 0):
        light_data.use_shadow = value
    else:
        light_data.cycles.cast_shadow = value


# === Motion blur position ===


def set_motion_blur_position(value):
    if V >= (4, 2, 0):
        bpy.context.scene.render.motion_blur_position = value
    else:
        bpy.context.scene.cycles.motion_blur_position = value


# === Animation fcurves ===


def get_fcurves(obj):
    """Get all fcurves for an animated object."""
    anim_data = obj.animation_data
    if anim_data is None or anim_data.action is None:
        return []
    if V >= (5, 0, 0):
        from bpy_extras.anim_utils import action_get_channelbag_for_slot
        channelbag = action_get_channelbag_for_slot(anim_data.action, anim_data.action_slot)
        if channelbag is None:
            return []
        return channelbag.fcurves
    else:
        return anim_data.action.fcurves
