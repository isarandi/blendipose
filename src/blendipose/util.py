import functools
import os.path as osp

import cv2
import numba
import numpy as np
from blendipose.blendify import scene
from blendipose.blendify.colors import TextureColors, VertexUV


def add_billboard(
    texture, material, scale, rotation=None, rotation_mode='quaternionWXYZ', translation=(0, 0, 0)
):
    uv_map = VertexUV(np.array([[0, 0], [0, 1], [1, 1], [1, 0]]))
    colors = TextureColors(texture, uv_map, interpolation='Closest')
    verts = np.array([[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]]) * scale / 2
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return scene.renderables.add_mesh(
        verts,
        faces,
        material,
        colors,
        rotation=rotation,
        rotation_mode=rotation_mode,
        translation=translation,
    )


#
# def make_emissive(obj):
#     mat = obj.data.materials[0]  # Get the first material
#     nodes = mat.node_tree.nodes
#     links = mat.node_tree.links
#
#     # Clear existing nodes (if starting fresh)
#     # nodes.clear()
#
#     # Create Emission Shader
#     emission = nodes.new(type='ShaderNodeEmission')
#     emission.location = -200, 300
#
#     # Create Mix Shader
#     mix_shader = nodes.new(type='ShaderNodeMixShader')
#     mix_shader.location = 50, 300
#
#     # Get Principled BSDF and Material Output
#     principled_bsdf = next(node for node in nodes if node.type == 'BSDF_PRINCIPLED')
#     material_output = next(node for node in nodes if node.type == 'OUTPUT_MATERIAL')
#
#     texture_node = next((node for node in nodes if node.type == 'TEX_IMAGE' and any(
#         link.to_node == principled_bsdf for link in node.outputs['Color'].links)), None)
#
#     if texture_node:
#         # Disconnect the texture node from the base color
#         for link in texture_node.outputs['Color'].links:
#             if link.to_node == principled_bsdf and link.to_socket.name == 'Base Color':
#                 links.remove(link)
#                 pass
#
#     principled_bsdf.inputs['Base Color'].default_value = (0.0, 0.0, 0.0, 1)
#     emission.inputs['Strength'].default_value = 10.0
#     # Connect Nodes
#     links.new(texture_node.outputs['Color'], emission.inputs['Color'])
#     links.new(emission.outputs['Emission'], mix_shader.inputs[2])
#
#     links.new(principled_bsdf.outputs['BSDF'], mix_shader.inputs[1])
#     links.new(mix_shader.outputs['Shader'], material_output.inputs['Surface'])
#
#     mix_shader.inputs['Fac'].default_value = 0.05


# def alpha_blend(im, overlay_rgba):
#     alpha = overlay_rgba[:, :, 3:].astype(np.float32) / 255
#     return (np.asarray(im, np.float32) * (1 - alpha) + overlay_rgba[:, :, :3].astype(
#         np.float32) * alpha).astype(np.uint8)
#
#
# def blend_with_background(img: np.ndarray, bkg_color=(1., 1., 1.)) -> np.ndarray:
#     """Blend the RGBA image with uniform colored background, return RGB image
#
#     Args:
#         img: RGBA foreground image
#         bkg_color: RGB uniform background color (default is white)
#
#     Returns:
#         np.ndarray: RGB image blended with background
#     """
#     bkg_color = np.array(bkg_color)
#     if img.dtype == np.uint8:
#         bkg_color_uint8 = (bkg_color * 255).astype(np.uint8)
#         alpha = img[:, :, 3:4].astype(np.int32)
#         img_with_bkg = ((img[:, :, :3] * alpha + bkg_color_uint8[None, None, :] * (
#                     255 - alpha)) // 255).astype(np.uint8)
#     else:
#         alpha = img[:, :, 3:4]
#         img_with_bkg = (img[:, :, :3] * alpha + bkg_color[None, None, :] * (1. - alpha))
#     return img_with_bkg


def resize_image(im, dst_shape):
    if im.shape[:2] == dst_shape[:2]:
        return im
    interp = cv2.INTER_LINEAR if dst_shape[0] > im.shape[0] else cv2.INTER_AREA
    return cv2.resize(im, (dst_shape[1], dst_shape[0]), interpolation=interp)


def smootherstep(x, x0=0, x1=1, y0=0, y1=1):
    a = np.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    b = a**3 * (a * (a * 6.0 - 15.0) + 10.0)
    return y0 + b * (y1 - y0)


def replace_extension(path, new_ext):
    return osp.splitext(path)[0] + new_ext


@functools.lru_cache
@numba.njit(error_model='numpy', cache=True)
def get_srgb_decoder_lut(encoded_dtype=np.uint8):
    maxval = np.iinfo(encoded_dtype).max
    length = int(maxval) + 1
    lut = np.zeros(length, np.float64)
    for i in numba.prange(length):
        x = i / maxval
        if x <= 0.04045:
            lut[i] = x / 12.92
        else:
            lut[i] = ((x + 0.055) / 1.055) ** 2.4
        if lut[i] < 0:
            lut[i] = 0
        elif lut[i] > 1:
            lut[i] = 1
    return (lut * (1 << 16 - 1)).astype(np.uint16)


@functools.lru_cache
@numba.njit(error_model='numpy', cache=True)
def get_srgb_encoder_lut(encoded_dtype=np.uint8):
    lut = np.zeros(1 << 16, np.float64)
    for i in numba.prange(1 << 16):
        x = i / (1 << 16 - 1)
        if x <= 0.0031308:
            lut[i] = x * 12.92
        else:
            lut[i] = 1.055 * x ** (1 / 2.4) - 0.055
        if lut[i] < 0:
            lut[i] = 0
        elif lut[i] > 1:
            lut[i] = 1

    maxval = np.iinfo(encoded_dtype).max
    return (lut * maxval).astype(encoded_dtype)


def alpha_blend(overlay_rgba, background=255, use_srgb_lut=True, dst=None, dtype=np.uint8):
    if np.isscalar(background):
        background = np.array([background, background, background], dtype)
    else:
        background = np.asarray(background, dtype)

    if background.ndim == 1:
        if use_srgb_lut:
            return _alpha_blend_singlecolor(
                overlay_rgba,
                background,
                get_srgb_encoder_lut(dtype),
                get_srgb_decoder_lut(dtype),
                dst,
                dtype,
            )
        else:
            return _alpha_blend_singlecolor_nolut(overlay_rgba, background, dst, dtype)
    else:
        if use_srgb_lut:
            return _alpha_blend(
                overlay_rgba,
                background,
                get_srgb_encoder_lut(dtype),
                get_srgb_decoder_lut(dtype),
                dst,
                dtype,
            )
        else:
            return _alpha_blend_nolut(overlay_rgba, background, dst, dtype)


@numba.njit(error_model='numpy', cache=True)
def _alpha_blend(overlay_rgba, im, srgb_encoder_lut, srgb_decoder_lut, dst=None, dtype=np.uint8):
    if dst is None:
        dst = np.empty((overlay_rgba.shape[0], overlay_rgba.shape[1], 3), dtype)

    rgba_flat = overlay_rgba.reshape(-1, 4)
    rgb_flat = im.reshape(-1, 3)
    dst_flat = dst.reshape(-1, 3)

    maxval = np.iinfo(dtype).max

    for i in numba.prange(rgba_flat.shape[0]):
        alpha = np.float32(rgba_flat[i, 3]) / maxval
        for c in range(3):
            rgb_linear1 = srgb_decoder_lut[rgba_flat[i, c]]
            rgb_linear2 = srgb_decoder_lut[rgb_flat[i, c]]
            combined_linear = (
                np.float32(rgb_linear2)
                + (np.float32(rgb_linear1) - np.float32(rgb_linear2)) * alpha
            )
            if combined_linear < 0:
                combined_linear = 0
            elif combined_linear > 65535:
                combined_linear = 65535

            dst_flat[i, c] = srgb_encoder_lut[np.uint16(combined_linear)]
    return dst


@numba.njit(error_model='numpy', cache=True)
def _alpha_blend_singlecolor(
    overlay_rgba, color, srgb_encoder_lut, srgb_decoder_lut, dst=None, dtype=np.uint8
):
    if dst is None:
        dst = np.empty((overlay_rgba.shape[0], overlay_rgba.shape[1], 3), dtype)

    rgba_flat = overlay_rgba.reshape(-1, 4)
    rgb_linear2_float = srgb_decoder_lut[color].astype(np.float32)
    dst_flat = dst.reshape(-1, 3)

    maxval = np.iinfo(dtype).max

    for i in numba.prange(rgba_flat.shape[0]):
        alpha = np.float32(rgba_flat[i, 3]) / maxval
        for c in range(3):
            rgb_linear1 = srgb_decoder_lut[rgba_flat[i, c]]
            combined_linear = (
                rgb_linear2_float[c] + (np.float32(rgb_linear1) - rgb_linear2_float[c]) * alpha
            )
            if combined_linear < 0:
                combined_linear = 0
            elif combined_linear > 65535:
                combined_linear = 65535

            dst_flat[i, c] = srgb_encoder_lut[np.uint16(combined_linear)]
    return dst


@numba.njit(error_model='numpy', cache=True)
def _alpha_blend_singlecolor_nolut(overlay_rgba, color, dst=None, dtype=np.uint8):
    if dst is None:
        dst = np.empty((overlay_rgba.shape[0], overlay_rgba.shape[1], 3), dtype)

    rgba_flat = overlay_rgba.reshape(-1, 4)
    rgb2_float = color.astype(np.float32)
    dst_flat = dst.reshape(-1, 3)

    maxval = np.iinfo(dtype).max

    for i in numba.prange(rgba_flat.shape[0]):
        alpha = np.float32(rgba_flat[i, 3]) / maxval
        for c in range(3):
            rgb1 = rgba_flat[i, c]
            combined = rgb2_float[c] + (np.float32(rgb1) - rgb2_float[c]) * alpha
            if combined < 0:
                combined = 0
            elif combined > maxval:
                combined = maxval
            dst_flat[i, c] = dtype(combined)
    return dst


@numba.njit(error_model='numpy', cache=True)
def _alpha_blend_nolut(overlay_rgba, im, dst=None, dtype=np.uint8):
    if dst is None:
        dst = np.empty((overlay_rgba.shape[0], overlay_rgba.shape[1], 3), dtype)

    rgba_flat = overlay_rgba.reshape(-1, 4)
    rgb_flat = im.reshape(-1, 3)
    dst_flat = dst.reshape(-1, 3)

    maxval = np.iinfo(dtype).max

    for i in numba.prange(rgba_flat.shape[0]):
        alpha = np.float32(rgba_flat[i, 3]) / maxval
        for c in range(3):
            rgb1 = rgba_flat[i, c]
            rgb2 = rgb_flat[i, c]
            combined = np.float32(rgb2) + (np.float32(rgb1) - np.float32(rgb2)) * alpha
            if combined < 0:
                combined = 0
            elif combined > maxval:
                combined = maxval

            dst_flat[i, c] = dtype(np.rint(combined))
    return dst


MM_TO_UNIT = 1 / 1000
UNIT_TO_MM = 1000
WORLD_UP = np.array([0, 0, 1])
WORLD_TO_BLENDERWORLD_ROTATION_MAT = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)

# Our convention follows OpenCV, where x is right, y is down, and z is forward
# Blender's convention is x is right, y is up, and z is backward
CAM_TO_BLENDERCAM_ROTATION_MAT = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)


def rotation_mat(up):
    up = unit_vector(up)
    rightlike = np.array([1, 0, 0])
    if np.allclose(up, rightlike):
        rightlike = np.array([0, 1, 0])

    forward = unit_vector(np.cross(up, rightlike))
    right = np.cross(forward, up)

    # In Blender, the world coordinate system is right-handed, with z pointing up
    return np.row_stack([right, forward, up])


def unit_vector(vectors, axis=-1):
    norm = np.linalg.norm(vectors, axis=axis, keepdims=True)
    return vectors / norm


def world_to_blender(points):
    points = np.asarray(points)
    if points.ndim == 1:
        return WORLD_TO_BLENDERWORLD_ROTATION_MAT @ points * MM_TO_UNIT
    else:
        return points @ WORLD_TO_BLENDERWORLD_ROTATION_MAT.T * MM_TO_UNIT


def blender_to_world(points):
    points = np.asarray(points)
    if points.ndim == 1:
        return WORLD_TO_BLENDERWORLD_ROTATION_MAT.T @ points * UNIT_TO_MM
    else:
        return points @ WORLD_TO_BLENDERWORLD_ROTATION_MAT * UNIT_TO_MM


def set_world_up(world_up):
    global WORLD_UP
    WORLD_UP = np.asarray(world_up)
    global WORLD_TO_BLENDERWORLD_ROTATION_MAT
    WORLD_TO_BLENDERWORLD_ROTATION_MAT = rotation_mat(WORLD_UP)
