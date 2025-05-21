import contextlib
import random

import blendipose.blendify.cameras
import blendipose.blendify.materials.base
import blendipose.util as util
import bpy
import cameravision
import cv2
import numpy as np
import shapely
import simplepyutils as spu
import trimesh
import framepump
from typing import List
from blendipose.blendify import scene
from blendipose.blendify.colors import TextureColorsViaTempFile, UniformColors, VertexUV
from blendipose.blendify.materials import PrincipledBSDFMaterial, PrincipledBSDFWireframeMaterial


class Renderer(contextlib.AbstractContextManager):
    def __init__(
        self,
        resolution,
        body_model_faces,
        out_video_path=None,
        out_fps=None,
        audio_path=None,
        show_image=True,
        num_views=1,
        frame_background=False,
        show_ground_plane=True,
        ground_plane_height=1000,
        world_up=(0, -1, 0),
        show_camera=True,
        preview_resolution=None,
        body_alpha=0.75,
        motion_blur=False,
        depth_of_field=False,
        adaptive_sampling_threshold=1,
        samples=16,
    ):
        self.viz = preview_resolution is not None

        cycles = bpy.context.scene.cycles
        # cycles.use_adaptive_sampling = True
        # cycles.adaptive_threshold = 10
        # cycles.min_samples = 1
        self.motion_blur = motion_blur

        render = bpy.context.scene.render
        if out_fps:
            render.fps = int(round(out_fps))
            render.fps_base = render.fps / out_fps
        if motion_blur:
            render.use_motion_blur = True
            render.motion_blur_shutter = 0.5
            cycles.motion_blur_position = 'CENTER'

        util.set_world_up(world_up)

        self.adaptive_sampling_threshold = adaptive_sampling_threshold
        self.samples = samples

        self.preview_resolution = (
            preview_resolution if preview_resolution is not None else resolution
        )

        self.show_ground = show_ground_plane
        self.show_camera = show_camera
        self.depth_of_field = depth_of_field
        self.ground = self._add_ground(ground_plane_height / 1000) if self.show_ground else None
        self._add_lights()
        if show_camera:
            self.camera_displays = [CameraDisplay(show_image=show_image) for _ in range(num_views)]
        else:
            self.camera_displays = []
        self.show_image = show_image
        self.body_meshes = BodyMeshCollection(body_model_faces, body_alpha)

        self.resolution = resolution
        self.fps = out_fps
        self.stencil_mesh = None
        self.distort_image = lambda x: x
        self.view_camera = None
        self.prev_undist_cam = None

        self.set_view_camera(cameravision.Camera.from_fov(55, resolution))
        self.frame_background = frame_background
        self.video_writer = framepump.VideoWriter(out_video_path, out_fps, audio_path)

    def new_sequence_output(self, out_video_path, fps, audio_source_path=None):
        self.video_writer.start_sequence(
            out_video_path, fps=fps, audio_source_path=audio_source_path
        )
        bpy.context.scene.frame_set(0)
        scene._frame_number = 0
        bpy.context.scene.render.fps = int(round(fps))
        bpy.context.scene.render.fps_base = bpy.context.scene.render.fps / fps
        self.body_meshes.prev_vertices = None
        self.fps = fps

    def update(
        self,
        frame=None,
        mask=None,
        boxes=(),
        poses=(),
        camera=None,
        poses_true=(),
        poses_alt=(),
        vertices=(),
        vertices_true=(),
        vertices_alt=(),
        viz_camera=None,
        viz_imshape=None,
        uncerts=None,
        uncerts_alt=None,
        colors=None,
    ):
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), np.float32)

        if uncerts is not None:
            vertices = [
                np.concatenate([v, uncert[..., np.newaxis]], axis=-1)
                for v, uncert in zip(vertices, uncerts)
            ]
        if uncerts_alt is not None:
            vertices_alt = [
                np.concatenate([v, uncert[..., np.newaxis]], axis=-1)
                for v, uncert in zip(vertices_alt, uncerts_alt)
            ]

        self.body_meshes.update(vertices, self.fps, colors=colors)

        if self.show_camera:
            self.camera_displays[0].update(camera, frame)

        if self.depth_of_field:
            if self.show_image and np.linalg.norm(viz_camera.t - camera.t) < 500:
                focus_distance = None
            elif len(vertices) > 0:
                focus_distance = np.linalg.norm(np.mean(vertices[0], axis=0) - viz_camera.t) / 1000
            else:
                focus_distance = 1.0

        else:
            focus_distance = None

        self.set_view_camera(viz_camera, focus_distance=focus_distance)
        rendered_im = self.render_image()
        if mask is not None:
            # current_mask = rendered_im[:, :, 3]
            # current_mask == 1.0
            rendered_im[:, :, 3] *= mask

        if self.frame_background:
            frame = resize_image(frame, rendered_im.shape, padding_value=65535)
            im_with_bg = util.alpha_blend(rendered_im, frame, dtype=np.uint16)
        else:
            im_with_bg = util.alpha_blend(rendered_im, 65535, dtype=np.uint16)

        if self.viz:
            im_show = util.resize_image(im_with_bg, self.preview_resolution[::-1])
            cv2.imshow('img', cv2.cvtColor(im_show, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if self.video_writer.is_active():
            self.video_writer.append_data(im_with_bg)

        return rendered_im, im_with_bg

    def render_image(self):
        bpy.context.scene.cycles.seed = random.randint(0, 10_000)
        undistorted_image = scene.render(
            use_denoiser=True,
            adaptive_sampling_threshold=self.adaptive_sampling_threshold,
            samples=self.samples,
            color_dtype=np.uint16,
        )
        return self.distort_image(undistorted_image)

    def set_view_camera(self, cameravision_camera, focus_distance=None):
        if False and self.view_camera is not None and self.view_camera.is_equal(cameravision_camera):
            return

        if self.stencil_mesh is not None:
            scene.renderables.remove(self.stencil_mesh)
            self.stencil_mesh = None

        c = cameravision_camera.copy()
        if c.has_distortion():
            undist_cam, box, poly = c.undistort_precise(
                alpha_balance=1.0, imshape_distorted=self.resolution[::-1], inplace=False
            )
            imshape_undistorted = np.ceil(box[2:][::-1]).astype(int)

            def distort_fn(image):
                return cameravision.reproject_image(
                    image,
                    undist_cam,
                    c,
                    output_imshape=self.resolution[::-1],
                    cache_maps=True,
                    use_linear_srgb=True,
                    antialias_factor=2,
                    interp=cv2.INTER_CUBIC,
                )[0]

            self.distort_image = distort_fn
            self._add_holdout_stencil(undist_cam, poly)
        else:
            undist_cam = c
            self.distort_image = lambda x: x
            imshape_undistorted = self.resolution[::-1]

        set_cameravision_camera(
            scene,
            undist_cam,
            imshape_undistorted[:2][::-1],
            near=0.1,
            far=10000,
            focus_distance=focus_distance,
        )

        if self.motion_blur:
            insert_camera_keyframe(scene.camera, bpy.context.scene.frame_current + 1)
            if self.prev_undist_cam is not None:
                undist_cam_extrap = extrapolate(self.prev_undist_cam, undist_cam)
                set_cameravision_camera(
                    scene, undist_cam_extrap, imshape_undistorted[:2][::-1], near=0.1, far=10000
                )
                insert_camera_keyframe(scene.camera, bpy.context.scene.frame_current + 2)
            self.prev_undist_cam = undist_cam

        self.view_camera = cameravision_camera.copy()

    def _add_holdout_stencil(self, camera, polygon):
        # Add a flat rectangle in front of the camera, with a cutout for the given polygon
        # ie, the polygon is a hole in the rectangle, and the rectangle has a holdout shader
        # material
        # This is useful for avoiding rendering unnecessary parts of the scene

        stencil_mesh = poly_to_stencil_trimesh(polygon)
        vs = camera.image_to_world(stencil_mesh.vertices[:, :2], camera_depth=0.101)
        self.stencil_mesh = scene.renderables.add_mesh(
            util.world_to_blender(vs),
            stencil_mesh.faces,
            material=HoldoutMaterial(),
            colors=UniformColors((0, 0, 0, 0)),
        )
        self.stencil_mesh.emit_shadows = False
        self.stencil_mesh._blender_mesh.is_holdout = True

    def _add_ground(self, height):
        texture = np.zeros((20, 20, 3), np.uint8)
        texture[::2] = 255 - texture[::2]
        texture[:, ::2] = 255 - texture[:, ::2]
        texture[texture == 255] = 155
        texture[texture == 0] = 100

        # up normal is the normal vector of the plane
        # height is the dot product of the points on the plane and the normal vector
        mesh = util.add_billboard(
            texture=texture,
            scale=20,
            translation=(0, 0, height),
            material=PrincipledBSDFMaterial(specular=0.0, alpha=0.6, roughness=1, metallic=0.0),
        )

        return mesh

    def _add_lights(self):
        strengths = np.array([0.9 * 2, 0.9 * 2, 0.7 * 2, 0.9 * 0.7, 0.9 * 0.7, 0.7 * 0.7])
        # by default the sun shines towards the negative z-axis (ie, downwards)
        rotations = [
            (10, 45, 0),
            (145, 25, 0),
            (-125, 35, 0),
            (10, 180 + 45, 0),
            (145, 180 + 25, 0),
            (-125, 180 + 35, 0),
        ]

        for strength, rotation in zip(strengths, rotations):
            scene.lights.add_sun(
                strength=strength,
                angular_diameter=0.1,
                rotation=rotation,
                rotation_mode='eulerZXY',
                cast_shadows=True,
            )
        scene.lights.set_background_light(strength=0.35, color=(1.0, 1.0, 1.0))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.video_writer.close()


def insert_camera_keyframe(camera, frame):
    camera = camera.blender_camera
    camera.keyframe_insert(data_path="location", frame=frame)
    camera.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    camera.data.keyframe_insert(data_path="lens", frame=frame)
    camera.data.keyframe_insert(data_path="shift_x", frame=frame)
    camera.data.keyframe_insert(data_path="shift_y", frame=frame)
    for fcurve in [
        *camera.animation_data.action.fcurves,
        *camera.data.animation_data.action.fcurves,
    ]:
        fcurve.keyframe_points[-1].interpolation = 'LINEAR'


class BodyMeshCollection:
    def __init__(self, faces, alpha, color=None, motion_blur=True):
        self.faces = faces
        self.meshes: List[blendipose.blendify.renderables.Mesh] = []
        color = (0 / 255, 100 / 255, 255 / 255, alpha)
        self.material = PrincipledBSDFMaterial(
            specular=1,
            metallic=0.25,
            roughness=0.54,
            emission_strength=0,  # 0.08,
            alpha=alpha,
            emission=color,
        )
        self.colors = UniformColors(color)

        self.prev_vertices = None
        self.motion_blur = motion_blur

    def _create_mesh(self, vertices, color=None):
        if color is None:
            colors = self.colors
        else:
            colors = UniformColors(color)

        mesh = scene.renderables.add_mesh(
            vertices=util.world_to_blender(vertices),
            faces=self.faces,
            material=self.material,
            colors=colors,
        )
        mesh.set_smooth()

        if self.motion_blur:
            mesh._blender_mesh.attributes.new(name='velocity', type='FLOAT_VECTOR', domain='POINT')
        return mesh

    def update(self, vertices, fps, colors=None):
        if colors is not None:
            for m in self.meshes:
                scene.renderables.remove(m)
            del self.meshes[:]
            self.meshes += [self._create_mesh(v, c) for v, c in zip(vertices, colors)]
            return

        n_people_new = len(vertices)
        n_people_old = len(self.meshes)

        if self.motion_blur:
            if self.prev_vertices is not None:
                velocities = [
                    (v - prev_v) * fps for v, prev_v in zip(vertices, self.prev_vertices)
                ]
                for m, vel in zip(self.meshes, velocities):
                    update_mesh_velocities(m, util.world_to_blender(vel))
            self.prev_vertices = vertices

        for m, v in zip(self.meshes, vertices):
            m.update_vertices(util.world_to_blender(v))

        if n_people_new < n_people_old:
            for m in self.meshes[n_people_new:]:
                scene.renderables.remove(m)
            del self.meshes[n_people_new:]
        elif n_people_new > n_people_old:
            self.meshes += [self._create_mesh(v) for v in vertices[n_people_old:]]


def update_mesh_velocities(mesh, velocities):
    for attr, vel in zip(mesh._blender_mesh.attributes['velocity'].data, velocities):
        attr.vector = vel


class CameraDisplay:
    def __init__(self, camera_depth=500, show_image=True, add_campoint=True):
        self.camera_depth = camera_depth
        self.add_campoint = add_campoint
        self.show_image = show_image
        self.texture_colors = None
        self.image_mesh = None
        self.camera_mesh = None
        self.imshape = None
        self.is_initialized = False
        self.uv_map = None

    def _initial_update(self, camera, frame):
        self.imshape = frame.shape
        h, w = frame.shape[:2]
        image_corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
        image_corners_world = camera.image_to_world(image_corners, camera_depth=self.camera_depth)
        if self.add_campoint:
            vertices = np.array([camera.t, *image_corners_world])
            faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]])
        else:
            vertices = np.array([camera.t - camera.R[2] * 0.5, *image_corners_world])
            faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]])

        vertices = util.world_to_blender(vertices)

        self.camera_mesh = scene.renderables.add_mesh(
            vertices,
            faces,
            material=PrincipledBSDFWireframeMaterial(),
            colors=UniformColors((0, 0, 0, 0)),
        )
        self.camera_mesh.emit_shadows = False
        self.uv_map = VertexUV(np.array([[0, 1], [1, 1], [1, 0], [0, 0]]))
        self.texture_colors = TextureColorsViaTempFile(frame, self.uv_map, has_alpha=False)
        if self.show_image:
            self.image_mesh = scene.renderables.add_mesh(
                vertices[1:],
                faces[:2],
                colors=self.texture_colors,
                material=PrincipledBSDFMaterial(
                    specular=0.0,
                    alpha=0.5,
                    roughness=1,
                    metallic=0.0,
                    emission_strength=1,
                    base_color=(0, 0, 0, 1),
                    use_colors_for_emission=True,
                ),
            )
            self.image_mesh.emit_shadows = False

        self.is_initialized = True

    def update(self, camera, frame):
        camera = camera.copy()

        if not self.is_initialized:
            self._initial_update(camera, frame)

        h, w = frame.shape[:2]
        image_corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
        image_corners_world = camera.image_to_world(image_corners, camera_depth=self.camera_depth)
        vertices = np.array([camera.t, *image_corners_world])
        vertices = util.world_to_blender(vertices)
        self.camera_mesh.update_vertices(vertices)
        if self.image_mesh is not None:
            self.image_mesh.update_vertices(vertices[1:])

            if frame.shape != self.imshape:
                self.imshape = frame.shape
                self.texture_colors = TextureColorsViaTempFile(frame, self.uv_map)
                self.image_mesh.update_colors(self.texture_colors)
            else:
                self.texture_colors.update_pixels(frame)


def set_cameravision_camera(
    scene,
    camera,
    resolution,
    near=0.1,
    far=100.0,
    tag='camera',
    resolution_percentage=100,
    focus_distance=None,
):
    # To go from blender world to blender cam, we go from
    # 1) blender world to world (util.WORLD_TO_BLENDERWORLD_ROTATION_MAT.T),
    # 2) world to cam (camera.R),
    # 3) cam to blender cam (util.CAM_TO_BLENDERCAM_ROTATION_MAT)

    R = util.CAM_TO_BLENDERCAM_ROTATION_MAT @ camera.R @ util.WORLD_TO_BLENDERWORLD_ROTATION_MAT.T
    t = util.world_to_blender(camera.t)
    f = camera.intrinsic_matrix[0, 0]
    center = camera.intrinsic_matrix[:2, 2] / resolution

    blendify_cam = scene.camera
    if isinstance(blendify_cam, blendipose.blendify.cameras.PerspectiveCamera) and np.all(
        blendify_cam.resolution == resolution
    ):
        # R is the rotation matrix from the world to camera, so we must transpose it
        # because this function expects the transformation from camera to world
        blendify_cam.set_position(translation=t, rotation=R.T, rotation_mode='rotmat')
        blendify_cam.focal_dist = f
        blendify_cam.center = center
        blendify_cam.far = far
        blendify_cam.near = near
    else:
        scene.set_perspective_camera(
            resolution,
            rotation=R.T,
            rotation_mode='rotmat',
            translation=t,
            focal_dist=f,
            center=center,
            near=near,
            far=far,
            tag=tag,
            resolution_percentage=resolution_percentage,
        )

    if focus_distance is not None:
        c = scene.camera.blender_camera
        c.data.dof.use_dof = True
        c.data.dof.focus_distance = focus_distance
        c.data.dof.aperture_fstop = 30
        c.data.dof.aperture_blades = 3


def extrapolate(cam1, cam_mid):
    t2 = 2 * cam_mid.t - cam1.t
    R2 = cam_mid.R @ cam1.R.T @ cam_mid.R

    center1 = cam1.intrinsic_matrix[:2, 2]
    center_mid = cam_mid.intrinsic_matrix[:2, 2]
    center2 = 2 * center_mid - center1

    f1 = cam1.intrinsic_matrix[0, 0]
    fmid = cam_mid.intrinsic_matrix[0, 0]
    f2 = 2 * fmid - f1

    cam2 = cam_mid.copy()
    cam2.t = t2
    cam2.R = R2

    cam2.intrinsic_matrix[0, 0] = f2
    cam2.intrinsic_matrix[1, 1] = f2
    cam2.intrinsic_matrix[:2, 2] = center2
    return cam2


def resize_image(im, dst_shape, padding_value=255):
    imshape = im.shape[:2]
    dst_shape = dst_shape[:2]
    if im.shape == dst_shape:
        return im

    interp = cv2.INTER_LINEAR if dst_shape[0] > imshape[0] else cv2.INTER_AREA
    factor = np.min(np.array(dst_shape, dtype=np.float32) / np.array(imshape, dtype=np.float32))
    dst_shape2 = spu.rounded_int_tuple(np.array(imshape, dtype=np.float32) * factor)
    resized = cv2.resize(im, (dst_shape2[1], dst_shape2[0]), interpolation=interp)
    # now pad
    padding_total = np.array(dst_shape) - np.array(dst_shape2)
    padding_top = padding_total[0] // 2
    padding_bottom = padding_total[0] - padding_top
    padding_left = padding_total[1] // 2
    padding_right = padding_total[1] - padding_left
    padded = cv2.copyMakeBorder(
        resized,
        padding_top,
        padding_bottom,
        padding_left,
        padding_right,
        cv2.BORDER_CONSTANT,
        value=padding_value,
    )
    return padded


def polygon_to_trimesh(polygon: shapely.Polygon):
    verts2d, faces = trimesh.creation.triangulate_polygon(polygon)
    verts = np.column_stack([verts2d, np.ones(len(verts2d))])
    return trimesh.Trimesh(vertices=verts, faces=faces)


def multipolygon_to_trimesh(multipolygon: shapely.MultiPolygon):
    return trimesh.util.concatenate([polygon_to_trimesh(p) for p in multipolygon.geoms])


def poly_to_stencil_trimesh(poly: shapely.Polygon):
    polybox = shapely.box(*poly.bounds)
    stencilpoly = polybox - poly.buffer(5)
    if isinstance(stencilpoly, shapely.Polygon):
        return polygon_to_trimesh(stencilpoly)
    else:
        return multipolygon_to_trimesh(stencilpoly)


class HoldoutMaterial(blendipose.blendify.materials.base.Material):
    """A class which manages a Holdout Blender material."""

    def __init__(self, use_backface_culling=True):
        super().__init__(use_backface_culling=use_backface_culling)

    def create_material(
        self, name: str = "holdout_material"
    ) -> blendipose.blendify.materials.base.MaterialInstance:
        object_material = bpy.data.materials.new(name=name)
        object_material.use_nodes = True
        object_material.use_backface_culling = self._use_backface_culling

        nodes = object_material.node_tree.nodes
        links = object_material.node_tree.links

        # Remove the default Principled BSDF node if it exists
        # if "Principled BSDF" in nodes:
        #    nodes.remove(nodes["Principled BSDF"])

        # Create the Holdout node
        holdout_node = nodes.new("ShaderNodeHoldout")
        holdout_node.location = (0, 0)

        # Link the Holdout node's output to the Material Output's Surface input
        material_output = nodes.get("Material Output")
        links.new(holdout_node.outputs[0], material_output.inputs["Surface"])

        material_instance = blendipose.blendify.materials.base.MaterialInstance(
            blender_material=object_material,
            inputs={
                'Color': nodes["Principled BSDF"].inputs["Base Color"],
                'Alpha': nodes["Principled BSDF"].inputs["Alpha"],
            },
        )
        return material_instance
