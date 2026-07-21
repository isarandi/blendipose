# Blendify Future Improvements

Ideas for making the blendify API more agent-friendly and robust.

## Tier 2: Moderate Effort

### Replace Singleton with Explicit Context
The `Scene` class uses a `Singleton` metaclass that raises `RuntimeError` on second instantiation,
which is hostile to retry logic and testing. Replace with a factory or explicit context manager
pattern (e.g. `with BlenderScene() as scene:`). The singleton instance in `__init__.py` can remain
as a convenience default.

### Typed Return Values for Render
`Scene.render()` returns raw file paths as strings. Consider returning a structured result
(e.g. a dataclass with `.color`, `.depth`, `.albedo` fields) so agents don't have to guess
what outputs are available.

### Expand `__all__` / Flat Imports
Currently users must do deep imports like `from blendify.materials.metal import MetalMaterial`.
Add re-exports in subpackage `__init__.py` files so `from blendify.materials import MetalMaterial`
works. Keep the top-level `__init__.py` minimal.

### Add `scene.remove()` / `scene.clear()`
No public API to remove individual objects or reset the scene. Agents doing iterative
rendering need this.

## Tier 3: Larger Refactors

### Split `Scene` into Focused Managers
`Scene` is a god-class (~600 lines) that manages renderables, lights, camera, and rendering.
Split into `RenderableManager`, `LightManager`, `CameraManager` with `Scene` as a thin facade.

### Decouple Material from Renderable Construction
Currently materials and colors must be passed at construction time and flow through 5+ levels
of inheritance. Consider a builder pattern or post-construction `.material = ...` assignment
(partially supported via `update_material()` but the constructor still requires them).

### Simplify the Inheritance Hierarchy
The renderable hierarchy is 5 levels deep:
`Positionable -> Renderable -> RenderableObject -> MeshPrimitive -> CubeMesh`.
Consider flattening with composition (e.g. `Positionable` as a mixin or component rather
than a base class).

### Better Error Messages
Many assertions use generic messages. Add context about what the agent/user should do
to fix the issue (e.g. "expected 3D array of shape (N, 3), got shape {arr.shape}").

## Tier 4: Nice-to-Haves

### Animation / Keyframe API
Expose Blender's keyframe API for property animation (e.g. camera moves, object transforms
over time). Currently only `update_vertices()` exists for per-frame mesh updates.

### Scene Serialization
Add `scene.to_dict()` / `Scene.from_dict()` for saving and restoring scene state,
useful for agent checkpointing and reproducibility.

### HDRI Environment Lighting
Add a convenience method for loading HDRI environment maps, a very common operation
that currently requires raw bpy calls.

### Compositor / Post-Processing API
Expose Blender's compositor nodes for post-processing (bloom, glare, color correction)
through a typed API.

## bpy 5 Support

`Scene.render()` and `Scene.preview()` are routed through `compat.py` and work on bpy 3.6-5.x.
On 5.0+ the compositor lives in `scene.compositing_node_group` and the color result is saved
via the regular render output (`write_still=True`) driven by the compositor group output
(`compat.set_composite_output`), because `CompositorNodeOutputFile` in 5.0+ can only write a
single multilayer EXR per node (`node.format.file_format` offers only `OPEN_EXR_MULTILAYER`;
per-item `override_node_format` is accepted but has no effect on the written container), which
OpenCV cannot read. On older versions the original file-output-node path is kept unchanged.

### `save_depth` / `save_albedo` on bpy 5+
Supported via the multilayer EXR that the 5.x file output node writes: the Depth and
Diffuse Color passes are added as raw (`save_as_render=False`) items of one data file
output node (`compat.add_data_file_output_node`), and `Scene._read_exr_layers` reads the
resulting EXR with the `OpenEXR` package (optional dependency, `pip install blendipose[exr]`;
only needed on 5.0+ when depth/albedo saving is used — OpenCV cannot read multilayer EXR).
Albedo is converted from scene-linear float to the sRGB-encoded integer image the pre-5 PNG
outputs produced (`Scene._encode_display_image`). Depth output is numerically identical
across 3.6/4.5/5.0.
