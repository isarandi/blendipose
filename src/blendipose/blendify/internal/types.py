from __future__ import annotations

from typing import TYPE_CHECKING, Literal

RotationMode = Literal[
    "quaternionWXYZ", "quaternionXYZW", "rotvec", "rotmat", "look_at", "lookat",
    # Tait-Bryan (intrinsic, uppercase)
    "eulerXYZ", "eulerXZY", "eulerYXZ", "eulerYZX", "eulerZXY", "eulerZYX",
    # Tait-Bryan (extrinsic, lowercase)
    "eulerxyz", "eulerxzy", "euleryxz", "euleryzx", "eulerzxy", "eulerzyx",
    # Proper Euler (intrinsic, uppercase)
    "eulerXYX", "eulerXZX", "eulerYXY", "eulerYZY", "eulerZXZ", "eulerZYZ",
    # Proper Euler (extrinsic, lowercase)
    "eulerxyx", "eulerxzx", "euleryxy", "euleryzy", "eulerzxz", "eulerzyz",
]
AreaLightShape = Literal["square", "circle", "rectangle", "ellipse"]
FillType = Literal["NOTHING", "NGON", "TRIFAN"]
BasePrimitive = Literal["PLANE", "CUBE", "SPHERE"]

if TYPE_CHECKING:
    from typing import Union, Tuple

    import bpy
    import numpy as np

    BlenderGroup = Union[bpy.types.Collection, bpy.types.Object]
    Vector2d = Union[np.ndarray, Tuple[float, float]]
    Vector2di = Union[np.ndarray, Tuple[int, int]]
    Vector3d = Union[np.ndarray, Tuple[float, float, float]]
    Vector3di = Union[np.ndarray, Tuple[int, int, int]]
    Vector4d = Union[np.ndarray, Tuple[float, float, float, float]]
    Mat3x3 = Union[np.ndarray, Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]
    RotationParams = Union[Vector4d, Vector3d, Mat3x3, "Positionable", Tuple[Vector3d, float], Tuple["Positionable", float]]