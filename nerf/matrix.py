import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("normalize must not be a zero vector")
    return v / norm


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Create a world-to-camera look-at matrix.

    The camera uses the NeRF/OpenGL convention: local ``-Z`` points forward,
    ``+X`` points right, and ``+Y`` points up.
    """
    c2w = look_at_c2w(eye, target, up)
    return np.linalg.inv(c2w)


def look_at_c2w(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Create a camera-to-world look-at pose matrix.

    This is suitable for the NeRF notebook in this directory, where rays are
    transformed with ``dirs @ pose[:3, :3].T`` and ``pose[:3, 3]`` is the camera
    origin in world coordinates.
    """

    if eye.shape != (3,) or target.shape != (3,) or up.shape != (3,):
        raise ValueError("eye, target, and up must all be shape (3,)")

    forward = normalize(target - eye)
    z_axis = -forward
    x_axis = normalize(np.cross(up, z_axis))
    y_axis = np.cross(z_axis, x_axis)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    c2w[:3, 3] = eye
    return c2w


def rotate(rad: float, axis: np.ndarray) -> np.ndarray:
    """Create a 4x4 rotation matrix around an arbitrary axis.

    ``rad`` is in radians. ``axis`` must be a non-zero 3D vector.
    """
    axis = np.asarray(axis, dtype=np.float64)
    if axis.shape != (3,):
        raise ValueError("axis must be shape (3,)")

    x, y, z = normalize(axis)
    c = np.cos(rad)
    s = np.sin(rad)
    t = 1.0 - c

    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )
    return mat
