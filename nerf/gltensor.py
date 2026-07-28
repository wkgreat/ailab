import torch


def normalize(v: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(v)
    if torch.any(norm < 1e-12):
        raise ValueError("normalize must not be a zero vector")
    return v / norm


def look_at(eye: torch.Tensor, center: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Create an OpenGL/GLM-style right-handed view matrix.

    The returned matrix is for column-vector math:
        p_camera = view @ p_world_h

    Camera local axes:
        +X right, +Y up, -Z forward.
    """
    if eye.shape != (3,) or center.shape != (3,) or up.shape != (3,):
        raise ValueError("eye, center, and up must all be shape (3,)")

    dtype = eye.dtype
    device = eye.device
    center = center.to(device=device, dtype=dtype)
    up = up.to(device=device, dtype=dtype)

    f = normalize(center - eye)
    s = normalize(torch.cross(f, up, dim=0))
    u = torch.cross(s, f, dim=0)

    view = torch.eye(4, dtype=dtype, device=device)
    view[0, :3] = s
    view[1, :3] = u
    view[2, :3] = -f
    view[0, 3] = -torch.dot(s, eye)
    view[1, 3] = -torch.dot(u, eye)
    view[2, 3] = torch.dot(f, eye)
    return view


def rotate(rad: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Create a 4x4 OpenGL/GLM-style rotation matrix.

    ``rad`` is in radians. ``axis`` must be a non-zero 3D vector.
    The returned matrix is for column-vector math:
        p_rotated = rot @ p_h
    """
    if axis.shape != (3,):
        raise ValueError("axis must be shape (3,)")

    dtype = axis.dtype
    device = axis.device
    rad = rad.to(device=device, dtype=dtype)

    x, y, z = normalize(axis)
    c = torch.cos(rad)
    s = torch.sin(rad)
    t = 1 - c

    rot = torch.eye(4, dtype=dtype, device=device)
    rot[:3, :3] = torch.stack(
        [
            torch.tensor([t * x * x + c, t * x * y - s * z, t * x * z + s * y]),
            torch.tensor([t * x * y + s * z, t * y * y + c, t * y * z - s * x]),
            torch.tensor([t * x * z - s * y, t * y * z + s * x, t * z * z + c]),
        ]
    )
    return rot
