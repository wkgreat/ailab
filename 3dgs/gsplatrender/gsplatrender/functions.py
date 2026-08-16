import numpy as np

from . import _C


def add_one(a: np.ndarray) -> np.ndarray:
    return _C.add_one(a)


def gsplat_rasterize(
    scene,
    pos3d: np.ndarray,
    scale: np.ndarray,
    rot: np.ndarray,
    opacity: np.ndarray,
    sh: np.ndarray,
) -> np.ndarray:
    return _C.gsplat_rasterize(scene, pos3d, scale, rot, opacity, sh)
