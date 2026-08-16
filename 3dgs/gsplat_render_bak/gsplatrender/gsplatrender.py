from . import _C


def add_one(a):
    return _C.add_one(a)


def gsplat_rasterize(scene, pos3d, scale, rot, opacity, sh):
    return _C.gsplat_rasterize(scene, pos3d, scale, rot, opacity, sh)
