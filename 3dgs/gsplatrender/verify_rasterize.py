"""复现 gsplat_rasterize_test.ipynb 的调用，验证 kernel 不再报 illegal memory access"""
import sys

sys.path.insert(0, r"E:\codes\projects\ailab\3dgs\gsplatrender")

import numpy as np
from plyfile import PlyData
from gsplatrender import gsplat_rasterize


def lookAt(eye, center, up):
    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3], m[1, 3], m[2, 3] = -s @ eye, -u @ eye, f @ eye
    return m


def perspective(fovy, aspect, near, far):
    f = 1.0 / np.tan(fovy / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


# ---- 数据加载（与 notebook SplatsData.from_ply 一致） ----
vertex = PlyData.read(r"../data/cactus_splat3.ply")["vertex"]
nsplats = len(vertex["x"])
pos3d = np.stack(
    [vertex["x"], vertex["y"], vertex["z"], np.ones(nsplats, dtype=np.float32)],
    axis=1,
).astype(np.float32)
opacity = np.asarray(vertex["opacity"], dtype=np.float32).reshape(nsplats, 1)
scale = np.stack(
    [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"],
     np.zeros(nsplats, dtype=np.float32)],
    axis=1,
).astype(np.float32)
rot = np.stack([vertex[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float32)
sh_dc = np.stack([vertex[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float32)[:, None, :]
sh_rest = np.stack([vertex[f"f_rest_{i}"] for i in range(45)], axis=1).astype(np.float32)
sh_rest = sh_rest.reshape(nsplats, 3, 15).transpose(0, 2, 1)
sh = np.ascontiguousarray(
    np.concatenate([sh_dc, sh_rest], axis=1).reshape(nsplats, 48)
)

# ---- 场景（与 notebook SplatScene 一致） ----
eye = np.array([0.0, 0.0, 2.0], dtype=np.float32)
viewmtx = lookAt(eye, np.array([0, 0, 0], dtype=np.float32), np.array([0, 1, 0], dtype=np.float32))
projmtx = perspective(np.pi / 2.0, 1024.0 / 768.0, 0.01, 1000.0)


class Scene:
    pass


scene = Scene()
scene.viewmtx, scene.projmtx, scene.eye = viewmtx, projmtx, eye
scene.near, scene.far, scene.fovy = 0.01, 1000.0, np.pi / 2.0
scene.width, scene.height = 1024, 768

# ---- 连续调用两次，确认没有残留 sticky error ----
for i in range(2):
    image = gsplat_rasterize(scene, pos3d, scale, rot, opacity, sh)
    print(f"call {i}: image shape={image.shape}, dtype={image.dtype}")
    print(f"  pixel[0,0]={image[0, 0]} pixel[767,1023]={image[767, 1023]}")
    print(f"  pixel[100,200]={image[100, 200]}")

print("OK: no kernel error")
