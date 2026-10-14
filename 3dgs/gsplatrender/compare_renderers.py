#!/usr/bin/env python3
"""
比较 CUDA 与 OpenGL renderer 各阶段结果，定位毛刺来源。

用法：
  1. 先编译 gsplatrender CUDA 扩展
  2. python compare_renderers.py /path/to/cactus_splat3.ply
"""

import os
import sys
import numpy as np
from pyglm import glm
from plyfile import PlyData
import moderngl
from typing import Any, cast


# ===== Utils =====

def get_uniform(shader: Any, name: str) -> moderngl.Uniform:
    return cast(moderngl.Uniform, shader[name])


# ===== Data loading (mirrors both notebooks) =====

def load_splats_cuda(path):
    """加载 ply → CUDA 格式的 SplatData"""
    vertex = PlyData.read(path)["vertex"]
    nsplats = len(vertex["x"])

    position = np.stack(
        [vertex["x"], vertex["y"], vertex["z"], np.ones((nsplats,), dtype=np.float32)], axis=1
    ).astype(np.float32)
    opacity = np.asarray(vertex["opacity"], dtype=np.float32).reshape(nsplats, 1)

    splat_scale = np.stack(
        [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"],
         np.zeros((nsplats,), dtype=np.float32)], axis=1
    ).astype(np.float32)

    splat_rot = np.stack([vertex[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float32)

    sh_dc = np.stack([vertex[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float32)[:, None, :]
    sh_rest = np.stack([vertex[f"f_rest_{i}"] for i in range(45)], axis=1).astype(np.float32)
    sh_rest = sh_rest.reshape(nsplats, 3, 15).transpose(0, 2, 1)
    sh_rgb = np.concatenate([sh_dc, sh_rest], axis=1)
    sh = np.ascontiguousarray(sh_rgb.reshape(nsplats, 48), dtype=np.float32)

    pos3d = np.ascontiguousarray(position, dtype=np.float32)
    scale = np.ascontiguousarray(splat_scale, dtype=np.float32)
    rot = np.ascontiguousarray(splat_rot, dtype=np.float32)
    opacity = np.ascontiguousarray(opacity, dtype=np.float32)

    return pos3d, scale, rot, opacity, sh


def load_splats_gl(path):
    """加载 ply → OpenGL SSBO 格式 (N, 76)"""
    vertex = PlyData.read(path)["vertex"]
    nsplats = len(vertex["x"])

    position = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    opacity = np.asarray(vertex["opacity"], dtype=np.float32).reshape(nsplats, 1)
    posopa = np.concatenate([position, opacity], axis=1)

    splat_scale = np.stack(
        [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"],
         np.zeros(nsplats, dtype=np.float32)], axis=1
    ).astype(np.float32)

    splat_rot = np.stack([vertex[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float32)

    sh_dc = np.stack([vertex[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float32)[:, None, :]
    sh_rest = np.stack([vertex[f"f_rest_{i}"] for i in range(45)], axis=1).astype(np.float32)
    sh_rest = sh_rest.reshape(nsplats, 3, 15).transpose(0, 2, 1)
    sh_rgb = np.concatenate([sh_dc, sh_rest], axis=1)
    sh_vec4 = np.zeros((nsplats, 16, 4), dtype=np.float32)
    sh_vec4[:, :, :3] = sh_rgb

    splats = np.concatenate(
        [posopa, splat_scale, splat_rot, sh_vec4.reshape(nsplats, 64)], axis=1
    )
    assert splats.shape == (nsplats, 76)
    return np.ascontiguousarray(splats, dtype=np.float32)


# ===== Scene setup =====

class Scene:
    def __init__(self, width, height):
        self.bgcolor = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.near = 0.01
        self.far = 1000
        self.fovy = glm.pi() / 2.0
        self.width = width
        self.height = height

        self.orbit_x = glm.radians(-90.0)
        self.orbit_y = 0.0
        self.orbit_r = 2.0
        self._update()

    def _update(self):
        x = self.orbit_r * glm.cos(self.orbit_y) * glm.cos(self.orbit_x)
        y = self.orbit_r * glm.cos(self.orbit_y) * glm.sin(self.orbit_x)
        z = self.orbit_r * glm.sin(self.orbit_y)
        self.eye = glm.vec3(x, y, z)
        self.center = glm.vec3(0, 0, 0)
        self.up = glm.vec3(0, 0, 1)
        _viewmtx = glm.lookAt(self.eye, self.center, self.up)
        # pyglm → F-order。moderngl write() 需要 C-contiguous，用 .T 转为列主序 C-order
        _v = np.array(_viewmtx, dtype=np.float32)
        self.viewmtx = _v  # F-order, 给 CUDA (PyArray_FROM_OTF 会自动转 C)
        self.viewmtx_gl = np.ascontiguousarray(_v.T)  # C-order列主序, 给 OpenGL write()
        _projmtx = glm.perspective(self.fovy, self.width * 1.0 / self.height,
                                   self.near, self.far)
        _p = np.array(_projmtx, dtype=np.float32)
        self.projmtx = _p
        self.projmtx_gl = np.ascontiguousarray(_p.T)


# ===== OpenGL compute =====

def read_shader(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def gl_compute_and_readback(gl_splats, scene: Scene, shader_dir: str):
    """运行 OpenGL compute shader，读取 SSBO 结果返回为 numpy"""
    ctx = moderngl.create_context(standalone=True, backend="egl")

    comp_code = read_shader(f"{shader_dir}/splat.comp.glsl")
    cs = ctx.compute_shader(comp_code)

    input_ssbo = ctx.buffer(gl_splats.tobytes())

    nsplats = gl_splats.shape[0]
    sort_count = 1 if nsplats == 0 else 1 << (nsplats - 1).bit_length()
    output_splats = np.zeros((sort_count, 16), dtype=np.float32)
    output_splats[nsplats:, 2] = -np.finfo(np.float32).max
    output_ssbo = ctx.buffer(output_splats.tobytes())

    input_ssbo.bind_to_storage_buffer(binding=0)
    output_ssbo.bind_to_storage_buffer(binding=1)

    get_uniform(cs, "camera.eye").value = (
        scene.eye.x, scene.eye.y, scene.eye.z, 1.0)
    get_uniform(cs, "camera.viewmtx").write(scene.viewmtx_gl)
    get_uniform(cs, "projection.fovy").value = scene.fovy
    get_uniform(cs, "projection.projmtx").write(scene.projmtx_gl)
    get_uniform(cs, "viewport.size").value = (float(scene.width), float(scene.height))
    get_uniform(cs, "nsplats").value = int(nsplats)

    group_count = (nsplats + 255) // 256
    cs.run(group_x=group_count)
    ctx.finish()

    # 读取 output SSBO: 每个 splat 16 个 float
    raw = np.frombuffer(output_ssbo.read(), dtype=np.float32)
    print(f"  DEBUG: raw SSBO len={len(raw)}, expected nsplats*16={nsplats*16}")
    print(f"  DEBUG: first 16 raw floats = {raw[:16]}")
    raw = raw[:nsplats * 16].reshape(nsplats, 16)

    # 解析: ndcpos[4], color[4], rect[4], cov2d[4] (mat2 stored as [a,b,c,pad])
    gl_ndcpos = raw[:, :4]
    gl_color = raw[:, 4:8]
    gl_rect = raw[:, 8:12]
    gl_cov2d = raw[:, 12:16]  # [a, b, c, pad?]

    ctx.release()
    return gl_ndcpos, gl_color, gl_rect, gl_cov2d


# ===== Comparison =====

def compare(path, shader_dir, width=500, height=500):
    print(f"PLY: {path}")
    print(f"Resolution: {width}x{height}")

    # Load data
    pos3d, scale, rot, opacity, sh_cuda = load_splats_cuda(path)
    gl_splats = load_splats_gl(path)
    nsplats = pos3d.shape[0]
    print(f"Splats: {nsplats}")

    # Scene
    scene = Scene(width, height)

    # --- CUDA compute ---
    print("\n=== CUDA compute (gsplat_compute_debug) ===")
    from gsplatrender import gsplat_compute_debug
    cuda_pos2d, cuda_depth, cuda_cov2d, cuda_rect, cuda_color = \
        gsplat_compute_debug(scene, pos3d, scale, rot, opacity, sh_cuda)

    # --- OpenGL compute ---
    print("=== OpenGL compute ===")
    gl_ndcpos, gl_color, gl_rect, gl_cov2d = gl_compute_and_readback(
        gl_splats, scene, shader_dir)

    # OpenGL compute 输出的是 ndcpos (4 floats)，需要转成 pixelpos 才能和 CUDA pos2d 比
    viewport = np.array([width, height], dtype=np.float32)
    gl_pos2d = (gl_ndcpos[:, :2] * 0.5 + 0.5) * viewport

    # --- 比较前 N 个 splat ---
    N = min(20, nsplats)
    print(f"\n{'='*80}")
    print(f"前 {N} 个 splat 对比")
    print(f"{'='*80}")

    for i in range(N):
        print(f"\n--- splat[{i}] ---")
        print(f"  pos2d  CUDA=({cuda_pos2d[i,0]:.6f}, {cuda_pos2d[i,1]:.6f})  "
              f"GL=({gl_pos2d[i,0]:.6f}, {gl_pos2d[i,1]:.6f})  "
              f"diff=({abs(cuda_pos2d[i,0]-gl_pos2d[i,0]):.2e}, {abs(cuda_pos2d[i,1]-gl_pos2d[i,1]):.2e})")

        print(f"  cov2d  CUDA=[{cuda_cov2d[i,0]:.6f}, {cuda_cov2d[i,1]:.6f}, "
              f"{cuda_cov2d[i,2]:.6f}, {cuda_cov2d[i,3]:.6f}]")
        print(f"  cov2d  GL  =[{gl_cov2d[i,0]:.6f}, {gl_cov2d[i,1]:.6f}, "
              f"{gl_cov2d[i,2]:.6f}, {gl_cov2d[i,3]:.6f}]")
        cov_diff = np.abs(cuda_cov2d[i] - gl_cov2d[i])
        print(f"  cov2d  diff=[{cov_diff[0]:.2e}, {cov_diff[1]:.2e}, "
              f"{cov_diff[2]:.2e}, {cov_diff[3]:.2e}]")

        print(f"  rect   CUDA=({cuda_rect[i,0]:.4f}, {cuda_rect[i,1]:.4f}, "
              f"{cuda_rect[i,2]:.4f}, {cuda_rect[i,3]:.4f})")
        print(f"  rect   GL  =({gl_rect[i,0]:.4f}, {gl_rect[i,1]:.4f}, "
              f"{gl_rect[i,2]:.4f}, {gl_rect[i,3]:.4f})")
        rect_diff = np.abs(cuda_rect[i] - gl_rect[i])
        print(f"  rect   diff=[{rect_diff[0]:.2e}, {rect_diff[1]:.2e}, "
              f"{rect_diff[2]:.2e}, {rect_diff[3]:.2e}]")

        print(f"  color  CUDA=({cuda_color[i,0]:.6f}, {cuda_color[i,1]:.6f}, "
              f"{cuda_color[i,2]:.6f}, {cuda_color[i,3]:.6f})")
        print(f"  color  GL  =({gl_color[i,0]:.6f}, {gl_color[i,1]:.6f}, "
              f"{gl_color[i,2]:.6f}, {gl_color[i,3]:.6f})")
        color_diff = np.abs(cuda_color[i] - gl_color[i])
        print(f"  color  diff=[{color_diff[0]:.2e}, {color_diff[1]:.2e}, "
              f"{color_diff[2]:.2e}, {color_diff[3]:.2e}]")

        print(f"  depth  CUDA={cuda_depth[i]:.6f}")

    # --- 统计 diff ---
    print(f"\n{'='*80}")
    print(f"全量统计 ({nsplats} splats)")
    print(f"{'='*80}")

    pos_diff = np.abs(cuda_pos2d - gl_pos2d)
    print(f"\npos2d max diff: ({pos_diff[:,0].max():.2e}, {pos_diff[:,1].max():.2e})")
    print(f"pos2d mean diff: ({pos_diff[:,0].mean():.2e}, {pos_diff[:,1].mean():.2e})")

    cov_diff = np.abs(cuda_cov2d - gl_cov2d)
    print(f"cov2d max diff: [{cov_diff[:,0].max():.2e}, {cov_diff[:,1].max():.2e}, "
          f"{cov_diff[:,2].max():.2e}, {cov_diff[:,3].max():.2e}]")
    print(f"cov2d mean diff: [{cov_diff[:,0].mean():.2e}, {cov_diff[:,1].mean():.2e}, "
          f"{cov_diff[:,2].mean():.2e}, {cov_diff[:,3].mean():.2e}]")

    rect_diff = np.abs(cuda_rect - gl_rect)
    print(f"rect max diff: [{rect_diff[:,0].max():.2e}, {rect_diff[:,1].max():.2e}, "
          f"{rect_diff[:,2].max():.2e}, {rect_diff[:,3].max():.2e}]")
    print(f"rect mean diff: [{rect_diff[:,0].mean():.2e}, {rect_diff[:,1].mean():.2e}, "
          f"{rect_diff[:,2].mean():.2e}, {rect_diff[:,3].mean():.2e}]")

    color_diff = np.abs(cuda_color - gl_color)
    print(f"color max diff: [{color_diff[:,0].max():.2e}, {color_diff[:,1].max():.2e}, "
          f"{color_diff[:,2].max():.2e}, {color_diff[:,3].max():.2e}]")
    print(f"color mean diff: [{color_diff[:,0].mean():.2e}, {color_diff[:,1].mean():.2e}, "
          f"{color_diff[:,2].mean():.2e}, {color_diff[:,3].mean():.2e}]")

    # --- 最终图像对比 ---
    print(f"\n{'='*80}")
    print(f"最终图像对比 (CUDA vs GL)")
    print(f"{'='*80}")

    from gsplatrender import gsplat_rasterize
    cuda_img = gsplat_rasterize(scene, pos3d, scale, rot, opacity, sh_cuda)

    # GL: render full pipeline
    ctx = moderngl.create_context(standalone=True, backend="egl")
    vert_code = read_shader(f"{shader_dir}/splat.vert.glsl")
    frag_code = read_shader(f"{shader_dir}/splat.frag.glsl")
    sort_code = read_shader(f"{shader_dir}/splat_sort.glsl")

    # redo compute with sort
    comp_cs = ctx.compute_shader(read_shader(f"{shader_dir}/splat.comp.glsl"))
    sort_cs = ctx.compute_shader(sort_code)
    program = ctx.program(vertex_shader=vert_code, fragment_shader=frag_code)

    input_ssbo = ctx.buffer(gl_splats.tobytes())
    sort_count = 1 << (nsplats - 1).bit_length()
    output_data = np.zeros((sort_count, 16), dtype=np.float32)
    output_data[nsplats:, 2] = -np.finfo(np.float32).max
    output_ssbo = ctx.buffer(output_data.tobytes())

    input_ssbo.bind_to_storage_buffer(binding=0)
    output_ssbo.bind_to_storage_buffer(binding=1)

    get_uniform(comp_cs, "camera.eye").value = (scene.eye.x, scene.eye.y, scene.eye.z, 1.0)
    get_uniform(comp_cs, "camera.viewmtx").write(scene.viewmtx_gl)
    get_uniform(comp_cs, "projection.fovy").value = scene.fovy
    get_uniform(comp_cs, "projection.projmtx").write(scene.projmtx_gl)
    get_uniform(comp_cs, "viewport.size").value = (float(width), float(height))
    get_uniform(comp_cs, "nsplats").value = int(nsplats)

    comp_cs.run(group_x=(nsplats + 255) // 256)
    ctx.memory_barrier()

    # sort: 需要把 output_ssbo 重新绑定到 binding=0
    output_ssbo.bind_to_storage_buffer(binding=0)
    sort_cs["sort_count"].value = int(sort_count)
    group_count = (sort_count + 255) // 256
    k = 2
    while k <= sort_count:
        j = k // 2
        while j > 0:
            sort_cs["k"].value = int(k)
            sort_cs["j"].value = int(j)
            sort_cs.run(group_x=group_count)
            ctx.memory_barrier()
            j //= 2
        k *= 2

    # render
    ctx.enable(moderngl.BLEND)
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)

    fbo = ctx.simple_framebuffer((width, height), components=4)
    fbo.use()
    ctx.clear(0.0, 0.0, 0.0, 1.0)

    vao = ctx.vertex_array(program, [])
    get_uniform(program, "viewport.size").value = (float(width), float(height))
    vao.render(mode=moderngl.TRIANGLES, vertices=6, instances=nsplats)
    ctx.memory_barrier()

    gl_img = np.frombuffer(fbo.read(components=4, alignment=1), dtype=np.uint8)
    gl_img = gl_img.reshape(height, width, 4).astype(np.float32) / 255.0
    ctx.release()

    img_diff = np.abs(cuda_img - gl_img)
    print(f"image max diff: {img_diff.max():.6f}")
    print(f"image mean diff: {img_diff.mean():.6f}")

    # 找出差异最大的像素
    per_pixel = img_diff.max(axis=2)
    flat_idx = np.argmax(per_pixel)
    py, px = divmod(flat_idx, width)
    print(f"max diff pixel at ({px}, {py}): "
          f"CUDA=({cuda_img[py,px,0]:.4f},{cuda_img[py,px,1]:.4f},{cuda_img[py,px,2]:.4f},{cuda_img[py,px,3]:.4f}) "
          f"GL=({gl_img[py,px,0]:.4f},{gl_img[py,px,1]:.4f},{gl_img[py,px,2]:.4f},{gl_img[py,px,3]:.4f})")

    # 非零像素数
    cuda_nonzero = (cuda_img.max(axis=2) > 0.01).sum()
    gl_nonzero = (gl_img.max(axis=2) > 0.01).sum()
    print(f"Non-zero pixels: CUDA={cuda_nonzero}, GL={gl_nonzero}")

    return cuda_img, gl_img, cuda_pos2d, gl_pos2d, cuda_cov2d, gl_cov2d, cuda_rect, gl_rect, cuda_color, gl_color


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../data/cactus_splat3.ply"
    shader_dir = os.path.join(os.path.dirname(__file__), "..", "gsplat_glrender")
    compare(path, shader_dir, width=500, height=500)