#include "gsplatrender.hpp"
#include "matrix.hpp"
#include <cuda_runtime.h>

void create_viewmtx_gpu(float **viewmtx_gpu, CameraInfo &camera) {

  cpu::mat4f viewmtx_cpu = cpu::lookAt(camera.eye, camera.center, camera.up);

  cudaMalloc(viewmtx_gpu, sizeof(viewmtx_cpu));

  cudaMemcpy(*viewmtx_gpu, &viewmtx_cpu, sizeof(viewmtx_cpu),
             cudaMemcpyHostToDevice);
}

__global__ void gsplat_compute_kernel(float3 *dpPos3d, float3 *dpScale,
                                      float4 *dpRot, float *dpOpacity,
                                      float *dpShcolor, float *dpViewmtx,
                                      int nsplats, float2 *out_dpPos2d,
                                      float4 *out_dpCov2d,
                                      float4 *out_dpColor) {
  // TODO
  return;
}

__global__ void gsplat_rasterize_kernel(float2 *dpPos2d, float4 *dpCov2d,
                                        float4 *dpColor, int nsplats,
                                        float *out_dpImage) {
  // TODO
  return;
}

__constant__ float viewmtx_gpu[16];

void gsplat_rasterize_gpu(GSplatData splats, SceneInfo scene,
                          float **out_image) {

  // Splats Data
  const int nsplats = splats.nsplats;

  float3 *dpPos3d;
  float3 *dpScale;
  float4 *dpRot;
  float *dpOpacity;
  float *dpShcolor;
  cudaMalloc(&dpPos3d, nsplats * sizeof(float3));
  cudaMalloc(&dpScale, nsplats * sizeof(float3));
  cudaMalloc(&dpRot, nsplats * sizeof(float4));
  cudaMalloc(&dpOpacity, nsplats * sizeof(float));
  cudaMalloc(&dpShcolor, 48 * nsplats * sizeof(float));
  // TODO cudaMemcpy

  // Scene

  // Column-major
  float *dpViewmtx;
  create_viewmtx_gpu(&dpViewmtx, scene.camera);

  // Splats 2D
  float2 *dpPos2d;
  float4 *dpCov2d;
  float4 *dpColor;
  cudaMalloc(&dpPos2d, nsplats * sizeof(float2));
  cudaMalloc(&dpCov2d, nsplats * sizeof(float4));
  cudaMalloc(&dpColor, nsplats * sizeof(float4));

  // Image
  int npixels = scene.viewport.width * scene.viewport.height;
  float *out_dpImage;
  cudaMalloc(&out_dpImage, npixels * 4 * sizeof(float));
  //// Sort TODO
  //// Compute

  const int blockDim = 512;
  const int blockNum = (nsplats + blockDim - 1) / blockDim;

  // TODO use stream
  gsplat_compute_kernel<<<blockNum, blockDim, 0>>>(
      dpPos3d, dpScale, dpRot, dpOpacity, dpShcolor, dpViewmtx, nsplats,
      dpPos2d, dpCov2d, dpColor);

  //// ...
  //// Rasterize
  //// ...

  // Copy Image to CPU
  *out_image = (float *)malloc(npixels * 4 * sizeof(float));
  // cudaMemcpy

  // Free
  cudaFree(out_dpImage);
  cudaFree(dpPos3d);
  cudaFree(dpScale);
  cudaFree(dpRot);
  cudaFree(dpOpacity);
  cudaFree(dpShcolor);

  return;
}
