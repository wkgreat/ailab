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

  int id = blockDim.x * blockIdx.x + threadIdx.x;

  out_dpPos2d[id] = {0, 0};
  out_dpCov2d[id] = {0, 0, 0, 0};
  out_dpColor[id] = {
      1,
      1,
      1,
      1,
  };

  return;
}

__global__ void gsplat_rasterize_kernel(float2 *dpPos2d, float4 *dpCov2d,
                                        float4 *dpColor, int nsplats,
                                        float4 *out_dpImage) {
  // TODO
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  out_dpImage[id] = {0, 1, 0, 1};
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
  cudaMemcpy(dpPos3d, splats.pos3d, nsplats * sizeof(float3),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dpScale, splats.scale, nsplats * sizeof(float3),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dpRot, splats.rot, nsplats * sizeof(float4),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dpOpacity, splats.opacity, nsplats * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dpShcolor, splats.sh, nsplats * 48 * sizeof(float),
             cudaMemcpyHostToDevice);

  // Scene
  // Column-major
  float *dpViewmtx;
  create_viewmtx_gpu(&dpViewmtx, scene.camera);

  // TODO Sort

  // Compute
  float2 *dpPos2d;
  float4 *dpCov2d;
  float4 *dpColor;
  cudaMalloc(&dpPos2d, nsplats * sizeof(float2));
  cudaMalloc(&dpCov2d, nsplats * sizeof(float4));
  cudaMalloc(&dpColor, nsplats * sizeof(float4));
  const int computeBlockDim = 512;
  const int computeBlockNum = (nsplats + computeBlockDim - 1) / computeBlockDim;
  gsplat_compute_kernel<<<computeBlockNum, computeBlockDim, 0>>>(
      dpPos3d, dpScale, dpRot, dpOpacity, dpShcolor, dpViewmtx, nsplats,
      dpPos2d, dpCov2d, dpColor);

  //// Rasterize
  int npixels = scene.viewport.width * scene.viewport.height;
  float4 *out_dpImage;
  const int imageByteSize = npixels * sizeof(float4);
  cudaMalloc(&out_dpImage, imageByteSize);
  const int rasterizeBlockDim = 512;
  const int rasterizeBlockNum =
      (npixels + rasterizeBlockDim - 1) / rasterizeBlockDim;
  gsplat_rasterize_kernel<<<rasterizeBlockNum, rasterizeBlockDim, 0>>>(
      dpPos2d, dpCov2d, dpColor, nsplats, out_dpImage);

  // Copy Image to CPU
  *out_image = (float *)malloc(imageByteSize);
  cudaMemcpy(*out_image, out_dpImage, imageByteSize, cudaMemcpyDeviceToHost);

  // Free
  cudaFree(out_dpImage);
  cudaFree(dpViewmtx);
  cudaFree(dpPos3d);
  cudaFree(dpScale);
  cudaFree(dpRot);
  cudaFree(dpOpacity);
  cudaFree(dpShcolor);
}
