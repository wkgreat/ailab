#include "add_one.hpp"

void add_one_cpu(const float *data, float *result, const size_t size) {
  for (size_t i = 0; i < size; ++i) {
    result[i] = data[i] + 1.0f;
  }
}

__global__ void add_one_kernel(const float *data, float *result, const int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    result[i] = data[i] + 1;
  }
}

void add_one_gpu(const float *data, float *result, const int n) {
  if (n <= 0) {
    return;
  }

  float *pData_gpu;
  cudaMalloc(&pData_gpu, sizeof(float) * n);
  cudaMemcpy(pData_gpu, data, sizeof(float) * n, cudaMemcpyHostToDevice);
  float *pRes_gpu;
  cudaMalloc(&pRes_gpu, sizeof(float) * n);

  const int blockDim = 512;
  const int blockNum = (n + blockDim - 1) / blockDim;

  add_one_kernel<<<blockNum, blockDim, 0>>>(pData_gpu, pRes_gpu, n);

  cudaMemcpy(result, pRes_gpu, sizeof(float) * n, cudaMemcpyDeviceToHost);
  cudaFree(pRes_gpu);
  cudaFree(pData_gpu);
}