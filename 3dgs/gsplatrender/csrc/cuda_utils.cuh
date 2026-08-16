#pragma once

#include <cuda_runtime.h>

namespace gsplat {
namespace cuda {
namespace kernel {

template <typename T> __global__ void max(T *nums, const int S, const int N) {

  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int id0 = id * S;
  int id1 = (id + 1) * S;

  if (id1 < N) {
    if (nums[id0] < nums[id1]) {
      nums[id0] = nums[id1];
    }
  }
}

template <typename T>
__global__ void divide(T *nums, const int d, const int N) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  if (id < N) {
    nums[id] /= d;
  }
}
} // namespace kernel

namespace device {}

namespace host {

// int getBlockDim1d(cudaDeviceProp prop) {
//   int mt = prop.maxThreadsPerBlock;
//   int mx = prop.maxThreadsDim[0];
//   return std::min(mt, mx);
// }

// dim3 getBlockDim2d(cudaDeviceProp prop) {
//   unsigned int mt = prop.maxThreadsPerBlock;
//   unsigned int s = (unsigned int)std::floor(std::sqrt((double)mt));
//   unsigned int mx = prop.maxThreadsDim[0];
//   unsigned int my = prop.maxThreadsDim[1];
//   return {std::min(mx, s), std::min(my, s), 1};
// }

// dim3 getBlockDim3d(cudaDeviceProp prop) {
//   unsigned int mt = prop.maxThreadsPerBlock;
//   unsigned int s = (unsigned int)std::floor(std::cbrt((double)mt));
//   unsigned int mx = prop.maxThreadsDim[0];
//   unsigned int my = prop.maxThreadsDim[1];
//   unsigned int mz = prop.maxThreadsDim[2];
//   return {std::min(mx, s), std::min(my, s), std::min(mz, s)};
// }

} // namespace host
} // namespace cuda
} // namespace gsplat
