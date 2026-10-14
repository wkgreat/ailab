#pragma once

#include <cuda_runtime.h>
#include <memory.h>

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

} // namespace cuda
} // namespace gsplat
