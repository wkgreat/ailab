#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

__global__ void mymuladd_kernel(const float *a_ptr, const float *b_ptr,
                                float c_value, float *out_ptr, int64_t n) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out_ptr[i] = a_ptr[i] * b_ptr[i] + c_value;
  }
}

torch::Tensor mymuladd_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(c.is_cuda(), "c must be a CUDA tensor");
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
  TORCH_CHECK(a.scalar_type() == torch::kFloat32, "a must be float32");
  TORCH_CHECK(b.scalar_type() == torch::kFloat32, "b must be float32");
  TORCH_CHECK(c.scalar_type() == torch::kFloat32, "c must be float32");

  auto a_contig = a.contiguous();
  auto b_contig = b.contiguous();
  auto out = torch::empty_like(a_contig);

  const int64_t n = out.numel();
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  const float c_value = c.item<float>();

  mymuladd_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      a_contig.data_ptr<float>(), b_contig.data_ptr<float>(), c_value,
      out.data_ptr<float>(), n);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}