#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <vector>

__global__ void mymuladd_backward_kernel(const float *grad_out_ptr,
                                         const float *a_ptr, const float *b_ptr,
                                         float *grad_a_ptr, float *grad_b_ptr,
                                         int64_t n) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    grad_a_ptr[i] = grad_out_ptr[i] * b_ptr[i];
    grad_b_ptr[i] = grad_out_ptr[i] * a_ptr[i];
  }
}

std::vector<torch::Tensor> mymuladd_backward_cuda(torch::Tensor grad_out,
                                                  torch::Tensor a,
                                                  torch::Tensor b,
                                                  torch::Tensor c) {
  TORCH_CHECK(grad_out.is_cuda(), "grad_out must be a CUDA tensor");
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
  TORCH_CHECK(grad_out.sizes() == a.sizes(), "grad_out shape must match a");
  TORCH_CHECK(a.scalar_type() == torch::kFloat32, "a must be float32");
  TORCH_CHECK(b.scalar_type() == torch::kFloat32, "b must be float32");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32,
              "grad_out must be float32");

  auto grad_out_contig = grad_out.contiguous();
  auto a_contig = a.contiguous();
  auto b_contig = b.contiguous();

  auto grad_a = torch::empty_like(a_contig);
  auto grad_b = torch::empty_like(b_contig);
  auto grad_c = grad_out.sum().reshape_as(c);

  const int64_t n = grad_out_contig.numel();
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);

  mymuladd_backward_kernel<<<blocks, threads, 0,
                             at::cuda::getCurrentCUDAStream()>>>(
      grad_out_contig.data_ptr<float>(), a_contig.data_ptr<float>(),
      b_contig.data_ptr<float>(), grad_a.data_ptr<float>(),
      grad_b.data_ptr<float>(), n);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_a, grad_b, grad_c};
}
