#include "ATen/core/TensorBody.h"
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> mymuladd_backward_cpu(torch::Tensor grad_out,
                                                 torch::Tensor a,
                                                 torch::Tensor b,
                                                 torch::Tensor c) {
  TORCH_CHECK(!grad_out.is_cuda(), "grad_out must be a CPU tensor");
  TORCH_CHECK(!a.is_cuda(), "a must be a CPU tensor");
  TORCH_CHECK(!b.is_cuda(), "b must be a CPU tensor");
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
  auto grad_c = grad_out_contig.sum().reshape_as(c);

  const float *grad_out_ptr = grad_out_contig.data_ptr<float>();
  const float *a_ptr = a_contig.data_ptr<float>();
  const float *b_ptr = b_contig.data_ptr<float>();
  float *grad_a_ptr = grad_a.data_ptr<float>();
  float *grad_b_ptr = grad_b.data_ptr<float>();

  const int64_t n = grad_out_contig.numel();
  for (int64_t i = 0; i < n; ++i) {
    grad_a_ptr[i] = grad_out_ptr[i] * b_ptr[i];
    grad_b_ptr[i] = grad_out_ptr[i] * a_ptr[i];
  }

  return {grad_a, grad_b, grad_c};
}