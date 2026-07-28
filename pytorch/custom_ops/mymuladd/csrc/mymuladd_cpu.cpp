#include "ATen/core/TensorBody.h"
#include "c10/util/Exception.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/types.h"
#include <cstdint>
#include <torch/extension.h>

torch::Tensor mymuladd_cpu(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  TORCH_CHECK(!a.is_cuda(), "a must be a CPU tensor!");
  TORCH_CHECK(!b.is_cuda(), "b must be a CPU tensor!");
  TORCH_CHECK(!c.is_cuda(), "c must be a CPU tensor!");
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
  TORCH_CHECK(a.scalar_type() == torch::kFloat32, "a must be float32");
  TORCH_CHECK(b.scalar_type() == torch::kFloat32, "b must be float32");
  TORCH_CHECK(c.scalar_type() == torch::kFloat32, "c must be float32");

  auto a_config = a.contiguous(); // 把tensor变成内存连续布局
  auto b_config = b.contiguous();
  auto out = torch::empty_like(a_config);

  const float *a_ptr = a_config.data_ptr<float>();
  const float *b_ptr = b_config.data_ptr<float>();
  float *out_ptr = out.data_ptr<float>();

  const int64_t n = out.numel(); // 元素个数
  const float c_value = c.item<float>();

  for (int64_t i = 0; i < n; ++i) {
    out_ptr[i] = a_ptr[i] * b_ptr[i] + c_value;
  }

  return out;
}
