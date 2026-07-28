#include "ATen/core/TensorBody.h"
#include <torch/autograd.h>
#include <torch/extension.h>
#include <vector>

torch::Tensor mymuladd_cpu(torch::Tensor a, torch::Tensor b, torch::Tensor c);
torch::Tensor mymuladd_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c);
std::vector<torch::Tensor> mymuladd_backward_raw(torch::Tensor grad_out,
                                                 torch::Tensor a,
                                                 torch::Tensor b,
                                                 torch::Tensor c);

torch::Tensor mymuladd_forward_raw(torch::Tensor a, torch::Tensor b,
                                   torch::Tensor c) {
  TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");

  if (a.is_cuda()) {
    return mymuladd_cuda(a, b, c);
  }
  return mymuladd_cpu(a, b, c);
}

std::vector<torch::Tensor> mymuladd_backward_cpu(torch::Tensor grad_out,
                                                 torch::Tensor a,
                                                 torch::Tensor b,
                                                 torch::Tensor c);

std::vector<torch::Tensor> mymuladd_backward_cuda(torch::Tensor grad_out,
                                                  torch::Tensor a,
                                                  torch::Tensor b,
                                                  torch::Tensor c);

std::vector<torch::Tensor> mymuladd_backward_raw(torch::Tensor grad_out,
                                                 torch::Tensor a,
                                                 torch::Tensor b,
                                                 torch::Tensor c) {
  TORCH_CHECK(grad_out.device() == a.device(),
              "grad_out and a must be on the same device");
  TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");
  TORCH_CHECK(a.device() == c.device(), "a and c must be on the same device");

  if (grad_out.is_cuda()) {
    return mymuladd_backward_cuda(grad_out, a, b, c);
  }
  return mymuladd_backward_cpu(grad_out, a, b, c);
}

class MyMulAddAutograd : public torch::autograd::Function<MyMulAddAutograd> {
public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor a, torch::Tensor b,
                               torch::Tensor c) {
    TORCH_CHECK(c.numel() == 1, "c must be a scalar tensor");
    ctx->save_for_backward({a, b, c});

    return mymuladd_forward_raw(a, b, c);
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto a = saved[0];
    auto b = saved[1];
    auto c = saved[2];
    auto grad_out = grad_outputs[0];

    auto grads = mymuladd_backward_raw(grad_out, a, b, c);
    auto grad_a = grads[0];
    auto grad_b = grads[1];
    auto grad_c = torch::Tensor();

    return {grad_a, grad_b, grad_c};
  }
};

torch::Tensor mymuladd(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  return MyMulAddAutograd::apply(a, b, c);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mymuladd", &mymuladd, "mymuladd with C++/CUDA custom backward");
}
