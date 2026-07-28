import mymuladd
import torch

a = torch.randn(8, device="cuda", requires_grad=True)
b = torch.randn(8, device="cuda", requires_grad=True)
c = torch.tensor([1.0], device="cuda", requires_grad=True)

loss = mymuladd.mymuladd(a, b, c).sum()
loss.backward()

print(loss)

torch.testing.assert_close(a.grad, b)
torch.testing.assert_close(b.grad, a)
