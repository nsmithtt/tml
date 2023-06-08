import torch

a = torch.randn(4, requires_grad=True)
grad = torch.randn(4)

out = torch.nn.functional.tanh(a)
out.backward(gradient=grad)

x = (1 - (out * out)) * grad
print(x)
print(a.grad)

assert torch.equal(a.grad, x)
