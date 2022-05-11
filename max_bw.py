import torch

a = torch.randn(4, requires_grad=True)
b, _ = torch.max(a, dim=0, keepdims=True)

grad = torch.randn(1)
b.backward(gradient=grad)
print(a.grad)
