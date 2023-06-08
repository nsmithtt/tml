import torch

a = torch.randn(4, requires_grad=True)
b = torch.clamp(a, min=0.1, max=0.5)

grad = torch.randn(4)
b.backward(gradient=grad)
print(a.grad)
print(a)
print(b)
