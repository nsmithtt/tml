import torch

a = torch.randn(4, requires_grad=True)
b = torch.argmax(a)

print(b)
grad = torch.tensor(0)
print(grad)
b.backward()

print(a)
print(a.grad)
