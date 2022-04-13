import torch

a = torch.randn(4, requires_grad=True)
b = torch.relu(a)

grad = torch.zeros(4)
b.backward(gradient=grad)

print(a)
print(a.grad)
