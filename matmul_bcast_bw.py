import torch

a = torch.rand(2, 4, 4, requires_grad=True)
b = torch.rand(4, 4, requires_grad=True)

c = torch.matmul(a, b)

grad = c - 0.1
c.backward(gradient=grad)
print(a.grad)
print(b.grad)

b_grad = torch.matmul(torch.transpose(a.detach(), 1, 2), grad.detach())
print(torch.sum(b_grad, dim=0))
