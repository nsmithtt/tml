import torch

dim = 1
a = torch.randn((1, 4), requires_grad=True)
b = torch.softmax(a, 1)
b_ = b.clone().detach()

print("a", a)
print("b", b)
grad = torch.randn((1, 4))
b.backward(gradient=grad)

print(a.grad)
d = b_ * torch.eye(a.shape[dim]) - b_.transpose(0, 1) @ b_
e = grad @ d
print(e)
