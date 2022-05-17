import torch

dim = 0
a = torch.randn((4, 2), requires_grad=True)
a_ = a.clone().detach()
b = torch.softmax(a, dim)
b_ = b.clone().detach()

print("a", a)
print("b", b)
grad = torch.randn(b.shape)
b.backward(gradient=grad)

print("a.grad", a.grad)
s = torch.sum(grad * b_, dim=dim, keepdim=True)
e = (grad - s) * b_
print(e)
