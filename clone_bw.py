import torch

a = torch.randn(4, requires_grad=True)
b = torch.randn(4, requires_grad=True)
c = a * b
d = c.clone()

a_ = a.detach().clone()
b_ = b.detach().clone()
a_.requires_grad = True
b_.requires_grad = True
c_ = a_ * b_

grad = torch.randn(4)
d.backward(gradient=grad)
c_.backward(gradient=grad)
print(a.grad)
print(a_.grad)
