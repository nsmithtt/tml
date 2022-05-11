import torch

a = torch.randn(4, requires_grad=True)
b = torch.randn(4, requires_grad=True)
p = torch.randn(4, requires_grad=True)
print(p)
w = torch.nn.parameter.Parameter(data=p, requires_grad=True)

d = a * w
e = b * w

optimizer = torch.optim.SGD([w], lr=0.1, momentum=0.9)
grad = torch.randn(4)
d.backward(gradient=grad)
e.backward(gradient=grad)
optimizer.step()
print(w)
