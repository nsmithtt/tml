import torch

dim = 1
a = torch.randn((2, 4), requires_grad=True)
b = torch.softmax(a, 1)
b_ = b.clone().detach()

print("a", a)
print("b", b)
grad = torch.randn(b.shape)
b.backward(gradient=grad)

print("a.grad", a.grad)
d = []
for i in range(b_.shape[0]):
    t = b_[i].unsqueeze(0)
    print("t", t.transpose(0, 1), t, t.transpose(0, 1) @ t)
    t = t * torch.eye(a.shape[dim]) - t.transpose(0, 1) @ t
    d.append(t)
d = torch.stack(d)
print("grad", grad)
print("d", d)
e = grad @ d
print(e)
