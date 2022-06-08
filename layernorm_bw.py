import torch

dim = 0
shape = (2, 4, 2)
eps = 1e-05
a = torch.randn(shape, requires_grad=True)
w = torch.randn(shape[-1], requires_grad=True)
b = torch.nn.functional.layer_norm(a, [shape[-1]], weight=w, bias=None, eps=eps)

x = a.clone().detach()
w_ = w.clone().detach()
b_ = b.clone().detach()

grad = torch.randn(b.shape)
b.backward(gradient=grad)

print("a.grad", a.grad)
print("w.grad", w.grad)

# FWD Calc
mean = torch.mean(x, dim=-1, keepdim=True)
a_minus_mean = x - mean
rstd = 1.0 / torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + eps)
out = a_minus_mean * rstd

# BWD Calc
dY = grad
gamma = w_.unsqueeze(0)

scale = 1.0 / shape[dim]
ds = (dY * x * gamma).sum(dim=-1, keepdim=True)
db = (dY * gamma).sum(dim=-1, keepdim=True)
ta = rstd
tb = (db * mean - ds) * ta * ta * ta * scale
tc = -tb * mean - db * ta * scale

dX = (ta * dY * gamma + tb * x + tc)
print("dX", dX)

dW = (dY * out).sum(-2).sum(-2)
print("dW", dW)
