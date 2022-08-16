import torch

inpt = torch.tensor([-0.3, 0.0, -1.5, 8.0, 0.0, 2.0], requires_grad=True)
vals = torch.randn(6, requires_grad=True)

golden = torch.heaviside(inpt, vals)
print(golden)

relu = lambda i, t: i * (i > t).to(i.dtype)
inv_relu = lambda i, t: i * (i < t).to(i.dtype)

i = inpt
v = vals

epsilon = 1.1920929e-07

# Total ~7 buda ops
# p = values that pass the piecewise function
# o = values that need ones in the piecewise function (i.e. positive)

# Compute p (~3 buda ops)
p = i + 1.0
p = relu(p, 1.0 - epsilon)
p = inv_relu(p, 1.0 + epsilon)
p = p * v

# Compute o (~3 buda ops)
o = relu(i, 0.0)
o = o + 1.0
o = inv_relu(o, 1.0 + epsilon)
o = 1.0 - o

# 1 buda op
out = p + o
print(out)

assert torch.allclose(out, golden)
