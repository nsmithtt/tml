import torch

def T(a):
    return a.transpose(0, 1)

a = torch.randn(16, 32)
b = torch.randn(32, 4)
c = torch.randn(4, 16)
golden = a @ b @ c
result = T(T(b) @ T(a)) @ c
r2 = T(T(b) @ T(a)) @ c
T(d) @ c
T(T(c) @ d)

assert torch.allclose(golden, result, atol=1e-05)
