import torch

act = torch.randn(1, 32)
w0 = torch.randn(32, 32)
w1 = torch.randn(32, 32)

a = (act @ w0) @ w1
b = act @ (w0 @ w1)

print(a, b)
assert torch.allclose(a, b)
