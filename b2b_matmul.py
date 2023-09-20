import torch

a = torch.randn(16, 16)
w0 = torch.randn(16, 16)
bias = torch.randn(1, 16)
w1 = torch.randn(16, 16)

golden = ((a @ w0) + bias) @ w1
result = (a @ (w0 @ w1)) + (bias @ w1)
print(golden[1])
print(result[1])
assert torch.allclose(golden, result, atol=1e-05)
