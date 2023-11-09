import torch

m = 128
k = 8192
n = 9216

a = torch.randn(m, k, dtype=torch.float32)
w = torch.randn(k, n, dtype=torch.float32)

golden = a @ w

tf32_mask = torch.tensor(0xffffe000)
a_tf32 = a.view(dtype=torch.int32).bitwise_and(tf32_mask).view(dtype=torch.float32)
w_tf32 = w.view(dtype=torch.int32).bitwise_and(tf32_mask).view(dtype=torch.float32)

result_tf32 = a_tf32 @ w_tf32

print(torch.nn.functional.mse_loss(result_tf32, golden))
