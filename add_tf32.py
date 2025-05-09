import torch

m = 8192
n = 8192

a = torch.randn(m, n, dtype=torch.float32)
w = torch.randn(m, n, dtype=torch.float32)

golden = a - (a + w)

tf32_mask = torch.tensor(0xffffe000)
a_tf32 = a.view(dtype=torch.int32).bitwise_and(tf32_mask).view(dtype=torch.float32)
w_tf32 = w.view(dtype=torch.int32).bitwise_and(tf32_mask).view(dtype=torch.float32)

result_tf32 = a_tf32 - (a_tf32 + w_tf32)

print(torch.nn.functional.mse_loss(result_tf32, golden))
print(torch.max(result_tf32 - golden))
