import torch

tf32_mask = torch.tensor(0xffffe000, dtype=torch.int32)
def test(x):
    x_tf32 = x.view(dtype=torch.int32).bitwise_and(tf32_mask).view(dtype=torch.float32)
    assert x_tf32 == x, f"{x_tf32} != {x}"

for i in range(1024):
    test(torch.tensor(float(i*1024), dtype=torch.float32))
