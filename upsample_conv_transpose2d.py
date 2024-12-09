import torch

batch = 20
inC = 32
iH = 24
iW = 24
scale_factor = 2

act = torch.randn(batch, inC, iH, iW)
golden = torch.nn.functional.upsample(act, scale_factor=2, mode='nearest')


upsample_weight = torch.ones(inC, 1, scale_factor, scale_factor)
result = torch.nn.functional.conv_transpose2d(act, upsample_weight, stride=scale_factor, groups=inC)

assert torch.allclose(result, golden)
