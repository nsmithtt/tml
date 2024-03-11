import torch

batch = 20
groups = 10
assert batch % groups == 0

inC = 3
outC = 32
iH = 24
iW = 24
kH = 3
kW = 3
stride = 2
padding = kH // 2

golden_act = torch.randn(batch, inC, iH, iW)
golden_weight = torch.randn(outC, inC, kH, kW)
golden = torch.nn.functional.conv2d(
    golden_act, golden_weight, stride=stride, padding=padding
)


grouped_act = golden_act.reshape(batch // groups, groups * inC, iH, iW)
grouped_weight = golden_weight.repeat((groups, 1, 1, 1))
result = torch.nn.functional.conv2d(
    grouped_act,
    grouped_weight,
    stride=stride,
    padding=padding,
    groups=groups,
)
result = result.reshape(batch, outC, result.shape[-2], result.shape[-1])

assert torch.allclose(result, golden, atol=1e-05)
