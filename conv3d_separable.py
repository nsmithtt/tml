import torch


# Channels last convenience function
def ttnn_conv2d(act, weight, **kwargs):
    assert len(act.shape) == 4
    assert len(weight.shape) == 4
    act = act.permute(0, 3, 1, 2)
    weight = weight.permute(3, 2, 0, 1)
    result = torch.nn.functional.conv2d(act, weight, **kwargs)
    return result.permute(0, 2, 3, 1)


batch = 2
inC = 3
outC = 32
iD = 24
iH = 24
iW = 24
kD = 3
kH = 3
kW = 3
stride = 2
padding = kH // 2


golden_act = torch.randn(batch, inC, iD, iH, iW)
golden_weight = torch.randn(outC, inC, kD, kH, kW)
golden = torch.nn.functional.conv3d(
    golden_act, golden_weight, stride=stride, padding=padding
)

# For convenience for working with TTNN shapes, we'll use channels last
# [batch, iD, iH, iW, inC]
channels_last_act = golden_act.permute(0, 2, 3, 4, 1)
# [kD, kH, kW, inC, outC]
channels_last_weight = golden_weight.permute(2, 3, 4, 1, 0)

act = channels_last_act
weight = channels_last_weight

# Collapse the batch and iD dimensions
# - [batch*iD, iH, iW, inC]
act = act.reshape(batch * iD, iH, iW, inC)

# Collapse the kD and outC dimensions
# - [kH, kW, inC, kD*outC]
weight = weight.permute(1, 2, 3, 0, 4).reshape(kH, kW, inC, kD * outC)

# Conv 1. run conv2d over the iH, iW dimensions, we get kD extra sets of output channels
# - [batch*iD, oH, oW, kD*outC]
conv1 = ttnn_conv2d(act, weight, stride=stride, padding=padding)

# Pull back out the iD dimension and collapse the oHoW dimensions
# - [batch, iD, oH*oW, kD*outC]
conv1 = conv1.reshape(batch, iD, conv1.shape[1] * conv1.shape[2], conv1.shape[3])

# Construct grouped identity weights
# - [kD, 1, kD, outC]
ident = torch.eye(kD).reshape(kD, 1, kD, 1).repeat(1, 1, 1, outC)
print(conv1.shape)
conv2 = ttnn_conv2d(conv1, ident, stride=(stride, 1), padding=(padding, 0), groups=outC)
print(conv2.shape)

result = conv2.reshape(batch, outC, golden.shape[-3], golden.shape[-2], golden.shape[-1])
print(result[0, 0, 0, 0])
print(golden[0, 0, 0, 0])
assert torch.allclose(result, golden, atol=1e-05)
