import torch

batch = 1
inC = 1
iH = 24
iW = 24
scale_factor = 2

act = torch.arange(batch*inC*iH*iW,dtype=torch.float).reshape(batch, inC, iH, iW)
golden = torch.nn.functional.upsample(act, scale_factor=2, mode='bilinear')

upsample_weight = torch.ones(inC, 1, scale_factor*2, scale_factor*2)
result = torch.nn.functional.conv_transpose2d(act, upsample_weight, stride=scale_factor*2, groups=inC)

bilinear_weightX = torch.ones(inC, 1, 1, scale_factor*2) / (scale_factor*2)
result = torch.nn.functional.pad(result, (1,1,0,0), mode='replicate', value=0)
result = torch.nn.functional.conv2d(result, bilinear_weightX, stride=(1, scale_factor), padding=0, groups=inC)

bilinear_weightY = torch.ones(inC, 1, scale_factor*2, 1) / (scale_factor*2)
result = torch.nn.functional.pad(result, (0,0,1,1), mode='replicate', value=0)
result = torch.nn.functional.conv2d(result, bilinear_weightY, stride=(scale_factor, 1), padding=0, groups=inC)

assert torch.allclose(result, golden)

a = torch.ones(32, 65)
b = torch.ones(65)

while len(a.shape) < len(b.shape):
    a = a.unsqueeze(0)

while len(b.shape) < len(a.shape):
    b = b.unsqueeze(0)

bcast_dims = []
for i in range(len(a.shape)):
    if a.shape[i] != b.shape[i]:
        assert a.shape[i] == 1 or b.shape[i] == 1
        bcast_dims.append(i)
