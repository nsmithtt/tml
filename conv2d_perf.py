import torch
import time

batch = 128
loop = 32
df = torch.float

iH = 224
iW = 224
inC = 3
outC = 64
kW = 7
stride = 2
padding = kW // 2

act = torch.randn(batch, inC, iH, iW, dtype=df)
layer = torch.nn.Conv2d(inC, outC, kW, stride=stride, padding=padding, dtype=df)

start = time.time()

for i in range(loop):
    layer(act)

stop = time.time()

t = stop - start
print("Samples/sec", (batch * loop) / t)
