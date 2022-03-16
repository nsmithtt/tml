import torch

inC = 8
outC = 20
iW = 25
iH = 25
kW = 5
kH = 5

a = torch.rand(1, inC, iH, iW, requires_grad=True)
w = torch.rand(outC, inC, kH, kW, requires_grad=True)

print("FW Pass:")
out = torch.nn.functional.conv2d(a, w, padding=(kH // 2))
print(f"  torch: conv2d({a.shape}, {w.shape}, ...) -> {out.shape}")
buda_a = a.detach().reshape(1, inC, -1).transpose(1, 2).unsqueeze(1)
buda_w = w.detach().reshape(outC, inC, -1).transpose(0, 2).unsqueeze(0)
buda_o = out.detach().reshape(1, outC, -1).transpose(1, 2).unsqueeze(0)
print(f"  buda : c2d_mm({buda_a.shape}, {buda_w.shape}, ...) -> {buda_o.shape}")

grad = out - 0.1
out.backward(gradient=grad)

buda_grads = []
for kY in range(kH):
    for kX in range(kW):
        y_offset = (kH // 2) - kY
        x_offset = (kW // 2) - kX
        shifted = buda_a.transpose(2, 3).reshape(1, inC, iH, iW)
        assert torch.allclose(a, shifted)
        shifted = torch.nn.functional.pad(shifted, (x_offset, -x_offset, y_offset, -y_offset))
        shifted = shifted.reshape(1, 1, inC, -1).transpose(2, 3)
        shifted = shifted.transpose(2, 3)
        buda_grads.append(shifted @ (buda_o - 0.1))
buda_grad = torch.stack(buda_grads, dim=1).squeeze(2)
print(buda_grad.shape)

buda_grad = buda_grad.transpose(1, 3).reshape(outC, inC, kH, kW)
if not torch.allclose(a.grad, buda_grad):
    print("FAILED")
    print(a.grad)
    print(buda_grad)
else:
    print("SUCCESS")
