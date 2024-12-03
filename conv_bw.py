import torch

inC = 4
outC = 4
iW = 25
iH = 25
kW = 3
kH = 3
stride = 1

a = torch.rand(1, inC, iH, iW, requires_grad=True)
w = torch.rand(outC, inC, kH, kW, requires_grad=True)

print("FW Pass:")
print(a.shape, w.shape)
out = torch.nn.functional.conv2d(a, w, padding=(kH // 2))
print(f"  torch: conv2d({a.shape}, {w.shape}, ...) -> {out.shape}")

grad = out - 0.1
out.backward(gradient=grad)
print(w.grad)

bw_a = a.detach()
bw_w = w.detach()
#bw_o = out.detach()

print(grad.shape, bw_w.shape)
bw_o = torch.nn.functional.conv_transpose2d(grad, bw_w, padding=(kH // 2))
print(w.grad.shape, bw_o.shape)

#assert torch.allclose(w.grad, bw_o, atol=1e-6)
