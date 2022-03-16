import torch

inC = 4
outC = 4
iW = 25
iH = 25
kW = 3
kH = 3
stride = 1
groups = inC
depthwise = inC == groups

a = torch.rand(1, inC, iH, iW, requires_grad=True)
w = torch.rand(outC, inC // groups, kH, kW, requires_grad=True)

print("FW Pass:")
print(a.shape, w.shape)
out = torch.nn.functional.conv2d(a, w, padding=(kH // 2), groups=groups)
print(f"  torch: conv2d({a.shape}, {w.shape}, ...) -> {out.shape}")
buda_a = a.detach().reshape(1, inC, -1).transpose(1, 2).unsqueeze(1)
buda_w = w.detach().reshape(outC, inC // groups, -1).transpose(0, 2).unsqueeze(0)
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
        if depthwise:
            shifted = shifted * (buda_o - 0.1)
            shifted = torch.sum(shifted, 2).unsqueeze(2)
        else:
            shifted = shifted.transpose(2, 3)
            shifted = shifted @ (buda_o - 0.1)
        buda_grads.append(shifted)
buda_grad = torch.stack(buda_grads, dim=1).squeeze(2)
print(buda_grad.shape)

buda_grad = buda_grad.transpose(1, 3).reshape(outC, inC // groups, kH, kW)
if not torch.allclose(w.grad, buda_grad):
    print("FAILED")
    print(w.grad)
    print(buda_grad)
else:
    print("SUCCESS")

"""
print()
print("BW Pass:")
g = out.repeat(1, inC // groups, 1, 1)
g = g.view(g.shape[0] * g.shape[1], 1, g.shape[2], g.shape[3])

gw = torch.nn.functional.conv2d(a, g, groups=inC)
print(f"  torch: conv2d({a.shape}, {g.shape}, groups={inC}, ...) -> {gw.shape}")
# sum is nop here since minibatch=1
gw = gw.sum(dim=0).view(inC // groups, outC, gw.shape[2], gw.shape[3]).transpose(0, 1)
print(f"  torch: reshape/transpose -> {gw.shape}")
buda_g = g.detach().reshape(g.shape[0], 1, -1).transpose(0, 2).unsqueeze(0)
buda_gw = gw.detach().reshape(outC, inC, -1).transpose(0, 2).unsqueeze(0)
print(f"  buda : c2d_??({buda_a.shape}, {buda_g.shape}, groups={inC}, ...) -> {buda_gw.shape}")
"""



"""
grad_output = grad_output.repeat(1, in_channels // groups, 1, 1)
grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2], grad_output.shape[3])

input = input.view(1, input.shape[0] * input.shape[1], input.shape[2], input.shape[3])

grad_weight = torch.conv2d(input, grad_output, None, dilation, padding, stride, in_channels * min_batch)

grad_weight = grad_weight.view(min_batch, grad_weight.shape[1] // min_batch, grad_weight.shape[2], grad_weight.shape[3])

return grad_weight.sum(dim=0).view(
    in_channels // groups, out_channels,
    grad_weight.shape[2], grad_weight.shape[3]).transpose(0, 1).narrow(
        2, 0, weight_size[2]).narrow(3, 0, weight_size[3])
"""
