import torch

torch.manual_seed(0)

a0 = torch.randn(4, requires_grad=True)
b0 = torch.randn(4, requires_grad=True)
c0 = torch.randn(4, requires_grad=True)
out0 = (a0 * c0) * b0

grad = torch.randn(4)
out0.backward(grad)

print(a0.grad)
print(b0.grad)

#tensor([-0.6682,  0.6209,  0.2578,  0.0187])
#tensor([ 0.9495,  0.1303, -1.3925,  0.0127])
