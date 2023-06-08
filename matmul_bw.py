import torch

param = torch.ones(1, requires_grad=True)

num_inputs = 128
mbatch = 16
loop_count = num_inputs // mbatch
for i in range(loop_count):
    a = torch.ones(16, 1, requires_grad=True)
    c = torch.multiply(a, param)
    loss = c
    grad = torch.ones(loss.shape)
    loss.backward(gradient=grad)
print(param.grad)
assert param.grad.item() == num_inputs
