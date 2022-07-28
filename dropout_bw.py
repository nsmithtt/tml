import torch

seed = 5
p=0.6
a = torch.randn(4, requires_grad=True)
grad = torch.randn(4)

torch.manual_seed(seed)
out = torch.nn.functional.dropout(a, p=p)
out.backward(gradient=grad)

print(a * (1.0 / (1.0 - p)))
print(out)
print(grad * (1.0 / (1.0 - p)))
print(a.grad)

torch.manual_seed(seed)
assert torch.equal(a.grad, torch.nn.functional.dropout(grad, p=p))
