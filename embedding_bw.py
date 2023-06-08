import torch

a = torch.tensor([2, 1, 0, 1], dtype=torch.int)
table = torch.randn((4, 4), requires_grad=True)
b = torch.nn.functional.embedding(a, table)

print(a)
print(table)
c = b.mean()
grad = torch.tensor(1.0)
print(grad)
c.backward(gradient=grad)

print(table.grad)
