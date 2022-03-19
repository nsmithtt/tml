import torch


def unstride(a, b):
    didentA = torch.eye(a.shape[-2] * 4)[::2][:,0:(a.shape[-2] * 2)].narrow(-2, 0, a.shape[-2])
    didentB = torch.nn.functional.pad(didentA, (1, -1))
    p0 = a.transpose(0, 1) @ didentA
    p1 = b.transpose(0, 1) @ didentB
    return (p0 + p1).transpose(0, 1)


a = torch.arange(64, dtype=torch.float32).reshape(8, 8) * 2.0
b = torch.arange(64, dtype=torch.float32).reshape(8, 8) * 2.0 + 1.0
print(unstride(a, b))
