import torch


def stride_shift(a, kernel_size=3, stride=2):
    didentA = torch.eye(a.shape[-2])[:,::stride].transpose(0, 1)
    didentB = torch.nn.functional.pad(didentA, (1, -1))
    ident = torch.stack([didentA, didentB], dim=0)
    #ident = torch.eye(a.shape[-2] * 4)[::2][:,0:(a.shape[-2] * 2)].narrow(-2, 0, a.shape[-2])
    print("ident:", ident.shape)
    print(ident)
    return ident @ a

a = torch.arange(64, dtype=torch.float32).reshape(8, 8)
print("activations:")
print(a)
out = stride_shift(a, stride=2)
print("out:")
print(out)
