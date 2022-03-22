import torch


def stride_shift(a, kernel_size=3, stride=2):
    ident = torch.eye(a.shape[-2])[:,::stride].transpose(0, 1)
    #ident = torch.eye(a.shape[-2] * 4)[::2][:,0:(a.shape[-2] * 2)].narrow(-2, 0, a.shape[-2])
    print("ident:")
    print(ident)
    shifted = []
    for k in range(kernel_size):
        x_offset = (kernel_size // 2) - k
        shifted_ident = torch.nn.functional.pad(ident, (x_offset, -x_offset))
        shifted.append(shifted_ident @ a)
    return shifted

a = torch.arange(64, dtype=torch.float32).reshape(8, 8)
print("activations:")
print(a)
out = stride_shift(a)
print("out:")
print(out)
