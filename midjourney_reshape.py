import torch

hslice = lambda t, factor: torch.cat(t.split(t.shape[-1] // factor, dim=-1))

i = torch.randn((1, 4096, 512))

golden = i.reshape((4096, 8, 64))
golden = golden.transpose(0, 1)
golden = golden.sum(dim=-1, keepdims=True)

out = hslice(i, 8)
out = out.sum(dim=-1, keepdims=True)

assert torch.allclose(out, golden)
