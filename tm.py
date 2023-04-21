import torch
import random


def hslice(tensor, factor):
    shape = list(tensor.shape)
    assert shape[-1] % factor == 0
    while len(shape) < 4:
        shape = [1] + shape
    ret = tensor.reshape(-1, shape[-2], factor, shape[-1] // factor)
    ret = ret.permute(0, 2, 1, 3)
    return ret.reshape(shape[:-3] + [shape[-3] * factor, shape[-2], shape[-1] // factor])


def hstack(tensor, factor):
    shape = list(tensor.shape)
    assert shape[-3] % factor == 0, f"HStack requires Z to be divisible by slice size"
    ret = tensor.reshape(-1, shape[-3] // factor, factor, shape[-2], shape[-1])
    ret = ret.permute(0, 1, 3, 2, 4)
    return ret.reshape(shape[:-3] + [shape[-3] // factor, shape[-2], shape[-1] * factor])


def vslice(tensor, factor):
    shape = list(tensor.shape)
    assert len(shape) >= 2
    assert shape[-2] % factor == 0
    if len(shape) < 3:
        shape = [1] + shape
    return tensor.reshape(shape[:-3] + (shape[-3] * factor, shape[-2] // factor, shape[-1]))


def vstack(tensor, factor):
    shape = tensor.shape
    assert shape[-3] % factor == 0, f"VStack requires Z to be divisible by slice size"
    return tensor.reshape(shape[:-3] + (shape[-3] // factor, shape[-2] * factor, shape[-1]))


def broadcast(tensor, dim, size):
    while len(tensor.shape) <= ((-dim - 1) if dim < 0 else dim):
        tensor = tensor.unsqueeze(0)
    target_shape = list(tensor.shape)
    assert dim < len(target_shape), f"Trying to broadcast on dim that doesn't exist: {dim} on {target_shape}"
    target_shape[dim] = size
    return torch.broadcast_to(tensor, target_shape)


def transpose(tensor):
    assert len(tensor.shape) > 1
    return tensor.transpose(-2, -1)


def random_factor(d):
    factors = [f for f in range(1, d+1) if d % f == 0]
    factor = factors[random.randint(0, len(factors) - 1)]
    return factor


def is_contiguous(tensor):
    volume = 1
    for d in tensor.shape:
        volume *= d
    return tensor.reshape(-1).tolist() == list(range(volume))


def test(shape, ublock_order_r, tms):
    volume = 1
    for d in shape:
        volume *= d
    t = torch.arange(volume).reshape(shape)
    if not ublock_order_r:
        t = transpose(t)
    out = t
    for tm in tms:
        print(tm)
        if "slice" in tm:
            v = tm[0] == "v"
            factor = random_factor(out.shape[-1 - v])
            out = vslice(out, factor) if v else hslice(out, factor)
        elif "stack" in tm:
            v = tm[0] == "v"
            factor = random_factor(out.shape[-3])
            out = vstack(out, factor) if v else hstack(out, factor)
        elif "transpose" == tm:
            out = transpose(out)
        else:
            assert False
    if not ublock_order_r:
        t = transpose(t)
        out = transpose(out)

    #print(t)
    #print(out)
    assert is_contiguous(out.reshape(-1)), f"{out}"


def random_tms():
    max_tms = 8
    available_tms = ["hslice", "vslice", "hstack", "vstack", "transpose"]
    return [available_tms[random.randint(0, len(available_tms) - 1)] for i in range(1, max_tms)]


def run():
    max_dim = 128
    for seed in range(0, 1):
        random.seed(seed)
        z = random.randint(0, max_dim)
        r = random.randint(0, max_dim)
        c = random.randint(0, max_dim)
        u = bool(random.randint(0, 1))
        test((z, r, c), u, random_tms())
run()
