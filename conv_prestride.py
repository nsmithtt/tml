import math
import torch

def calculate_conv2d_output_dimensions(original_y, original_x, kernel_size, stride, padding):
    if isinstance(stride, int):
        stride = [stride] * 2

    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)

    # Pooling layers (max, avg)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    y = (original_y + padding[2] + padding[3] - (kernel_size[0] - 1) - 1) // stride[0] + 1
    x = (original_x + padding[0] + padding[1] - (kernel_size[1] - 1) - 1) // stride[1] + 1
    return y, x


def round_up_mod(n, d):
    m = n % d
    return 0 if m == 0 else d - m


pad = torch.torch.nn.functional.pad


def run_tests():
    if True:
        #test2((8, 8), (3, 3), 2, 3)
        #test2((8, 8), (3, 3), stride=2, padding=0)
        #test2((9, 9), (3, 3), stride=2, padding=0)
        #test2((8, 8), (3, 3), stride=2, padding=1)
        #test2((8, 8), (3, 3), stride=3, padding=0)
        #test2((8, 8), (3, 3), stride=3, padding=1)
        #test2((8, 8), (3, 3), stride=5, padding=1)
        #test2((8, 8), (4, 4), stride=2, padding=0)
        #test2((8, 8), (4, 4), stride=4, padding=2)
        #test2((8, 8), (4, 4), stride=4, padding=0)
        #test2((8, 8), (4, 4), stride=5, padding=2)
        #test2((8, 8), (7, 7), stride=2, padding=3)
        #test2((8, 8), (3, 3), stride=3, padding=1)
        #test2((8, 8), (5, 5), stride=5, padding=1)
        #test2((8, 8), (5, 5), stride=3, padding=0)
        #test2((8, 8), (5, 5), stride=3, padding=1)
        #test2((8, 8), (5, 5), stride=3, padding=2)
        #test2((8, 8), (5, 5), stride=3, padding=4)
        test2((8, 8), (5, 5), stride=3, padding=7)
        #test2((8, 8), (5, 5), stride=4, padding=1)
        #test2((8, 8), (6, 6), stride=2, padding=1)
        #test2((8, 8), (6, 6), stride=4, padding=1)
        #test2((8, 8), (7, 7), stride=3, padding=3)
        #test2((18, 18), (7, 7), stride=5, padding=3)
        #test2((8, 8), (3, 3), stride=4, padding=1)
        #test2((8, 8), (3, 3), stride=4, padding=1, m_override=4)
        #test2((8, 8), (3, 3), stride=4, padding=1)
        return
    if True:
        for o in range(8, 14):
            for k in range(3, 9):
                for s in range(2, k+1):
                    for p in range(0, 5):
                        test2((o, o), (k, k), s, p)
    if False:
        cache = {}
        for o in range(8, 8 + 1):
            for k in range(3, 9):
                for s in range(2, k + 1):
                    for p in range(0, 8):
                        key = (k, s, p)# % s == 0)
                        #if p % s == 0:
                        #    continue
                        #r = (cache[key], cache[key]+1) if key in cache else (0, 8)
                        r = (0, 8)
                        for m in range(*r):
                            try:
                                test2((o, o), (k, k), s, p, m_override=m)
                                print(f"({k}, {s}, {p}) -> {m}")
                                break
                            except:
                                continue
                        if key in cache:
                            assert cache[key] == m, f"{cache[key]} != {m}"
                        else:
                            cache[key] = m


def test2(original_shape, kernel_size, stride, padding, m_override=None):
    print(f"test2({original_shape}, {kernel_size}, stride={stride}, padding={padding})")
    orig_out = calculate_conv2d_output_dimensions(original_shape[1], original_shape[0], kernel_size, stride, (padding, padding, padding, padding))

    in_channels = 1
    out_channels = 1

    #
    # Setup + Golden
    #
    #torch_activations = torch.randn((1, in_channels, original_shape[0], original_shape[1]))
    torch_activations = torch.arange(original_shape[0] * original_shape[1], dtype=torch.float).reshape(1, 1, original_shape[0], original_shape[1])
    #torch_weights = torch.randn((out_channels, in_channels, kernel_size[0], kernel_size[1]))
    torch_weights = torch.arange(kernel_size[0] * kernel_size[1], dtype=torch.float).reshape((1, 1, *kernel_size)) + 1.0
    if False:
        tmp = torch_weights
        torch_weights = torch.zeros((out_channels, in_channels, kernel_size[0], kernel_size[1]))
        k3_s2_4 = [
            (0, 0, 0, 0),
            (0, 0, 0, 2),
            (0, 0, 2, 0),
            (0, 0, 2, 2),
            ]
        k3_s2_1 = [
            (0, 0, 0, 2),
            ]
        k5 = [
            (0, 0, 4, 4),
            ]
        k4 = [
            (0, 0, 3, 3),
            ]
        for s in k5:
            torch_weights[s] = tmp[s]

    #print(torch_activations)

    torch_output = torch.nn.functional.conv2d(torch_activations, torch_weights, stride=stride, padding=padding)


    # Make activations a multiple of stride
    torch_activations = pad(torch_activations, (0, round_up_mod(torch_activations.shape[-2], stride), 0, round_up_mod(torch_activations.shape[-1], stride)))

    # Calculate reduced padding

    ps_in_shape = ((original_shape[0] + stride - 1) // stride, (original_shape[1] + stride - 1) // stride)

    #y = (original_y + padding[2] + padding[3] - (kernel_size[0] - 1) - 1) // stride[0] + 1
    #orig_out[0] = (ps_in_shape[0] + init_padding + tail_padding - (ps_kernel_size - 1) - 1) // 1 + 1
    #orig_out[0] = ps_in_shape[0] + init_padding + tail_padding - (ps_kernel_size - 1)
    #orig_out[0] + ps_kernel_size = ps_in_shape[0] + init_padding + tail_padding + 1
    #ps_kernel_size = ps_in_shape[0] + init_padding + tail_padding + 1 - orig_out[0]
    #ps_kernel_size >= ps_in_shape[0] + init_padding + 1 - orig_out[0]

    #
    # Weight striding
    #
    init_padding = (padding + stride - 1) // stride
    print("init_padding", init_padding)


    m = m_override if m_override is not None else (stride - (kernel_size[0] % stride))

    inequality = ps_in_shape[0] + init_padding - orig_out[0] + 1
    print("asdf", ((kernel_size[0] + m) // stride) , inequality)
    if ((kernel_size[0] + m) // stride) <= inequality:
        m += stride

    print("m", m)
    mx = m
    my = m
    weights_view = torch_weights
    weights_view = pad(weights_view, (0, mx, 0, my))
    weights_view = weights_view.roll((-padding, -padding), (-2, -1))
    print("weights_view", weights_view)

    def get_offsets(k, stride, padding):
        offsets = torch.zeros(k, dtype=torch.int)
        offsets = pad(offsets, (0, max(0, stride - k)), value=1)
        offsets = offsets.roll(-padding, -1)
        offsets *= offsets.sum()
        return offsets

    y_offsets = get_offsets(kernel_size[1], stride, padding)
    x_offsets = get_offsets(kernel_size[0], stride, padding)
    print("asdf", y_offsets, x_offsets, x_offsets.sum())

    ps_activations = []
    ps_weights = []

    pre_striders_per_dim = min(kernel_size[0], stride)
    for y in range(pre_striders_per_dim):
        for x in range(pre_striders_per_dim):
            start_y = y + y_offsets[y]
            start_x = x + x_offsets[x]
            ps_activations.append(torch_activations[:, :, start_y::stride, start_x::stride])

            w = weights_view
            w = w[:, :, y::stride, x::stride]
            print((y, x), w, init_padding)
            w = w.roll((init_padding, init_padding), (-2, -1))
            #print((y, x), w.shape)
            ps_weights.append(w)

    ps_activations = torch.cat(ps_activations, dim=-3)
    ps_weights = torch.cat(ps_weights, dim=-3)
    print(ps_weights)
    #print(torch_weights.shape, ps_weights.shape)
    v0 = torch_weights.shape[-3] * torch_weights.shape[-2] * torch_weights.shape[-1]
    v1 = ps_weights.shape[-3] * ps_weights.shape[-2] * ps_weights.shape[-1]
    #print("  asdf", torch_weights.shape, ps_weights.shape)
    #if v1 > v0:
    #    print("asdf", v0, v1, v1 / v0)


    #
    # Modified padding
    #
    ps_kernel_size = ps_weights.shape[-2]
    for tail_padding in [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7]:
        ps_padding = (
            init_padding,
            tail_padding,
            init_padding,
            tail_padding,
        )
        strided_out = calculate_conv2d_output_dimensions(ps_in_shape[1], ps_in_shape[0], ps_kernel_size, 1, ps_padding)
        if strided_out == orig_out:
            break
    print("tail_padding", tail_padding)

    #
    # Verify
    #
    ps_activations = pad(ps_activations, ps_padding)
    #print(ps_padding)
    print(ps_activations)
    print(ps_weights)
    ps_output = torch.nn.functional.conv2d(ps_activations, ps_weights, stride=1, padding=0)
    assert torch.allclose(torch_output, ps_output, atol=1e-04), f"test2({original_shape}, {kernel_size}, stride={stride}, padding={padding})\n{torch_output}\n{ps_output}\nFAILURE"
    print(f"test2({original_shape}, {kernel_size}, stride={stride}, padding={padding})\n{torch_output}\n{ps_output}\nSUCCESS")
    return m


def test(original_shape, kernel_size, stride, padding):
    print(f"test({original_shape}, {kernel_size}, stride={stride}, padding={padding})")
    pre_strided_in_shape = ((original_shape[0] + stride - 1) // stride, (original_shape[1] + stride - 1) // stride)
    orig_out = calculate_conv2d_output_dimensions(original_shape[1], original_shape[0], kernel_size, stride, (padding, padding, padding, padding))

    pad_up = 0 if (kernel_size[0] >= stride or padding == 0) else (stride - padding)
    init_window = padding % kernel_size[0] != 0

    init_padding = (padding + stride - 1) // stride
    tail_padding = (round_up_mod((original_shape[0] + padding - kernel_size[0] + 1), stride) + padding) // stride + init_window

    pre_strided_kernel_size = (kernel_size[0] + stride - 1) // stride + init_window + pad_up
    pre_volume = pre_strided_kernel_size * pre_strided_kernel_size * stride * stride
    assert pre_volume >= kernel_size[0] * kernel_size[1], f"({pre_strided_kernel_size}, {pre_strided_kernel_size}) {stride} v={pre_volume} {kernel_size[0] * kernel_size[1]}"

    print("  asdf", (init_padding, tail_padding), pad_up, init_window)

    pre_strided_padding = (padding + stride - 1) // stride
    pre_strided_padding = (
        init_padding,
        tail_padding,
        init_padding,
        tail_padding,
    )

    strided_out = calculate_conv2d_output_dimensions(pre_strided_in_shape[1], pre_strided_in_shape[0], pre_strided_kernel_size, 1, pre_strided_padding)

    print("  pre_strided_kernel_size", (pre_strided_kernel_size, pre_strided_kernel_size), "padding", pre_strided_padding, "orig", original_shape, "pre_in", pre_strided_in_shape, "out", orig_out)
    assert orig_out == strided_out, f"{orig_out} != {strided_out}"
    print("  PASSED" if orig_out == strided_out else "  FAILED")
    return orig_out == strided_out


if False:
    num_fails = 0
    for o in range(8, 16):
        for k in range(2, 8):
            for s in range(2, 5):
                for p in range(0, 5):
                    num_fails += not test((o, o), (k, k), s, p)
    print("Num fails", num_fails)
elif False:
    test((8, 8), (4, 4), 2, 1)
    test((8, 8), (4, 4), 2, 2)
    test((10, 10), (4, 4), 2, 2)
run_tests()
