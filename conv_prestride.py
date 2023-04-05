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
    test2((8, 8), (3, 3), 2, 3)
    for o in range(8, 9):
        for k in range(3, 9, 2):
            for s in range(2, 3):
                for p in range(1, 5):
                    test2((o, o), (k, k), s, p)


def test2(original_shape, kernel_size, stride, padding):
    print(f"test2({original_shape}, {kernel_size}, stride={stride}, padding={padding})")
    orig_out = calculate_conv2d_output_dimensions(original_shape[1], original_shape[0], kernel_size, stride, (padding, padding, padding, padding))

    in_channels = 1
    out_channels = 1

    #
    # Setup + Golden
    #
    torch_activations = torch.randn((1, in_channels, original_shape[0], original_shape[1]))
    #torch_weights = torch.randn((out_channels, in_channels, kernel_size[0], kernel_size[1]))
    torch_weights = torch.arange(kernel_size[0] * kernel_size[1], dtype=torch.float).reshape((1, 1, *kernel_size)) + 1.0
    if False:
        tmp = torch_weights
        torch_weights = torch.zeros((out_channels, in_channels, kernel_size[0], kernel_size[1]))
        s = (0, 0, 0, 0)
        torch_weights[s] = tmp[s]

    torch_output = torch.nn.functional.conv2d(torch_activations, torch_weights, stride=stride, padding=padding)

    #
    # Weight striding
    #
    ps_activations = []
    ps_weights = []

    pre_striders_per_dim = min(kernel_size[0], stride)
    for y in range(pre_striders_per_dim):
        for x in range(pre_striders_per_dim):
            ps_activations.append(torch_activations[:, :, y::stride, x::stride])
            w = torch_weights
            #w = torch_weights[:, :, (padding - y)::stride, (padding - x)::stride]

            iy = (padding & 1) - y
            ix = (padding & 1) - x
            w = pad(w, (ix, -ix, iy, -iy))
            w = w[:, :, ::stride, ::stride]
            inside_padding = x < padding or y < padding
            if not inside_padding:
                w = pad(w, (-ix, ix, -iy, iy))
            print((y, x), w, inside_padding)
            ps_weights.append(w)

    ps_activations = torch.cat(ps_activations, dim=-3)
    ps_weights = torch.cat(ps_weights, dim=-3)

    #
    # Modified padding
    #
    ps_in_shape = ((original_shape[0] + stride - 1) // stride, (original_shape[1] + stride - 1) // stride)
    ps_kernel_size = ps_weights.shape[-2]
    init_padding = (padding + stride - 1) // stride

    for tail_padding in [0, 1]:
        ps_padding = (
            init_padding,
            tail_padding,
            init_padding,
            tail_padding,
        )
        strided_out = calculate_conv2d_output_dimensions(ps_in_shape[1], ps_in_shape[0], ps_kernel_size, 1, ps_padding)
        if strided_out == orig_out:
            break

    #
    # Verify
    #
    ps_activations = pad(ps_activations, ps_padding)
    print(ps_padding)
    print(ps_activations)
    print(ps_weights)
    ps_output = torch.nn.functional.conv2d(ps_activations, ps_weights, stride=1, padding=0)
    print("Golden", torch_output)
    print("Output", ps_output)
    print(f"test2({original_shape}, {kernel_size}, stride={stride}, padding={padding})")
    assert torch.allclose(torch_output, ps_output, atol=1e-04)


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
