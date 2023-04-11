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
    if False:
        #test2((8, 8), (3, 3), 2, 3)
        #test2((8, 8), (3, 3), stride=2, padding=0)
        #test2((9, 9), (3, 3), stride=2, padding=0)
        #test2((8, 8), (3, 3), stride=2, padding=1)
        #test2((8, 8), (3, 3), stride=3, padding=0)
        #test2((8, 8), (3, 3), stride=3, padding=1)
        #test2((8, 8), (3, 3), stride=3, padding=2)
        #test2((8, 8), (3, 3), stride=5, padding=1)
        #test2((8, 8), (4, 4), stride=2, padding=0)
        #test2((8, 8), (4, 4), stride=4, padding=2)
        #test2((8, 8), (4, 4), stride=4, padding=0)
        #test2((8, 8), (4, 4), stride=5, padding=2)
        #test2((8, 8), (3, 3), stride=3, padding=1)
        #test2((8, 8), (5, 5), stride=5, padding=1)
        #test2((8, 8), (5, 5), stride=3, padding=0)

        #test2((8, 8), (5, 5), stride=3, padding=1)
        test2((8, 8), (5, 5), stride=3, padding=2)
        #test2((9, 9), (5, 5), stride=3, padding=1)

        #test2((8, 8), (5, 5), stride=3, padding=4)
        #test2((8, 8), (5, 5), stride=3, padding=7)
        #test2((8, 8), (5, 5), stride=4, padding=1)
        #test2((8, 8), (6, 6), stride=2, padding=1)
        #test2((8, 8), (6, 6), stride=4, padding=1)
        #test2((8, 8), (7, 7), stride=3, padding=3)
        #test2((8, 8), (7, 7), stride=2, padding=3)
        #test2((8, 8), (7, 7), stride=4, padding=2)
        #test2((18, 18), (7, 7), stride=5, padding=3)
        #test2((8, 8), (3, 3), stride=4, padding=1)
        #test2((8, 8), (3, 3), stride=4, padding=1)
        #test2((8, 8), (3, 3), stride=4, padding=1)
        return
    if True:
        total = 0
        for o in range(8, 14):
            for k in range(3, 9):
                for s in range(2, k+1):
                    for p in range(0, 9):
                        total += 1
                        try:
                            test2((o, o), (k, k), s, p)
                        except ValueError:
                            test2((o, o), (k, k), s, p, add_one=1)
        print("Total", total)


def test2(original_shape, kernel_size, stride, padding, add_one=0):
    #print(f"test2({original_shape}, {kernel_size}, stride={stride}, padding={padding})")
    orig_out = calculate_conv2d_output_dimensions(original_shape[1], original_shape[0], kernel_size, stride, (padding, padding, padding, padding))

    in_channels = 1
    out_channels = 1

    #
    # Setup + Golden
    #
    torch_activations = torch.arange(original_shape[0] * original_shape[1], dtype=torch.float).reshape(1, 1, original_shape[0], original_shape[1])
    torch_weights = torch.arange(kernel_size[0] * kernel_size[1], dtype=torch.float).reshape((1, 1, *kernel_size)) + 1.0
    #if True:
    #    torch_weights = torch.zeros((1, 1, *kernel_size))
    #    torch_weights[:, :, 4, 4] = 1.0

    torch_output = torch.nn.functional.conv2d(torch_activations, torch_weights, stride=stride, padding=padding)
    assert orig_out == (torch_output.shape[-2], torch_output.shape[-1])

    # Make activations a multiple of stride
    torch_activations = pad(torch_activations, (0, round_up_mod(torch_activations.shape[-2], stride), 0, round_up_mod(torch_activations.shape[-1], stride)))

    # Calculate reduced padding
    ps_in_shape = ((original_shape[0] + stride - 1) // stride, (original_shape[1] + stride - 1) // stride)
    init_padding = (padding + stride - 1) // stride
    tail_padding = padding // stride

    # Special case where if the bottom right kernel point ends up in the upper left and tail_padding < init_padding
    # Then we must bump up the tail padding otherwise the bottom right kernel point isn't able to reach its
    # respective prestride data
    tail_padding += int(((kernel_size[0] - padding) % stride == 1) and tail_padding < init_padding)

    def get_offsets(k, stride, padding):
        offsets = torch.zeros(k, dtype=torch.int)
        offsets = pad(offsets, (0, max(0, stride - k)), value=1)
        offsets = offsets.roll(-padding, -1)
        offsets *= offsets.sum()
        return offsets

    kernel_round_up = stride - (kernel_size[0] % stride)
    final_out_diff = ps_in_shape[0] + init_padding + tail_padding - orig_out[0] + 1
    if ((kernel_size[0] + kernel_round_up) // stride) < final_out_diff:
        kernel_round_up += stride

    mx = kernel_round_up
    my = kernel_round_up
    weights_view = torch_weights
    weights_view = pad(weights_view, (0, mx, 0, my))
    weights_view = weights_view.roll((-padding, -padding), (-2, -1))

    y_offsets = get_offsets(kernel_size[1], stride, padding)
    x_offsets = get_offsets(kernel_size[0], stride, padding)

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
            w = w.roll((init_padding, init_padding), (-2, -1))
            ps_weights.append(w)

    ps_activations = torch.cat(ps_activations, dim=-3)
    ps_weights = torch.cat(ps_weights, dim=-3)

    v0 = torch_weights.shape[-3] * torch_weights.shape[-2] * torch_weights.shape[-1]
    v1 = ps_weights.shape[-3] * ps_weights.shape[-2] * ps_weights.shape[-1]

    #
    # Modified padding
    #
    ps_kernel_size = ps_weights.shape[-2]
    tail_padding = orig_out[0] - ps_in_shape[0] - init_padding + (ps_kernel_size - 1)
    ps_padding = (
        init_padding,
        tail_padding,
        init_padding,
        tail_padding,
        )

    #
    # Verify
    #
    ps_activations = pad(ps_activations, ps_padding)
    ps_output = torch.nn.functional.conv2d(ps_activations, ps_weights, stride=1, padding=0)
    assert torch.allclose(torch_output, ps_output, atol=1e-04), f"test2({original_shape}, {kernel_size}, stride={stride}, padding={padding})\n{torch_output}\n{ps_output}\n{ps_weights}\n{tail_padding}\nFAILURE"
    print(f"test2({original_shape}, {kernel_size}, stride={stride}, padding={padding}) {v1 / v0}")

run_tests()
