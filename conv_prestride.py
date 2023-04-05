import math

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

def test(original_shape, kernel_size, stride, padding):
    print(f"test({original_shape}, {kernel_size}, stride={stride}, padding={padding})")
    pre_strided_in_shape = ((original_shape[0] + stride - 1) // stride, (original_shape[1] + stride - 1) // stride)
    orig_out = calculate_conv2d_output_dimensions(original_shape[1], original_shape[0], kernel_size, stride, (padding, padding, padding, padding))

    pre_striders_per_dim = min(kernel_size[0], stride)

    pre_strided_kernel_size = (kernel_size[0] + pre_striders_per_dim - 1) // pre_striders_per_dim
    pre_volume = pre_strided_kernel_size * pre_strided_kernel_size * stride * stride
    assert pre_volume >= kernel_size[0] * kernel_size[1], f"({pre_strided_kernel_size}, {pre_strided_kernel_size}) {stride} v={pre_volume} {kernel_size[0] * kernel_size[1]}"

    # x = (original_x + padding[0] + padding[1] - (kernel_size[1] - 1) - 1) // stride[1] + 1
    #orig_out[0] = pre_strided_in_shape[0] + asdf + asdf - (pre_strided_kernel_size - 1)
    #-2asdf = pre_strided_in_shape[0] - pre_strided_kernel_size - orig_out[0] + 1
    offset = (pre_strided_kernel_size + orig_out[0] - pre_strided_in_shape[0] - 1)
    offset = (offset & 1)

    pre_strided_padding = (padding + stride - 1) // stride
    pre_strided_padding = (
        pre_strided_padding,
        pre_strided_padding - offset,
        pre_strided_padding,
        pre_strided_padding - offset,
    )

    strided_out = calculate_conv2d_output_dimensions(pre_strided_in_shape[1], pre_strided_in_shape[0], pre_strided_kernel_size, 1, pre_strided_padding)

    print("  pre_strided_kernel_size", (pre_strided_kernel_size, pre_strided_kernel_size), "padding", pre_strided_padding, "orig", original_shape, "pre_in", pre_strided_in_shape, "out", orig_out, offset)
    assert orig_out == strided_out, f"{orig_out} != {strided_out}"
    print("  PASSED" if orig_out == strided_out else "  FAILED")
    return orig_out == strided_out


if True:
    num_fails = 0
    for o in range(8, 16):
        for k in range(2, 8):
            for s in range(2, 5):
                for p in range(0, 5):
                    num_fails += not test((o, o), (k, k), s, p)
    print("Num fails", num_fails)
else:
    test((8, 8), (4, 4), 2, 1)
    test((8, 8), (4, 4), 2, 2)
    test((10, 10), (4, 4), 2, 2)
