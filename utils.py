import torch


def calculate_conv2d_output_dimensions(
    original_y, original_x, kernel_size, stride, padding, dilation=1, ceil_mode=False
):
    if isinstance(stride, int):
        stride = [stride] * 2

    assert len(padding) == 4 and all(
        isinstance(x, int) for x in padding
    ), "Padding should be list of four ints"

    # Pooling layers (max, avg)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    # Padding is [left, right, top, bottom]
    if ceil_mode:
        y = (
            math.ceil(
                (
                    original_y
                    + padding[2]
                    + padding[3]
                    - dilation * (kernel_size[0] - 1)
                    - 1
                )
                / stride[0]
            )
            + 1
        )
        x = (
            math.ceil(
                (
                    original_x
                    + padding[0]
                    + padding[1]
                    - dilation * (kernel_size[1] - 1)
                    - 1
                )
                / stride[1]
            )
            + 1
        )
    else:
        y = (
            original_y + padding[2] + padding[3] - dilation * (kernel_size[0] - 1) - 1
        ) // stride[0] + 1
        x = (
            original_x + padding[0] + padding[1] - dilation * (kernel_size[1] - 1) - 1
        ) // stride[1] + 1
    return y, x


def create_conv2d_sparse_picker_matrix(
    y,
    x,
    y_shift,
    x_shift,
    k_y,
    k_x,
    stride,
    padding,
    dilation,
    tile_align=False,
    pad_x_only=False,
    sparse_r_pad=0,
    sparse_c_pad=0,
):
    cols = torch.arange(start=1, end=y * x + 1).view(y, x)

    # pad
    cols = torch.nn.functional.pad(cols, padding)
    # shift
    shift_y = dilation * ((k_y - 1) // 2 - y_shift)
    shift_x = dilation * ((k_x - 1) // 2 - x_shift)
    cols = torch.nn.functional.pad(cols, (-shift_x, shift_x, -shift_y, shift_y))
    # stride
    cols = cols[:: stride[0], :: stride[1]]
    # clamp to output dims
    out_y, out_x = calculate_conv2d_output_dimensions(
        y, x, [k_y, k_x], stride, padding, dilation
    )

    cols = torch.nn.functional.pad(
        cols, (0, out_x - cols.shape[1], 0, out_y - cols.shape[0])
    )

    cols = cols.reshape(-1)
    rows = torch.arange(cols.shape[0])
    rows = rows.index_select(0, cols.nonzero().flatten())
    cols = cols.index_select(0, cols.nonzero().flatten())
    cols -= 1

    if pad_x_only:
        # Channel last conv
        sparse_r = align_up_tile(out_x) * out_y
        sparse_c = align_up_tile(x) * y
    elif tile_align:
        sparse_r = align_up_tile(out_y * out_x)
        sparse_c = align_up_tile(y * x)
        if sparse_r_pad:
            sparse_r_tile = align_up_tile(out_y * out_x) // 32
            sparse_r = (sparse_r_tile + sparse_r_pad) * 32
        if sparse_c_pad:
            sparse_c += sparse_c_pad * 32
    else:
        sparse_r = out_y * out_x
        sparse_c = y * x
    return torch.sparse_coo_tensor(
        [rows.tolist(), cols.tolist()],
        torch.ones(cols.shape[0]),
        (sparse_r, sparse_c),
        dtype=torch.float32,
    ).coalesce()


def create_conv2d_sparse_matrix(
    y,
    x,
    kH,
    kW,
    stride,
    padding,
    dilation,
):
    orig_padding = [*padding]
    pickers = []
    for kY in range(kH):
        for kX in range(kW):
            # pickers are created row-major, starting from top-left kernel pixel
            y_shift = ((kH - 1) // 2) - kY
            x_shift = ((kW - 1) // 2) - kX
            picker = create_conv2d_sparse_picker_matrix(
                y, x, y_shift, x_shift, kH, kW, stride, padding, dilation
            )
            pickers.append(picker)
    return torch.stack(pickers)


def hslice(x, factor):
    shape = list(x.shape)
    assert shape[-1] % factor == 0
    while len(shape) < 4:
        shape = [1] + shape
    ret = x.reshape(-1, shape[-2], factor, shape[-1] // factor)
    ret = ret.permute(0, 2, 1, 3)
    return ret.reshape(
        shape[:-3] + [shape[-3] * factor, shape[-2], shape[-1] // factor]
    )


def hstack(x, factor):
    shape = list(x.shape)
    assert shape[-3] % factor == 0, f"HStack requires Z to be divisible by slice size"
    ret = x.reshape(-1, shape[-3] // factor, factor, shape[-2], shape[-1])
    ret = ret.permute(0, 1, 3, 2, 4)
    return ret.reshape(
        shape[:-3] + [shape[-3] // factor, shape[-2], shape[-1] * factor]
    )


def vslice(x, factor):
    shape = x.shape
    assert len(shape) >= 2
    assert shape[-2] % factor == 0
    if len(shape) < 3:
        shape = (1,) + shape
    return x.reshape(shape[:-3] + (shape[-3] * factor, shape[-2] // factor, shape[-1]))


def vstack(x, factor):
    shape = x.shape
    assert shape[-3] % factor == 0, f"VStack requires Z to be divisible by slice size"
    return x.reshape(shape[:-3] + (shape[-3] // factor, shape[-2] * factor, shape[-1]))


def volume(l):
    v = 1
    for e in l:
        v *= e
    return v


def clamp(x, min_x, max_x):
    return min(max(x, min_x), max_x)
