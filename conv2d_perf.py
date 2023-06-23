import torch
import time

def calculate_conv2d_output_dimensions(
    original_y, original_x, kernel_size, stride, padding, dilation=1, ceil_mode=False
):
    if isinstance(stride, int):
        stride = [stride] * 2

    assert len(padding) == 4 and all(isinstance(x, int) for x in padding), "Padding should be list of four ints"

    # Pooling layers (max, avg)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    # Padding is [left, right, top, bottom]
    if ceil_mode:
        y = (
            math.ceil(
                (original_y + padding[2] + padding[3] - dilation * (kernel_size[0] - 1) - 1) / stride[0]
            )
            + 1
        )
        x = (
            math.ceil(
                (original_x + padding[0] + padding[1] - dilation * (kernel_size[1] - 1) - 1) / stride[1]
            )
            + 1
        )
    else:
        y = (original_y + padding[2] + padding[3] - dilation * (kernel_size[0] - 1) - 1) // stride[0] + 1
        x = (original_x + padding[0] + padding[1] - dilation * (kernel_size[1] - 1) - 1) // stride[1] + 1
    return y, x

def create_conv2d_sparse_picker_matrix(y, x, y_shift, x_shift, k_y, k_x, stride, padding, dilation=1, cols_only=False):
    cols = torch.arange(start=1, end=y * x + 1).view(y, x)

    # pad
    cols = torch.nn.functional.pad(cols, padding)
    # shift
    shift_y = dilation * ((k_y - 1) // 2 - y_shift)
    shift_x = dilation * ((k_x - 1) // 2 - x_shift)
    cols = torch.nn.functional.pad(cols, (-shift_x, shift_x, -shift_y, shift_y))
    # stride
    cols = cols[::stride[0], ::stride[1]]
    # clamp to output dims
    out_y, out_x = calculate_conv2d_output_dimensions(y, x, [k_y, k_x], stride, padding)

    cols = torch.nn.functional.pad(
        cols, (0, out_x - cols.shape[1], 0, out_y - cols.shape[0])
    )

    cols = cols.reshape(-1)

    if cols_only:
        cols -= 1
        return cols

    rows = torch.arange(cols.shape[0])
    rows = rows.index_select(0, cols.nonzero().flatten())
    cols = cols.index_select(0, cols.nonzero().flatten())
    cols -= 1

    sparse_r = out_y * out_x
    sparse_c = y * x
    return torch.sparse_coo_tensor(
        [rows.tolist(), cols.tolist()],
        torch.ones(cols.shape[0]),
        (sparse_r, sparse_c),
        dtype=torch.float32,
    ).coalesce()


def conv2d_shift(act, y, x, y_shift, x_shift, k_y, k_x, stride, padding, dilation=1, cols_only=False):

    # shift
    shift_y = dilation * ((k_y - 1) // 2 - y_shift)
    shift_x = dilation * ((k_x - 1) // 2 - x_shift)
    #act = torch.nn.functional.pad(act, (-shift_x, shift_x, -shift_y, shift_y))

    # stride
    act = act[:, :, shift_y::stride[0], shift_x::stride[1]]

    # clamp to output dims
    out_y, out_x = calculate_conv2d_output_dimensions(y, x, [k_y, k_x], stride, padding)

    act = act[:, :, :out_y, :out_x]

    return act

    act = torch.nn.functional.pad(
        act, (0, out_x - act.shape[1], 0, out_y - act.shape[0])
    )

    return act


batch = 32
loop = 32
df = torch.float16

iH = 224
iW = 224
inC = 3
outC = 64
kW = 7
kH = 7
stride = 2
padding = kW // 2

pickers = []
for kY in range(kH):
    for kX in range(kW):
        y_shift = ((kH - 1) // 2) - kY
        x_shift = ((kW - 1) // 2) - kX
        picker = create_conv2d_sparse_picker_matrix(iH, iW, y_shift, x_shift, kH, kW, (stride, stride), (padding, padding, padding, padding))
        pickers.append(picker)

if False:
    act = torch.randn(batch, inC, iH, iW, dtype=df)
    layer = torch.nn.Conv2d(inC, outC, kW, stride=stride, padding=padding, dtype=df)
elif False:
    sparse = torch.stack([torch.cat(pickers, dim=-2)]*batch)
    act = torch.randn(batch, iH*iW, inC, dtype=df)
    layer = lambda x: torch.bmm(sparse, x)
else:
    act = torch.zeros(batch, inC, iH, iW, dtype=df, requires_grad=False)
    def layer(x):
        # pad
        xs = []
        x = torch.nn.functional.pad(x, (padding, padding, padding, padding))
        for kY in range(kH):
            for kX in range(kW):
                y_shift = ((kH - 1) // 2) - kY
                x_shift = ((kW - 1) // 2) - kX
                xs.append(conv2d_shift(x, iH, iW, y_shift, x_shift, kH, kW, (stride, stride), (padding, padding, padding, padding)))

        torch.cat(xs, dim=-3)

start = time.time()

for i in range(loop):
    layer(act)

stop = time.time()

t = stop - start
print("Samples/sec", (batch * loop) / t)
