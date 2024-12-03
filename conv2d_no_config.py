import torch


NOC_MAX_BURST_SIZE = 512


def calculate_conv2d_output_dimensions(
    original_y, original_x, kernel_size, stride, padding, dilation=1
):
    assert len(padding) == 4 and all(
        isinstance(x, int) for x in padding
    ), "Padding should be list of four ints"

    if isinstance(stride, int):
        stride = [stride] * 2

    # Pooling layers (max, avg)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    y = (
        original_y + padding[2] + padding[3] - dilation * (kernel_size[0] - 1) - 1
    ) // stride[0] + 1
    x = (
        original_x + padding[0] + padding[1] - dilation * (kernel_size[1] - 1) - 1
    ) // stride[1] + 1
    return y, x


def pad(act, padding):
    assert len(padding) == 4 and all(
        isinstance(x, int) for x in padding
    ), "Padding should be list of four ints"
    assert len(input.shape) == 4

    (pl, pr, pt, pb) = padding
    (n, h, w, c) = act.shape
    act = act.reshape(act.shape[0], -1, act.shape[-1])
    out_h, out_w = (h + pt + pb, w + pl + pr)
    out = torch.rand((act.shape[0], out_h * out_w, act.shape[-1]))
    zero = torch.zeros((NOC_MAX_BURST_SIZE // 2) * c)

    for ni in range(n):
        # pad top

        # pad middle
        for hi in range(h):
            # pad left
            # copy act
            # pad right

        # pad bottom

