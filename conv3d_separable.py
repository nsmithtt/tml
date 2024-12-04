import torch


# Channels last convenience function
def ttnn_conv2d(act, weight, **kwargs):
    assert len(act.shape) == 4
    assert len(weight.shape) == 4
    act = act.permute(0, 3, 1, 2)
    weight = weight.permute(3, 2, 0, 1)
    result = torch.nn.functional.conv2d(act, weight, **kwargs)
    return result.permute(0, 2, 3, 1)


def test_conv3d(
    batch,
    inC,
    outC,
    iD,
    iH,
    iW,
    kD,
    kH,
    kW,
    stride,
    padding,
    verbose=True,
):
    golden_act = torch.randn(batch, inC, iD, iH, iW)
    golden_weight = torch.randn(outC, inC, kD, kH, kW)
    golden = torch.nn.functional.conv3d(
        golden_act, golden_weight, stride=stride, padding=padding
    )

    # For convenience for working with TTNN shapes, we'll use channels last
    # [batch, iD, iH, iW, inC]
    channels_last_act = golden_act.permute(0, 2, 3, 4, 1)
    # [kD, kH, kW, inC, outC]
    channels_last_weight = golden_weight.permute(2, 3, 4, 1, 0)

    act = channels_last_act
    weight = channels_last_weight

    # Collapse the batch and iD dimensions
    # - [batch*iD, iH, iW, inC]
    act = act.reshape(batch * iD, iH, iW, inC)

    # Collapse the kD and outC dimensions
    # - [kH, kW, inC, outC*kD]
    weight = weight.permute(1, 2, 3, 4, 0).reshape(kH, kW, inC, outC * kD)

    # Conv 1. run conv2d over the iH, iW dimensions, we get kD extra sets of output channels
    # - [batch*iD, oH, oW, outC*kD]
    conv1 = ttnn_conv2d(act, weight, stride=stride, padding=padding)

    # Pull back out the iD dimension and collapse the oHoW dimensions
    # - [batch, iD, oH*oW, outC*kD]
    conv1 = conv1.reshape(batch, iD, conv1.shape[1] * conv1.shape[2], conv1.shape[3])

    # Conv 2. run conv2d over the iD dimension, we get outC sets of output channels
    # Construct grouped identity weights
    # - [kD, 1, kD, outC]
    ident = torch.eye(kD).reshape(kD, 1, kD, 1).repeat(1, 1, 1, outC)
    conv2 = ttnn_conv2d(
        conv1, ident, stride=(stride, 1), padding=(padding, 0), groups=outC
    )

    # Move back to torch style channels first
    result = conv2.reshape(
        batch, golden.shape[-3], golden.shape[-2], golden.shape[-1], outC
    ).permute(0, 4, 1, 2, 3)

    test_str = f"test_conv3d({batch}, {inC}, {outC}, {iD}, {iH}, {iW}, {kD}, {kH}, {kW}, {stride}, {padding})"
    if verbose:
        print(test_str)
    try:
        assert torch.allclose(result, golden, atol=1e-05)
    except:
        print("result", result)
        print("golden", golden)
        raise Exception(test_str)


# Failures
test_conv3d(1, 4, 3, 4, 6, 7, 3, 4, 4, 2, 0)
test_conv3d(1, 3, 3, 5, 5, 4, 3, 4, 4, 1, 2)


def test_sweep():
    i = 0
    for batch in range(1, 3):
        for inC in range(3, 5):
            for outC in range(3, 5):
                for kD in range(1, 5):
                    for kH in range(1, 5):
                        for kW in range(1, 5):
                            for iD in range(kD, 8):
                                for iH in range(kH, 8):
                                    for iW in range(kW, 8):
                                        for stride in range(1, 4):
                                            for padding in range(0, 3):
                                                print(i)
                                                i += 1
                                                test_conv3d(
                                                    batch,
                                                    inC,
                                                    outC,
                                                    iD,
                                                    iH,
                                                    iW,
                                                    kD,
                                                    kH,
                                                    kW,
                                                    stride,
                                                    padding,
                                                    verbose=False,
                                                )


test_sweep()
