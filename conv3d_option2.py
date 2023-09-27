import torch
from utils import (
    calculate_conv2d_output_dimensions,
    create_conv2d_sparse_matrix,
    hstack,
    vstack,
    volume,
)


def test_conv3d(
    inC,
    iD,
    iH,
    iW,
    outC,
    kD,
    kH,
    kW,
    stride,
    padding,
    dilation,
    verbose=True,
):
    assert iD >= kD, "todo"
    outD = (iD + padding * 2 - dilation * (kD - 1) - 1) // stride + 1
    outH, outW = calculate_conv2d_output_dimensions(
        iH, iW, (kH, kW), (stride, stride), [padding] * 4
    )
    if verbose:
        print(
            f"Activations[{inC}, {iD}, {iH}, {iW}] Kernel[{kD}, {kH}, {kW}] stride={stride} padding={padding} -> Result[{outC}, {outD}, {outH}, {outW}]"
        )

    ################
    # Golden
    ################

    golden_act = torch.randn(inC, iD, iH, iW)
    golden_weight = torch.randn(outC, inC, kD, kH, kW)
    golden = torch.nn.functional.conv3d(
        golden_act, golden_weight, stride=stride, padding=padding
    )

    ################
    # Decomp
    ################

    # Setup the activations into buda conv layout
    act = golden_act.reshape(inC * iD, iH * iW).transpose(0, 1)

    act_sparse = create_conv2d_sparse_matrix(
        iH,
        iW,
        kH,
        kW,
        (stride, stride),
        [padding] * 4,
        dilation=dilation,
    )

    # Sparse mm
    act_shifted = act_sparse.to_dense() @ act

    # Setup the weights into 3D conv layout
    weight = golden_weight.reshape(outC, inC, kD, kH * kW).permute(3, 0, 2, 1)
    # -> (kH * kW, outC, kD, inC)

    weight_sparse = create_conv2d_sparse_matrix(
        1,
        iD,
        1,
        kD,
        (1, stride),
        [padding, padding, 0, 0],
        dilation=dilation,
    )

    weight_sparse = weight_sparse.to_dense()  # (kD, outD, iD)
    weight_sparse = weight_sparse.permute(1, 2, 0)  # (outD, iD, kD)
    weight_sparse = vstack(weight_sparse, weight_sparse.shape[0]) # (1, outD * iD, iD)
    weight_sparse = weight_sparse.squeeze(0)  # (outD * iD, iD)
    weight_shifted = weight_sparse @ weight  # (kH * kW, outC, outD * iD, inC)
    weight_shifted = weight_shifted.reshape(kH * kW, outC, outD, iD, inC)
    weight_shifted = weight_shifted.transpose(-2, -1)
    weight_shifted = weight_shifted.reshape(kH * kW, outC * outD, inC * iD)
    weight_shifted = weight_shifted.transpose(-2, -1)
    #print("Sparsity", len(weight_shifted.to_sparse().values()) / volume(weight_shifted.shape), kD / (iD + padding * 2))

    act_shifted = hstack(act_shifted, act_shifted.shape[0]).squeeze(0)
    weight_shifted = vstack(weight_shifted, weight_shifted.shape[0]).squeeze(0)
    result = act_shifted @ weight_shifted

    result = result.transpose(0, 1).reshape(outC, outD, outH, outW)
    # print(golden)
    # print(result)
    assert torch.allclose(result, golden, atol=1e-4)


# Simple test
def test_simple():
    inC = 2
    iD = 5
    iH = 5
    iW = 5
    outC = 4
    kD = 3
    kH = 3
    kW = 3
    stride = 1
    padding = 0
    dilation = 1

    test_conv3d(
        inC,
        iD,
        iH,
        iW,
        outC,
        kD,
        kH,
        kW,
        stride,
        padding,
        dilation,
    )


def test_sweep():
    i = 0
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
                                            for dilation in range(1, 2):
                                                print(i)
                                                i += 1
                                                try:
                                                    test_conv3d(
                                                        inC,
                                                        iD,
                                                        iH,
                                                        iW,
                                                        outC,
                                                        kD,
                                                        kH,
                                                        kW,
                                                        stride,
                                                        padding,
                                                        dilation,
                                                        verbose=False,
                                                    )
                                                except:
                                                    print("Repro")
                                                    print(
                                                        f"  test_conv3d({inC}, {iD}, {iH}, {iW}, {outC}, {kD}, {kH}, {kW}, {stride}, {padding}, {dilation})"
                                                    )
                                                    raise


# test_simple()
#test_conv3d(3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1)
# test_conv3d(3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1)
# test_conv3d(3, 1, 1, 1, 3, 1, 1, 1, 1, 2, 1)
# test_conv3d(3, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1)
# test_conv3d(3, 1, 1, 1, 3, 1, 1, 1, 1, 2, 1)
# test_conv3d(3, 1, 1, 1, 3, 1, 1, 1, 2, 2, 1)
# test_conv3d(3, 1, 1, 1, 3, 1, 1, 1, 2, 2, 1)
# test_conv3d(3, 1, 1, 1, 3, 1, 1, 1, 2, 0, 1)
# test_conv3d(3, 1, 1, 1, 3, 1, 1, 1, 2, 2, 1)
# test_conv3d(3, 1, 2, 2, 3, 1, 1, 1, 2, 1, 1)
test_sweep()
