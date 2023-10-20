import torch
from utils import (
    calculate_conv2d_output_dimensions,
    create_conv2d_sparse_matrix,
    hstack,
    vstack,
    volume,
)

inC = 3
iD = 8
iH = 8
iW = 8
outC = inC
kD = 3
kH = 1
kW = 1
stride = 1
padding = 0
dilation = 1

outD = (iD + padding * 2 - dilation * (kD - 1) - 1) // stride + 1
outH, outW = calculate_conv2d_output_dimensions(
    iH, iW, (kH, kW), (stride, stride), [padding] * 4
)

################
# Golden
################

golden_act = torch.randn(inC, iD, iH, iW)
golden = torch.nn.functional.max_pool3d(
    golden_act,
    (kD, kH, kW),
    stride=stride,
    padding=padding,
    dilation=dilation,
)

################
# Decomp
################
# Setup the activations into buda conv layout
act = golden_act.reshape(inC * iD, iH * iW).transpose(0, 1)

# Max pool2d on all of the 2D faces first, per usual
act_sparse = create_conv2d_sparse_matrix(
    iH,
    iW,
    kH,
    kW,
    (stride, stride),
    [padding] * 4,
    dilation=dilation,
)

act_shifted = act_sparse.to_dense() @ act
result_2d = torch.max(act_shifted, -3)[0]

# Run max pool on the depth dimension in a separate step
depth_sparse = create_conv2d_sparse_matrix(
    inC,
    iD,
    1,
    kD,
    (1, stride),
    [padding, padding, 0, 0],
    dilation=dilation,
)

# Transpose the activations to allow sparse mm to work on the depth dim
result_2d_T = result_2d.transpose(-2, -1)
result_2d_T_shifted = depth_sparse.to_dense() @ result_2d_T
result_3d_T = torch.max(result_2d_T_shifted, -3)[0]
# Transpose back
result_3d = result_3d_T.transpose(-2, -1)

# Undo buda conv shape for golden check
result = result_3d.transpose(-2, -1).reshape(outC, outD, outH, outW)

assert torch.allclose(result, golden, atol=1e-4)
