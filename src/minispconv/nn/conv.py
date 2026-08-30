import torch.nn as nn


class SpConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: str | int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: str | None = None,
    ):
        pass
