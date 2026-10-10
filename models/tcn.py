import torch
import torch.nn as nn


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.pointwise(self.depthwise(x)))))


class MDTCMBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations, dropout: float):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                DepthwiseSeparableConv1d(channels, kernel_size, int(dilation), dropout)
                for dilation in dilations
            ]
        )
        self.fuse = nn.Conv1d(channels * len(dilations), channels, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))
        if fused.size(-1) != x.size(-1):
            fused = _align_time(fused, x.size(-1))
        out = (fused + x).transpose(1, 2)
        return self.norm(out).transpose(1, 2)


class MDTCM(nn.Module):
    def __init__(self, channels: int, layers: int, kernel_size: int, dilations, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MDTCMBlock(channels, kernel_size, dilations, dropout)
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _align_time(tensor: torch.Tensor, length: int) -> torch.Tensor:
    current = tensor.size(-1)
    if current == length:
        return tensor
    if current > length:
        start = (current - length) // 2
        return tensor[..., start:start + length]
    pad_left = (length - current) // 2
    pad_right = length - current - pad_left
    return nn.functional.pad(tensor, (pad_left, pad_right))
