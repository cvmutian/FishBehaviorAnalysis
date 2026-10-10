import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tcn import MDTCM

PART_NAMES = ("head", "trunk", "tail")
DEFAULT_PART_CHANNELS = 32


class FBPEM(nn.Module):
    def __init__(self, part_indices, embed_channels: int, part_channels: int):
        super().__init__()
        if len(part_indices) != len(PART_NAMES):
            raise ValueError("FBPEM 需要头部、躯干和尾部三个部位")
        if int(part_channels) <= 0 or int(embed_channels) <= 0:
            raise ValueError("FBPEM 通道数必须为正整数")
        self.part_indices = {
            name: list(indices) for name, indices in zip(PART_NAMES, part_indices)
        }
        self.part_maps = nn.ModuleDict(
            {name: nn.Linear(2, int(part_channels)) for name in PART_NAMES}
        )
        self.fuse = nn.Conv1d(int(part_channels) * len(PART_NAMES), int(embed_channels), kernel_size=1)

    def forward(self, coord: torch.Tensor) -> torch.Tensor:
        mapped = []
        for name in PART_NAMES:
            pooled = coord[:, :, self.part_indices[name], :].mean(dim=2)
            mapped.append(self.part_maps[name](pooled))
        stacked = torch.cat(mapped, dim=-1)
        return self.fuse(stacked.transpose(1, 2)).transpose(1, 2)


class PQTGM(nn.Module):
    def __init__(self, channels: int, kernel_size: int, smooth_window: int):
        super().__init__()
        padding = kernel_size // 2
        self.gate = nn.Conv1d(channels + 1, channels, kernel_size, padding=padding)
        if smooth_window % 2 == 0:
            raise ValueError("PQTGM 平滑窗口需要是奇数")
        self.smooth = nn.AvgPool1d(smooth_window, stride=1, padding=smooth_window // 2)

    def forward(self, fused: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        sequence = fused.transpose(1, 2)
        gate_input = torch.cat([fused, quality], dim=-1).transpose(1, 2)
        gate = torch.sigmoid(self.gate(gate_input))
        smoothed = self.smooth(sequence)
        if smoothed.size(-1) != sequence.size(-1):
            smoothed = _match_time(smoothed, sequence.size(-1))
        if gate.size(-1) != sequence.size(-1):
            gate = _match_time(gate, sequence.size(-1))
        return (gate * sequence + (1.0 - gate) * smoothed).transpose(1, 2)


class MLPHead(nn.Module):
    def __init__(self, channels, dropout: float):
        super().__init__()
        if len(channels) != 3:
            raise ValueError("输出头通道应按 [输入, 隐藏, 输出] 配置")
        self.net = nn.Sequential(
            nn.Linear(int(channels[0]), int(channels[1])),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(int(channels[1]), int(channels[2])),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiStreamTCN(nn.Module):
    def __init__(self, config: dict, motion_dim: int, geometry_dim: int):
        super().__init__()
        model_cfg = config["model"]
        keypoints_cfg = config["keypoints"]
        part_indices = [
            keypoints_cfg["parts"]["head"],
            keypoints_cfg["parts"]["trunk"],
            keypoints_cfg["parts"]["tail"],
        ]
        fusion_channels = int(model_cfg["fusion_channels"])
        dropout = float(model_cfg["dropout"])
        self.num_keypoints = int(config["num_keypoints"])
        part_channels = int(model_cfg.get("fbpem", {}).get("part_channels", DEFAULT_PART_CHANNELS))
        self.fbpem = FBPEM(part_indices, int(model_cfg["coord_embed"]), part_channels)
        self.motion_map = nn.Linear(motion_dim, int(model_cfg["motion_embed"]))
        self.geometry_map = nn.Linear(geometry_dim, int(model_cfg["geometry_embed"]))
        embed_sum = int(model_cfg["coord_embed"]) + int(model_cfg["motion_embed"]) + int(model_cfg["geometry_embed"])
        if embed_sum != fusion_channels:
            raise ValueError("三流拼接通道与 fusion_channels 不一致")
        self.stream_fusion = nn.Conv1d(fusion_channels, fusion_channels, kernel_size=1)
        pqtgm_cfg = model_cfg["pqtgm"]
        self.pqtgm = PQTGM(
            fusion_channels,
            int(pqtgm_cfg["kernel_size"]),
            int(pqtgm_cfg["smooth_window"]),
        )
        mdtcm_cfg = model_cfg["mdtcm"]
        self.mdtcm = MDTCM(
            fusion_channels,
            int(mdtcm_cfg["layers"]),
            int(mdtcm_cfg["kernel_size"]),
            [int(item) for item in mdtcm_cfg["dilations"]],
            dropout,
        )
        self.classifier = MLPHead(model_cfg["classifier_channels"], dropout)
        self.regressor = MLPHead(model_cfg["regressor_channels"], dropout)

    def forward(self, coord, motion, geometry, confidence=None) -> dict:
        if coord.dim() != 4:
            raise ValueError("坐标流形状应为 [B, T, K, 2]")
        if not (coord.size(1) == motion.size(1) == geometry.size(1)):
            raise ValueError("三个分支的时间长度不一致")
        if coord.size(2) != self.num_keypoints or coord.size(-1) != 2:
            raise ValueError(f"坐标流应为 [B, T, {self.num_keypoints}, 2]")
        coord_embed = self.fbpem(coord)
        concatenated = torch.cat(
            [coord_embed, self.motion_map(motion), self.geometry_map(geometry)],
            dim=-1,
        )
        if concatenated.size(-1) != self.stream_fusion.in_channels:
            raise ValueError(
                f"三流拼接通道为 {concatenated.size(-1)}，卷积融合需要 {self.stream_fusion.in_channels}"
            )
        fused = self.stream_fusion(concatenated.transpose(1, 2)).transpose(1, 2)
        quality = _quality(coord, confidence)
        gated = self.pqtgm(fused, quality)
        temporal = self.mdtcm(gated.transpose(1, 2))
        pooled = temporal.mean(dim=-1)
        return {
            "logits": self.classifier(pooled),
            "aux": self.regressor(pooled),
        }


def build_model(config: dict, motion_dim: int, geometry_dim: int) -> MultiStreamTCN:
    return MultiStreamTCN(config, motion_dim, geometry_dim)


def _quality(coord: torch.Tensor, confidence) -> torch.Tensor:
    delta = torch.linalg.norm(coord[:, 1:] - coord[:, :-1], dim=-1).mean(dim=-1)
    delta = F.pad(delta, (1, 0))
    stability = torch.exp(-delta).unsqueeze(-1)
    if confidence is None:
        mean_confidence = torch.ones(coord.shape[0], coord.shape[1], 1, device=coord.device, dtype=coord.dtype)
    else:
        mean_confidence = confidence.mean(dim=-1, keepdim=True)
    return mean_confidence * stability


def _match_time(tensor: torch.Tensor, length: int) -> torch.Tensor:
    current = tensor.size(-1)
    if current == length:
        return tensor
    if current > length:
        start = (current - length) // 2
        return tensor[..., start:start + length]
    pad_left = (length - current) // 2
    pad_right = length - current - pad_left
    return F.pad(tensor, (pad_left, pad_right))

