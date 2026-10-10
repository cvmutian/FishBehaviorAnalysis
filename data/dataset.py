from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from features.feature_extractor import FeatureExtractor


class BehaviorDataset(Dataset):
    def __init__(self, data_dir: str, config: dict, extractor: FeatureExtractor = None, augment: bool = False):
        self.data_dir = Path(data_dir)
        self.config = config
        self.window_size = int(config["window_size"])
        self.num_keypoints = int(config["num_keypoints"])
        self.num_classes = int(config["num_classes"])
        self.extractor = extractor if extractor is not None else FeatureExtractor(config)
        self.augment = augment
        self.augment_cfg = config["augment"]
        labeling = config.get("labeling", {})
        self.use_unverified = bool(labeling.get("use_unverified", False))
        self.allow_legacy_samples = bool(labeling.get("allow_legacy_samples", False))
        self.files = []
        self.skip_reasons = {}
        self.scanned = 0
        self.skipped_unverified = 0
        self.accepted_legacy = 0
        self.accepted_prelabel = 0
        if self.data_dir.is_dir():
            paths = sorted(self.data_dir.glob("*.npz"))
            self.scanned = len(paths)
            for path in paths:
                decision, reason = _sample_status(
                    path,
                    self.num_classes,
                    self.use_unverified,
                    self.allow_legacy_samples,
                )
                if decision == "accept":
                    self.files.append(path)
                    if reason == "legacy":
                        self.accepted_legacy += 1
                    elif reason == "prelabel":
                        self.accepted_prelabel += 1
                else:
                    self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1
            self.skipped_unverified = self.scanned - len(self.files)

    def __len__(self) -> int:
        return len(self.files)

    def filter_report(self) -> str:
        if self.scanned == 0:
            return "没有找到 npz 样本。"
        verified = len(self.files) - self.accepted_legacy - self.accepted_prelabel
        parts = [f"已核验 {verified} 个"]
        if self.accepted_legacy:
            parts.append(f"旧格式兼容 {self.accepted_legacy} 个")
        if self.accepted_prelabel:
            parts.append(f"显式纳入的预标注 {self.accepted_prelabel} 个")
        text = f"扫描 {self.scanned} 个样本，加载 {len(self.files)} 个（{'，'.join(parts)}）"
        if self.skip_reasons:
            detail = "，".join(
                f"{_REASON_TEXT[key]} {self.skip_reasons[key]} 个" for key in sorted(self.skip_reasons)
            )
            text += f"，过滤 {self.skipped_unverified} 个（{detail}）"
        return text + "。"

    def __getitem__(self, index: int) -> dict:
        path = self.files[index]
        with np.load(path) as data:
            keypoints = np.asarray(data["keypoints"], dtype=np.float32)
            label = int(np.asarray(data["label"]).reshape(-1)[0])
            track_id = int(np.asarray(data["track_id"]).reshape(-1)[0]) if "track_id" in data.files else -1
        if keypoints.shape[0] != self.window_size or keypoints.shape[1] != self.num_keypoints:
            raise ValueError(
                f"{path.name} 的关键点形状为 {keypoints.shape}，"
                f"需要 [{self.window_size}, {self.num_keypoints}, 2] 或 [{self.window_size}, {self.num_keypoints}, 3]"
            )
        if label < 0 or label >= self.num_classes:
            raise ValueError(f"{path.name} 的标签 {label} 不在 0–{self.num_classes - 1}")
        dt_scale = 1.0
        if self.augment:
            keypoints, dt_scale = self._augment(keypoints)
        features = self.extractor.extract(keypoints, dt=self.extractor.dt * dt_scale)
        return {
            "coord": torch.from_numpy(np.ascontiguousarray(features["coord"])),
            "motion": torch.from_numpy(np.ascontiguousarray(features["motion"])),
            "geometry": torch.from_numpy(np.ascontiguousarray(features["geometry"])),
            "confidence": torch.from_numpy(np.ascontiguousarray(features["confidence"])),
            "aux": torch.from_numpy(np.ascontiguousarray(features["aux"])),
            "label": torch.tensor(label, dtype=torch.long),
            "track_id": torch.tensor(track_id, dtype=torch.long),
        }

    def _augment(self, keypoints: np.ndarray):
        cropped, dt_scale = _temporal_crop(
            keypoints,
            float(self.augment_cfg["time_crop_min"]),
            float(self.augment_cfg["time_crop_max"]),
        )
        noisy = _coordinate_noise(
            cropped,
            float(self.augment_cfg["coord_noise_std"]),
            int(self.config["keypoints"]["mouth_index"]),
            int(self.config["keypoints"]["tail_index"]),
            float(self.config["preprocess"]["eps"]),
        )
        dropped = _drop_keypoints(noisy, float(self.augment_cfg["keypoint_drop_prob"]))
        return dropped, dt_scale


def _temporal_crop(keypoints: np.ndarray, low: float, high: float):
    time_steps = keypoints.shape[0]
    ratio = float(np.random.uniform(low, high))
    length = min(time_steps, max(2, int(round(time_steps * ratio))))
    start = 0 if length == time_steps else int(np.random.randint(0, time_steps - length + 1))
    clip = keypoints[start:start + length]
    if length == time_steps:
        return clip.copy(), 1.0
    dt_scale = (length - 1) / max(time_steps - 1, 1)
    source = np.linspace(0, length - 1, time_steps)
    output = np.empty_like(keypoints)
    positions = np.arange(length)
    for keypoint_id in range(keypoints.shape[1]):
        for channel in range(keypoints.shape[2]):
            output[:, keypoint_id, channel] = np.interp(source, positions, clip[:, keypoint_id, channel])
    return output.astype(np.float32), float(dt_scale)


def _coordinate_noise(keypoints: np.ndarray, std: float, mouth: int, tail: int, eps: float) -> np.ndarray:
    if std <= 0:
        return keypoints
    output = keypoints.copy()
    body_length = np.linalg.norm(output[:, mouth, :2] - output[:, tail, :2], axis=-1)
    body_length = np.maximum(body_length, eps)
    noise = np.random.normal(0.0, std, size=output[:, :, :2].shape).astype(np.float32)
    output[:, :, :2] = output[:, :, :2] + noise * body_length[:, None, None].astype(np.float32)
    return output


_REASON_TEXT = {
    "missing_label_source": "缺少 label_source",
    "prelabel": "未核验预标注",
    "review": "待复核",
    "invalid_label": "无效标签",
    "unknown_source": "未知标签来源",
}
_VERIFIED_SOURCES = {"manual"}


def _sample_status(path: Path, num_classes: int, use_unverified: bool, allow_legacy: bool):
    with np.load(path) as data:
        label = int(np.asarray(data["label"]).reshape(-1)[0])
        source = _read_label_source(data)
    if source == "review" or label < 0 or label >= int(num_classes):
        if source == "review":
            return "reject", "review"
        return "reject", "invalid_label"
    if source is None:
        if allow_legacy:
            return "accept", "legacy"
        return "reject", "missing_label_source"
    if source in _VERIFIED_SOURCES:
        return "accept", "manual"
    if source == "prelabel":
        if use_unverified:
            return "accept", "prelabel"
        return "reject", "prelabel"
    return "reject", "unknown_source"


def _read_label_source(data):
    if "label_source" not in data.files:
        return None
    value = np.asarray(data["label_source"]).reshape(-1)
    if value.size == 0:
        return None
    item = value[0]
    if isinstance(item, bytes):
        text = item.decode("utf-8")
    else:
        text = str(item)
    text = text.strip()
    return text or None


def _drop_keypoints(keypoints: np.ndarray, probability: float) -> np.ndarray:
    if probability <= 0:
        return keypoints
    output = keypoints.copy()
    drop = np.random.random(output.shape[:2]) < probability
    if output.shape[-1] == 3:
        output[drop, 2] = 0.0
    output[drop, :2] = np.nan
    for frame_id in range(output.shape[0]):
        if np.isnan(output[frame_id, :, 0]).all():
            restore = int(np.random.randint(0, output.shape[1]))
            output[frame_id, restore] = keypoints[frame_id, restore]
    return output

