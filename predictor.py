import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.feature_extractor import FeatureExtractor
from models.multistream_tcn import build_model
from utils.config import MODEL_NAME, select_device, validate_config


def load_checkpoint(model_path: str, map_location="cpu") -> dict:
    try:
        checkpoint = torch.load(model_path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=map_location)
    if checkpoint.get("model_name") != MODEL_NAME:
        raise RuntimeError("权重与当前多流行为网络不匹配")
    if "model_state_dict" not in checkpoint or "config" not in checkpoint:
        raise RuntimeError("权重文件缺少 model_state_dict 或 config")
    return checkpoint


class BehaviorPredictor:
    def __init__(self, model_path: str, device: str = None):
        checkpoint = load_checkpoint(model_path)
        self.config = checkpoint["config"]
        validate_config(self.config)
        self.behavior_names = list(self.config["behavior_names"])
        self.window_size = int(self.config["window_size"])
        self.num_keypoints = int(self.config["num_keypoints"])
        self.extractor = FeatureExtractor(self.config)
        self._check_feature_dims(checkpoint.get("feature_dims"))
        resolved_device = device or self.config.get("online", {}).get("device", "auto")
        self.device = select_device(resolved_device)
        self.model = build_model(self.config, self.extractor.motion_dim, self.extractor.geometry_dim)
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except RuntimeError as exc:
            detail = str(exc)
            hint = ""
            if "fbpem." in detail or "stream_fusion" in detail:
                hint = "FBPEM 部位编码和三流卷积融合层已变更，旧权重不能继续使用，需要重新训练。"
            raise RuntimeError(f"权重参数与当前网络不匹配，已拒绝加载。{hint}{detail}") from exc
        self.model.to(self.device)
        self.model.eval()

    def _check_feature_dims(self, saved_dims) -> None:
        if not saved_dims:
            return
        if list(saved_dims["coord"]) != [self.num_keypoints, 2]:
            raise RuntimeError("坐标流维度与权重不一致")
        if int(saved_dims["motion"]) != self.extractor.motion_dim or int(saved_dims["geometry"]) != self.extractor.geometry_dim:
            raise RuntimeError("运动流或几何流维度与权重不一致")

    @torch.no_grad()
    def predict(self, keypoints_sequence) -> dict:
        sequence = np.asarray(keypoints_sequence, dtype=np.float32)
        if sequence.ndim != 3 or sequence.shape[1] != self.num_keypoints or sequence.shape[2] not in (2, 3):
            raise ValueError(
                f"输入形状应为 [{self.window_size}, {self.num_keypoints}, 2]，当前为 {tuple(sequence.shape)}"
            )
        if sequence.shape[0] != self.window_size:
            raise ValueError(
                f"单窗口推理需要连续 {self.window_size} 帧，当前为 {sequence.shape[0]} 帧。"
                "帧数不足时不要补齐，请改用 OnlineBehaviorAnalyzer。"
            )
        features = self.extractor.extract(sequence)
        coord = torch.from_numpy(features["coord"]).unsqueeze(0).to(self.device)
        motion = torch.from_numpy(features["motion"]).unsqueeze(0).to(self.device)
        geometry = torch.from_numpy(features["geometry"]).unsqueeze(0).to(self.device)
        confidence = torch.from_numpy(features["confidence"]).unsqueeze(0).to(self.device)
        outputs = self.model(coord, motion, geometry, confidence)
        probabilities = torch.softmax(outputs["logits"], dim=-1)[0]
        behavior_id = int(torch.argmax(probabilities).item())
        return {
            "behavior_id": behavior_id,
            "behavior_name": self.behavior_names[behavior_id],
            "confidence": float(probabilities[behavior_id].item()),
            "probabilities": [float(value) for value in probabilities.detach().cpu().tolist()],
            "kinematics": features["kinematics"],
        }

