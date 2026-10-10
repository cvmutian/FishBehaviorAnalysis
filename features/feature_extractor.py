import numpy as np

from data.preprocessing import KeypointPreprocessor


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def unsigned_angle(vector_a: np.ndarray, vector_b: np.ndarray, eps: float) -> np.ndarray:
    norm_a = np.linalg.norm(vector_a, axis=-1)
    norm_b = np.linalg.norm(vector_b, axis=-1)
    denom = np.maximum(norm_a * norm_b, eps)
    cosine = np.clip(np.sum(vector_a * vector_b, axis=-1) / denom, -1.0, 1.0)
    angle = np.arccos(cosine)
    angle = np.where((norm_a < eps) | (norm_b < eps), 0.0, angle)
    return angle.astype(np.float32)


def clip_statistics(motion: np.ndarray, geometry: np.ndarray) -> dict:
    return {
        "speed": _mean_from(motion[:, 0], 1),
        "accel": _mean_from(motion[:, 1], 2),
        "yaw": _mean_from(np.abs(motion[:, 2]), 1),
        "bend": _mean_from(geometry[:, 0], 0),
        "tail_swing": _mean_from(geometry[:, 1], 1),
    }


def _mean_from(values: np.ndarray, start: int) -> float:
    count = int(values.shape[0])
    if count <= start:
        return 0.0
    return float(np.sum(values[start:]) / (count - start))


def signed_angle(vector_a: np.ndarray, vector_b: np.ndarray, eps: float) -> np.ndarray:
    cross = vector_a[:, 0] * vector_b[:, 1] - vector_a[:, 1] * vector_b[:, 0]
    dot = np.sum(vector_a * vector_b, axis=-1)
    angle = np.arctan2(cross, dot)
    norm_a = np.linalg.norm(vector_a, axis=-1)
    norm_b = np.linalg.norm(vector_b, axis=-1)
    angle = np.where((norm_a < eps) | (norm_b < eps), 0.0, angle)
    return angle.astype(np.float32)


class FeatureExtractor:
    def __init__(self, config: dict):
        self.config = config
        self.num_keypoints = int(config["num_keypoints"])
        self.fps = float(config["fps"])
        self.dt = 1.0 / self.fps
        self.preprocessor = KeypointPreprocessor(config)
        self.eps = float(config["preprocess"]["eps"])
        keypoints_cfg = config["keypoints"]
        self.mouth_index = int(keypoints_cfg["mouth_index"])
        self.tail_index = int(keypoints_cfg["tail_index"])
        self.gill_indices = [int(index) for index in keypoints_cfg["gill_indices"]]
        self.peduncle_indices = [int(index) for index in keypoints_cfg["peduncle_indices"]]
        self.part_indices = {
            name: [int(index) for index in indices]
            for name, indices in keypoints_cfg["parts"].items()
        }
        self.aux_norm = {key: float(value) for key, value in config["aux_norm"].items()}
        self.motion_dim = 3
        self.geometry_dim = 2
        self._check_indices()

    def _check_indices(self) -> None:
        indices = [self.mouth_index, self.tail_index, *self.gill_indices, *self.peduncle_indices]
        for part in self.part_indices.values():
            indices.extend(part)
        for index in indices:
            if index < 0 or index >= self.num_keypoints:
                raise ValueError(f"关键点编号 {index} 超出范围 0–{self.num_keypoints - 1}")

    def extract(self, keypoints: np.ndarray, dt: float = None) -> dict:
        prepared = self.preprocessor.prepare(keypoints)
        step = self.dt if dt is None else float(dt)
        motion = self._motion_features(prepared, step)
        geometry = self._geometry_features(prepared, step)
        features = {
            "coord": np.nan_to_num(prepared["pose"], nan=0.0, posinf=0.0, neginf=0.0),
            "motion": np.nan_to_num(motion, nan=0.0, posinf=0.0, neginf=0.0),
            "geometry": np.nan_to_num(geometry, nan=0.0, posinf=0.0, neginf=0.0),
            "confidence": np.nan_to_num(prepared["confidence"], nan=0.0, posinf=0.0, neginf=0.0),
        }
        features["aux"] = self.clip_targets(features["motion"], features["geometry"])
        features["kinematics"] = self.kinematics_summary(features["motion"], features["geometry"])
        self._check_feature_shapes(features)
        return features

    def _motion_features(self, prepared: dict, dt: float) -> np.ndarray:
        center = prepared["center"].astype(np.float64)
        body_length = np.maximum(prepared["body_length"].astype(np.float64), self.eps)
        axis_angle = prepared["axis_angle"].astype(np.float64)
        time_steps = center.shape[0]
        step = max(float(dt), self.eps)
        velocity = np.zeros((time_steps, 2), dtype=np.float64)
        if time_steps > 1:
            length_avg = np.maximum((body_length[1:] + body_length[:-1]) * 0.5, self.eps)
            velocity[1:] = (center[1:] - center[:-1]) / length_avg[:, None] / step
        acceleration = np.zeros((time_steps, 2), dtype=np.float64)
        if time_steps > 2:
            acceleration[2:] = (velocity[2:] - velocity[1:-1]) / step
        yaw_rate = np.zeros(time_steps, dtype=np.float64)
        if time_steps > 1:
            yaw_rate[1:] = wrap_angle(axis_angle[1:] - axis_angle[:-1]) / step
        motion = np.stack(
            [
                np.linalg.norm(velocity, axis=-1),
                np.linalg.norm(acceleration, axis=-1),
                yaw_rate,
            ],
            axis=1,
        )
        return motion.astype(np.float32)

    def _geometry_features(self, prepared: dict, dt: float) -> np.ndarray:
        xy = prepared["xy"].astype(np.float64)
        mouth = xy[:, self.mouth_index]
        tail = xy[:, self.tail_index]
        gill = xy[:, self.gill_indices].mean(axis=1)
        peduncle = xy[:, self.peduncle_indices].mean(axis=1)
        axis = mouth - tail
        bend = unsigned_angle(gill - mouth, tail - peduncle, self.eps).astype(np.float64)
        tail_rel = signed_angle(axis, tail - peduncle, self.eps).astype(np.float64)
        tail_swing = np.zeros(xy.shape[0], dtype=np.float64)
        step = max(float(dt), self.eps)
        if xy.shape[0] > 1:
            tail_swing[1:] = np.abs(wrap_angle(tail_rel[1:] - tail_rel[:-1])) / step
        geometry = np.stack([bend, tail_swing], axis=1)
        return geometry.astype(np.float32)

    def clip_targets(self, motion: np.ndarray, geometry: np.ndarray) -> np.ndarray:
        summary = self.kinematics_summary(motion, geometry)
        return np.array(
            [
                summary["speed"] / self.aux_norm["speed"],
                summary["accel"] / self.aux_norm["accel"],
                summary["yaw"] / self.aux_norm["yaw"],
                summary["bend"] / self.aux_norm["bend"],
                summary["tail_swing"] / self.aux_norm["tail_swing"],
            ],
            dtype=np.float32,
        )

    def kinematics_summary(self, motion: np.ndarray, geometry: np.ndarray) -> dict:
        return clip_statistics(motion, geometry)

    def _check_feature_shapes(self, features: dict) -> None:
        time_steps = features["coord"].shape[0]
        expected = {
            "coord": (time_steps, self.num_keypoints, 2),
            "motion": (time_steps, self.motion_dim),
            "geometry": (time_steps, self.geometry_dim),
            "confidence": (time_steps, self.num_keypoints),
            "aux": (5,),
        }
        for name, shape in expected.items():
            if features[name].shape != shape:
                raise RuntimeError(f"{name} 特征形状应为 {shape}，实际为 {features[name].shape}")
