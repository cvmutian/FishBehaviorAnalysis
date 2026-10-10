import numpy as np


def _interpolate_series(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    filled = values.copy()
    time_steps = filled.shape[0]
    timeline = np.arange(time_steps)
    observed = np.where(valid)[0]
    if observed.size == 0:
        return filled
    for coord_id in range(filled.shape[-1]):
        filled[:, coord_id] = np.interp(timeline, observed, values[observed, coord_id])
    return filled


def sanitize_keypoints(keypoints: np.ndarray, num_keypoints: int, conf_threshold: float) -> dict:
    array = np.asarray(keypoints, dtype=np.float64)
    if array.ndim != 3 or array.shape[1] != num_keypoints or array.shape[2] not in (2, 3):
        raise ValueError(
            f"关键点序列形状应为 [T, {num_keypoints}, 2] 或 [T, {num_keypoints}, 3]，当前为 {tuple(array.shape)}"
        )

    xy = array[:, :, :2].copy()
    finite_xy = np.isfinite(xy).all(axis=-1)
    if array.shape[2] == 3:
        confidence = np.nan_to_num(array[:, :, 2], nan=0.0, posinf=0.0, neginf=0.0)
        valid = finite_xy & np.isfinite(array[:, :, 2]) & (confidence >= conf_threshold)
    else:
        confidence = np.ones(xy.shape[:2], dtype=np.float64)
        valid = finite_xy
        confidence[~valid] = 0.0

    xy[~valid] = np.nan
    for keypoint_id in range(num_keypoints):
        xy[:, keypoint_id] = _interpolate_series(xy[:, keypoint_id], valid[:, keypoint_id])

    observed_confidence = np.where(valid, confidence, 0.0)
    for frame_id in range(xy.shape[0]):
        known = np.isfinite(xy[frame_id]).all(axis=-1)
        if not np.any(known):
            xy[frame_id] = 0.0
            continue
        center = xy[frame_id, known].mean(axis=0)
        missing = ~known
        xy[frame_id, missing] = center

    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "xy": xy.astype(np.float32),
        "confidence": observed_confidence.astype(np.float32),
        "valid": valid,
    }


class KeypointPreprocessor:
    def __init__(self, config: dict):
        self.num_keypoints = int(config["num_keypoints"])
        self.conf_threshold = float(config["preprocess"]["conf_threshold"])
        self.eps = float(config["preprocess"]["eps"])
        keypoints_cfg = config["keypoints"]
        self.mouth_index = int(keypoints_cfg["mouth_index"])
        self.tail_index = int(keypoints_cfg["tail_index"])
        self.fps = float(config["fps"])

    def prepare(self, keypoints: np.ndarray) -> dict:
        cleaned = sanitize_keypoints(keypoints, self.num_keypoints, self.conf_threshold)
        xy = cleaned["xy"]
        valid = cleaned["valid"]
        center = np.zeros((xy.shape[0], 2), dtype=np.float64)
        for frame_id in range(xy.shape[0]):
            known = valid[frame_id]
            if np.any(known):
                center[frame_id] = xy[frame_id, known].mean(axis=0)
            else:
                center[frame_id] = xy[frame_id].mean(axis=0)

        mouth = xy[:, self.mouth_index]
        tail = xy[:, self.tail_index]
        body_axis = mouth - tail
        body_length = np.linalg.norm(body_axis, axis=-1)
        axis_angle = np.arctan2(body_axis[:, 1], body_axis[:, 0])

        scale = np.maximum(body_length, self.eps)
        translated = xy - center[:, None, :]
        scaled = translated / scale[:, None, None]
        pose = _rotate_points(scaled, -axis_angle)
        return {
            "pose": pose.astype(np.float32),
            "confidence": cleaned["confidence"],
            "center": center.astype(np.float32),
            "body_length": body_length.astype(np.float32),
            "axis_angle": axis_angle.astype(np.float32),
            "xy": xy.astype(np.float32),
        }


def _rotate_points(points: np.ndarray, angles: np.ndarray) -> np.ndarray:
    cos = np.cos(angles).astype(np.float64)
    sin = np.sin(angles).astype(np.float64)
    x = points[:, :, 0]
    y = points[:, :, 1]
    rotated = np.empty_like(points, dtype=np.float64)
    rotated[:, :, 0] = cos[:, None] * x - sin[:, None] * y
    rotated[:, :, 1] = sin[:, None] * x + cos[:, None] * y
    return rotated
