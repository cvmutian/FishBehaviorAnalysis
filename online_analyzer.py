import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predictor import BehaviorPredictor
from data.association import densify_track, suppress_jumps, window_is_reliable
from utils.statistics import summarize_school_status

INSUFFICIENT = "数据不足"


class OnlineBehaviorAnalyzer:
    def __init__(self, model_path: str, device: str = None):
        self.predictor = BehaviorPredictor(model_path, device=device)
        self.config = self.predictor.config
        self.window_size = int(self.config["window_size"])
        self.stride = int(self.config["stride"])
        self.num_keypoints = int(self.config["num_keypoints"])
        self.behavior_names = list(self.config["behavior_names"])
        self.fps = float(self.config["fps"])
        online_cfg = self.config.get("online", {})
        self.max_idle_frames = int(online_cfg.get("max_idle_frames", self.window_size))
        self.max_idle_seconds = float(online_cfg.get("max_idle_seconds", self.window_size / self.fps))
        self.buffers = {}
        self.indices = {}
        self.segment_counts = {}
        self.latest = {}
        self.last_seen_frame = {}
        self.last_seen_time = {}

    def update(self, track_id, keypoints, frame_idx=None) -> dict:
        track_id = _normalize_track_id(track_id)
        self.cleanup(frame_idx)
        if keypoints is None:
            return self._handle_missing(track_id, frame_idx)
        frame = self._with_confidence(self._normalize_frame(keypoints))
        index = self._resolve_index(track_id, frame_idx)
        if index is None:
            timeline = int(frame_idx) if frame_idx is not None else self.last_seen_frame.get(track_id)
            return self._current(track_id, timeline)
        self._append_frame(track_id, index, frame)
        self.last_seen_time[track_id] = time.monotonic()
        count = self.segment_counts[track_id]
        if count < self.window_size or not self._window_span_ok(track_id):
            return self._insufficient(track_id, index)
        need_update = (count - self.window_size) % self.stride == 0
        if need_update or track_id not in self.latest:
            sequence = np.stack(list(self.buffers[track_id]), axis=0)
            if not window_is_reliable(sequence, self.config):
                return self._insufficient(track_id, index)
            sequence = suppress_jumps(sequence, self.config)
            self.latest[track_id] = self.predictor.predict(sequence)
            self.latest[track_id]["update_frame"] = index
            self.latest[track_id]["update_time_s"] = index / self.fps
            updated = True
        else:
            updated = False
        return self._pack(track_id, self.latest[track_id], updated)

    def cleanup(self, frame_idx=None) -> None:
        if frame_idx is None:
            now = time.monotonic()
            stale = [
                track_id
                for track_id, seen_at in self.last_seen_time.items()
                if now - seen_at > self.max_idle_seconds
            ]
        else:
            frame_idx = int(frame_idx)
            stale = [
                track_id
                for track_id, seen_frame in self.last_seen_frame.items()
                if frame_idx - seen_frame > self.max_idle_frames
            ]
        for track_id in stale:
            self._drop(track_id)

    def summarize(self, results) -> dict:
        summary = summarize_school_status(results, self.behavior_names)
        if summary["num_valid"] == 0:
            summary["activity"]["dominant_behavior"] = None
        return summary

    def _normalize_frame(self, keypoints) -> np.ndarray:
        array = np.asarray(keypoints, dtype=np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2 or array.shape[0] != self.num_keypoints or array.shape[1] not in (2, 3):
            raise ValueError(
                f"单帧关键点形状应为 [{self.num_keypoints}, 2] 或 [{self.num_keypoints}, 3]，当前为 {tuple(array.shape)}"
            )
        return array.copy()

    def _with_confidence(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] == 3:
            return frame
        output = np.ones((self.num_keypoints, 3), dtype=np.float32)
        output[:, :2] = frame[:, :2]
        return output

    def _resolve_index(self, track_id, frame_idx):
        if frame_idx is None:
            if track_id not in self.last_seen_frame:
                return 0
            return int(self.last_seen_frame[track_id]) + 1
        index = int(frame_idx)
        if track_id not in self.last_seen_frame:
            return index
        if index <= int(self.last_seen_frame[track_id]):
            return None
        return index

    def _append_frame(self, track_id, index: int, frame: np.ndarray) -> None:
        max_gap = int(self.config["association"]["max_consecutive_missing"])
        if track_id not in self.buffers:
            self._start_segment(track_id, index, frame)
            return
        gap = int(index) - int(self.last_seen_frame[track_id]) - 1
        if gap > max_gap:
            self._drop(track_id)
            self._start_segment(track_id, index, frame)
            return
        if gap > 0:
            filled = self._interpolate(self.buffers[track_id][-1], frame, gap)
            if filled is None:
                self._drop(track_id)
                self._start_segment(track_id, index, frame)
                return
            last = int(self.last_seen_frame[track_id])
            for step, item in enumerate(filled, start=1):
                self._push(track_id, last + step, item)
        self._push(track_id, index, frame)

    def _start_segment(self, track_id, index: int, frame: np.ndarray) -> None:
        self.buffers[track_id] = deque(maxlen=self.window_size)
        self.indices[track_id] = deque(maxlen=self.window_size)
        self.segment_counts[track_id] = 0
        self._push(track_id, index, frame)

    def _push(self, track_id, index: int, frame: np.ndarray) -> None:
        self.buffers[track_id].append(np.asarray(frame, dtype=np.float32))
        self.indices[track_id].append(int(index))
        self.segment_counts[track_id] += 1
        self.last_seen_frame[track_id] = int(index)

    def _interpolate(self, previous, current, gap: int):
        dense = densify_track(
            [(0, previous), (gap + 1, current)],
            self.num_keypoints,
            int(self.config["association"]["max_consecutive_missing"]),
        )
        if len(dense) != 1:
            return None
        sequence = dense[0][1]
        if sequence.shape[0] != gap + 2:
            return None
        filled = []
        for item in sequence[1:-1]:
            frame = np.array(item, dtype=np.float32, copy=True)
            frame[:, 2] = 0.0
            filled.append(frame)
        return filled

    def _window_span_ok(self, track_id) -> bool:
        indices = list(self.indices[track_id])
        if len(indices) != self.window_size:
            return False
        span = int(indices[-1]) - int(indices[0])
        if span != self.window_size - 1:
            return False
        if span / self.fps > self.max_idle_seconds:
            return False
        return True

    def _handle_missing(self, track_id, frame_idx) -> dict:
        if track_id not in self.buffers:
            return self._insufficient(track_id, frame_idx)
        if frame_idx is not None and track_id in self.last_seen_frame:
            gap = int(frame_idx) - int(self.last_seen_frame[track_id])
            if gap > int(self.config["association"]["max_consecutive_missing"]):
                self._drop(track_id)
                return self._insufficient(track_id, frame_idx)
        return self._current(track_id, frame_idx)

    def _current(self, track_id, timeline) -> dict:
        if track_id in self.latest:
            return self._pack(track_id, self.latest[track_id], False)
        return self._insufficient(track_id, timeline)

    def _insufficient(self, track_id, timeline) -> dict:
        return {
            "track_id": track_id,
            "behavior": INSUFFICIENT,
            "behavior_id": -1,
            "confidence": 0.0,
            "probabilities": [],
            "updated": False,
            "update_frame": None if timeline is None else int(timeline),
            "update_time_s": None,
            "kinematics": None,
        }

    def _pack(self, track_id, prediction: dict, updated: bool) -> dict:
        return {
            "track_id": track_id,
            "behavior": prediction["behavior_name"],
            "behavior_id": int(prediction["behavior_id"]),
            "confidence": float(prediction["confidence"]),
            "probabilities": list(prediction["probabilities"]),
            "updated": updated,
            "update_frame": prediction.get("update_frame"),
            "update_time_s": prediction.get("update_time_s"),
            "kinematics": prediction.get("kinematics"),
        }

    def _drop(self, track_id) -> None:
        self.buffers.pop(track_id, None)
        self.indices.pop(track_id, None)
        self.segment_counts.pop(track_id, None)
        self.latest.pop(track_id, None)
        self.last_seen_frame.pop(track_id, None)
        self.last_seen_time.pop(track_id, None)


def _normalize_track_id(track_id):
    if isinstance(track_id, (np.integer,)):
        return int(track_id)
    return track_id
