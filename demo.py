import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.feature_extractor import FeatureExtractor
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="检查行为分析接口")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()
    config = load_config(args.config)
    sequence = make_synthetic_sequence(config)
    features = FeatureExtractor(config).extract(sequence)
    print(f"coord {tuple(features['coord'].shape)}")
    print(f"motion {tuple(features['motion'].shape)}")
    print(f"geometry {tuple(features['geometry'].shape)}")
    if not args.checkpoint:
        return
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"找不到模型权重: {checkpoint}")
    from predictor import BehaviorPredictor

    result = BehaviorPredictor(str(checkpoint)).predict(sequence)
    print(
        f"behavior_id={result['behavior_id']} "
        f"behavior_name={result['behavior_name']} "
        f"confidence={result['confidence']:.4f}"
    )


def make_synthetic_sequence(config: dict) -> np.ndarray:
    window_size = int(config["window_size"])
    num_keypoints = int(config["num_keypoints"])
    sequence = np.zeros((window_size, num_keypoints, 3), dtype=np.float32)
    for frame_id in range(window_size):
        origin_x = 40.0 + frame_id * 1.5
        origin_y = 80.0 + 2.0 * np.sin(frame_id / window_size * np.pi)
        bend = 0.15 * np.sin(frame_id / 8.0)
        sequence[frame_id, :, :2] = _fish_shape(origin_x, origin_y, bend)
        sequence[frame_id, :, 2] = 0.95
    return sequence


def _fish_shape(origin_x: float, origin_y: float, bend: float) -> np.ndarray:
    local = np.array(
        [
            [28, 0],
            [20, 2],
            [14, 6],
            [4, 7],
            [-8, 5],
            [-16, 7],
            [-22, 0],
            [-16, -7],
            [-8, -5],
            [2, -6],
            [14, -6],
        ],
        dtype=np.float32,
    )
    local[4:, 1] += bend * 10.0
    local[:, 0] += origin_x
    local[:, 1] += origin_y
    return local


if __name__ == "__main__":
    main()
