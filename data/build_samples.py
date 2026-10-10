import csv
import json
from pathlib import Path

import numpy as np

from data.association import (
    associate_detections,
    densify_track,
    keypoints_to_box,
    suppress_jumps,
    window_is_reliable,
)
from data.labeling import resolve_label, window_indicators
from features.feature_extractor import FeatureExtractor
from utils.config import load_config, resolve_path


def assign_video_splits(video_ids, train_ratio: float, val_ratio: float, seed: int) -> dict:
    ids = sorted(set(video_ids))
    if not ids:
        return {}
    order = np.array(ids)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(order)
    count = len(order)
    if count == 1:
        return {str(order[0]): "train"}
    train_count = min(count - 1, max(1, int(round(count * float(train_ratio)))))
    val_count = max(0, int(round(count * float(val_ratio))))
    if train_count + val_count > count - 1:
        val_count = count - 1 - train_count
    mapping = {}
    for index, video_id in enumerate(order):
        if index < train_count:
            split = "train"
        elif index < train_count + val_count:
            split = "val"
        else:
            split = "test"
        mapping[str(video_id)] = split
    return mapping


def load_split_file(path: Path) -> dict:
    mapping = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = row["split"].strip()
            if split not in {"train", "val", "test"}:
                raise ValueError(f"未知数据划分: {split}")
            mapping[row["video_id"].strip()] = split
    return mapping


def load_manual_labels(path: Path) -> dict:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["labels"] if isinstance(payload, dict) else payload
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    labels = {}
    for row in rows:
        start = int(row["start_frame"])
        key = (str(row["video_id"]).strip(), int(row["track_id"]), None if start < 0 else start)
        labels[key] = int(row["label"])
    return labels


def manual_label_for(labels: dict, video_id: str, track_id: int, start_frame: int):
    if not labels:
        return None
    if (video_id, track_id, start_frame) in labels:
        return labels[(video_id, track_id, start_frame)]
    return labels.get((video_id, track_id, None))


def iter_track_windows(sequence: np.ndarray, window_size: int, stride: int):
    length = int(sequence.shape[0])
    if length < window_size:
        return
    for start in range(0, length - window_size + 1, stride):
        yield start, sequence[start:start + window_size]


def build_dataset(input_dir: Path, config: dict, output_root: Path, split_file: Path = None, label_file: Path = None) -> dict:
    videos = sorted(Path(input_dir).glob("*.npz"))
    if not videos:
        raise FileNotFoundError(f"没有找到视频序列: {input_dir}")
    if split_file is not None:
        splits = load_split_file(split_file)
    else:
        splits = assign_video_splits(
            [path.stem for path in videos],
            config["split"]["train_ratio"],
            config["split"]["val_ratio"],
            config["split"]["seed"],
        )
    manual = load_manual_labels(label_file) if label_file is not None else {}
    extractor = FeatureExtractor(config)
    counts = {"train": 0, "val": 0, "test": 0}
    for path in videos:
        video_id = path.stem
        if video_id not in splits:
            raise KeyError(f"视频 {video_id} 没有数据划分")
        split = splits[video_id]
        output_dir = output_root / split
        output_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples_from_video(path, config, extractor, manual):
            filename = f"{video_id}_track{sample['track_id']}_{sample['start_frame']:06d}.npz"
            np.savez_compressed(output_dir / filename, **sample)
            counts[split] += 1
    manifest = output_root / "splits.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "split"])
        writer.writeheader()
        for video_id, split in sorted(splits.items()):
            writer.writerow({"video_id": video_id, "split": split})
    return counts


def samples_from_video(path: Path, config: dict, extractor: FeatureExtractor, manual: dict):
    video = np.load(path)
    keypoints = np.asarray(video["keypoints"], dtype=np.float32)
    if keypoints.ndim != 4 or keypoints.shape[2] != int(config["num_keypoints"]) or keypoints.shape[3] not in (2, 3):
        raise ValueError(f"{path.name} 的 keypoints 形状应为 [F, M, 11, 2或3]")
    valid = np.asarray(video["valid"], dtype=bool) if "valid" in video.files else np.ones(keypoints.shape[:2], dtype=bool)
    frame_idx = np.asarray(video["frame_idx"], dtype=np.int64) if "frame_idx" in video.files else np.arange(keypoints.shape[0])
    frames = []
    for index, frame_number in enumerate(frame_idx):
        detections = []
        for det_id in range(keypoints.shape[1]):
            if not valid[index, det_id]:
                continue
            pose = keypoints[index, det_id]
            detections.append({"keypoints": pose, "box": keypoints_to_box(pose)})
        frames.append((int(frame_number), detections))
    window_size = int(config["window_size"])
    stride = int(config["stride"])
    video_id = path.stem
    for track in associate_detections(frames, config):
        for segment_start, sequence in densify_track(
            track["frames"],
            int(config["num_keypoints"]),
            int(config["association"]["max_consecutive_missing"]),
        ):
            for offset, window in iter_track_windows(sequence, window_size, stride):
                if not window_is_reliable(window, config):
                    continue
                window = suppress_jumps(window, config)
                start_frame = segment_start + offset
                features = extractor.extract(window)
                label, source = resolve_label(
                    window_indicators(features["motion"], features["geometry"]),
                    config["labeling"],
                    manual_label_for(manual, video_id, int(track["track_id"]), start_frame),
                )
                yield {
                    "keypoints": window.astype(np.float32),
                    "label": np.int64(label),
                    "track_id": np.int64(track["track_id"]),
                    "video_id": np.array(video_id),
                    "start_frame": np.int64(start_frame),
                    "label_source": np.array(source),
                }


def main():
    import argparse
    import sys

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    parser = argparse.ArgumentParser(description="由姿态序列构建单鱼行为样本")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="dataset")
    parser.add_argument("--split-file", default="")
    parser.add_argument("--labels", default="")
    args = parser.parse_args()
    config = load_config(args.config)
    counts = build_dataset(
        Path(args.input),
        config,
        resolve_path(root, args.output),
        Path(args.split_file) if args.split_file else None,
        Path(args.labels) if args.labels else None,
    )
    print(f"train={counts['train']} val={counts['val']} test={counts['test']}")
    if not args.labels:
        print("未提供人工核验标签。预标注只作为候选，不能直接作为正式训练标签。")


if __name__ == "__main__":
    main()
