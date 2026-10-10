import copy
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.association import (
    associate_detections,
    association_cost,
    densify_track,
    jump_frame_ratio,
    linear_sum_assignment,
    pose_similarity,
    suppress_jumps,
    window_is_reliable,
)
from data.dataset import BehaviorDataset, _coordinate_noise, _drop_keypoints, _temporal_crop
from features.feature_extractor import clip_statistics
from data.build_samples import assign_video_splits, build_dataset
from data.labeling import prelabel, resolve_label
from demo import make_synthetic_sequence
from features.feature_extractor import FeatureExtractor
from models.multistream_tcn import _quality, build_model
from online_analyzer import INSUFFICIENT, OnlineBehaviorAnalyzer
from predictor import BehaviorPredictor
from train import epoch_learning_rate, run_train
from utils.config import MODEL_NAME, load_config
from utils.visualization import skeleton_edges


class ConfigAndFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(str(ROOT / "config" / "config.yaml"))
        cls.extractor = FeatureExtractor(cls.config)

    def test_paper_settings(self):
        config = self.config
        self.assertEqual(config["fps"], 30)
        self.assertEqual(config["window_size"], 96)
        self.assertEqual(config["stride"], 48)
        self.assertEqual(config["behavior_names"], ["低活动", "正常游动", "快速游动", "急转避让"])
        self.assertEqual(config["train"]["batch_size"], 32)
        self.assertEqual(config["train"]["epochs"], 120)
        self.assertEqual(config["train"]["learning_rate"], 0.001)
        self.assertEqual(config["train"]["warmup_epochs"], 5)
        self.assertEqual(config["train"]["aux_loss_weight"], 0.2)
        self.assertEqual(config["model"]["dropout"], 0.3)
        self.assertEqual(config["model"]["coord_embed"], 64)
        self.assertEqual(config["model"]["motion_embed"], 32)
        self.assertEqual(config["model"]["geometry_embed"], 32)
        self.assertEqual(config["model"]["fusion_channels"], 128)
        self.assertEqual(config["model"]["classifier_channels"], [128, 64, 4])
        self.assertEqual(config["model"]["regressor_channels"], [128, 64, 5])
        self.assertEqual(config["augment"]["coord_noise_std"], 0.01)
        self.assertEqual(config["augment"]["keypoint_drop_prob"], 0.1)
        self.assertEqual(config["augment"]["time_crop_min"], 0.8)
        self.assertEqual(config["augment"]["time_crop_max"], 1.0)

    def test_keypoint_indices_and_skeleton(self):
        keypoints = self.config["keypoints"]
        self.assertEqual(keypoints["mouth_index"], 0)
        self.assertEqual(keypoints["tail_index"], 6)
        self.assertEqual(keypoints["gill_indices"], [2, 10])
        self.assertEqual(keypoints["peduncle_indices"], [4, 8])
        edges = skeleton_edges(self.config)
        self.assertIn((0, 1), edges)
        self.assertIn((10, 0), edges)
        self.assertTrue(all(0 <= start < 11 and 0 <= end < 11 for start, end in edges))

    def test_coordinate_normalization_and_units(self):
        length = 30.0
        base = _straight_fish(length)
        sequence = np.repeat(base[None], 8, axis=0)
        features = self.extractor.extract(sequence)
        pose = features["coord"]
        mouth = pose[:, 0]
        tail = pose[:, 6]
        self.assertTrue(np.allclose(pose.mean(axis=1), 0.0, atol=1e-4))
        self.assertTrue(np.allclose(np.linalg.norm(mouth - tail, axis=-1), 1.0, atol=1e-4))
        self.assertTrue(np.all(mouth[:, 0] > tail[:, 0]))
        shifted = sequence.copy()
        shifted[:, :, 0] += 500
        shifted[:, :, 1] += 200
        shifted_pose = self.extractor.extract(shifted)["coord"]
        self.assertTrue(np.allclose(pose, shifted_pose, atol=1e-4))
        moving = sequence.copy()
        for frame_id in range(sequence.shape[0]):
            moving[frame_id, :, 0] += frame_id
        motion = self.extractor.extract(moving)["motion"]
        self.assertTrue(np.allclose(motion[1:, 0], 1.0, atol=1e-3))
        self.assertTrue(np.allclose(motion[2:, 1], 0.0, atol=1e-3))
        self.assertTrue(np.allclose(motion[:, 2], 0.0, atol=1e-4))

    def test_motion_uses_heading_and_tail_swing_does_not(self):
        length = 30.0
        base = _straight_fish(length)
        center = base[:, :2].mean(axis=0)
        angles = np.arange(10, dtype=np.float64) * 0.1
        rotated = np.zeros((len(angles), 11, 3), dtype=np.float32)
        for frame_id, angle in enumerate(angles):
            xy = base[:, :2] - center
            cos = np.cos(angle)
            sin = np.sin(angle)
            turned = np.stack([cos * xy[:, 0] - sin * xy[:, 1], sin * xy[:, 0] + cos * xy[:, 1]], axis=1)
            rotated[frame_id, :, :2] = turned + center
            rotated[frame_id, :, 2] = 1.0
        features = self.extractor.extract(rotated)
        self.assertTrue(np.allclose(features["motion"][1:, 2], 0.1 * 30.0, atol=1e-3))
        self.assertEqual(features["geometry"].shape[1], 2)
        self.assertTrue(np.allclose(features["geometry"][:, 1], 0.0, atol=1e-3))
        self.assertTrue(np.allclose(features["coord"][0], features["coord"][-1], atol=1e-4))

        tail_only = np.repeat(base[None], 8, axis=0)
        for frame_id in range(8):
            tail_only[frame_id, 4, 1] += frame_id * 3.0
            tail_only[frame_id, 8, 1] += frame_id * 3.0
        tail_features = self.extractor.extract(tail_only)
        self.assertTrue(np.allclose(tail_features["motion"][:, 2], 0.0, atol=1e-4))
        self.assertGreater(tail_features["geometry"][-1, 1], 1.0)

    def test_bend_uses_front_and_rear_segments(self):
        pose = np.zeros((2, 11, 3), dtype=np.float32)
        pose[:, :, 2] = 1.0
        pose[:, 0, :2] = [10.0, 0.0]
        pose[:, 2, :2] = [6.0, 1.0]
        pose[:, 10, :2] = [6.0, -1.0]
        pose[:, 4, :2] = [4.0, 1.0]
        pose[:, 8, :2] = [4.0, -1.0]
        pose[:, 6, :2] = [4.0, 4.0]
        bent = self.extractor.extract(pose)
        self.assertEqual(bent["geometry"].shape[1], 2)
        self.assertAlmostEqual(float(bent["geometry"][0, 0]), np.pi / 2, places=3)
        straight = pose.copy()
        straight[:, 6, :2] = [0.0, 0.0]
        flat = self.extractor.extract(straight)
        self.assertLess(float(flat["geometry"][0, 0]), 0.2)
        collapsed = straight.copy()
        collapsed[:, 6, :2] = [4.0, 0.0]
        protected = self.extractor.extract(collapsed)
        self.assertAlmostEqual(float(protected["geometry"][0, 0]), 0.0, places=4)

    def test_length_average_and_clip_denominators(self):
        fish = _straight_fish(30.0)
        pair = np.stack([fish, fish.copy()], axis=0)
        pair[1, :, 0] += 20.0
        tail = pair[1, 6, :2].copy()
        pair[1, :, :2] = tail + (pair[1, :, :2] - tail) * (10.0 / 30.0)
        motion = self.extractor.extract(pair)["motion"]
        center_delta = pair[1, :, :2].mean(axis=0) - pair[0, :, :2].mean(axis=0)
        expected = float(np.linalg.norm(center_delta) / 20.0 / (1.0 / 30.0))
        current_only = float(np.linalg.norm(center_delta) / 10.0 / (1.0 / 30.0))
        self.assertAlmostEqual(float(motion[1, 0]), expected, places=2)
        self.assertNotAlmostEqual(float(motion[1, 0]), current_only, places=1)
        values = np.array([9.0, 1.0, 2.0, 3.0], dtype=np.float32)
        motion_stats = np.stack([values, values, values], axis=1)
        geometry_stats = np.stack([values, values], axis=1)
        stats = clip_statistics(motion_stats, geometry_stats)
        self.assertAlmostEqual(stats["speed"], (1.0 + 2.0 + 3.0) / 3.0)
        self.assertAlmostEqual(stats["accel"], (2.0 + 3.0) / 2.0)
        self.assertAlmostEqual(stats["yaw"], (1.0 + 2.0 + 3.0) / 3.0)
        self.assertAlmostEqual(stats["bend"], (9.0 + 1.0 + 2.0 + 3.0) / 4.0)
        self.assertAlmostEqual(stats["tail_swing"], (1.0 + 2.0 + 3.0) / 3.0)

    def test_prelabel_review_and_degree_thresholds(self):
        rules = self.config["labeling"]
        self.assertEqual(
            prelabel({"speed": 0.5, "accel": 0.1, "yaw": 0.0, "bend": 0.0, "tail_swing": 0.0}, rules),
            -1,
        )
        yaw = np.deg2rad(100.0)
        self.assertEqual(
            prelabel({"speed": 0.1, "accel": 0.0, "yaw": yaw, "bend": 0.0, "tail_swing": 0.0}, rules),
            -1,
        )

    def test_augmented_pose_feeds_all_streams(self):
        sequence = np.repeat(_straight_fish(30.0)[None], 96, axis=0)
        for frame_id in range(96):
            sequence[frame_id, :, 0] += frame_id
        np.random.seed(0)
        cropped, dt_scale = _temporal_crop(sequence, 0.8, 0.8)
        noisy = _coordinate_noise(cropped, 0.01, 0, 6, 1e-6)
        dropped = _drop_keypoints(noisy, 0.0)
        features = self.extractor.extract(dropped, dt=self.extractor.dt * dt_scale)
        direct = self.extractor.extract(dropped, dt=self.extractor.dt * dt_scale)
        self.assertTrue(np.allclose(features["coord"], direct["coord"]))
        self.assertTrue(np.allclose(features["motion"], direct["motion"]))
        self.assertTrue(np.allclose(features["geometry"], direct["geometry"]))
        clean = self.extractor.extract(sequence)
        self.assertFalse(np.allclose(features["motion"], clean["motion"]))
        self.assertEqual(features["coord"].shape, (96, 11, 2))
        self.assertEqual(features["motion"].shape, (96, 3))
        self.assertEqual(features["geometry"].shape, (96, 2))

    def test_prelabel_does_not_require_all_thresholds(self):
        rules = self.config["labeling"]
        self.assertEqual(prelabel({"speed": 0.1, "accel": 0.0, "yaw": 0.0, "bend": 0.0, "tail_swing": 0.0}, rules), 0)
        self.assertEqual(prelabel({"speed": 3.0, "accel": 0.0, "yaw": 0.0, "bend": 0.0, "tail_swing": 0.0}, rules), -1)
        self.assertEqual(prelabel({"speed": 0.1, "accel": 0.0, "yaw": 3.0, "bend": 0.0, "tail_swing": 0.0}, rules), -1)
        sharp = {
            "speed": 0.4,
            "accel": 0.2,
            "yaw": np.deg2rad(100.0),
            "bend": np.deg2rad(18.0),
            "tail_swing": np.deg2rad(20.0),
        }
        fast = {
            "speed": 1.3,
            "accel": 0.2,
            "yaw": 0.0,
            "bend": np.deg2rad(10.0),
            "tail_swing": np.deg2rad(150.0),
        }
        self.assertEqual(prelabel(sharp, rules), 3)
        self.assertEqual(prelabel(fast, rules), 2)
        self.assertEqual(resolve_label(sharp, rules, 1), (1, "manual"))
        self.assertEqual(resolve_label({"speed": 0.0, "accel": 0.0, "yaw": 0.0, "bend": 0.0, "tail_swing": 0.0}, rules, None)[1], "prelabel")

    def test_learning_rate_schedule(self):
        self.assertAlmostEqual(epoch_learning_rate(1, 120, 0.001, 5, 1e-6), 0.0002)
        self.assertAlmostEqual(epoch_learning_rate(5, 120, 0.001, 5, 1e-6), 0.001)
        self.assertAlmostEqual(epoch_learning_rate(120, 120, 0.001, 5, 1e-6), 1e-6)


class ModelAndTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(str(ROOT / "config" / "config.yaml"))

    def test_forward_shapes(self):
        extractor = FeatureExtractor(self.config)
        model = build_model(self.config, extractor.motion_dim, extractor.geometry_dim)
        model.eval()
        batch = 2
        frames = self.config["window_size"]
        coord = torch.zeros(batch, frames, 11, 2)
        motion = torch.zeros(batch, frames, extractor.motion_dim)
        geometry = torch.zeros(batch, frames, extractor.geometry_dim)
        confidence = torch.ones(batch, frames, 11)
        outputs = model(coord, motion, geometry, confidence)
        self.assertEqual(tuple(outputs["logits"].shape), (batch, 4))
        self.assertEqual(tuple(outputs["aux"].shape), (batch, 5))
        self.assertEqual(len(model.mdtcm.layers), 2)
        self.assertEqual(len(model.mdtcm.layers[0].branches), 4)
        self.assertEqual(set(model.fbpem.part_maps.keys()), {"head", "trunk", "tail"})
        part_channels = int(self.config["model"]["fbpem"]["part_channels"])
        for name in ("head", "trunk", "tail"):
            self.assertEqual(model.fbpem.part_maps[name].in_features, 2)
            self.assertEqual(model.fbpem.part_maps[name].out_features, part_channels)
        self.assertEqual(len({id(model.fbpem.part_maps[name].weight) for name in ("head", "trunk", "tail")}), 3)
        self.assertEqual(model.fbpem.fuse.in_channels, part_channels * 3)
        self.assertEqual(model.fbpem.fuse.out_channels, 64)
        self.assertEqual(model.fbpem.fuse.kernel_size, (1,))
        self.assertEqual(model.stream_fusion.in_channels, 128)
        self.assertEqual(model.stream_fusion.out_channels, 128)
        self.assertEqual(model.stream_fusion.kernel_size, (1,))
        self.assertEqual(model.geometry_map.in_features, 2)
        self.assertEqual(model.geometry_map.out_features, 32)
        self.assertEqual(model.motion_map.out_features, 32)
        self.assertEqual(model.pqtgm.gate.in_channels, int(self.config["model"]["fusion_channels"]) + 1)
        confidence = torch.full((batch, frames, 11), 0.5)
        quality = _quality(coord, confidence)
        self.assertEqual(tuple(quality.shape), (batch, frames, 1))
        self.assertTrue(torch.allclose(quality, torch.full((batch, frames, 1), 0.5)))

    def test_part_encoding_and_stream_fusion(self):
        extractor = FeatureExtractor(self.config)
        model = build_model(self.config, extractor.motion_dim, extractor.geometry_dim)
        model.eval()
        frames = self.config["window_size"]
        parts = self.config["keypoints"]["parts"]
        for batch in (1, 4, 32):
            coord = torch.randn(batch, frames, 11, 2)
            mapped = torch.cat(
                [
                    model.fbpem.part_maps[name](coord[:, :, parts[name], :].mean(dim=2))
                    for name in ("head", "trunk", "tail")
                ],
                dim=-1,
            )
            self.assertEqual(tuple(mapped.shape), (batch, frames, 96))
            encoded = model.fbpem(coord)
            expected = model.fbpem.fuse(mapped.transpose(1, 2)).transpose(1, 2)
            self.assertEqual(tuple(encoded.shape), (batch, frames, 64))
            self.assertTrue(torch.allclose(encoded, expected))

        calls = {"fbpem": 0, "fusion": 0, "pqtgm": 0, "mdtcm": 0}
        captured = {}

        def remember(name):
            def hook(module, inputs, output):
                calls[name] += 1
                captured[name] = {
                    "input": inputs[0].detach().clone(),
                    "output": output.detach().clone(),
                }
            return hook

        model.fbpem.fuse.register_forward_hook(remember("fbpem"))
        model.stream_fusion.register_forward_hook(remember("fusion"))
        model.pqtgm.register_forward_hook(remember("pqtgm"))
        model.mdtcm.register_forward_hook(remember("mdtcm"))
        for batch in (1, 4, 32):
            outputs = model(
                torch.randn(batch, frames, 11, 2),
                torch.randn(batch, frames, extractor.motion_dim),
                torch.randn(batch, frames, extractor.geometry_dim),
                torch.rand(batch, frames, 11),
            )
            self.assertEqual(tuple(outputs["logits"].shape), (batch, 4))
            self.assertEqual(tuple(outputs["aux"].shape), (batch, 5))
        self.assertEqual(calls, {"fbpem": 3, "fusion": 3, "pqtgm": 3, "mdtcm": 3})
        self.assertEqual(tuple(captured["fbpem"]["input"].shape), (32, 96, frames))
        self.assertEqual(tuple(captured["fbpem"]["output"].shape), (32, 64, frames))
        self.assertEqual(tuple(captured["fusion"]["input"].shape), (32, 128, frames))
        self.assertEqual(tuple(captured["fusion"]["output"].shape), (32, 128, frames))
        self.assertEqual(tuple(captured["pqtgm"]["input"].shape), (32, frames, 128))
        self.assertTrue(
            torch.allclose(captured["pqtgm"]["input"], captured["fusion"]["output"].transpose(1, 2))
        )
        self.assertEqual(tuple(captured["mdtcm"]["input"].shape), (32, 128, frames))
        self.assertEqual(tuple(captured["mdtcm"]["output"].shape), (32, 128, frames))

    def test_new_layers_receive_gradients(self):
        extractor = FeatureExtractor(self.config)
        model = build_model(self.config, extractor.motion_dim, extractor.geometry_dim)
        model.train()
        torch.manual_seed(0)
        batch = 4
        frames = self.config["window_size"]
        outputs = model(
            torch.randn(batch, frames, 11, 2),
            torch.randn(batch, frames, extractor.motion_dim),
            torch.randn(batch, frames, extractor.geometry_dim),
            torch.rand(batch, frames, 11),
        )
        (outputs["logits"].sum() + outputs["aux"].sum()).backward()
        watched = [
            "fbpem.part_maps.head.weight",
            "fbpem.part_maps.trunk.weight",
            "fbpem.part_maps.tail.weight",
            "fbpem.fuse.weight",
            "stream_fusion.weight",
        ]
        parameters = dict(model.named_parameters())
        for name in watched:
            grad = parameters[name].grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_previous_checkpoint_cannot_load(self):
        extractor = FeatureExtractor(self.config)
        model = build_model(self.config, extractor.motion_dim, extractor.geometry_dim)
        current = {key: value.detach().clone() for key, value in model.state_dict().items()}
        legacy = {
            key: value
            for key, value in current.items()
            if not key.startswith("fbpem.") and not key.startswith("stream_fusion.")
        }
        legacy["fbpem.map.weight"] = torch.zeros(64, 6)
        legacy["fbpem.map.bias"] = torch.zeros(64)
        result = model.load_state_dict(legacy, strict=False)
        self.assertIn("fbpem.part_maps.head.weight", result.missing_keys)
        self.assertIn("fbpem.fuse.weight", result.missing_keys)
        self.assertIn("stream_fusion.weight", result.missing_keys)
        self.assertIn("fbpem.map.weight", result.unexpected_keys)
        with self.assertRaises(RuntimeError):
            model.load_state_dict(legacy, strict=True)
        fresh = build_model(self.config, extractor.motion_dim, extractor.geometry_dim)
        fresh.load_state_dict(current, strict=True)
        root = Path(tempfile.mkdtemp(dir=str(ROOT)))
        try:
            checkpoint = _checkpoint(fresh, self.config, extractor)
            checkpoint["model_state_dict"] = legacy
            path = root / "legacy.pth"
            torch.save(checkpoint, path)
            with self.assertRaises(RuntimeError) as raised:
                BehaviorPredictor(str(path))
            self.assertIn("不匹配", str(raised.exception))
            self.assertIn("重新训练", str(raised.exception))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_pose_similarity_and_track_quality(self):
        fish = _straight_fish(30.0)
        self.assertAlmostEqual(pose_similarity(fish, fish, self.config), 1.0, places=5)
        shifted = fish.copy()
        shifted[:, 0] += 80.0
        self.assertAlmostEqual(pose_similarity(fish, shifted, self.config), 1.0, places=4)
        distorted = fish.copy()
        distorted[4, 1] += 40.0
        distorted[8, 1] -= 40.0
        self.assertLess(pose_similarity(fish, distorted, self.config), 1.0)
        cost = association_cost(fish, fish, np.array([0, 0, 40, 20], dtype=np.float64), self.config)
        self.assertGreaterEqual(cost, 0.0)
        self.assertLessEqual(cost, 1.0)
        low_conf = np.repeat(fish[None], 10, axis=0)
        low_conf[:, :, 2] = 0.3
        self.assertFalse(window_is_reliable(low_conf, self.config))
        reliable = np.repeat(fish[None], 10, axis=0)
        self.assertTrue(window_is_reliable(reliable, self.config))
        jumped = np.repeat(fish[None], 4, axis=0)
        jumped[1, 3, :2] += 500
        jumped[3, 5, :2] += 500
        self.assertGreaterEqual(jump_frame_ratio(jumped, self.config), 0.2)
        self.assertFalse(window_is_reliable(jumped, self.config))
        long_gap = reliable.copy()
        long_gap[1:10, :, 2] = 0.0
        self.assertFalse(window_is_reliable(long_gap, self.config))

    def test_hungarian_and_tracks(self):
        cost = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
        rows, cols = linear_sum_assignment(cost)
        assignment = dict(zip(rows.tolist(), cols.tolist()))
        self.assertEqual(assignment, {0: 1, 1: 0, 2: 2})
        frames = []
        for frame_id in range(6):
            left = _straight_fish(30.0)
            right = _straight_fish(30.0)
            left[:, 0] += frame_id
            right[:, 0] += 400 + frame_id
            frames.append((frame_id, [
                {"keypoints": left, "box": np.array([0, 0, 40, 20], dtype=np.float64)},
                {"keypoints": right, "box": np.array([400, 0, 440, 20], dtype=np.float64)},
            ]))
        tracks = associate_detections(frames, self.config)
        self.assertEqual(len(tracks), 2)
        self.assertEqual({len(track["frames"]) for track in tracks}, {6})
        gapped = frames[:3] + [(frame_id, []) for frame_id in range(3, 6)] + [
            (frame_id, [{"keypoints": _straight_fish(30.0), "box": np.array([0, 0, 40, 20], dtype=np.float64)}])
            for frame_id in range(6, 10)
        ]
        gapped_tracks = associate_detections(gapped, self.config)
        continued = max(gapped_tracks, key=lambda track: len(track["frames"]))
        dense = densify_track(
            continued["frames"],
            11,
            int(self.config["association"]["max_missing_frames"]),
        )
        self.assertEqual(len(dense), 1)
        sequence = dense[0][1]
        self.assertGreaterEqual(sequence.shape[0], 10)
        self.assertTrue(np.all(sequence[3:6, :, 2] == 0))
        jumped = np.repeat(_straight_fish(30.0)[None], 4, axis=0)
        jumped[2, 3, :2] += 500
        filtered = suppress_jumps(jumped, self.config)
        self.assertEqual(float(filtered[2, 3, 2]), 0.0)

    def test_video_split_has_no_window_leakage(self):
        mapping = assign_video_splits(["v1", "v1", "v2", "v3", "v4"], 0.5, 0.0, 1)
        self.assertEqual(len(mapping), 4)
        self.assertIn("test", mapping.values())
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            source = root / "sequences"
            source.mkdir()
            output = root / "dataset"
            for name in ("alpha", "beta"):
                _write_video(source / f"{name}.npz", frames=96)
            counts = build_dataset(source, self.config, output)
            self.assertEqual(sum(counts.values()), 2)
            self.assertNotEqual(counts["train"], 5120)
            owners = {}
            for split in ("train", "val", "test"):
                for path in (output / split).glob("*.npz"):
                    with np.load(path) as data:
                        self.assertEqual(data["keypoints"].shape, (96, 11, 3))
                        video_id = str(data["video_id"])
                        self.assertIn(str(data["label_source"]), {"prelabel", "review"})
                    owners.setdefault(video_id, set()).add(split)
            self.assertTrue(all(len(splits) == 1 for splits in owners.values()))

    def test_manual_labels_override_candidates(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            source = root / "sequences"
            source.mkdir()
            output = root / "dataset"
            _write_video(source / "alpha.npz", frames=96)
            labels = root / "labels.csv"
            labels.write_text("video_id,track_id,start_frame,label\nalpha,1,0,1\n", encoding="utf-8")
            build_dataset(source, self.config, output, label_file=labels)
            saved = list(output.glob("*/*.npz"))
            self.assertEqual(len(saved), 1)
            with np.load(saved[0]) as data:
                self.assertEqual(int(data["label"]), 1)
                self.assertEqual(str(data["label_source"]), "manual")
            candidate_dir = root / "candidates"
            candidate_dir.mkdir()
            np.savez_compressed(
                candidate_dir / "prelabel.npz",
                keypoints=np.zeros((96, 11, 3), dtype=np.float32),
                label=np.int64(2),
                track_id=np.int64(1),
                label_source=np.array("prelabel"),
            )
            dataset = BehaviorDataset(candidate_dir, self.config, augment=False)
            self.assertEqual(len(dataset), 0)
            self.assertEqual(dataset.skipped_unverified, 1)
            self.assertEqual(dataset.skip_reasons, {"prelabel": 1})

    def test_verified_samples_and_legacy_opt_in(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            folder = root / "samples"
            folder.mkdir()
            _save_window(folder / "manual.npz", 1, "manual")
            _save_window(folder / "prelabel.npz", 2, "prelabel")
            _save_window(folder / "negative.npz", -1, "prelabel")
            _save_window(folder / "review.npz", 1, "review")
            _save_window(folder / "unknown.npz", 0, "auto")
            _save_window(folder / "legacy.npz", 3, None)
            _save_window(folder / "legacy_negative.npz", -1, None)

            dataset = BehaviorDataset(folder, self.config, augment=False)
            self.assertEqual([path.name for path in dataset.files], ["manual.npz"])
            self.assertEqual(dataset.skip_reasons["prelabel"], 1)
            self.assertEqual(dataset.skip_reasons["invalid_label"], 2)
            self.assertEqual(dataset.skip_reasons["review"], 1)
            self.assertEqual(dataset.skip_reasons["unknown_source"], 1)
            self.assertEqual(dataset.skip_reasons["missing_label_source"], 1)
            item = dataset[0]
            self.assertEqual(tuple(item["coord"].shape), (96, 11, 2))
            self.assertEqual(tuple(item["motion"].shape), (96, 3))
            self.assertEqual(tuple(item["geometry"].shape), (96, 2))
            self.assertEqual(int(item["label"]), 1)
            self.assertIn("缺少 label_source", dataset.filter_report())

            legacy_cfg = copy.deepcopy(self.config)
            legacy_cfg["labeling"]["allow_legacy_samples"] = True
            legacy = BehaviorDataset(folder, legacy_cfg, augment=False)
            self.assertEqual([path.name for path in legacy.files], ["legacy.npz", "manual.npz"])
            self.assertEqual(legacy.accepted_legacy, 1)
            self.assertEqual(legacy.skip_reasons["invalid_label"], 2)
            self.assertEqual(int(legacy[0]["label"]), 3)
            self.assertEqual(tuple(legacy[0]["coord"].shape), (96, 11, 2))
            with np.load(folder / "legacy.npz") as data:
                self.assertNotIn("label_source", data.files)

            prelabel_cfg = copy.deepcopy(self.config)
            prelabel_cfg["labeling"]["use_unverified"] = True
            prelabel = BehaviorDataset(folder, prelabel_cfg, augment=False)
            self.assertEqual([path.name for path in prelabel.files], ["manual.npz", "prelabel.npz"])
            self.assertEqual(prelabel.skip_reasons["missing_label_source"], 1)
            with np.load(folder / "legacy.npz") as data:
                self.assertNotIn("label_source", data.files)

            empty = BehaviorDataset(root / "missing", self.config, augment=False)
            self.assertEqual(len(empty), 0)
            self.assertIn("没有找到", empty.filter_report())

    def test_unverified_training_set_stops(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            empty_dir = root / "empty"
            empty_dir.mkdir()
            empty_config = _tiny_config(root, empty_dir)
            with self.assertRaises(SystemExit) as empty_raised:
                run_train(str(empty_config))
            self.assertIn("为空", str(empty_raised.exception))

            unverified_dir = root / "unverified"
            unverified_dir.mkdir()
            _save_window(unverified_dir / "old.npz", 1, None)
            _save_window(unverified_dir / "candidate.npz", 2, "prelabel")
            unverified_config = _tiny_config(root, unverified_dir, name="unverified.yaml")
            with self.assertRaises(SystemExit) as raised:
                run_train(str(unverified_config))
            message = str(raised.exception)
            self.assertIn("人工核验", message)
            self.assertIn("缺少 label_source", message)
            self.assertIn("未核验预标注", message)


class InferenceTests(unittest.TestCase):
    def test_predictor_online_and_one_epoch(self):
        config = load_config(str(ROOT / "config" / "config.yaml"))
        extractor = FeatureExtractor(config)
        model = build_model(config, extractor.motion_dim, extractor.geometry_dim)
        sequence = make_synthetic_sequence(config)
        root = Path(tempfile.mkdtemp(dir=str(ROOT)))
        try:
            checkpoint = root / "model.pth"
            torch.save(_checkpoint(model, config, extractor), checkpoint)
            predictor = BehaviorPredictor(str(checkpoint))
            result = predictor.predict(sequence)
            self.assertIn(result["behavior_id"], range(4))
            self.assertEqual(result["behavior_name"], config["behavior_names"][result["behavior_id"]])
            self.assertEqual(len(result["probabilities"]), 4)
            features = extractor.extract(sequence)
            batch = {name: torch.from_numpy(features[name]).unsqueeze(0) for name in ("coord", "motion", "geometry", "confidence")}
            logits = predictor.model(batch["coord"], batch["motion"], batch["geometry"], batch["confidence"])["logits"]
            self.assertEqual(int(torch.argmax(logits, dim=-1)), result["behavior_id"])

            analyzer = OnlineBehaviorAnalyzer(str(checkpoint))
            frame = sequence[0]
            for _ in range(95):
                early = analyzer.update(1, frame)
            self.assertEqual(early["behavior"], INSUFFICIENT)
            self.assertEqual(early["behavior_id"], -1)
            ready = analyzer.update(1, frame, frame_idx=95)
            self.assertTrue(ready["updated"])
            self.assertIn(ready["behavior_id"], range(4))
            for step in range(47):
                held = analyzer.update(1, frame, frame_idx=96 + step)
                self.assertFalse(held["updated"])
            nxt = analyzer.update(1, frame, frame_idx=143)
            self.assertTrue(nxt["updated"])
            missing = analyzer.update(7, None, frame_idx=1)
            self.assertEqual(missing["behavior"], INSUFFICIENT)
            with self.assertRaises(ValueError):
                analyzer.update(1, np.zeros((10, 2), dtype=np.float32))
            summary = analyzer.summarize([early, missing])
            self.assertIsNone(summary["activity"]["dominant_behavior"])
            other = analyzer.update(2, frame)
            self.assertEqual(other["behavior"], INSUFFICIENT)
            fresh = OnlineBehaviorAnalyzer(str(checkpoint))
            filled = None
            for index in range(2):
                filled = fresh.update(4, frame, frame_idx=index * 3)
            self.assertEqual(len(fresh.buffers[4]), 4)
            self.assertTrue(np.all(fresh.buffers[4][1][:, 2] == 0))
            self.assertTrue(np.all(fresh.buffers[4][2][:, 2] == 0))
            self.assertGreater(float(np.mean(fresh.buffers[4][3][:, 2])), 0.4)
            ready_track = None
            for index in range(96):
                ready_track = fresh.update(5, frame, frame_idx=index)
            self.assertTrue(ready_track["updated"])
            duplicate = fresh.update(5, frame, frame_idx=95)
            self.assertFalse(duplicate["updated"])
            self.assertEqual(duplicate["behavior_id"], ready_track["behavior_id"])
            reorder = fresh.update(5, frame, frame_idx=90)
            self.assertFalse(reorder["updated"])
            broken = fresh.update(5, frame, frame_idx=95 + 40)
            self.assertEqual(broken["behavior"], INSUFFICIENT)
            self.assertEqual(broken["behavior_id"], -1)
            bad_state = model.state_dict()
            bad_state["pqtgm.gate.weight"] = torch.zeros(128, 130, 3)
            bad_model = build_model(config, extractor.motion_dim, extractor.geometry_dim)
            with self.assertRaises(RuntimeError):
                bad_model.load_state_dict(bad_state, strict=True)
            bad_checkpoint = _checkpoint(model, config, extractor)
            bad_checkpoint["model_state_dict"] = bad_state
            torch.save(bad_checkpoint, root / "bad.pth")
            with self.assertRaises(RuntimeError) as raised:
                BehaviorPredictor(str(root / "bad.pth"))
            self.assertIn("不匹配", str(raised.exception))

            train_dir = root / "train"
            train_dir.mkdir()
            for index in range(4):
                sample = sequence.copy()
                sample[:, :, 0] += index
                np.savez_compressed(
                    train_dir / f"{index}.npz",
                    keypoints=sample,
                    label=np.int64(index % 4),
                    track_id=np.int64(index),
                    label_source=np.array("manual"),
                )
            tiny = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
            tiny["dataset"]["train_dir"] = str(train_dir)
            tiny["dataset"]["val_dir"] = str(root / "empty_val")
            tiny["dataset"]["test_dir"] = str(root / "empty_test")
            tiny["train"]["epochs"] = 1
            tiny["train"]["batch_size"] = 2
            tiny["train"]["checkpoint_dir"] = str(root / "checkpoints")
            config_path = root / "tiny.yaml"
            config_path.write_text(yaml.safe_dump(tiny, allow_unicode=True), encoding="utf-8")
            run_train(str(config_path))
            self.assertTrue((root / "checkpoints" / "best.pth").is_file())
            self.assertTrue((root / "checkpoints" / "last.pth").is_file())
        finally:
            shutil.rmtree(root, ignore_errors=True)


def _straight_fish(length: float) -> np.ndarray:
    xy = np.zeros((11, 2), dtype=np.float32)
    xy[6] = [0.0, 0.0]
    xy[0] = [length, 0.0]
    xy[1] = [length * 0.82, 1.5]
    xy[2] = [length * 0.68, 5.0]
    xy[10] = [length * 0.68, -5.0]
    xy[3] = [length * 0.48, 6.0]
    xy[9] = [length * 0.42, -5.5]
    xy[4] = [length * 0.22, 3.5]
    xy[8] = [length * 0.22, -3.5]
    xy[5] = [length * 0.06, 5.0]
    xy[7] = [length * 0.06, -5.0]
    pose = np.ones((11, 3), dtype=np.float32)
    pose[:, :2] = xy
    return pose


def _save_window(path: Path, label: int, source) -> None:
    payload = {
        "keypoints": np.repeat(_straight_fish(30.0)[None], 96, axis=0),
        "label": np.int64(label),
        "track_id": np.int64(1),
    }
    if source is not None:
        payload["label_source"] = np.array(source)
    np.savez_compressed(path, **payload)


def _tiny_config(root: Path, train_dir: Path, name: str = "tiny.yaml") -> Path:
    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    config["dataset"]["train_dir"] = str(train_dir)
    config["dataset"]["val_dir"] = str(root / "empty_val")
    config["dataset"]["test_dir"] = str(root / "empty_test")
    config["train"]["epochs"] = 1
    config["train"]["batch_size"] = 2
    config["train"]["checkpoint_dir"] = str(root / "checkpoints")
    path = root / name
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


def _write_video(path: Path, frames: int) -> None:
    fish = _straight_fish(30.0)
    keypoints = np.repeat(fish[None, None], frames, axis=0)
    for frame_id in range(frames):
        keypoints[frame_id, 0, :, 0] += frame_id * 2.0
    np.savez_compressed(
        path,
        keypoints=keypoints,
        valid=np.ones((frames, 1), dtype=bool),
        frame_idx=np.arange(frames),
    )


def _checkpoint(model, config, extractor) -> dict:
    return {
        "model_name": MODEL_NAME,
        "model_state_dict": model.state_dict(),
        "config": config,
        "feature_dims": {
            "coord": [extractor.num_keypoints, 2],
            "motion": extractor.motion_dim,
            "geometry": extractor.geometry_dim,
        },
    }


if __name__ == "__main__":
    unittest.main()

