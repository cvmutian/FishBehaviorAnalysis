import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import BehaviorDataset
from predictor import BehaviorPredictor
from utils.config import load_config, resolve_path
from utils.metrics import classification_scores, confusion_matrix
from utils.visualization import plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="测试多流鱼体行为分类模型")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    file_config = load_config(args.config)
    checkpoint_path = args.checkpoint or file_config["test"]["checkpoint"]
    checkpoint_path = resolve_path(ROOT, checkpoint_path)
    if not checkpoint_path.is_file():
        raise SystemExit(f"找不到模型权重: {checkpoint_path}")

    predictor = BehaviorPredictor(str(checkpoint_path), device=file_config["test"].get("device", "auto"))
    config = predictor.config
    test_dir = resolve_path(ROOT, file_config["dataset"]["test_dir"])
    dataset = BehaviorDataset(test_dir, config, predictor.extractor, augment=False)
    if len(dataset) == 0:
        if dataset.scanned == 0:
            raise SystemExit(f"测试集为空: {test_dir}")
        raise SystemExit(f"测试集没有通过人工核验的样本。{dataset.filter_report()}")
    print(f"测试集{dataset.filter_report()}")

    loader = DataLoader(
        dataset,
        batch_size=int(file_config["test"]["batch_size"]),
        shuffle=False,
        num_workers=int(file_config["test"]["num_workers"]),
    )
    labels, predictions = _predict_dataset(predictor.model, loader, predictor.device)
    matrix = confusion_matrix(labels, predictions, int(config["num_classes"]))
    scores = classification_scores(matrix)
    _print_report(scores, config["behavior_names"])
    output_dir = resolve_path(ROOT, file_config["test"]["output_dir"])
    figure_path = output_dir / file_config["test"]["confusion_matrix_name"]
    plot_confusion_matrix(scores["confusion_matrix"], config["behavior_names"], figure_path)
    print(f"混淆矩阵已保存: {figure_path}")


@torch.no_grad()
def _predict_dataset(model, loader, device):
    model.eval()
    labels = []
    predictions = []
    for batch in loader:
        coord = batch["coord"].to(device)
        motion = batch["motion"].to(device)
        geometry = batch["geometry"].to(device)
        confidence = batch["confidence"].to(device)
        logits = model(coord, motion, geometry, confidence)["logits"]
        pred = torch.argmax(logits, dim=-1)
        labels.extend(batch["label"].cpu().numpy().tolist())
        predictions.extend(pred.cpu().numpy().tolist())
    return labels, predictions


def _print_report(scores: dict, class_names) -> None:
    print(f"Accuracy: {scores['accuracy']:.4f}")
    print(f"Macro-Precision: {scores['macro_precision']:.4f}")
    print(f"Macro-Recall: {scores['macro_recall']:.4f}")
    print(f"Macro-F1: {scores['macro_f1']:.4f}")
    print("类别        Precision  Recall     F1         Support")
    for name, item in zip(class_names, scores["per_class"]):
        print(
            f"{name:<10}  {item['precision']:.4f}     {item['recall']:.4f}     "
            f"{item['f1']:.4f}     {item['support']}"
        )
    print("混淆矩阵（行=真实类别，列=预测类别）")
    print(scores["confusion_matrix"])


if __name__ == "__main__":
    main()

