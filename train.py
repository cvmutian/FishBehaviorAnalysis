import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import BehaviorDataset
from features.feature_extractor import FeatureExtractor
from models.multistream_tcn import build_model
from utils.config import MODEL_NAME, load_config, resolve_path, select_device
from utils.metrics import classification_scores, confusion_matrix
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="训练多流鱼体行为分类模型")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run_train(args.config)


def run_train(config_path: str) -> None:
    config = load_config(config_path)
    train_cfg = config["train"]
    set_seed(int(train_cfg["seed"]))
    extractor = FeatureExtractor(config)
    model = build_model(config, extractor.motion_dim, extractor.geometry_dim)
    device = select_device(train_cfg["device"])
    model.to(device)

    train_dir = resolve_path(ROOT, config["dataset"]["train_dir"])
    val_dir = resolve_path(ROOT, config["dataset"]["val_dir"])
    train_dataset = BehaviorDataset(train_dir, config, extractor, augment=True)
    val_dataset = BehaviorDataset(val_dir, config, extractor, augment=False)
    _require_verified_samples(train_dataset, "训练集", train_dir, int(config["window_size"]))
    if val_dataset.scanned:
        print(f"验证集{val_dataset.filter_report()}")
    if len(train_dataset) < 2:
        raise SystemExit("训练集至少需要 2 个样本，否则 BatchNorm 无法训练。")

    batch_size = int(train_cfg["batch_size"])
    drop_last = len(train_dataset) >= batch_size and len(train_dataset) % batch_size == 1
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        drop_last=drop_last,
        generator=_seeded_generator(int(train_cfg["seed"])),
    )
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(train_cfg["num_workers"]),
        )

    criterion = torch.nn.CrossEntropyLoss()
    aux_criterion = torch.nn.SmoothL1Loss()
    aux_weight = float(train_cfg["aux_loss_weight"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    checkpoint_dir = resolve_path(ROOT, train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / train_cfg["best_name"]
    last_path = checkpoint_dir / train_cfg["last_name"]
    best_score = -1.0
    best_loss = float("inf")
    epochs = int(train_cfg["epochs"])

    for epoch in range(1, epochs + 1):
        lr = epoch_learning_rate(
            epoch,
            epochs,
            float(train_cfg["learning_rate"]),
            int(train_cfg["warmup_epochs"]),
            float(train_cfg["eta_min"]),
        )
        _set_learning_rate(optimizer, lr)
        train_loss = _train_one_epoch(model, train_loader, criterion, aux_criterion, aux_weight, optimizer, device)
        message = f"Epoch {epoch}/{epochs}  lr={lr:.6g}  train_loss={train_loss:.4f}"
        improved = False
        if val_loader is not None:
            val_result = _evaluate(model, val_loader, device, int(config["num_classes"]))
            message += f"  val_acc={val_result['accuracy']:.4f}  val_macro_f1={val_result['macro_f1']:.4f}"
            score = val_result["macro_f1"]
            reference = {"accuracy": val_result["accuracy"], "macro_f1": val_result["macro_f1"]}
            improved = score > best_score
            if improved:
                best_score = score
        else:
            reference = {"train_loss": train_loss}
            improved = train_loss < best_loss
            if improved:
                best_loss = train_loss
        print(message)
        payload = _checkpoint_payload(model, config, extractor, epoch, reference)
        torch.save(payload, last_path)
        if improved:
            torch.save(payload, best_path)

    print(f"last model: {last_path}")
    print(f"best model: {best_path}")


def _require_verified_samples(dataset: BehaviorDataset, split_name: str, directory: Path, window_size: int) -> None:
    if len(dataset) == 0:
        if dataset.scanned == 0:
            raise SystemExit(
                f"{split_name}为空：{directory} 中没有 npz 样本。"
                f"正式训练只使用已人工核验、标签有效的 {window_size} 帧样本。"
            )
        raise SystemExit(f"{split_name}没有通过人工核验的样本，已停止训练。{dataset.filter_report()}")
    print(f"{split_name}{dataset.filter_report()}")


def epoch_learning_rate(epoch: int, total_epochs: int, base_lr: float, warmup_epochs: int, eta_min: float) -> float:
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    span = max(total_epochs - warmup_epochs, 1)
    progress = min(max((epoch - warmup_epochs) / span, 0.0), 1.0)
    return eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * progress))


def _set_learning_rate(optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _seeded_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _train_one_epoch(model, loader, criterion, aux_criterion, aux_weight, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        if batch["label"].size(0) < 2:
            continue
        coord = batch["coord"].to(device)
        motion = batch["motion"].to(device)
        geometry = batch["geometry"].to(device)
        confidence = batch["confidence"].to(device)
        labels = batch["label"].to(device)
        targets = batch["aux"].to(device)
        optimizer.zero_grad()
        outputs = model(coord, motion, geometry, confidence)
        loss = criterion(outputs["logits"], labels) + aux_weight * aux_criterion(outputs["aux"], targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        total_count += labels.size(0)
    if total_count == 0:
        return 0.0
    return total_loss / total_count


@torch.no_grad()
def _evaluate(model, loader, device, num_classes: int) -> dict:
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
    matrix = confusion_matrix(labels, predictions, num_classes)
    return classification_scores(matrix)


def _checkpoint_payload(model, config, extractor, epoch: int, reference: dict) -> dict:
    return {
        "model_name": MODEL_NAME,
        "model_state_dict": model.state_dict(),
        "config": config,
        "feature_dims": {
            "coord": [extractor.num_keypoints, 2],
            "motion": extractor.motion_dim,
            "geometry": extractor.geometry_dim,
        },
        "epoch": epoch,
        "reference": reference,
    }


if __name__ == "__main__":
    main()

