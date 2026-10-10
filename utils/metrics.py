import numpy as np


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            matrix[true_label, pred_label] += 1
    return matrix


def classification_scores(matrix: np.ndarray) -> dict:
    num_classes = matrix.shape[0]
    per_class = []
    for class_id in range(num_classes):
        tp = float(matrix[class_id, class_id])
        fp = float(matrix[:, class_id].sum() - tp)
        fn = float(matrix[class_id, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class.append({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(matrix[class_id, :].sum()),
        })

    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total > 0 else 0.0
    macro_precision = float(np.mean([item["precision"] for item in per_class])) if per_class else 0.0
    macro_recall = float(np.mean([item["recall"] for item in per_class])) if per_class else 0.0
    macro_f1 = float(np.mean([item["f1"] for item in per_class])) if per_class else 0.0
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }
