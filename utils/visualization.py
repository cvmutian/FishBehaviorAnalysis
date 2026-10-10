from pathlib import Path

import numpy as np


def skeleton_edges(config: dict) -> list:
    edges = []
    for edge in config["keypoints"]["skeleton"]:
        start, end = int(edge[0]), int(edge[1])
        if not (0 <= start < 11 and 0 <= end < 11):
            raise ValueError(f"无效骨架连线: {edge}")
        edges.append((start, end))
    return edges


def plot_confusion_matrix(matrix: np.ndarray, class_names, save_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("行为分类混淆矩阵")

    threshold = matrix.max() / 2 if matrix.size and matrix.max() > 0 else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            color = "white" if value > threshold else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=color)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
