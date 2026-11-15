# MODULE: visualize.py
#
# Vai trò: tạo hình từ logs/metrics/predictions đã lưu: learning curves,
# confusion matrix và các ảnh phân loại sai.
#
# Nguyên tắc: chỉ đọc artifacts đã lưu, KHÔNG chạy lại training trong module này.
# Hình test chỉ để báo cáo; KHÔNG dùng để chọn epoch/hyperparameter.
#
# Hướng dẫn triển khai:
# 1. Đọc artifacts đã lưu thay vì chạy lại training.
# 2. Ghi tên model, class order, split và metric lên biểu đồ.
# 3. Hiển thị ảnh sai kèm nhãn thật, nhãn dự đoán và confidence.
# 4. Lưu hình vào artifacts/figures/ để báo cáo có thể tái tạo.

# MODULE: visualize.py
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.cifar_compare.data import build_transforms, denormalize
else:
    from .data import build_transforms, denormalize


def _load_test_metrics(run_name, output_dir="artifacts"):
    path = Path(output_dir) / "logs" / run_name / "test_metrics.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_summary(run_name, output_dir="artifacts"):
    path = Path(output_dir) / "logs" / run_name / "summary.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _title_with_conditions(base, metrics, summary):
    parts = [base]
    if summary.get("best_epoch") is not None:
        parts.append(f"best epoch={summary['best_epoch']}")
    if summary.get("best_val_accuracy") is not None:
        parts.append(f"val acc={summary['best_val_accuracy']:.4f}")
    if metrics.get("accuracy") is not None:
        parts.append(f"test acc={metrics['accuracy']:.4f}")
    if metrics.get("macro_f1") is not None:
        parts.append(f"macro-F1={metrics['macro_f1']:.4f}")
    return " | ".join(parts)


def plot_learning_curves(run_name, output_dir="artifacts"):
    metrics_path = Path(output_dir) / "logs" / run_name / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Không thấy {metrics_path}")

    epochs, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_acc.append(float(row["train_accuracy"]))
            val_acc.append(float(row["val_accuracy"]))

    summary = _load_summary(run_name, output_dir)
    metrics = _load_test_metrics(run_name, output_dir)
    title = _title_with_conditions(run_name, metrics, summary)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_loss, label="Train", marker="o", ms=3)
    axes[0].plot(epochs, val_loss, label="Validation", marker="o", ms=3)
    if summary.get("best_epoch"):
        axes[0].axvline(summary["best_epoch"], color="gray", ls="--", lw=1,
                        label=f"Best (ep {summary['best_epoch']})")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_acc, label="Train", marker="o", ms=3)
    axes[1].plot(epochs, val_acc, label="Validation", marker="o", ms=3)
    if summary.get("best_epoch"):
        axes[1].axvline(summary["best_epoch"], color="gray", ls="--", lw=1)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out = Path(output_dir) / "figures" / run_name / "learning_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[visualize] Learning curves -> {out}")
    return str(out)


def plot_confusion_matrix(run_name, output_dir="artifacts", normalize=False):
    pred_dir = Path(output_dir) / "predictions" / run_name
    cm = np.load(pred_dir / "confusion_matrix.npy")
    data = np.load(pred_dir / "test_predictions.npz", allow_pickle=True)
    class_names = list(data["class_names"])

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).astype(float)
        row_sums[row_sums == 0] = 1.0
        cm_show = cm / row_sums
        fmt = "{:.2f}"
        vmax = 1.0
    else:
        cm_show = cm
        fmt = "{:d}"
        vmax = cm.max()

    metrics = _load_test_metrics(run_name, output_dir)
    summary = _load_summary(run_name, output_dir)
    base = (f"{run_name} — Confusion Matrix "
            f"({'row-normalized' if normalize else 'counts'})\n"
            f"Split: CIFAR-10 official test (10.000 mẫu)")
    title = _title_with_conditions(base, metrics, summary)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_show, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j] == 0:
                continue
            val = cm_show[i, j]
            ax.text(j, i, fmt.format(val), ha="center", va="center",
                    color="white" if val > vmax / 2 else "black", fontsize=7)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    suffix = "confusion_matrix_normalized" if normalize else "confusion_matrix"
    out = Path(output_dir) / "figures" / run_name / f"{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[visualize] Confusion matrix ({'norm' if normalize else 'count'}) -> {out}")
    return str(out)


def plot_misclassified(run_name, config, output_dir="artifacts",
                       num_images=16, criterion="top_conf"):
    pred_dir = Path(output_dir) / "predictions" / run_name
    metrics = _load_test_metrics(run_name, output_dir)
    data = np.load(pred_dir / "test_predictions.npz", allow_pickle=True)
    indices = data["indices"]
    true_labels = data["true_labels"]
    pred_labels = data["pred_labels"]
    confidences = data["confidences"]
    class_names = list(data["class_names"])

    wrong_pos = np.where(true_labels != pred_labels)[0]
    if len(wrong_pos) == 0:
        print("[visualize] Không có ảnh sai.")
        return None

    if criterion == "random":
        rng = np.random.default_rng(seed=0)
        take = min(num_images, len(wrong_pos))
        selected = rng.choice(wrong_pos, size=take, replace=False)
        order_desc = "ngẫu nhiên"
    else:
        wrong_conf = confidences[wrong_pos]
        order = np.argsort(wrong_conf)[::-1]
        take = min(num_images, len(wrong_pos))
        selected = wrong_pos[order[:take]]
        order_desc = "confidence cao nhất"

    input_size = metrics.get("input_size", config.get("input_size", 224))
    eval_tf = build_transforms(input_size)[1]
    test_tf = datasets.CIFAR10(config.get("data_root", "data/raw"),
                               train=False, download=False, transform=eval_tf)

    n = len(selected)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = np.array(axes).flatten()

    for ax_idx in range(rows * cols):
        ax = axes[ax_idx]
        ax.axis("off")
        if ax_idx >= n:
            continue
        pos = selected[ax_idx]
        idx = int(indices[pos])
        tensor, _ = test_tf[idx]
        img = denormalize(tensor).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(
            f"True: {class_names[true_labels[pos]]}\n"
            f"Pred: {class_names[pred_labels[pos]]} ({confidences[pos]:.2f})",
            fontsize=8, color="red",
        )

    summary = _load_summary(run_name, output_dir)
    base = (f"{run_name} — Top {n} misclassified ({order_desc})\n"
            f"Split: CIFAR-10 official test (10.000 mẫu)")
    title = _title_with_conditions(base, metrics, summary)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out = Path(output_dir) / "figures" / run_name / "misclassified.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[visualize] Misclassified -> {out}")
    return str(out)


def plot_per_class_f1(run_name, output_dir="artifacts"):
    metrics = _load_test_metrics(run_name, output_dir)
    if "per_class" not in metrics:
        print("[visualize] Chưa có test_metrics.json — bỏ qua per-class F1.")
        return None
    per_class = metrics["per_class"]
    class_names = [c for c in per_class if c not in
                   ("accuracy", "macro avg", "weighted avg")]
    f1_scores = [per_class[c]["f1-score"] for c in class_names]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(class_names, f1_scores, color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score")
    ax.set_title(
        f"{run_name} — Per-class F1 trên test "
        f"(macro-F1={metrics['macro_f1']:.4f})",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.01,
                f"{score:.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    out = Path(output_dir) / "figures" / run_name / "per_class_f1.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[visualize] Per-class F1 -> {out}")
    return str(out)


def visualize_all(run_name, config):
    output_dir = config.get("output_dir", "artifacts")
    results = {
        "learning_curves": plot_learning_curves(run_name, output_dir),
        "confusion_matrix": plot_confusion_matrix(run_name, output_dir, normalize=False),
        "confusion_matrix_normalized": plot_confusion_matrix(run_name, output_dir, normalize=True),
        "misclassified": plot_misclassified(run_name, config, output_dir),
        "per_class_f1": plot_per_class_f1(run_name, output_dir),
    }
    manifest = {
        "run_name": run_name,
        "split": "CIFAR-10 official test (10.000 mẫu)",
        "note": "Hình test chỉ để báo cáo. "
                "Không dùng để chọn epoch/hyperparameter.",
        "figures": {k: v for k, v in results.items() if v},
    }
    mpath = Path(output_dir) / "figures" / run_name / "manifest.json"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[visualize] Manifest -> {mpath}")
    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create plots from saved training and evaluation artifacts"
    )
    parser.add_argument("--run", required=True,
                        help="Run name, e.g. resnet18")
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8-sig") as f:
        _cfg = json.load(f)
    visualize_all(args.run, _cfg)