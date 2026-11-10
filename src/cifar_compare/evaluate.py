# MODULE: evaluate.py
#
# Vai trò: nạp checkpoint validation-best, đánh giá trên test set và lưu
# metrics/dự đoán để lập bảng so sánh và phân tích lỗi.
#
# Hướng dẫn triển khai:
# 1. Dùng config/transforms và class order tương ứng với checkpoint.
# 2. Chạy eval/no_grad; tính accuracy, macro-F1, per-class precision/recall,
#    loss và confusion matrix.
# 3. Lưu true label, predicted label, confidence/logits và index mẫu.
# 4. Báo rõ checkpoint, split và các điều kiện đánh giá.
# Không cập nhật weights hay điều chỉnh hyperparameter bằng test results.

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torchvision.datasets import CIFAR10

from .data import build_transforms, get_test_loader
from .models import build_model
from .utils import resolve_device


def evaluate_checkpoint(checkpoint_path, config, output_dir=None, device=None):
    """
    Nạp checkpoint best.pt, chạy đánh giá MỘT LẦN trên test set.
    Trả về dict metrics và lưu artifacts (metrics + predictions + confusion matrix).
    """
    device = device or resolve_device(config.get("device", "auto"))
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 1. Nạp checkpoint và lấy metadata
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = ckpt["model_name"]
    num_classes = int(ckpt["num_classes"])
    class_names = list(ckpt["class_names"])
    input_size = int(ckpt["config"]["input_size"])

    # 2. Dựng model, nạp state_dict (KHÔNG train)
    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    # 3. Test loader — chỉ eval transform, không đụng train/val split
    eval_cfg = dict(config)
    eval_cfg["input_size"] = input_size
    eval_cfg["batch_size"] = config.get("eval_batch_size", 64)
    test_loader = get_test_loader(eval_cfg)

    # Kiểm tra class order khớp CIFAR-10 gốc
    official = CIFAR10(config["data_root"], train=False, download=False).classes
    if class_names != official:
        raise ValueError(
            f"Class order trong checkpoint không khớp CIFAR-10.\n"
            f"Checkpoint: {class_names}\nCIFAR-10: {official}"
        )

    # 4. Suy luận
    all_indices, all_true, all_pred, all_conf = [], [], [], []
    all_logits = []
    total_loss = 0.0
    total_samples = 0
    sample_offset = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction="sum")
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

            n = labels.size(0)
            all_indices.extend(range(sample_offset, sample_offset + n))
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())
            all_conf.extend(conf.cpu().tolist())
            all_logits.append(logits.cpu().numpy())
            total_loss += loss.item()
            total_samples += n
            sample_offset += n

    all_true = np.array(all_true, dtype=np.int64)
    all_pred = np.array(all_pred, dtype=np.int64)
    all_conf = np.array(all_conf, dtype=np.float32)
    all_indices = np.array(all_indices, dtype=np.int64)
    all_logits = np.concatenate(all_logits, axis=0).astype(np.float32)

    # 5. Metrics + metadata điều kiện đánh giá
    cm = confusion_matrix(all_true, all_pred)
    metrics = {
        "checkpoint": str(checkpoint_path),
        "model": model_name,
        "num_classes": num_classes,
        "class_names": class_names,
        "num_samples": int(total_samples),

        # Điều kiện đánh giá
        "split": "CIFAR-10 official test set (10.000 mẫu, không dùng để chọn model)",
        "split_path": config.get("split_path", "data/splits/split_indices.json"),
        "seed": config.get("seed"),
        "input_size": input_size,
        "eval_batch_size": eval_cfg["batch_size"],
        "device": str(device),
        "eval_transform": str(build_transforms(input_size)[1]),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "note": "Test set chỉ dùng để đánh giá cuối. "
                "Không dùng để chọn epoch/hyperparameter.",

        # Metrics
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "macro_f1": float(f1_score(all_true, all_pred, average="macro",
                                   zero_division=0)),
        "loss": float(total_loss / total_samples),
        "per_class": classification_report(
            all_true, all_pred, target_names=class_names,
            output_dict=True, zero_division=0,
        ),
    }

    # 6. Lưu artifacts
    run_name = Path(checkpoint_path).parent.name
    out_root = Path(output_dir or config.get("output_dir", "artifacts"))
    pred_dir = out_root / "predictions" / run_name
    log_dir = out_root / "logs" / run_name
    pred_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    metrics["confusion_matrix_path"] = str(pred_dir / "confusion_matrix.npy")
    metrics["predictions_path"] = str(pred_dir / "test_predictions.npz")

    with open(log_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    np.savez_compressed(
        pred_dir / "test_predictions.npz",
        indices=all_indices,
        true_labels=all_true,
        pred_labels=all_pred,
        confidences=all_conf,
        logits=all_logits,
        class_names=np.array(class_names),
    )
    np.save(pred_dir / "confusion_matrix.npy", cm)

    # 7. Log
    print(f"[evaluate] Checkpoint: {checkpoint_path}")
    print("[evaluate] Split: CIFAR-10 official test (10.000 mẫu, không dùng để chọn model)")
    print(f"[evaluate] Device: {device} | input_size: {input_size} "
          f"| batch: {eval_cfg['batch_size']}")
    print(f"[evaluate] Accuracy={metrics['accuracy']:.4f} | "
          f"Macro-F1={metrics['macro_f1']:.4f} | Loss={metrics['loss']:.4f}")
    print(f"[evaluate] Đã lưu: {log_dir / 'test_metrics.json'}")
    print(f"[evaluate] Đã lưu: {pred_dir / 'test_predictions.npz'}")
    return metrics