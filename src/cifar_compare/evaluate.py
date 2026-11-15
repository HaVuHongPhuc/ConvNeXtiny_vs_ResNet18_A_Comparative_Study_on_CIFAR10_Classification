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

# MODULE: evaluate.py
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

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.cifar_compare.data import build_transforms
    from src.cifar_compare.models import build_model
    from src.cifar_compare.utils import resolve_device
else:
    from .data import build_transforms
    from .models import build_model
    from .utils import resolve_device

# --- Defensive import: data.py có thể đặt tên khác ---
try:
    if __package__ in (None, ""):
        from src.cifar_compare.data import get_test_loader  # kiểu cũ
    else:
        from .data import get_test_loader  # kiểu cũ
except ImportError:
    get_test_loader = None

try:
    if __package__ in (None, ""):
        from src.cifar_compare.data import get_dataloaders
    else:
        from .data import get_dataloaders
except ImportError:
    get_dataloaders = None

if get_test_loader is None and get_dataloaders is None:
    raise ImportError(
        "data.py phải export get_test_loader hoặc get_dataloaders. "
        "Hãy kiểm tra data.py."
    )


def _build_test_loader(eval_cfg):
    """Lấy test_loader bất kể data.py đặt tên/return kiểu gì."""
    if get_test_loader is not None:
        return get_test_loader(eval_cfg)
    result = get_dataloaders(eval_cfg)
    if isinstance(result, tuple):
        if len(result) >= 3:
            # get_dataloaders returns (train, val, test, class_names).
            return result[2]
        raise ValueError(
            "get_dataloaders() phải trả ít nhất (train, val, test)."
        )
    if isinstance(result, dict):
        for key in ("test", "test_loader", "test_dataloader"):
            if key in result:
                return result[key]
        raise KeyError(
            f"get_dataloaders() trả dict nhưng thiếu key test. "
            f"Các key có: {list(result.keys())}"
        )
    # trả về chính nó (loader trực tiếp)
    return result


def evaluate_checkpoint(checkpoint_path, config, output_dir=None, device=None):
    device = device or resolve_device(config.get("device", "auto"))
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = ckpt["model_name"]
    num_classes = int(ckpt["num_classes"])
    class_names = list(ckpt["class_names"])
    input_size = int(ckpt["config"]["input_size"])

    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    eval_cfg = dict(config)
    eval_cfg["input_size"] = input_size
    eval_cfg["batch_size"] = config.get("eval_batch_size", 64)
    test_loader = _build_test_loader(eval_cfg)

    official = CIFAR10(config["data_root"], train=False, download=False).classes
    if class_names != official:
        raise ValueError(
            f"Class order trong checkpoint không khớp CIFAR-10.\n"
            f"Checkpoint: {class_names}\nCIFAR-10: {official}"
        )

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

    cm = confusion_matrix(all_true, all_pred)
    metrics = {
        "checkpoint": str(checkpoint_path),
        "model": model_name,
        "num_classes": num_classes,
        "class_names": class_names,
        "num_samples": int(total_samples),
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
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "macro_f1": float(f1_score(all_true, all_pred, average="macro",
                                   zero_division=0)),
        "loss": float(total_loss / total_samples),
        "per_class": classification_report(
            all_true, all_pred, target_names=class_names,
            output_dict=True, zero_division=0,
        ),
    }

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

    print(f"[evaluate] Checkpoint: {checkpoint_path}")
    print("[evaluate] Split: CIFAR-10 official test (10.000 mẫu)")
    print(f"[evaluate] Device: {device} | input_size: {input_size} "
          f"| batch: {eval_cfg['batch_size']}")
    print(f"[evaluate] Accuracy={metrics['accuracy']:.4f} | "
          f"Macro-F1={metrics['macro_f1']:.4f} | Loss={metrics['loss']:.4f}")
    print(f"[evaluate] Đã lưu: {log_dir / 'test_metrics.json'}")
    print(f"[evaluate] Đã lưu: {pred_dir / 'test_predictions.npz'}")
    return metrics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the CIFAR-10 test set"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best.pt, e.g. artifacts/checkpoints/resnet18/best.pt")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8-sig") as f:
        _cfg = json.load(f)
    evaluate_checkpoint(args.checkpoint, _cfg, output_dir=args.output_dir)