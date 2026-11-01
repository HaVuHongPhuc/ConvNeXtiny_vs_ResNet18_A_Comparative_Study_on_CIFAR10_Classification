"""Training loop dùng chung cho ConvNeXt-Tiny và ResNet18.

Chạy từ thư mục gốc sau khi môi trường và dữ liệu đã sẵn sàng:
python -m src.cifar_compare.train --model resnet18
python -m src.cifar_compare.train --model convnext_tiny
"""

import argparse
import csv
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision

from .data import get_data_info, get_dataloaders
from .models import build_model, count_parameters
from .utils import load_config, resolve_device, set_seed


DEFAULTS = {
    "seed": 42,
    "model": "convnext_tiny",
    "weights": "IMAGENET1K_V1",
    "pretrained": True,
    "freeze_backbone": False,
    "num_classes": 10,
    "data_root": "data/raw",
    "split_path": "data/splits/split_indices.json",
    "val_size": 5000,
    "input_size": 224,
    "batch_size": 32,
    "num_workers": 4,
    "epochs": 10,
    "learning_rate": 0.0001,
    "head_learning_rate": 0.001,
    "weight_decay": 0.05,
    "scheduler": "cosine",
    "amp": True,
    "early_stopping_patience": 5,
    "deterministic": False,
    "device": "auto",
    "output_dir": "artifacts",
    "run_name": None,
}


def validate_config(config):
    if config["model"] not in ("convnext_tiny", "resnet18", "convnext", "convnext-tiny"):
        raise ValueError("model phải là convnext_tiny hoặc resnet18.")
    for key in ("epochs", "batch_size", "input_size"):
        if int(config[key]) < 1:
            raise ValueError(f"{key} phải lớn hơn 0.")
    if int(config["num_classes"]) < 2 or int(config["val_size"]) < 1:
        raise ValueError("num_classes phải >= 2 và val_size phải > 0.")
    if config["scheduler"] not in ("cosine", "none"):
        raise ValueError("scheduler chỉ nhận cosine hoặc none.")


def build_optimizer(model, config):
    """AdamW; dùng learning rate riêng cho backbone và classification head."""
    backbone, head = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (head if name.startswith(("classifier.", "fc.")) else backbone).append(parameter)

    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": float(config["learning_rate"])})
    if head:
        groups.append({"params": head, "lr": float(config["head_learning_rate"])})
    if not groups:
        raise ValueError("Không có tham số trainable trong model.")
    return torch.optim.AdamW(groups, weight_decay=float(config["weight_decay"]))


def run_epoch(model, loader, device, optimizer=None, scaler=None, amp_enabled=False):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        batch_count = labels.size(0)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        loss_sum += loss.detach().item() * batch_count
        correct += (logits.detach().argmax(1) == labels).sum().item()
        count += batch_count

    if count == 0:
        raise RuntimeError("DataLoader không có mẫu.")
    return {"loss": loss_sum / count, "accuracy": correct / count, "samples": count}


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def save_checkpoint(path, model, optimizer, scheduler, epoch, config,
                    best_accuracy, best_loss, class_names):
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    checkpoint = {
        "epoch": epoch,
        "model_name": config["model"],
        "weights": config["weights"] if config["pretrained"] else None,
        "num_classes": config["num_classes"],
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_val_accuracy": best_accuracy,
        "best_val_loss": best_loss,
        "class_names": list(class_names),
        "config": config,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train_model(model_name, config, train_loader, val_loader, class_names, device=None):
    """Huấn luyện một model từ loaders đã tạo bởi data.get_dataloaders."""
    config = dict(config)
    config["model"] = model_name
    validate_config(config)
    set_seed(int(config["seed"]), bool(config["deterministic"]))
    device = device or resolve_device(config["device"])

    model = build_model(
        model_name, int(config["num_classes"]), bool(config["pretrained"]),
        config["weights"], bool(config["freeze_backbone"]),
    ).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))
        if config["scheduler"] == "cosine" else None
    )
    amp_enabled = bool(config["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    run_name = config.get("run_name") or (
        f"{config['model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_name)).strip("._")
    if not run_name:
        raise ValueError("run_name không được rỗng.")

    root = Path(config["output_dir"])
    checkpoint_dir = root / "checkpoints" / run_name
    log_dir = root / "logs" / run_name
    # Không ghi đè run cũ; đổi run_name nếu thư mục đã tồn tại.
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)

    resolved = dict(config)
    resolved.update({
        "device_resolved": str(device),
        "amp_enabled": amp_enabled,
        "total_parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "data_info": get_data_info(config),
    })
    write_json(log_dir / "config.json", resolved)

    history = []
    best_accuracy, best_loss = -1.0, float("inf")
    stale_epochs = 0
    patience = int(config["early_stopping_patience"])
    metrics_path = log_dir / "metrics.csv"

    logging.info("Model=%s device=%s AMP=%s", model_name, device, amp_enabled)
    logging.info("Parameters total=%d trainable=%d",
                 resolved["total_parameters"], resolved["trainable_parameters"])

    for epoch in range(1, int(config["epochs"]) + 1):
        learning_rate_used = optimizer.param_groups[0]["lr"]
        started = time.perf_counter()
        train_metrics = run_epoch(
            model, train_loader, device, optimizer=optimizer,
            scaler=scaler, amp_enabled=amp_enabled,
        )
        val_metrics = run_epoch(model, val_loader, device, amp_enabled=amp_enabled)
        if scheduler:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate_backbone": learning_rate_used,
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(row)
        with metrics_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)

        logging.info(
            "Epoch %d/%d train loss=%.4f acc=%.4f val loss=%.4f acc=%.4f time=%.1fs",
            epoch, config["epochs"], row["train_loss"], row["train_accuracy"],
            row["val_loss"], row["val_accuracy"], row["epoch_seconds"],
        )

        improved = (
            val_metrics["accuracy"] > best_accuracy
            or (val_metrics["accuracy"] == best_accuracy and val_metrics["loss"] < best_loss)
        )
        if improved:
            best_accuracy, best_loss = val_metrics["accuracy"], val_metrics["loss"]
            stale_epochs = 0
            save_checkpoint(
                checkpoint_dir / "best.pt", model, optimizer, scheduler, epoch,
                resolved, best_accuracy, best_loss, class_names,
            )
        else:
            stale_epochs += 1

        save_checkpoint(
            checkpoint_dir / "last.pt", model, optimizer, scheduler, epoch,
            resolved, best_accuracy, best_loss, class_names,
        )
        if patience > 0 and stale_epochs >= patience:
            logging.info("Early stopping: validation chưa cải thiện trong %d epoch.", patience)
            break

    best_row = next(
        row for row in history
        if row["val_accuracy"] == best_accuracy and row["val_loss"] == best_loss
    )
    summary = {
        "model": model_name,
        "best_epoch": best_row["epoch"],
        "best_val_accuracy": best_accuracy,
        "best_val_loss": best_loss,
        "epochs_completed": len(history),
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "last_checkpoint": str(checkpoint_dir / "last.pt"),
        "metrics_csv": str(metrics_path),
        "device": str(device),
    }
    write_json(log_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained CIFAR-10 classifier.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--model", choices=("convnext_tiny", "resnet18"))
    parser.add_argument("--device", help="auto, cuda, cuda:0 hoặc cpu")
    parser.add_argument("--run-name")
    args = parser.parse_args()

    config = load_config(args.config, DEFAULTS)
    if args.model:
        config["model"] = args.model
    if args.device:
        config["device"] = args.device
    if args.run_name:
        config["run_name"] = args.run_name
    validate_config(config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    device = resolve_device(config["device"])
    set_seed(int(config["seed"]), bool(config["deterministic"]))
    train_loader, val_loader, _test_loader, class_names = get_dataloaders(config)
    summary = train_model(
        config["model"], config, train_loader, val_loader, class_names, device
    )
    logging.info("Training hoàn tất: %s", summary)


if __name__ == "__main__":
    main()

