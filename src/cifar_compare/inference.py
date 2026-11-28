"""Nạp checkpoint chính thức và dự đoán một ảnh trong giao diện demo."""

import csv
import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import torch
from torch.torch_version import TorchVersion

from .data import build_transforms
from .models import build_model


MODEL_NAMES = {"resnet18": "ResNet18", "convnext_tiny": "ConvNeXt-Tiny"}
CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
_CHECKPOINT_LOAD_LOCK = Lock()


@dataclass(frozen=True)
class CheckpointSpec:
    model_name: str
    run_name: str
    input_size: int
    epoch: int
    seed: int
    checkpoint: Path
    sha256: str


def read_demo_runs(root):
    """Đọc cặp run đã chốt từ manifest và giới hạn checkpoint trong project."""
    root = Path(root).resolve()
    checkpoint_root = root / "artifacts" / "checkpoints"
    runs = {}
    with (root / "reports" / "run_manifest.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            name = row["model"]
            if name not in MODEL_NAMES:
                continue
            if name in runs:
                raise ValueError(f"Manifest có nhiều run cho {MODEL_NAMES[name]}.")
            path = (root / row["best_checkpoint"]).resolve()
            run_name = row["run_name"]
            if (
                not run_name
                or Path(run_name).name != run_name
                or not path.is_relative_to(checkpoint_root)
                or path != checkpoint_root / run_name / "best.pt"
            ):
                raise ValueError(f"Đường dẫn best.pt không khớp manifest cho {name}.")
            digest = row["best_checkpoint_sha256"].lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"SHA-256 của {name} không hợp lệ.")
            runs[name] = CheckpointSpec(
                name, run_name, int(row["input_size"]), int(row["best_epoch"]),
                int(row["seed"]), path, digest,
            )
    if set(runs) != set(MODEL_NAMES):
        raise ValueError("Manifest cần có một run cho ResNet18 và một run cho ConvNeXt-Tiny.")
    if len({run.input_size for run in runs.values()}) != 1 or runs["resnet18"].input_size < 32:
        raise ValueError("Hai checkpoint cần dùng cùng kích thước đầu vào hợp lệ.")
    return runs


def load_demo_model(run, device):
    """Xác minh file từ manifest rồi dựng model và nạp state_dict."""
    with run.checkpoint.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
        if digest != run.sha256:
            raise ValueError(f"SHA-256 của {MODEL_NAMES[run.model_name]} không khớp manifest.")
        file.seek(0)
        # train.py lưu phiên bản PyTorch dưới dạng TorchVersion trong checkpoint hiện có.
        try:
            with _CHECKPOINT_LOAD_LOCK, torch.serialization.safe_globals([TorchVersion]):
                checkpoint = torch.load(file, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError as error:
            raise ValueError(f"Không đọc được checkpoint {MODEL_NAMES[run.model_name]}.") from error

    if (
        checkpoint["model_name"] != run.model_name
        or int(checkpoint["epoch"]) != run.epoch
        or int(checkpoint["config"]["seed"]) != run.seed
        or checkpoint["config"]["run_name"] != run.run_name
        or int(checkpoint["config"]["input_size"]) != run.input_size
        or int(checkpoint["num_classes"]) != len(CLASS_NAMES)
        or tuple(checkpoint["class_names"]) != CLASS_NAMES
    ):
        raise ValueError(f"Metadata của {MODEL_NAMES[run.model_name]} không khớp manifest hoặc CIFAR-10.")
    model = build_model(run.model_name, num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def prepare_demo_image(image, input_size, device):
    """Dùng chính transform validation/test và thêm chiều batch."""
    _, evaluation_transform = build_transforms(input_size)
    return evaluation_transform(image.convert("RGB")).unsqueeze(0).to(device)


@torch.inference_mode()
def predict_logits(model, batch):
    """Trả về 10 logit thô theo đúng thứ tự lớp CIFAR-10."""
    logits = model(batch)
    if tuple(logits.shape) != (1, len(CLASS_NAMES)):
        raise ValueError("Đầu ra của mô hình cần có dạng [1, 10].")
    logits = logits[0].cpu()
    if not torch.isfinite(logits).all():
        raise ValueError("Mô hình trả về logit không hữu hạn.")
    return tuple(logits.tolist())
