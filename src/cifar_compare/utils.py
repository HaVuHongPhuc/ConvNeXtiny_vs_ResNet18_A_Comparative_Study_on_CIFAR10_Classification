"""Tiện ích dùng chung cho config, seed và chọn thiết bị."""

import json
import random
from pathlib import Path

import numpy as np
import torch


def load_config(config_path, defaults):
    """Đọc JSON và ghi đè giá trị mặc định bằng các trường trong file."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8-sig") as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        raise ValueError("Cấu hình phải là JSON object.")
    config = dict(defaults)
    config.update(loaded)
    return config


def set_seed(seed, deterministic=False):
    """Đặt seed cho Python, NumPy và PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)


def resolve_device(name="auto"):
    """Chọn auto/cuda/cuda:0/cpu và kiểm tra CUDA có khả dụng."""
    name = str(name).lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Đã chọn CUDA nhưng torch.cuda.is_available() là False.")
    return device
