# MODULE: data.py
#
# Vai trò: nạp CIFAR-10, tạo split train/validation có thể tái lập,
# định nghĩa transforms và cung cấp DataLoader cho train/validation/test.
# Không đặt training loop hoặc logic chọn checkpoint trong module này.

import json
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Mean/std ImageNet - dùng chung cho cả ConvNeXt-Tiny và ResNet18
# (đã kiểm tra bằng weights.transforms() của torchvision: hai bộ weights giống nhau)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INTERPOLATION = transforms.InterpolationMode.BILINEAR

DEFAULT_SPLIT_PATH = "data/splits/split_indices.json"


def build_transforms(input_size):
    """Tạo transform cho train (có augmentation) và cho val/test (không augmentation)."""
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),     # augmentation trên ảnh gốc 32x32
        transforms.RandomHorizontalFlip(),        # lật ngang ngẫu nhiên 50%
        transforms.Resize((input_size, input_size), interpolation=INTERPOLATION),
        transforms.ToTensor(),                    # PIL [0,255] -> tensor [3,H,W] trong [0,1]
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=INTERPOLATION),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def get_split_indices(labels, val_size, seed, split_path):
    """Chia 50.000 ảnh train thành train/val có phân tầng theo lớp.
    Nếu file split đã tồn tại thì đọc lại, để mọi lần chạy dùng đúng một split."""
    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            split = json.load(f)
        return split["train_indices"], split["val_indices"]

    all_indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=val_size,
        stratify=labels,       # giữ tỷ lệ 10 lớp giống nhau ở train và val
        random_state=seed,
    )
    train_idx = sorted(train_idx.tolist())
    val_idx = sorted(val_idx.tolist())

    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, "w") as f:
        json.dump({"seed": seed, "val_size": val_size,
                   "train_indices": train_idx, "val_indices": val_idx}, f)
    return train_idx, val_idx


def get_dataloaders(config):
    """Trả về train_loader, val_loader, test_loader, class_names."""
    root = config["data_root"]
    seed = config["seed"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    split_path = config.get("split_path", DEFAULT_SPLIT_PATH)

    train_tf, eval_tf = build_transforms(config["input_size"])

    # Hai object cùng đọc 50.000 ảnh train, nhưng gắn transform khác nhau:
    # một cái cho train (có augmentation), một cái cho validation (không augmentation)
    train_full_aug = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    train_full_eval = datasets.CIFAR10(root, train=True, download=False, transform=eval_tf)
    test_set = datasets.CIFAR10(root, train=False, download=True, transform=eval_tf)

    train_idx, val_idx = get_split_indices(
        train_full_aug.targets, config["val_size"], seed, split_path)

    train_set = Subset(train_full_aug, train_idx)
    val_set = Subset(train_full_eval, val_idx)

    # Generator cố định để thứ tự shuffle của train tái lập được theo seed
    g = torch.Generator()
    g.manual_seed(seed)
    pin = torch.cuda.is_available()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, generator=g)
    # Val/test: KHÔNG shuffle (để index khớp dữ liệu gốc), KHÔNG bỏ batch cuối
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin, drop_last=False)

    return train_loader, val_loader, test_loader, train_full_aug.classes


def get_data_info(config):
    """Metadata về dữ liệu/transforms để train.py ghi vào log của mỗi run (phục vụ tái lập)."""
    train_tf, eval_tf = build_transforms(config["input_size"])
    return {
        "dataset": "CIFAR-10 (torchvision)",
        "data_root": config["data_root"],
        "split_path": config.get("split_path", DEFAULT_SPLIT_PATH),
        "seed": config["seed"],
        "val_size": config["val_size"],
        "input_size": config["input_size"],
        "interpolation": INTERPOLATION.value,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "train_transform": str(train_tf),
        "eval_transform": str(eval_tf),
    }


def denormalize(images):
    """Đưa tensor đã normalize về lại [0,1] để vẽ ảnh. Nhận [3,H,W] hoặc [B,3,H,W]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (images * std + mean).clamp(0, 1)


if __name__ == "__main__":
    # Kiểm tra nhanh (Phase 2-3): đứng ở thư mục gốc project, chạy `python src/cifar_compare/data.py`
    config = {"data_root": "data/raw", "seed": 42, "val_size": 5000,
              "batch_size": 64, "num_workers": 0, "input_size": 224}
    train_loader, val_loader, test_loader, class_names = get_dataloaders(config)

    # 1. Dữ liệu gốc (trước transform): ảnh 32x32x3, kiểu uint8, giá trị 0-255
    raw = train_loader.dataset.dataset.data
    print("Ảnh gốc train: shape", raw.shape, "| dtype", raw.dtype,
          "| pixel min/max", raw.min(), raw.max())
    print("Tên lớp:", class_names)

    # 2. Số mẫu và phân bố lớp (phải ~4500/lớp ở train, 500/lớp ở val, 1000/lớp ở test)
    print("Số mẫu  train / val / test:",
          len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    all_targets = np.array(train_loader.dataset.dataset.targets)
    print("Phân bố lớp train:", np.bincount(all_targets[train_loader.dataset.indices]))
    print("Phân bố lớp val:  ", np.bincount(all_targets[val_loader.dataset.indices]))
    print("Phân bố lớp test: ", np.bincount(test_loader.dataset.targets))

    # 3. Train và val không được trùng ảnh nào
    overlap = set(train_loader.dataset.indices) & set(val_loader.dataset.indices)
    print("Số ảnh trùng giữa train và val:", len(overlap))

    # 4. Batch sau transform
    images, labels = next(iter(train_loader))
    print("Shape batch ảnh:", tuple(images.shape), "| shape nhãn:", tuple(labels.shape))
    print("Nhãn min/max:", labels.min().item(), labels.max().item())
    print("Giá trị pixel sau normalize min/max: %.2f / %.2f" % (images.min(), images.max()))
    print("Toàn bộ giá trị hữu hạn (không NaN/Inf):", torch.isfinite(images).all().item())
