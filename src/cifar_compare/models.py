"""Model factory cho ConvNeXt-Tiny và ResNet18 trên CIFAR-10."""

from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights, ResNet18_Weights, convnext_tiny, resnet18,
)

_BUILDERS = {
    "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights),
    "resnet18": (resnet18, ResNet18_Weights),
}
_ALIASES = {"convnext": "convnext_tiny", "convnext-tiny": "convnext_tiny"}


def build_model(model_name, num_classes=10, pretrained=True,
                weights_name="IMAGENET1K_V1", freeze_backbone=False):
    """Tạo model, thay head thành num_classes và tùy chọn đóng băng backbone."""
    name = _ALIASES.get(str(model_name).lower().strip(),
                        str(model_name).lower().strip())
    if name not in _BUILDERS:
        raise ValueError("model_name phải là convnext_tiny hoặc resnet18.")
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes phải là số nguyên >= 2.")

    builder, weights_enum = _BUILDERS[name]
    if pretrained:
        try:
            weights = getattr(weights_enum, weights_name or "DEFAULT")
        except AttributeError as exc:
            valid = ", ".join(item.name for item in weights_enum)
            raise ValueError(f"Weights không hợp lệ: {weights_name!r}. Chọn: {valid}") from exc
    else:
        weights = None

    model = builder(weights=weights)
    if name == "convnext_tiny":
        old_head = model.classifier[-1]
        model.classifier[-1] = nn.Linear(old_head.in_features, num_classes)
        head = model.classifier
    else:
        old_head = model.fc
        model.fc = nn.Linear(old_head.in_features, num_classes)
        head = model.fc

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in head.parameters():
            parameter.requires_grad = True
    return model


def count_parameters(model, trainable_only=False):
    """Đếm tổng tham số hoặc chỉ tham số đang được huấn luyện."""
    return sum(p.numel() for p in model.parameters()
               if not trainable_only or p.requires_grad)
