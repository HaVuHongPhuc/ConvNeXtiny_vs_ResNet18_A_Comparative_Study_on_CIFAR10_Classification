# MODULE: profiling.py
#
# Vai trò: đo chi phí của model, gồm số tham số, FLOPs/MACs, latency,
# throughput và peak memory nếu thiết bị hỗ trợ.
#
# Hướng dẫn triển khai:
# 1. Ghi input size, device, dtype, batch size và công cụ/định nghĩa FLOPs.
# 2. Đặt model ở eval mode và warm-up trước khi đo thời gian.
# 3. Nếu dùng CUDA, đồng bộ thiết bị trước/sau vùng đo.
# 4. Lặp nhiều lần, báo cách tổng hợp và nói rõ có tính data loading không.
# Không xem FLOPs/MACs là thay thế cho latency thực tế.

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision

from .models import build_model, count_parameters
from .utils import resolve_device


# FLOPs tham chiếu từ torchvision classification_table (ImageNet-1K, 224×224, batch=1)
TORCHVISION_FLOPS = {
    "convnext_tiny": 4.46,
    "resnet18": 1.81,
}
TORCHVISION_PARAMS = {
    "convnext_tiny": 28.6,
    "resnet18": 11.7,
}


def measure_latency(model, input_size, batch_size, device,
                    warmup=20, repeats=100, dtype=torch.float32):
    """Đo latency forward pass, trả về mean/std/p50/p95/p99 (ms)."""
    model.eval()
    dummy = torch.randn(batch_size, 3, input_size, input_size,
                        device=device, dtype=dtype)

    with torch.no_grad():
        # Warm-up
        for _ in range(warmup):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        timings = []
        for _ in range(repeats):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(dummy)
                end.record()
                torch.cuda.synchronize()
                timings.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                model(dummy)
                timings.append((time.perf_counter() - t0) * 1000)

    timings = np.array(timings, dtype=np.float64)
    return {
        "mean_ms": float(timings.mean()),
        "std_ms": float(timings.std()),
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "p99_ms": float(np.percentile(timings, 99)),
        "min_ms": float(timings.min()),
        "max_ms": float(timings.max()),
    }


def measure_peak_memory(model, input_size, batch_size, device,
                        dtype=torch.float32):
    """Đo peak CUDA memory khi forward (chỉ CUDA)."""
    if device.type != "cuda":
        return None
    model.eval()
    dummy = torch.randn(batch_size, 3, input_size, input_size,
                        device=device, dtype=dtype)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        model(dummy)
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return float(peak_bytes / (1024 ** 2))


def profile_model(model_name, config, device=None, dtype=torch.float32):
    """Đo đầy đủ cho một model, trả về dict kết quả và lưu JSON."""
    device = device or resolve_device(config.get("device", "auto"))
    input_size = int(config.get("input_size", 224))
    batch_size = int(config.get("batch_size", 32))
    warmup = int(config.get("warmup", 20))
    repeats = int(config.get("repeats", 100))

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"[profiling] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[profiling] VRAM: {props.total_memory / 1e9:.1f} GB "
              f"| sm_{props.major}{props.minor}")

    model = build_model(
        model_name, num_classes=10, pretrained=True,
        weights_name=config.get("weights", "IMAGENET1K_V1"),
    )
    model.to(device=device, dtype=dtype)

    latency = measure_latency(
        model, input_size, batch_size, device,
        warmup=warmup, repeats=repeats, dtype=dtype,
    )
    peak_mb = measure_peak_memory(model, input_size, batch_size, device, dtype=dtype)

    result = {
        "model": model_name,
        "input_size": input_size,
        "batch_size": batch_size,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "timestamp": datetime.now().isoformat(timespec="seconds"),

        # Params
        "params_m": TORCHVISION_PARAMS.get(model_name),
        "params_total_actual": int(count_parameters(model)),

        # FLOPs — định nghĩa rõ ràng
        "flops_gflops": TORCHVISION_FLOPS.get(model_name),
        "flops_definition": {
            "source": "torchvision classification_table (ImageNet-1K)",
            "reference_url": "https://pytorch.org/vision/stable/models.html",
            "input": "224×224, batch=1, FP32",
            "definition": "GFLOPs = 10^9 FLOPs. Giá trị tham chiếu từ "
                          "torchvision, không đo trực tiếp trên model đã thay head 10 lớp.",
            "note": "FLOPs của head 10 lớp khác không đáng kể so với backbone.",
        },

        # Latency — cách tổng hợp + phạm vi
        "latency": latency,
        "throughput_samples_per_sec": float(batch_size / (latency["mean_ms"] / 1000)),
        "latency_aggregation": (
            f"mean/std/p50/p95/p99/min/max của {repeats} lần đo sau "
            f"{warmup} lần warm-up; mỗi lần dùng torch.cuda.Event (CUDA) "
            f"hoặc time.perf_counter (CPU)."
        ),
        "latency_scope": "Chỉ forward pass của model, không gồm data loading, "
                         "không gồm khởi tạo model.",

        # Memory
        "peak_memory_mb": peak_mb,
        "peak_memory_scope": (
            "CUDA max_memory_allocated sau reset_peak_memory_stats, "
            "chỉ áp dụng khi device là CUDA."
        ) if peak_mb is not None else None,

        # Backend state
        "cudnn_benchmark": (torch.backends.cudnn.benchmark
                            if device.type == "cuda" else None),
        "tf32_matmul": (torch.backends.cuda.matmul.allow_tf32
                        if device.type == "cuda" else None),
        "tf32_cudnn": (torch.backends.cudnn.allow_tf32
                       if device.type == "cuda" else None),

        "note": "FLOPs là giá trị tham chiếu, KHÔNG thay thế latency thực tế. "
                "Luôn báo cả hai khi so sánh mô hình.",
    }

    out_dir = Path(config.get("output_dir", "artifacts")) / "profiling"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_profile.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"[profiling] {model_name}: "
        f"Latency mean={latency['mean_ms']:.2f}ms "
        f"p95={latency['p95_ms']:.2f}ms "
        f"| Throughput={result['throughput_samples_per_sec']:.1f} samples/s "
        f"| Peak Mem={peak_mb:.0f}MB" if peak_mb is not None else
        f"[profiling] {model_name}: Latency mean={latency['mean_ms']:.2f}ms"
    )
    print(f"[profiling] Đã lưu: {out_path}")
    return result


def run_all_profiling(config):
    """Chạy profiling cho cả hai model với cùng protocol, lưu summary."""
    results = {}
    for name in ("resnet18", "convnext_tiny"):
        results[name] = profile_model(name, config)

    out_dir = Path(config.get("output_dir", "artifacts")) / "profiling"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": {
            "input_size": results["resnet18"]["input_size"],
            "batch_size": results["resnet18"]["batch_size"],
            "device": results["resnet18"]["device"],
            "dtype": results["resnet18"]["dtype"],
            "warmup": int(config.get("warmup", 20)),
            "repeats": int(config.get("repeats", 100)),
            "note": "Cùng protocol cho cả hai model. "
                    "Latency không gồm data loading.",
        },
        "results": results,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[profiling] Summary -> {out_dir / 'summary.json'}")
    return results