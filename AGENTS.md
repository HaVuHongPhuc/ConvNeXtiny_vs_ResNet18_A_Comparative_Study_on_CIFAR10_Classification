# Project Context for Codex and Contributors

## Mục đích của file này

File này là bộ nhớ bền vững của project. Khi mở lại project sau khi clear hoặc tóm gọn session, hãy đọc file này trước khi sửa nội dung. Sau mọi thay đổi đáng kể đối với project — thêm, sửa, đổi tên hoặc xóa code, cấu hình, dữ liệu mẫu hay tài liệu — phải cập nhật phần liên quan trong `AGENTS.md` cùng lượt làm việc. Giữ thông tin trong file khớp với trạng thái thực tế; phân biệt rõ nội dung đã có với kế hoạch chưa triển khai.

## Mục tiêu project

Đồ án môn Deep Learning: tìm hiểu ConvNeXt như một ConvNet hiện đại, sau đó dùng ConvNeXt-Tiny pretrained để phân loại CIFAR-10 và so sánh với ResNet18 pretrained theo:

- Chất lượng phân loại.
- Chi phí tính toán.
- Các lỗi phân loại và những lớp thường bị nhầm.

Ngôn ngữ tài liệu và hướng dẫn: tiếng Việt. Đối tượng đọc là sinh viên đã học nền tảng neural network/MLP nhưng có thể mới bắt đầu với CNN và PyTorch. Ưu tiên giải thích từ nguyên lý, sau đó mới đưa ví dụ và bài tập.

## Trạng thái hiện tại

- Project hiện có tài liệu DOCX và scaffold thư mục/code placeholder; chưa có logic mã nguồn huấn luyện, cấu hình thực nghiệm hoàn chỉnh, dữ liệu CIFAR-10 tải về, checkpoint hoặc kết quả chạy.
- Các file Python trong `src/cifar_compare/` chỉ có comment hướng dẫn vai trò/cách triển khai, chưa có logic thực thi. `configs/default.json` hiện là `{}`; `requirements.txt` chưa có danh sách dependency thực sự.
- `PROJECT_STRUCTURE.md` ghi vai trò từng file và thứ tự triển khai. Cấu trúc đã được tạo nhưng tính năng bên trong chưa được code.
- Pipeline trong tài liệu đặc tả là kế hoạch triển khai, không phải pipeline đã được chạy hoặc xác nhận.
- Chưa chốt các quyết định thực nghiệm: input size cuối cùng, transforms/augmentation, seed và split, chiến lược fine-tuning, optimizer/hyperparameters, epoch/batch size, thiết bị đo và ngân sách thời gian.
- Không bịa số liệu kết quả. Chỉ ghi accuracy, FLOPs, latency, memory hoặc nhận xét mô hình sau khi có run thực tế và điều kiện đo kèm theo.

## Tài liệu trong project

Các file dưới đây là tài liệu DOCX hiện có:

- `01_Ly_thuyet_kien_truc_ConvNeXt.docx`: lý thuyết ConvNeXt, cấu trúc stage/block, giải thích các thành phần, ví dụ shape và bài tập.
- `02_Nen_tang_ly_thuyet_ConvNet.docx`: nền tảng ConvNet nối từ MLP, tensor ảnh, convolution, shape/parameters, normalization, residual learning, huấn luyện và bài tập.
- `03_ConvNeXt_nhu_mot_ConvNet_hien_dai.docx`: tổng hợp ConvNet → ResNet → ConvNeXt, transfer learning CIFAR-10, giao thức so sánh và phân tích lỗi.
- `04_Dac_ta_project_va_pipeline_CIFAR10.docx`: đặc tả mục tiêu, phạm vi, pipeline 9 phase, cấu trúc code đề xuất, cấu hình, rủi ro và tiêu chí thành công.

Các file DOCX không có số thứ tự đầu tên cũng đang hiện diện: `ConvNet_co_ban.docx`, `ConvNeXt_nhu_mot_ConvNet_hien_dai.docx`, `kien_truc_ConvNeXt.docx`. Đây có thể là các bản cũ/khác tên trùng nội dung. Không tự xóa chúng. Trước khi sửa DOCX, kiểm tra danh sách file và xác định đúng bản chuẩn; các file đánh số `01`–`04` là các bản được cập nhật trong phiên gần đây và nên được coi là bản chuẩn trừ khi người dùng chỉ định khác.

## Scaffold code đã tạo

Cây dưới đây hiện tồn tại; các file Python chỉ có comment hướng dẫn. Tham khảo `PROJECT_STRUCTURE.md` để biết hướng dẫn thực hiện chi tiết:

```text
project-root/
├── AGENTS.md
├── README.md
├── PROJECT_STRUCTURE.md
├── .gitignore
├── requirements.txt                 # Placeholder, chưa pin dependency
├── configs/
│   └── default.json                 # Placeholder {}
├── data/
│   ├── raw/.gitkeep                 # Chưa tải dữ liệu
│   └── splits/.gitkeep
├── src/
│   └── cifar_compare/
│       ├── __init__.py              # Python package
│       ├── data.py                  # CIFAR-10, split, transforms, DataLoaders
│       ├── models.py                # ConvNeXt-Tiny/ResNet18 pretrained + 10-class head
│       ├── train.py                 # Training, validation, checkpoint selection
│       ├── evaluate.py              # Test metrics and predictions
│       ├── profiling.py             # Params, FLOPs/MACs, latency, throughput, memory
│       ├── visualize.py             # Learning curves, confusion matrix, error images
│       └── utils.py                 # Seed, config, logging, shared helpers
├── artifacts/
│   ├── checkpoints/                 # Model checkpoints
│   ├── logs/                        # Resolved config and epoch metrics
│   ├── predictions/                 # Predictions used for error analysis
│   └── figures/                     # Generated charts and figures
├── reports/.gitkeep
└── [tài liệu DOCX học tập và đặc tả]
```

Use one shared data/training/evaluation pipeline for both models and select architecture through configuration or CLI. Avoid duplicating training loops by model. Keep large datasets and checkpoints out of Git unless explicitly required. Use configurable project-relative paths, not paths tied to one machine. Các bước triển khai chi tiết và vai trò từng file nằm trong `PROJECT_STRUCTURE.md`.

## Pipeline theo phase

### Phase 0 — Chốt giao thức

Xác định split, seed, input resolution, preprocessing, augmentation, fine-tuning policy, epochs, optimizer, checkpoint criterion, metrics and profiling method before long runs. A reasonable initial split is 45,000 train / 5,000 validation derived stratified from the 50,000 training images; reserve the canonical 10,000-image test set for final evaluation only. Record any different choice.

### Phase 1 — Environment and configuration

Record Python, PyTorch, torchvision, CUDA/device versions; use a resolved config; seed Python/NumPy/Torch/DataLoader where applicable; save run metadata and logs.

### Phase 2 — Dataset and split

Load CIFAR-10; check sample counts, `[3,32,32]` image shape, pixel range, labels and class mapping; create reproducible stratified train/validation split. Never select hyperparameters using the test set.

### Phase 3 — Transforms and DataLoaders

Define train augmentation and deterministic validation/test transforms. Use and record the selected pretrained weights' expected normalization/interpolation. Apply equivalent policy to both models; document any unavoidable transform difference. Check batch shapes and finite values.

### Phase 4 — Model setup and smoke check

Load exact torchvision weights enums for ConvNeXt-Tiny and ResNet18; replace final classifier with 10 outputs; check `[B,10]` logits, finite CrossEntropy loss, gradients, and intended trainable parameters on a small batch before full training.

### Phase 5 — Training

Use a shared loop. Track train and validation metrics by epoch. Choose the best checkpoint using validation only. Save model/optimizer state, epoch, resolved config, seed and metrics. Keep training budget/protocol comparable; disclose model-specific learning rates if used.

### Phase 6 — Final test and compute profile

After selecting checkpoints, evaluate each on the held-out test set and report accuracy, macro-F1, per-class precision/recall, loss and confusion matrix. Measure parameters and FLOPs/MACs at a stated input resolution. Measure latency/throughput on the same device, dtype, batch size and eval mode, with warm-up and correct GPU synchronization; specify whether data loading/preprocessing is included.

### Phase 7 — Error analysis

Inspect confusion matrices and representative errors with true/predicted labels and confidence. Discuss observations separately from hypotheses. If test errors inspire a new tuning decision, treat it as a new experiment and use validation for that decision.

### Phase 8 — Report and reproducibility

Provide paired tables/plots, method details, compute conditions, limitations, README commands and artifacts. Report the exact weights, versions, split, transforms, training settings and hardware. Limit conclusions to the measured configuration.

## Experimental principles

- CIFAR-10 has 60,000 RGB 32×32 images in 10 classes: 50,000 train and 10,000 test. Source: <https://www.cs.toronto.edu/~kriz/cifar.html>.
- Use exact torchvision weight enums and associated metadata/transforms; APIs can vary by version. Official docs: <https://docs.pytorch.org/vision/stable/models/convnext.html> and <https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html>.
- ConvNeXt is a pure ConvNet architecture presented in “A ConvNet for the 2020s”: <https://doi.org/10.1109/CVPR52688.2022.01167>.
- ResNet residual learning reference: <https://doi.org/10.1109/CVPR.2016.90>.
- Accuracy alone is not sufficient: include macro-F1/per-class behavior, compute measures and qualitative error analysis.
- FLOPs/MACs do not equal wall-clock speed. State tools/conventions and measure latency under controlled conditions.
- Do not claim one architecture is universally better based on a single split/run. If feasible, add multiple seeds and report mean plus variation.

## Working rules for future changes

1. Read this file and inspect the current workspace before editing.
2. Follow the user's latest request; do not assume a proposed structure or phase has already been implemented.
3. When adding, editing, renaming or deleting project files, update this file in the same work session: file map, actual structure, pipeline/status, configuration or known limitations as relevant.
4. Do not delete or overwrite user data, datasets, checkpoints, results or older documents without a clear request. Inspect target files first.
5. Do not run long training jobs, background monitors or repeated repair/test loops unless the user explicitly asks. Prefer making the requested change and giving the user precise commands plus expected outputs when execution is needed.
6. Do not expose secrets in code, logs, documentation or prompts.
7. Do not describe paper results as results of this project. Separate published benchmark findings from local experiment outcomes.
8. Keep documents and code in Vietnamese where user-facing explanation is needed; preserve standard English names for APIs, metrics and architecture components.

## Update record

- 2026-10-06: Created this project context file. Recorded current DOCX-only state, proposed code structure, research scope and nine-phase experimental pipeline.
- 2026-10-06: Created scaffold only: directories, Python placeholders containing guidance comments, `{}` config, README, `.gitignore` and `PROJECT_STRUCTURE.md`. No model/data/training logic was added or run.
- 2026-10-06: Added Vietnamese role and implementation notes to each Python placeholder. They contain comments only; no executable implementation was added.
- 2026-10-06: Expanded README with annotated scaffold, role and implementation guidance for every file/folder, and step-by-step implementation order. Python modules remain comments-only.
- 2026-10-06: Updated `.gitignore` to ignore `AGENTS.py` at any depth and all `*.docx`; documented the rule in README. The workspace is not currently a Git repository, so tracked status/push behavior could not be inspected.
