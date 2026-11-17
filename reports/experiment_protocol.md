# Protocol thực nghiệm ConvNeXt-Tiny và ResNet18 trên CIFAR-10

Mã protocol: **cifar10-baseline-v1**. Ngày ghi nhận: **16/11/2026**.

**Trạng thái ngày 17/11/2026: Hai model đã hoàn tất training chính thức và reload checkpoint PASS (T018–T021). ResNet18 best epoch9/validation96,46%; ConvNeXt-Tiny best epoch10/validation98,50%. Chưa đánh giá test hoặc benchmark. T022/T023 đã chốt cặp best.pt tại run_manifest.csv theo baseline một seed42; tiếp theo bàn giao Quy (T024).**
Phúc đã cung cấp xác nhận rằng Khang đồng ý cấu hình dữ liệu và Quy đồng ý metrics/điều kiện đo được hỏi trong cuộc trò chuyện. Các xác nhận và phạm vi cụ thể được lưu trong [protocol_confirmations.json](protocol_confirmations.json). Assistant không tự gửi tin nhắn cho thành viên khác.

Cấu hình máy đọc là [configs/default.json](../configs/default.json). Hai mô hình dùng cùng cấu hình này; chọn model bằng tham số `--model` hoặc biến model trong notebook huấn luyện sau này. Việc ghi cấu hình không khởi chạy training.

## 1. Mục tiêu và giới hạn kết luận

So sánh hai mô hình pretrained về chất lượng phân loại, chi phí tính toán và đặc điểm lỗi trên CIFAR-10. So sánh này phản ánh hai kiến trúc cùng các bộ trọng số pretrained tương ứng, không cô lập hoàn toàn ảnh hưởng của kiến trúc khỏi lịch sử pretraining.

Vòng baseline dùng **một seed 42**, mỗi model một run chính thức hợp lệ; không tuning riêng trong vòng đầu. Không báo độ lệch chuẩn giữa các seed khi chỉ chạy một seed. Mười epoch là ngân sách ban đầu đã chọn, không phải cam kết mô hình đã hội tụ hoặc đạt độ chính xác cụ thể.

## 2. Dữ liệu đã thống nhất với Khang

| Thành phần | Thiết lập |
|---|---|
| Dataset | CIFAR-10, ảnh gốc RGB 32 × 32 |
| Training / validation / test | 45.000 / 5.000 / 10.000 |
| Split | Phân tầng theo lớp; seed 42, val_size 5.000 |
| Split dùng lại | `data/splits/split_indices.json`; không tạo lại giữa hai model |
| Phân bố | Mỗi lớp 4.500 train / 500 validation / 1.000 test |
| Class order | 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck |
| Input sau xử lý | [B, 3, 224, 224] |
| Batch size / workers | 32 / 4 |
| Train sampler | Shuffle; tạo generator mới với seed 42 cho mỗi run |
| Validation/test sampler | Không shuffle, không bỏ batch cuối |

Chuỗi train theo đúng `data.py`: RandomCrop(32, padding=4) → RandomHorizontalFlip(p=0.5) → Resize(224,224) nội suy bilinear → ToTensor → Normalize.

Validation/test: Resize(224,224) bilinear → ToTensor → Normalize. Mean ImageNet = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]. Augmentation chỉ áp dụng trên training; không dùng crop ngẫu nhiên cho validation/test.

Notebook chuẩn bị chỉ đọc indices và metadata, không tải lại dữ liệu hoặc chạy DataLoader. Các batch đã được kiểm chứng trong notebook 01; nếu đổi code/cấu hình, kiểm tra lại phần bị ảnh hưởng trước training.

## 3. Cấu hình huấn luyện baseline

| Lựa chọn | Giá trị và ý nghĩa |
|---|---|
| Models | `resnet18`, `convnext_tiny` |
| Pretrained | `IMAGENET1K_V1` của từng kiến trúc |
| Classifier | Thay đầu cuối thành 10 logits |
| Fine-tuning | Toàn mạng; `freeze_backbone=false` |
| Loss | CrossEntropyLoss nhận logits; không softmax trước loss |
| Optimizer | AdamW |
| LR backbone / head | 1e-4 / 1e-3, cùng quy tắc chia nhóm ở hai model |
| Weight decay | 0.05 theo implementation hiện có |
| Scheduler | CosineAnnealingLR, T_max=10, bước scheduler mỗi epoch |
| Ngân sách | Tối đa 10 epoch/model |
| Early stopping | Patience=5 epoch không cải thiện theo tiêu chí chọn checkpoint |
| AMP khi train/validation | Bật trên CUDA theo `train.py` |
| Device | `cuda`, tránh tự chuyển sang CPU khi cấu hình run chính thức |
| Seed | 42 cho Python/NumPy/PyTorch và generator DataLoader |
| Deterministic | false, đúng code hiện có; không cam kết tái lập bitwise |
| Tuning riêng | 0 run tuning thêm/model ở vòng baseline |

Chọn `best.pt` bằng validation accuracy cao nhất; nếu bằng nhau thì chọn validation loss thấp hơn. Một lần cải thiện theo quy tắc này sẽ reset bộ đếm patience. `last.pt` không mặc nhiên là checkpoint tốt nhất.

Hai model có cùng ngân sách tối đa và chính sách early stopping, nhưng số epoch thực chạy có thể khác. Không gọi đây là so sánh với thời gian chạy tuyệt đối bằng nhau. Khi chạy nối tiếp, phải tạo model và loaders/generator mới cho từng run.

Nếu cần đổi số epoch, batch, learning rate hoặc chính sách tuning, ghi một phiên bản protocol mới trước khi chạy lại. Chỉ dùng training/validation để quyết định; không xem test rồi chọn lại cấu hình.

## 4. Đánh giá chất lượng đã thống nhất với Quy

| Thành phần | Thiết lập |
|---|---|
| Đầu vào | `best.pt` đã chốt của từng model, đúng class order/transforms |
| Tập đánh giá cuối | 10.000 ảnh test chuẩn CIFAR-10 |
| Eval batch size | 32 (`eval_batch_size` trong config) |
| Chế độ | eval, no_grad, float32; không dùng AMP training làm giả định cho phép đánh giá này |
| Chỉ số | Accuracy, macro-F1, loss, precision/recall/F1 theo lớp |
| Lỗi phân loại | Confusion matrix, nhãn thật/dự đoán, confidence, logits và index mẫu |

Cùng class order 10 lớp cho tất cả bảng/hình. Khi truy nguyên ảnh sai, indices do evaluator hiện tại sinh là vị trí trong loader; vì vậy test phải giữ đúng thứ tự và không bỏ mẫu.

Test chỉ dùng cho đánh giá cuối sau khi chốt checkpoint. Không dùng test để chọn epoch, model configuration hoặc ngân sách huấn luyện. Các kết quả 200/100 ảnh và kiểm tra 10 ảnh validation trước đây chỉ là kiểm chứng pipeline.

## 5. Điều kiện profiling đã thống nhất với Quy

| Thành phần | Thiết lập |
|---|---|
| Hardware | NVIDIA GeForce RTX 5070 Ti của Phúc |
| Input / batch | [32,3,224,224] |
| Dtype | float32, không autocast; ghi các cờ TF32/cuDNN thực dùng và giữ giống nhau cho cả hai model |
| Warm-up / repeats | 20 forward làm nóng / 100 forward được đo |
| Mode | eval, no_grad, không training đồng thời trên GPU |
| Timing CUDA | CUDA Events và synchronize theo code profiling |
| Phạm vi | Chỉ forward; không data loading, preprocessing, tải trọng số hoặc khởi tạo model |
| Tổng hợp | Mean/std/p50/p95/p99/min/max; throughput = batch_size / thời gian batch tính bằng giây |
| Memory | CUDA max_memory_allocated sau reset; nêu rõ đây là bộ nhớ tensor PyTorch được cấp phát, không phải toàn bộ VRAM của hệ thống |

`mean_ms` là thời gian của **cả batch 32 ảnh**, không phải latency của một ảnh chạy riêng. Nếu chia cho 32, chỉ gọi là thời gian trung bình quy đổi mỗi ảnh. Chưa có phép đo batch=1 cho single-image latency trong baseline này.

`warmup=20` và `repeats=100` là cấu hình profiling, không phải số epoch warm-up của training. Các trường này hiện được `profiling.py` đọc trực tiếp.

### Chính sách parameters và FLOPs/MACs

- Số tham số chính thức lấy từ model đã thay head 10 lớp; trong kết quả profiling dùng `params_total_actual`. Không dùng `params_m` tham chiếu ImageNet như số đếm chính xác của model 10 lớp.
- `profiling.py` hiện lấy FLOPs từ hằng số tham chiếu torchvision cho model ImageNet, input 224/batch1. Nếu đưa vào báo cáo, đặt tên rõ **FLOPs tham chiếu ImageNet** và dẫn nguồn được ghi trong output; không ghi là FLOPs trực tiếp của model đã fine-tune.
- Chưa có phép đếm FLOPs/MACs trực tiếp trong code. Không tự điền MACs hoặc chuyển đổi khi chưa xác định quy ước công cụ. Nếu Quy bổ sung bộ đếm, ghi tên/phiên bản công cụ, input [1,3,224,224], quy ước phép toán, các toán tử không được hỗ trợ và model head 10 lớp.
- Xác nhận của Quy do Phúc cung cấp bao phủ metrics và điều kiện timing được hỏi; không được hiểu là công cụ FLOPs trực tiếp đã hoàn thành. Phần C1 tương ứng trong checklist vẫn còn mở.

## 6. Môi trường và artifacts

Môi trường dự kiến là kernel `.venv-1`. Không chép các số phiên bản cũ sang kết quả mới. Chạy [notebook chuẩn bị](../notebooks/Chot_protocol_va_ghi_moi_truong.ipynb) để ghi nhận từ kernel đang dùng:

- `reports/environment_phuc.txt`: bản đọc bằng người, có Python, executable/kernel, torch/torchvision, CUDA build, GPU, driver, các thư viện chính.
- `reports/environment_phuc.json`: dữ liệu có cấu trúc và các điều kiện môi trường đã/ chưa đạt.
- `reports/requirements_phuc_snapshot.txt`: tên và phiên bản distributions thực cài; không thu thập biến môi trường hay URL chứa thông tin đăng nhập.
- `reports/phase2_readiness.json`: đối chiếu cấu hình, split, xác nhận và môi trường. Đây là kiểm tra chuẩn bị Phase 2, không thay thế T013–T015 hoặc chứng minh training thành công.

Các file trên chỉ xuất hiện khi **Phúc tự chạy notebook**; lượt chuẩn bị này không tự tạo kết quả môi trường.

Run name dự kiến: `<model>_seed42_<YYYYMMDD_HHMMSS>`; mỗi run một thư mục riêng, không ghi đè.
Checkpoints: `artifacts/checkpoints/<run_name>/best.pt`, `last.pt`.
Logs: `artifacts/logs/<run_name>/config.json`, `metrics.csv`, `summary.json`.

Không commit dataset/checkpoint lên Git. Chọn nơi bàn giao các tệp lớn khi có artifacts thật; nơi bàn giao hiện chưa được xác nhận. `epoch_seconds` hiện bao gồm train/validation, không bao gồm toàn bộ thời gian tải dữ liệu, tạo model và ghi xong mọi artifact; không gọi tổng trường này là toàn bộ thời gian end-to-end.

## 7. Trạng thái công việc và cách xác nhận

| Task | Trạng thái sau lượt chuẩn bị |
|---|---|
| T008 | Đã ghi nhận xác nhận dữ liệu của Khang qua Phúc |
| T009 | Đã hoàn thành tài liệu protocol và lựa chọn baseline |
| T010 | Đã ghi nhận xác nhận metrics/timing của Quy; giới hạn FLOPs được nêu riêng |
| T011 | Đã cập nhật cấu hình baseline dùng chung cho hai model |
| T012 | Phúc đã chạy/lưu ngày17/11, 4 cell code không lỗi, PHASE 2 PASS; 4 file kết quả môi trường đã được đọc và xác nhận |

Phúc đã chạy và lưu notebook bằng kernel .venv-1 ngày17/11/2026, dữ liệu thu thập lúc17:07:30+08:00. Các file môi trường/readiness và output notebook đều đạt; fingerprints khớp config/data.py/split hiện tại. Assistant chỉ đọc các kết quả này và tick T012, không chạy lại. Matplotlib vẫn chưa cài. T013/T014 đã được chuẩn bị và T015 đã được Phúc chạy/lưu PASS (report reports/phase3_preflight_20261117_174231_303611.json). Sau preflight, Phúc đã hoàn tất run ResNet18 T018; checkpoint reload T019 đã PASS, ConvNeXt-Tiny chưa chạy.

Từ yêu cầu ngày 16/11/2026, assistant chỉ chuẩn bị/sửa file và hướng dẫn; không tự chạy code project, notebook, test, training, benchmark hoặc cài gói nếu Phúc chưa yêu cầu lại rõ ràng.


**Rà soát run ResNet18 ngày17/11/2026:** run resnet18_seed42_20261117_175014_404116 đã hoàn tất10 epoch trên45.000 train/5.000 validation, CUDA/AMP, đúng protocol. Best validation accuracy96,46% ở epoch9, loss0,1373025; epoch10 đạt96,42%. CSV hữu hạn, LR cosine/config/best epoch khớp, best.pt/last.pt và ba file logs tồn tại. Tick T018/X2, tổng19/40 task. Chưa strict load hoặc tính validation lại cho checkpoint chính thức: T019 còn mở; chưa chạy ConvNeXt-Tiny chính thức, test hoặc benchmark. Bằng chứng file-only: reports/training_review_resnet18_seed42_20261117_175014_404116.json. Notebook05 đã chuẩn bị8 cell/4 code, chưa chạy; Phúc chọn .venv-1, Run All, Ctrl+S và gửi output/report checkpoint_reload để kiểm tra T019. Assistant chỉ đọc/ghi file, không chạy project, không sửa notebook04 hoặc kết quả gốc.


**Xác nhận T019 ngày 17/11/2026:** Phúc đã chạy và lưu notebook05, 4 cell code counts1–4 không error; T019/T021 CHECKPOINT RELOAD: PASS áp dụng cho ResNet18 (không phải T021 ConvNeXt-Tiny). Run resnet18_seed42_20261117_175014_404116, best.pt epoch9: strict load/metadata/class order/fingerprints/weights finite đạt, forward FP32 batch(32,10) hữu hạn trên CUDA. Tính lại đủ5.000 ảnh validation với AMP đạt accuracy0,9646 (khớp log), loss0,13729811782836915 so với0,13730251471996308; chênh lệch-0,000004396891594 nằm trong tolerance. Report: reports/checkpoint_reload_resnet18_seed42_20261117_175014_404116_20261117_181643_179508.json. Tick T019, tổng20/40 task; tiếp theo T020 ConvNeXt-Tiny rồi T021. Không training/test/benchmark trong notebook05. Assistant chỉ đọc kết quả Phúc lưu và cập nhật tracking, không chạy code hoặc sửa notebook/raw report.


**Cập nhật bổ sung ngày 17/11/2026 — T020:** Phúc đã có output notebook04.1 counts1–4 không lỗi và run convnext_tiny_seed42_20261117_191514_964199 hoàn tất10 epoch trên45.000/5.000 ảnh với CUDA/AMP, đúng baseline. Best epoch10: validation accuracy98,50%, loss0,0583064739. CSV hữu hạn, summary khớp trong sai số đọc số thực và đủ best/last/config/CSV/summary. Tick T020/X3, tổng21/40 task. Notebook05 hiện vẫn lưu PASS ResNet18 T019; chưa có reload ConvNeXt-Tiny, T021 còn mở. Tiếp theo: đổi RUN_NAME trong notebook05 thành convnext_tiny_seed42_20261117_191514_964199, chọn .venv-1, Restart Kernel → Run All → Ctrl+S rồi gửi kết quả. Không chạy lại training hoặc test. Hai notebook training đã được Phúc đặt tên riêng: 04_Huan_luyen_chinh_thuc_CIFAR10_resnet18.ipynb và 04.1_Huan_luyen_chinh_thuc_CIFAR10_convnext_tiny.ipynb. Assistant chỉ đọc/ghi tracking; giữ nguyên notebook và kết quả gốc.


**17/11/2026 — T021 hoàn thành:** Phúc đã chạy và lưu notebook05 cho run convnext_tiny_seed42_20261117_191514_964199; 4 cell code counts1–4, không lỗi, CHECKPOINT RELOAD: PASS. Best checkpoint epoch10 nạp strict=True, metadata/class order/fingerprints/weights hữu hạn khớp; forward validation FP32(32,10) hữu hạn trên CUDA RTX5070Ti. Tính lại đủ5.000 ảnh validation với AMP: accuracy98,50% khớp log, loss0,058303282952308653 so với0,058306473898887635; chênh lệch-0,000003190946579 trong tolerance. Bằng chứng: reports/checkpoint_reload_convnext_tiny_seed42_20261117_191514_964199_20261117_195006_056631.json. Tick T021, tổng22/40 task. Cả hai run chính thức đã training/reload đạt; chưa đánh giá test hoặc benchmark. Bước tiếp T022 (ghi nhận baseline chỉ seed42, không chạy thêm seed) và T023 (chốt manifest hai run), sau đó T024 bàn giao artifacts cho Quy. Assistant chỉ đọc output/file và cập nhật tracking; không chạy code hoặc sửa notebook/raw report/checkpoint.


## Chốt số run và checkpoint trước test (T022/T023)

Không áp dụng chạy thêm seed: baseline dùng duy nhất seed42, một run cho mỗi kiến trúc. Hai run chính thức đã hoàn tất và reload PASS; không cần huấn luyện lặp để hoàn tất T022.

Danh sách cố định tại [run_manifest.csv](run_manifest.csv), chọn ResNet18 epoch9 (validation accuracy0,9646/loss0,1373025147) và ConvNeXt-Tiny epoch10 (0,985/loss0,0583064739). Đây là best.pt chọn theo validation accuracy, hòa xét validation loss; accuracy không phải test accuracy. CSV lưu đường dẫn và SHA256 để nhận diện đúng checkpoint/split. Cấu hình, ngân sách và class order của hai run đã được đối chiếu với cùng protocol.

Chốt manifest không đồng nghĩa bàn giao thành công hoặc đã đánh giá test; T024 và T027 còn phải có bằng chứng thực hiện.
