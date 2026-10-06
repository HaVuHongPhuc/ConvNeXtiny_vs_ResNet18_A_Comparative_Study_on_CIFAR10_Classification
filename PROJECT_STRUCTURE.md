# Cấu trúc project và hướng dẫn triển khai

## Trạng thái scaffold

Các thư mục và file dưới đây đã được tạo để chuẩn bị triển khai. Mỗi file `.py` hiện chỉ có ghi chú comment về vai trò và hướng dẫn thực hiện; chưa có logic Python. Đây chưa phải project có thể chạy. Chưa tải dữ liệu CIFAR-10, cài thư viện, chạy huấn luyện hoặc tạo kết quả. Không xem các file placeholder là tính năng đã hoàn thành.

## Cây thư mục

```text
project-root/
├── AGENTS.md                         # Bộ nhớ bền vững và quy tắc cập nhật project
├── README.md                         # Giới thiệu trạng thái và điểm bắt đầu
├── PROJECT_STRUCTURE.md              # Vai trò từng file và trình tự code
├── requirements.txt                  # Placeholder; chưa chọn/pin dependency
├── .gitignore                        # Bỏ qua cache Python, dữ liệu tải về và artifacts
├── configs/
│   └── default.json                  # Placeholder cấu hình ban đầu: {}
├── data/
│   ├── raw/.gitkeep                  # Vị trí dự kiến cho CIFAR-10/cache tải về
│   └── splits/.gitkeep               # Vị trí dự kiến lưu indices train/validation
├── src/
│   └── cifar_compare/
│       ├── __init__.py               # Đánh dấu package Python; hiện để trống
│       ├── data.py                   # Tải dữ liệu, split, transforms và DataLoaders
│       ├── models.py                 # ConvNeXt-Tiny/ResNet18 pretrained và head 10 lớp
│       ├── train.py                  # Training loop, validation, log và checkpoint
│       ├── evaluate.py               # Test metrics và lưu dự đoán
│       ├── profiling.py              # Parameters, FLOPs/MACs, latency/throughput, memory
│       ├── visualize.py              # Learning curves, confusion matrix, ảnh lỗi
│       └── utils.py                  # Seed, config, logging và hàm dùng chung
├── artifacts/
│   ├── checkpoints/.gitkeep          # Vị trí lưu checkpoint sau khi huấn luyện
│   ├── logs/.gitkeep                 # Vị trí log/config resolved/metrics theo epoch
│   ├── predictions/.gitkeep          # Vị trí dự đoán test phục vụ phân tích lỗi
│   └── figures/.gitkeep              # Vị trí biểu đồ và hình sinh ra
├── reports/.gitkeep                  # Vị trí bảng/hình và nội dung báo cáo cuối
└── [các tài liệu DOCX đã có]
```

## Vai trò và nguyên tắc file

### Cấu hình và dữ liệu

- `configs/default.json`: sẽ chứa seed, đường dẫn dữ liệu/artifacts, tên model/weights, input size, transforms, batch size, số epoch, optimizer, learning rate và tùy chọn profiling. Chưa điền giá trị thực nghiệm vì các quyết định này chưa được chốt.
- `data/raw/`: nơi cache dataset CIFAR-10. Dữ liệu tải về không nên đưa lên Git.
- `data/splits/`: tùy chọn lưu indices train/validation để lần chạy sau giữ nguyên split. Không lưu hoặc dùng test để tuning.
- `requirements.txt`: sẽ ghi các package và phiên bản đã chọn sau khi chốt môi trường PyTorch/torchvision. Hiện chỉ là placeholder, chưa phải danh sách cài đặt.

### Mã nguồn dự kiến

- `data.py`: tạo dataset, split phân tầng từ training set, train/eval transforms và DataLoaders. Augmentation ngẫu nhiên chỉ áp dụng trên train; validation/test transforms phải xác định.
- `models.py`: khởi tạo đúng pretrained weights enum cho hai kiến trúc; thay classifier cuối thành 10 logits; hỗ trợ freeze/unfreeze nếu protocol cần.
- `train.py`: dùng chung một training loop cho cả hai model. Mỗi epoch ghi train/validation metrics và lưu checkpoint tốt nhất dựa trên validation.
- `evaluate.py`: nạp checkpoint validation-best; chạy test cuối; tính accuracy, macro-F1, per-class precision/recall, loss và confusion matrix; lưu dự đoán.
- `profiling.py`: đo số tham số, FLOPs/MACs tại input size đã nêu, latency và throughput trên cùng thiết bị/điều kiện; ghi rõ warm-up, batch size và có tính data loading hay không.
- `visualize.py`: vẽ đường cong train/validation, confusion matrix và montage các ảnh dự đoán sai có true/predicted label.
- `utils.py`: chỉ chứa tiện ích dùng chung như seed, config loading, tạo thư mục, logging và metadata môi trường; tránh gom logic nghiệp vụ lớn vào đây.
- `__init__.py`: để trống ban đầu; chỉ thêm export nếu cần khi package được triển khai.

### Kết quả sinh ra

- `artifacts/checkpoints/`: checkpoint có model state, optimizer state (nếu cần resume), epoch, metrics và config của run.
- `artifacts/logs/`: cấu hình đã resolve, metadata runtime và lịch sử metrics theo epoch.
- `artifacts/predictions/`: nhãn thật, nhãn dự đoán, confidence/logits và index mẫu test để truy nguyên lỗi.
- `artifacts/figures/`: plots sinh tự động. File này có thể tạo lại từ logs/predictions.
- `reports/`: bảng tổng hợp và thành phần dùng cho báo cáo. Báo cáo cuối nên ghi cả điều kiện đo và limitations.
- `.gitignore`: loại `__pycache__`, môi trường ảo, dataset tải về và artifacts lớn khỏi Git, nhưng giữ các file `.gitkeep` để thư mục rỗng hiện diện.

## Thứ tự triển khai đề xuất

1. **Chốt thí nghiệm và cấu hình:** quyết định seed, split train/validation, input resolution, transforms, fine-tuning policy, epochs, optimizer, checkpoint criterion và cách đo compute. Ghi các lựa chọn đã thống nhất vào `default.json` và `AGENTS.md`.
2. **Triển khai `data.py`:** tải CIFAR-10, xác minh shape/nhãn, tạo split tái lập và kiểm tra một batch. Test set giữ nguyên cho đánh giá cuối.
3. **Triển khai `models.py`:** load ConvNeXt-Tiny/ResNet18 pretrained, thay head; smoke-check đầu ra `[B,10]` và loss hữu hạn.
4. **Triển khai `utils.py` rồi `train.py`:** seed/logging/checkpoint và một vòng train/validation dùng chung; hoàn thiện run ngắn trước run đầy đủ.
5. **Triển khai `evaluate.py`:** chọn checkpoint theo validation, sau đó đánh giá test và lưu metrics/predictions.
6. **Triển khai `profiling.py`:** đo params/FLOPs/MACs và latency/throughput với cùng điều kiện.
7. **Triển khai `visualize.py`:** xuất learning curves, confusion matrices và ví dụ lỗi.
8. **Hoàn thiện README và báo cáo:** ghi lệnh chạy thực tế, versions, cấu hình, kết quả, limitations và cách tái lập.

Mỗi bước chỉ được đánh dấu hoàn thành sau khi phần tương ứng đã được code và xác minh. Hiện các bước trên đều là kế hoạch; chưa có file pipeline nào được triển khai.

## Pipeline dữ liệu dự kiến

```text
CIFAR-10 train set
       │
       ├── split phân tầng/tái lập ──> train ──> augment ─┐
       │                                                  ├─> model + 10-class head
       └────────────────────────────> validation ────────┘      │
                                                                 ├─> train + validation
                                                                 └─> chọn checkpoint tốt nhất

CIFAR-10 test set ──> chỉ sau khi chốt checkpoint ──> metrics + profiling + phân tích lỗi
```

Để so sánh công bằng, dùng chung split, ngân sách huấn luyện, augmentation policy, checkpoint rule và điều kiện đo. Dùng transforms đúng với pretrained weights và ghi chúng ra config; nếu transforms khác giữa hai weights, nêu khác biệt trong báo cáo.

## Tiêu chí scaffold hoàn chỉnh

- Tất cả thư mục/file trong cây đã có mặt; các `.py` chỉ có comment hướng dẫn, chưa có logic thực thi.
- `default.json` parse được nhưng chưa áp đặt hyperparameters.
- Chưa tải dữ liệu, chạy train, cài dependency hoặc tạo checkpoint.
- Sau mỗi lần code thêm/sửa/xóa, cập nhật `AGENTS.md` và phần trạng thái file này.
