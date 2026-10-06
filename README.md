# ConvNeXt Tiny vs ResNet18 trên CIFAR-10

Project nghiên cứu ConvNeXt như một ConvNet hiện đại, sau đó fine-tune ConvNeXt-Tiny pretrained trên CIFAR-10 và so sánh với ResNet18 pretrained về chất lượng phân loại, chi phí tính toán và lỗi phân loại.

## Trạng thái hiện tại

Đã có tài liệu lý thuyết, đặc tả pipeline và scaffold thư mục. Các file Python trong `src/cifar_compare/` hiện chỉ có comment hướng dẫn, chưa có logic thực thi. Chưa tải CIFAR-10, chưa cài/pin dependency, chưa huấn luyện model và chưa có kết quả thực nghiệm. `configs/default.json` hiện là `{}`.

## Cấu trúc project và vai trò

```text
project-root/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.json
├── data/
│   ├── raw/
│   └── splits/
├── src/
│   └── cifar_compare/
│       ├── __init__.py
│       ├── data.py
│       ├── models.py
│       ├── train.py
│       ├── evaluate.py
│       ├── profiling.py
│       ├── visualize.py
│       └── utils.py
├── artifacts/
│   ├── checkpoints/
│   ├── figures/
│   ├── logs/
│   └── predictions/
└── reports/
```

### File cấu hình, tài liệu và dữ liệu

| File/thư mục | Vai trò | Hướng dẫn thực hiện |
|---|---|---|
| `README.md` | Điểm vào để hiểu mục tiêu, trạng thái và cách đi tiếp. | Giữ hướng dẫn chạy khớp với code thực tế; chưa ghi lệnh train/evaluate là khả dụng trước khi triển khai. |
| `requirements.txt` | Danh sách thư viện Python và phiên bản. | Chọn và pin phiên bản sau khi chốt môi trường PyTorch/torchvision; hiện chưa có dependency thực sự. |
| `.gitignore` | Bỏ qua cache Python, `AGENTS.py`, mọi file `.docx`, dữ liệu tải về và artifacts lớn. | Các tài liệu Word và file `AGENTS.py` không được thêm mới vào Git; file dữ liệu/checkpoint lớn cũng được bỏ qua. |
| `configs/default.json` | Cấu hình thí nghiệm: seed, split, weights, input size, transforms, optimizer, epochs, paths và profiling. | Điền các lựa chọn sau khi chốt protocol; lưu config thực tế bên cạnh mỗi run để tái lập. |
| `data/raw/` | Vị trí cache dữ liệu CIFAR-10 gốc. | Loader có thể tải dữ liệu vào đây; không chỉnh sửa ảnh/nhãn gốc và không commit dataset. |
| `data/splits/` | Vị trí lưu chỉ số mẫu train/validation nếu cần cố định split. | Tạo split phân tầng từ tập train gốc và lưu seed/indices; không dùng test để tuning. |

### File Python

| File | Vai trò | Hướng dẫn triển khai |
|---|---|---|
| `__init__.py` | Đánh dấu `cifar_compare` là Python package. | Để trống ban đầu; chỉ thêm export nếu module khác thật sự cần API package-level. |
| `data.py` | Nạp CIFAR-10, tạo split, transforms và DataLoaders. | Kiểm tra số mẫu, nhãn và shape `[3,32,32]`; augmentation ngẫu nhiên chỉ cho train; validation/test transforms xác định; giữ test riêng. |
| `models.py` | Khởi tạo ConvNeXt-Tiny/ResNet18 pretrained và classifier 10 lớp. | Dùng weights enum cụ thể của torchvision; thay head cuối; hỗ trợ freeze/unfreeze khi protocol yêu cầu; đầu ra phải là 10 logits mỗi ảnh. |
| `train.py` | Huấn luyện và validation bằng một loop dùng chung. | Thực hiện forward, CrossEntropy, backward và optimizer step; ghi metrics mỗi epoch; chọn/lưu checkpoint chỉ dựa trên validation; không dùng test để chọn model. |
| `evaluate.py` | Đánh giá checkpoint đã chọn trên test và lưu dự đoán. | Tính accuracy, macro-F1, precision/recall theo lớp, loss và confusion matrix; lưu nhãn thật/dự đoán, confidence và index mẫu. Không cập nhật weights. |
| `profiling.py` | Đo parameters, FLOPs/MACs, latency, throughput và memory nếu có thể. | Ghi input size, thiết bị, dtype, batch size, warm-up và số lần lặp; đồng bộ CUDA khi đo; tách hoặc nêu rõ data-loading time. |
| `visualize.py` | Vẽ learning curves, confusion matrix và hình ảnh lỗi. | Đọc logs/predictions đã lưu, ghi tên model và class order trên hình; không chạy huấn luyện bên trong module này. |
| `utils.py` | Hàm hỗ trợ chung cho cấu hình, seed, logging và đường dẫn. | Giữ helper nhỏ, dùng project-relative paths; không để module này thành nơi chứa logic model/data/training. |

### Thư mục kết quả

| Thư mục | Nội dung dự kiến | Lưu ý |
|---|---|---|
| `artifacts/checkpoints/` | Checkpoint tốt nhất theo validation; có thể kèm optimizer state để resume. | Gắn checkpoint với model, epoch, config và metrics của run. |
| `artifacts/figures/` | Learning curves, confusion matrices và ảnh phân loại sai. | Nên tái tạo được từ logs/predictions. |
| `artifacts/logs/` | Config đã resolve, môi trường chạy và metrics theo epoch. | Không ghi secrets; lưu phiên bản thư viện và thông tin thiết bị. |
| `artifacts/predictions/` | Dự đoán test dùng cho bảng metrics và phân tích lỗi. | Lưu class order và index mẫu để truy nguyên kết quả. |
| `reports/` | Bảng, biểu đồ và thành phần báo cáo cuối. | Ghi rõ phương pháp, điều kiện benchmark và giới hạn kết luận. |

## Hướng dẫn thực hiện theo thứ tự

1. **Chốt protocol và cấu hình:** quyết định seed, train/validation split, input resolution, transforms, fine-tuning policy, epochs, optimizer, checkpoint criterion và cách đo compute. Ghi lại trong `configs/default.json` và `AGENTS.md`.
2. **Hoàn thiện `data.py`:** nạp CIFAR-10, kiểm tra dữ liệu và nhãn, tạo split tái lập, dựng loaders. Chỉ dùng augmentation cho train; giữ test set cho đánh giá cuối.
3. **Hoàn thiện `models.py`:** load từng pretrained model, thay classifier bằng 10 lớp và xác nhận logits có shape `[B,10]`.
4. **Hoàn thiện `utils.py` và `train.py`:** thêm seed/config/logging/checkpoint, sau đó hoàn thiện training loop chung. Bắt đầu với một run nhỏ để kiểm tra luồng trước khi dùng ngân sách đầy đủ.
5. **Hoàn thiện `evaluate.py`:** load checkpoint validation-best rồi tính metrics trên test đúng một lần cho cấu hình cuối.
6. **Hoàn thiện `profiling.py`:** đo parameters/FLOPs và tốc độ trong cùng điều kiện cho cả hai model.
7. **Hoàn thiện `visualize.py`:** tạo đường cong học tập, confusion matrices và ví dụ phân loại sai.
8. **Hoàn thiện báo cáo và hướng dẫn chạy:** cập nhật README bằng lệnh thực tế, versions, config, kết quả và limitations sau khi code đã được xác minh.

Các bước trên là kế hoạch, chưa được triển khai. Không có lệnh train/evaluate khả dụng cho đến khi các module Python có logic thực thi.

## Nguyên tắc đánh giá

- Chọn checkpoint bằng validation; không dùng test để chọn epoch hoặc hyperparameter.
- So sánh trên cùng split, ngân sách huấn luyện và điều kiện profiling; nêu rõ khác biệt nếu preprocessing/weights đòi hỏi transforms khác.
- Báo accuracy cùng macro-F1/per-class metrics và phân tích lỗi, không chỉ một con số tổng.
- FLOPs/MACs không đồng nghĩa thời gian chạy thực tế; cần đo latency/throughput cùng thiết bị, batch size và dtype.
- Không khẳng định mô hình nào luôn tốt hơn từ một cấu hình hoặc một seed.

## Tài liệu liên quan

- `01_Ly_thuyet_kien_truc_ConvNeXt.docx`: lý thuyết ConvNeXt.
- `02_Nen_tang_ly_thuyet_ConvNet.docx`: nền tảng ConvNet.
- `03_ConvNeXt_nhu_mot_ConvNet_hien_dai.docx`: tổng hợp kiến trúc và liên hệ CIFAR-10.
- `04_Dac_ta_project_va_pipeline_CIFAR10.docx`: đặc tả project theo phase.
