# ConvNeXt Tiny vs ResNet18 trên CIFAR-10

Project nghiên cứu ConvNeXt như một ConvNet hiện đại, sau đó fine-tune ConvNeXt-Tiny pretrained trên CIFAR-10 và so sánh với ResNet18 pretrained về chất lượng phân loại, chi phí tính toán và lỗi phân loại.

## Trạng thái hiện tại

Cập nhật 09/11/2026: data.py có pipeline dữ liệu; models.py và train.py có model factory/training loop; utils.py có helper config, seed và device. Notebook đã xác nhận forward pretrained của cả hai model và smoke training ResNet18 trên dữ liệu giả. CIFAR-10 đã tải/giải nén đủ, MD5 khớp; số mẫu, phân bố lớp, split và một batch train thật đã đạt với num_workers=0. Notebook 01 đã kiểm chứng batch cả ba loader, dtype/class order và chính sách validation/test với 224/32/worker=0. Ngày 09/11, Phúc cung cấp thêm ảnh output PASS của một batch ở cả ba loader với cấu hình chính thức 224/32/worker=4; nếu thay cấu hình về sau cần kiểm chứng lại. Notebook 03 đã train thử cả hai pretrained model một epoch trên 200 train/100 validation và nạp lại checkpoint thành công; chưa có run trên đầy đủ split để báo cáo. evaluate.py, profiling.py, visualize.py vẫn là placeholder.

Theo dõi việc đã làm, việc còn thiếu và ai cần chờ ai tại [checklist.md](checklist.md). Theo [file phân công chính thức](PhanCong_DL_CNXtinyVSRN18onCIFAR10.txt): **Phúc** phụ trách model/training, chạy GPU, cấu hình và tích hợp; **Khang** phụ trách dữ liệu/lý thuyết; **Quy** phụ trách đánh giá, profiling, trực quan hóa và tổng hợp kết quả.
## Cấu trúc project và vai trò

```text
project-root/
├── README.md
├── checklist.md
├── PhanCong_DL_CNXtinyVSRN18onCIFAR10.txt
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_Kiem_tra_DataLoader_CIFAR10.ipynb
│   ├── 02_Kiem_tra_model_factory_va_training.ipynb
│   └── 03_Kiem_tra_backward_training_CIFAR10_va_checkpoint.ipynb
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
│   ├── predictions/
│   └── notebook_smoke/
└── reports/
```

### File cấu hình, tài liệu và dữ liệu

| File/thư mục | Vai trò | Hướng dẫn thực hiện |
|---|---|---|
| `README.md` | Điểm vào để hiểu mục tiêu, trạng thái và cách đi tiếp. | Giữ hướng dẫn chạy và trạng thái kiểm chứng khớp với code, notebook và artifacts thực tế. |
| `checklist.md` | Theo dõi việc đã xong, còn thiếu và ai chờ ai. | Ghi rõ Phúc/Khang/Quy, đầu ra bàn giao và bằng chứng hoàn thành; đồng bộ với file phân công. |
| `PhanCong_DL_CNXtinyVSRN18onCIFAR10.txt` | Phân công chính thức của ba thành viên. | Dùng làm căn cứ xác định người phụ trách và phụ thuộc khi cập nhật tài liệu. |
| `requirements.txt` | Danh sách thư viện Python và phiên bản. | Đã có torch, torchvision, NumPy và scikit-learn với phiên bản tối thiểu; cần ghi/chốt phiên bản môi trường thực nghiệm để cài lại được. |
| `.gitignore` | Bỏ qua cache Python, `AGENTS.md`, mọi file `.docx`, dữ liệu tải về và artifacts sinh ra, gồm notebook smoke. | Giữ tài liệu Word, `AGENTS.md`, dataset và checkpoint ở local theo quy tắc ignore hiện có; bàn giao file lớn ngoài Git. |
| `configs/default.json` | Cấu hình thí nghiệm: seed, split, weights, input size, transforms, optimizer, epochs, paths và profiling. | Điền các lựa chọn sau khi chốt protocol; lưu config thực tế bên cạnh mỗi run để tái lập. |
| `data/raw/` | Vị trí cache dữ liệu CIFAR-10 gốc. | Loader có thể tải dữ liệu vào đây; không chỉnh sửa ảnh/nhãn gốc và không commit dataset. |
| `data/splits/` | Vị trí lưu chỉ số mẫu train/validation nếu cần cố định split. | Tạo split phân tầng từ tập train gốc và lưu seed/indices; không dùng test để tuning. |

### File Python

| File | Vai trò | Hướng dẫn triển khai |
|---|---|---|
| `__init__.py` | Đánh dấu `cifar_compare` là Python package. | Để trống ban đầu; chỉ thêm export nếu module khác thật sự cần API package-level. |
| `data.py` | Nạp CIFAR-10, tạo split, transforms và DataLoaders. | Kiểm tra ảnh gốc `[3,32,32]` và batch sau resize `[B,3,224,224]` theo cấu hình hiện tại; augmentation ngẫu nhiên chỉ cho train; validation/test transforms xác định; giữ test riêng. |
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
| `artifacts/notebook_smoke/` | Checkpoint, config và logs khi kiểm tra training bằng dữ liệu giả. | Được Git bỏ qua; không dùng các metrics này làm kết quả CIFAR-10. |
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

Đến 08/11/2026, data.py, models.py, utils.py và training loop trong train.py đã được triển khai. Forward hai model và smoke ResNet18 đã chạy qua; tích hợp CIFAR-10, training thật, đánh giá cuối, profiling và trực quan hóa còn thiếu. Thứ tự và điều kiện hoàn thành chi tiết nằm trong checklist.md.

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

## Model factory và training loop

- models.py tạo ConvNeXt-Tiny hoặc ResNet18 bằng weights enum cụ thể, thay head thành 10 logits và hỗ trợ đóng băng backbone.
- train.py dùng một loop chung, AdamW với learning rate riêng cho backbone/head, validation theo epoch, cosine scheduler, AMP khi chạy CUDA, early stopping và lưu best.pt/last.pt.
- utils.py đọc JSON, đặt seed và chọn device. default.json chứa cấu hình khởi đầu có thể chỉnh.
- Training chỉ dùng train/validation để cập nhật model và chọn checkpoint. Test set không dùng trong training.
- Lệnh huấn luyện từ thư mục gốc, sau khi dữ liệu và cấu hình đã được kiểm chứng:
  - python -m src.cifar_compare.train --model resnet18
  - python -m src.cifar_compare.train --model convnext_tiny
- Chưa có kết quả chạy training thật bằng các lệnh này. Chúng có thể tải CIFAR-10/pretrained weights và bắt đầu huấn luyện; kiểm tra các điều kiện Phase X trong checklist trước khi chạy.
- Cài torch/torchvision theo đúng CUDA từ trang chính thức: https://pytorch.org/get-started/locally/

## Notebook kiểm tra

Mở notebooks/02_Kiem_tra_model_factory_va_training.ipynb từ thư mục gốc và chạy từng cell. Output đã lưu xác nhận môi trường/GPU, forward pretrained của hai model và một epoch smoke ResNet18 bằng dữ liệu giả. Cell CIFAR hiện bật, đã đọc được batch train `(32,3,224,224)` với num_workers=4 và không lưu error. Lần kiểm tra terminal sau đó ngày 08/11 đã đạt integrity, split và batch train. Notebook 01 riêng cho DataLoader đã chạy và lưu output PASS; dùng notebook 01 khi muốn kiểm tra dữ liệu trực tiếp trong notebook. Cell training thật vẫn tắt. Checkpoint/logs smoke nằm trong artifacts/notebook_smoke/ và được Git bỏ qua. Xem checklist.md trước khi chuyển sang dữ liệu thật.

## Kiểm chứng DataLoader trên CIFAR-10 thật

Mở [notebook 01 — Kiểm tra DataLoader](notebooks/01_Kiem_tra_DataLoader_CIFAR10.ipynb) trong VS Code:

1. Chọn **Select Kernel → Python Environments → `.venv-1`**.
2. Chọn **Run All**, hoặc chạy từng cell từ trên xuống và đọc giải thích ngay trước cell.
3. Kết quả cuối cần là `PASS — DataLoader CIFAR-10 đạt kiểm tra cơ bản.`

Notebook gồm 18 cell, trong đó 9 cell code. **Đã thực thi thành công ngày 08/11/2026 trong `.venv-1` và lưu đầy đủ output vào file.** Nó đọc cấu hình JSON, dùng input 224, batch size 32 và worker=0 cho bước kiểm chứng; các overrides chỉ ở trong notebook.

| Nội dung | Kết quả đã xác minh |
|---|---|
| CIFAR-10 gốc | 50.000 train, 10.000 test; ảnh 32×32×3, uint8 |
| Train/validation/test | 45.000 / 5.000 / 10.000 |
| Split | Seed 42; indices không lặp/giao nhau, phủ đủ 50.000 ảnh train; khớp file đã lưu |
| Phân bố lớp | Mỗi lớp 4.500 train / 500 validation / 1.000 test |
| Class order | Cùng thứ tự và ánh xạ ID ở ba tập |
| Một batch mỗi loader | Ảnh `(32,3,224,224)` float32; nhãn `(32,)` int64, trong 0–9, ảnh hữu hạn |
| Chính sách validation/test | Sequential sampler, không bỏ batch cuối; ba mẫu mỗi tập đọc lại cho tensor/label giống nhau |

Batch là một nhóm ảnh đưa vào model cùng lúc. Pixel sau normalize có thể âm/lớn hơn 1. DataLoader trả tensor CPU; training loop sẽ chuyển batch lên thiết bị đã chọn.

Notebook 01 kiểm tra dữ liệu/split/batch, không chạy model hoặc training. Cell dữ liệu gốc dùng `download=False` để báo lỗi nếu dataset đã tải thiếu/hỏng. Thư viện `nbformat`/`nbclient` đã được cài trong `.venv-1` để kiểm chứng và lưu outputs notebook; người dùng có thể chạy bằng VS Code với kernel tương ứng.

**Phạm vi:** một batch của mỗi loader, ba mẫu validation/test và kiểm tra metadata/indices/nhãn toàn tập; chưa duyệt mọi batch và chưa đo tốc độ. Worker=4 chỉ cần kiểm chứng riêng nếu nhóm chọn sử dụng; worker=0 vẫn dùng được cho smoke/training.

**Trạng thái checklist:** D1/D2 và D3 cơ bản đã đạt. Phúc có thể smoke tích hợp với cùng cấu hình, chia sẻ output cho Khang để thống nhất pipeline. Config training cần khớp config đã dùng tạo DataLoader.

Archive hiện có 170.498.071 byte, MD5 `c58f30108f718f92721af3b95e74349a` khớp. Kết quả terminal trước đó dùng batch64, còn notebook 01 dùng batch32; cả hai đều là cấu hình kiểm tra có chủ đích. Notebook 02 hiện dùng 224/32/worker4, đã qua một batch train. Cả hai notebook đều có output hoàn tất; notebook 01 có 9 cell code, notebook 02 có 5. Cell training thật vẫn tắt; nên dùng `final_config = dict(data_config)` để config training đi cùng config tạo loader khi chỉnh các run sau.

## Khắc phục lỗi PyTorch DLL trong .venv-1

**Đã khôi phục và kiểm chứng ngày 08/11/2026.** Khi chạy `python -m src.cifar_compare.data`, lỗi xảy ra ở `import torch` trước khi DataLoader khởi chạy. Trong `.venv-1` có thư mục torch nhưng thiếu `torch/lib` và metadata cài đặt; phần cài còn sót đã được sao lưu tại `.venv-1/torch_recovery_20261108_97acf904/`.

Lần cài lại gặp `WinError 206` ở đường dẫn file license quá dài, trong khi `LongPathsEnabled=0`. Đã hoàn tất cài bằng đường dẫn Windows mở rộng `\\?\` ngay trong virtualenv hiện tại. Thư mục project, registry và mã nguồn không được thay đổi trong bước khôi phục.

**Môi trường đã xác minh:**

| Thành phần | Phiên bản/trạng thái |
|---|---|
| Python | 3.14.4, Windows 64 bit |
| torch | `2.14.1+cu130` |
| torchvision | `0.29.1+cu130` |
| scikit-learn | `1.9.1` |
| NumPy | `2.5.4` |
| CUDA của PyTorch | 13.0; `torch.cuda.is_available() == True` |
| GPU | NVIDIA GeForce RTX 5070 Ti; driver 617.42 |
| Notebook | Đã cài ipykernel vào `.venv-1` |
| Kiểm tra | Import torch/torchvision/sklearn/ipykernel và `src.cifar_compare.data` qua; phép nhân ma trận CUDA qua; `pip check` không báo broken requirements |

Cặp torch/torchvision và kho CUDA sử dụng theo [hướng dẫn chính thức PyTorch](https://pytorch.org/get-started/previous-versions/). Kho wheel có bản Windows CPython 3.14 tương ứng. Phiên bản tối thiểu trong requirements.txt vẫn được giữ; bảng trên ghi môi trường đã kiểm tra thực tế.

**Lệnh xác nhận sau khi mở terminal mới ở thư mục project:**

```powershell
.\.venv-1\Scripts\python.exe -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available())"
.\.venv-1\Scripts\python.exe -m src.cifar_compare.data
```

Lệnh thứ hai đã được Phúc chạy thành công. Notebook 01 sau đó cũng đạt kiểm tra mở rộng cả ba loader với 224/32/worker=0; cấu hình khác cần kiểm chứng nếu chọn dùng. Trong VS Code, chọn kernel `.venv-1` rồi restart kernel trước khi chạy notebook.

**Cài dependencies trong virtualenv có đường dẫn dài:** từ thư mục project, có thể dùng cách đã kiểm chứng sau để tránh WinError 206:

```powershell
$projectPythonLongPath = '\\?\' + (Resolve-Path '.\.venv-1\Scripts\python.exe').Path
& $projectPythonLongPath -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
& $projectPythonLongPath -m pip install -r requirements.txt ipykernel
```

Ghi chú này xác nhận môi trường Python/GPU đã qua kiểm tra; các thí nghiệm CIFAR-10 và notebook smoke cũ có trạng thái riêng trong checklist.

**Cập nhật 09/11/2026 — cấu hình DataLoader chính thức:** ảnh output Phúc cung cấp xác nhận `input_size=224`, `batch_size=32`, `num_workers=4` đã đọc được một batch từ cả train, validation và test, đều PASS. Mục kiểm chứng cấu hình chính thức trong checklist đã hoàn thành; chưa có phép đo tốc độ hoặc duyệt mọi batch. Bản notebook 01 trên đĩa tại thời điểm rà soát vẫn chỉ có các cell baseline, nên cần Ctrl+S để lưu cell `official_config` và output mới.

## Notebook 03 — Backward, CIFAR-10 nhỏ và checkpoint

Mở [03_Kiem_tra_backward_training_CIFAR10_va_checkpoint.ipynb](notebooks/03_Kiem_tra_backward_training_CIFAR10_va_checkpoint.ipynb), chọn kernel `.venv-1`, rồi chạy từng cell từ trên xuống. Notebook độc lập về biến với notebook 01/02, có giải thích trước mỗi bước.

- Phần A: forward/loss/backward/optimizer step của ConvNeXt-Tiny pretrained, dùng hai tensor giả FP32. Assert gradient hữu hạn, gradient khác 0 và backbone/classifier được theo dõi có cập nhật trọng số.
- Phần B: chọn cùng 200 train/100 validation cân bằng 10 lớp từ split đã lưu, chạy từng pretrained model một epoch bằng train_model của project. Input224/batch32/worker0, AMP theo config/CUDA. Cấu hình và DataLoader dùng chung smoke_config; manifest lưu indices gốc.
- Phần C: dựng model pretrained=False, nạp checkpoint strict=True, xác nhận metadata/class order, tính validation lại so với log và kiểm tra logits qua lưu–nạp state_dict bằng bộ đệm RAM với cùng batch eval/FP32.

**Đã chạy thành công ngày 09/11/2026:** 19 cell, trong đó 9 cell code; kết thúc bằng PASS A/B/C. Cả hai model nạp lại cho validation khớp log; logits roundtrip max abs diff 0.0 ở lần kiểm chứng đã lưu. Output đã có trong notebook.

Checkpoints/logs và indices đặt dưới `artifacts/notebook_smoke/cifar_subset/`, được Git bỏ qua. Bằng chứng: [artifacts/notebook_smoke/cifar_subset/checks/20261109_004702_079834/verification_results.json](artifacts/notebook_smoke/cifar_subset/checks/20261109_004702_079834/verification_results.json). Mỗi lần chạy tạo run group mới, có thể dùng thêm dung lượng checkpoint; không ghi đè run trước.

Notebook tạo các run ngắn để kiểm chứng luồng code. Accuracy từ 200/100 ảnh chưa dùng để kết luận so sánh; training đầy đủ, test evaluation, profiling và kiểm tra resume là các việc riêng. Việc chốt protocol cuối vẫn do Phúc phối hợp Khang/Quy.
