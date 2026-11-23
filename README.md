# ConvNeXt Tiny vs ResNet18 trên CIFAR-10

Project nghiên cứu ConvNeXt như một ConvNet hiện đại, sau đó fine-tune ConvNeXt-Tiny pretrained trên CIFAR-10 và so sánh với ResNet18 pretrained về chất lượng phân loại, chi phí tính toán và lỗi phân loại.

## Trạng thái hiện tại

Cập nhật 17/11/2026: data.py có pipeline dữ liệu; models.py và train.py có model factory/training loop; utils.py có helper config, seed và device. Notebook đã xác nhận forward pretrained của cả hai model và smoke training ResNet18 trên dữ liệu giả. CIFAR-10 đã tải/giải nén đủ, MD5 khớp; số mẫu, phân bố lớp, split và một batch train thật đã đạt với num_workers=0. Notebook 01 đã kiểm chứng batch cả ba loader, dtype/class order và chính sách validation/test với 224/32/worker=0. Ngày 09/11, Phúc cung cấp thêm ảnh output PASS của một batch ở cả ba loader với cấu hình chính thức 224/32/worker=4; nếu thay cấu hình về sau cần kiểm chứng lại. Notebook 03 đã train thử cả hai pretrained model một epoch trên 200 train/100 validation và nạp lại checkpoint thành công; đã có run ResNet18 chính thức10 epoch trên45.000/5.000 ảnh, best validation96,46% ở epoch9; checkpoint ResNet18 chính thức đã reload/validation PASS ở T019; ConvNeXt-Tiny đã hoàn tất10 epoch, best validation98,50% ở epoch10; reload ConvNeXt-Tiny T021 đã PASS trên5.000 validation; cả hai model đã hoàn tất training/reload; T022/T023 đã chốt theo baseline một seed, danh sách tại reports/run_manifest.csv. Sau pull, evaluate.py/profiling.py/visualize.py đã có code. Evaluate/profiling đã import và kiểm tra CLI được; đánh giá hai checkpoint smoke trên 10 ảnh validation trong harness đạt trên CUDA, không phải đánh giá test cuối. Ảnh chụp môi trường ngày 17/11 ghi nhận thiếu matplotlib trong `.venv-1`. Đã bổ sung pandas 3.1.1 và matplotlib 3.11.2 vào .venv-1 cho notebook 06, import và pip check đạt. Trong VS Code, chọn .venv-1 và Restart Kernel để nạp các thư viện vừa cài. FLOPs hiện là số tham chiếu ImageNet, chưa đo trực tiếp hai model 10 lớp; chưa chạy benchmark.

Kế hoạch riêng của Phúc, chia nhỏ theo thứ tự thực hiện và ghi rõ đầu vào cần chờ: [tasks.md](tasks.md). Theo dõi trạng thái toàn project và các phụ thuộc tại [checklist.md](checklist.md). Theo [file phân công chính thức](PhanCong_DL_CNXtinyVSRN18onCIFAR10.txt): **Phúc** phụ trách model/training, chạy GPU, cấu hình và tích hợp; **Khang** phụ trách dữ liệu/lý thuyết; **Quy** phụ trách đánh giá, profiling, trực quan hóa và tổng hợp kết quả.
## Cấu trúc project và vai trò

```text
project-root/
├── README.md
├── checklist.md
├── tasks.md
├── PhanCong_DL_CNXtinyVSRN18onCIFAR10.txt
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_Kiem_tra_DataLoader_CIFAR10.ipynb
│   ├── 02_Kiem_tra_model_factory_va_training.ipynb
│   ├── 03_Kiem_tra_backward_training_CIFAR10_va_checkpoint.ipynb
│   ├── 04_Huan_luyen_chinh_thuc_CIFAR10_resnet18.ipynb
│   ├── 04.1_Huan_luyen_chinh_thuc_CIFAR10_convnext_tiny.ipynb
│   ├── 05_Kiem_tra_checkpoint_chinh_thuc.ipynb
│   └── Chot_protocol_va_ghi_moi_truong.ipynb
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

Archive hiện có 170.498.071 byte, MD5 `c58f30108f718f92721af3b95e74349a` khớp. Các output trước Phase3 ghi nhận batch64 ở terminal, 224/32 ở notebook01 và một batch train workers4 trong notebook02. Sau sửa Phase3, notebook02 đọc JSON chung và dùng final_config=deepcopy(data_config); source mới chưa chạy, output cũ nằm trong backup, không phải kết quả bản sửa.

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

**Cập nhật 09/11/2026 — cấu hình DataLoader chính thức:** ảnh output Phúc cung cấp xác nhận `input_size=224`, `batch_size=32`, `num_workers=4` đã đọc được một batch từ cả train, validation và test, đều PASS. Mục kiểm chứng cấu hình chính thức trong checklist đã hoàn thành; chưa có phép đo tốc độ hoặc duyệt mọi batch. Lúc rà soát ban đầu cell chưa được lưu; các lượt rà soát sau đã xác nhận notebook 01 có đủ 10 cell code và output official_config, không còn việc chờ Ctrl+S.

## Notebook 03 — Backward, CIFAR-10 nhỏ và checkpoint

Mở [03_Kiem_tra_backward_training_CIFAR10_va_checkpoint.ipynb](notebooks/03_Kiem_tra_backward_training_CIFAR10_va_checkpoint.ipynb), chọn kernel `.venv-1`, rồi chạy từng cell từ trên xuống. Notebook độc lập về biến với notebook 01/02, có giải thích trước mỗi bước.

- Phần A: forward/loss/backward/optimizer step của ConvNeXt-Tiny pretrained, dùng hai tensor giả FP32. Assert gradient hữu hạn, gradient khác 0 và backbone/classifier được theo dõi có cập nhật trọng số.
- Phần B: chọn cùng 200 train/100 validation cân bằng 10 lớp từ split đã lưu, chạy từng pretrained model một epoch bằng train_model của project. Input224/batch32/worker0, AMP theo config/CUDA. Cấu hình và DataLoader dùng chung smoke_config; manifest lưu indices gốc.
- Phần C: dựng model pretrained=False, nạp checkpoint strict=True, xác nhận metadata/class order, tính validation lại so với log và kiểm tra logits qua lưu–nạp state_dict bằng bộ đệm RAM với cùng batch eval/FP32.

**Đã chạy thành công ngày 09/11/2026:** 19 cell, trong đó 9 cell code; kết thúc bằng PASS A/B/C. Cả hai model nạp lại cho validation khớp log; logits roundtrip max abs diff 0.0 ở lần kiểm chứng đã lưu. Output đã có trong notebook.

Checkpoints/logs và indices đặt dưới `artifacts/notebook_smoke/cifar_subset/`, được Git bỏ qua. Bằng chứng: [artifacts/notebook_smoke/cifar_subset/checks/20261109_005858_994753/verification_results.json](artifacts/notebook_smoke/cifar_subset/checks/20261109_005858_994753/verification_results.json). Mỗi lần chạy tạo run group mới, có thể dùng thêm dung lượng checkpoint; không ghi đè run trước.

Notebook tạo các run ngắn để kiểm chứng luồng code. Accuracy từ 200/100 ảnh chưa dùng để kết luận so sánh; training đầy đủ, test evaluation, profiling và kiểm tra resume là các việc riêng. Việc chốt protocol cuối vẫn do Phúc phối hợp Khang/Quy.

**Xác nhận lần Phúc tự chạy ngày 09/11/2026:** notebook 03, run `20261109_005858_994753`, hoàn tất cả 9 cell code không lỗi. Backward/update, training nhỏ hai pretrained model, nạp checkpoint và so sánh logits đều đạt; bằng chứng M1/T1/T2/T3 đã trỏ tới run này. Việc chốt protocol nhóm và training/đánh giá đầy đủ vẫn còn.

## Bản báo cáo theo mẫu giảng viên

Đã nhận bản [BÁO CÁO DEEP LEARNING_copy.docx](<BÁO CÁO DEEP LEARNING_copy.docx>) ngày 14/11/2026. Chương 2 đã được điền đủ mô tả bài toán, input/output, ví dụ minh họa, dữ liệu sử dụng, giới hạn và bố cục bốn chương. Nội dung các chương khác được giữ nguyên. Tài liệu sử dụng các mục tài liệu tham khảo [22]/[23] đã có và chưa đưa kết quả chạy thử vào bảng kết quả chính thức. File DOCX vẫn là tài liệu local được Git bỏ qua.

Cả nhóm còn hoàn thiện các chương còn lại và rà soát bản cuối; không còn chờ giảng viên cung cấp mẫu. Word được đặt cập nhật các trường khi mở; nếu mục lục vẫn hiển thị nội dung cũ, chọn **Update Table → Update entire table**.

## Rà soát sau pull ngày 16/11/2026

Bản kiểm tra: HEAD `c04b0de`, `origin/main` `341bf0f`. Notebook 01/02/03 giữ outputs không lỗi (10/5/9 cell code). Đã tick thêm T007 (chia sẻ notebook/config qua main) và T026 (nhận/tích hợp code evaluate/profiling) trong tasks.md. Nhóm vẫn cần chốt protocol và chạy training đầy đủ.

Bằng chứng: [artifacts/notebook_smoke/post_pull_review/20261116_c04b0de/review_report.json](artifacts/notebook_smoke/post_pull_review/20261116_c04b0de/review_report.json). Kiểm tra dùng hai checkpoint smoke và cùng 10 ảnh validation, không duyệt test chính thức; import/CLI evaluate/profiling PASS. Module visualize còn thiếu matplotlib trong môi trường/dependencies; chưa cài thêm, chưa vẽ hình hoặc đo benchmark. FLOPs trong profiling là tham chiếu ImageNet, cần nêu đúng nguồn hoặc bổ sung phép đếm trực tiếp trước báo cáo cuối.

## Protocol baseline và notebook ghi môi trường

Ngày 16/11/2026, Phúc cung cấp xác nhận Khang đồng ý pipeline và Quy đồng ý metrics/điều kiện timing. [Protocol baseline](reports/experiment_protocol.md) và [bản ghi xác nhận](reports/protocol_confirmations.json) đã có; `configs/default.json` dùng CUDA, eval batch32, warmup20/repeats100 bên cạnh thiết lập training hiện có. Một seed42, fine-tune toàn mạng, tối đa10 epoch/patience5, không tuning riêng vòng đầu. Hai model dùng chung config; chưa có run chính thức mới.

[Chot_protocol_va_ghi_moi_truong.ipynb](notebooks/Chot_protocol_va_ghi_moi_truong.ipynb) đã được Phúc chạy/lưu ngày17/11/2026: 4 cell code (counts6–9), không lỗi, PHASE 2: PASS. Notebook đọc config/split và ghi môi trường; không tải dữ liệu, tạo model, train/inference/benchmark/cài gói.

Đã có [environment_phuc.txt](reports/environment_phuc.txt), [environment_phuc.json](reports/environment_phuc.json), [requirements_phuc_snapshot.txt](reports/requirements_phuc_snapshot.txt) và [phase2_readiness.json](reports/phase2_readiness.json), thu thập lúc17:07:30+08:00 ngày17/11. Đúng .venv-1/Python3.14.4, torch2.14.1+cu130/torchvision0.29.1+cu130, CUDA13.0, RTX5070Ti, driver617.42; fingerprints và mọi readiness check đạt. T012 đã tick, tasks13/40; tiếp theo T013. Matplotlib vẫn thiếu cho visualize. Assistant chỉ đọc kết quả Phúc lưu, không chạy lại notebook hoặc lệnh môi trường.

**Quy tắc làm việc từ 16/11/2026:** sau khi sửa/chuẩn bị xong, assistant hướng dẫn Phúc tự chạy và chỉ rà soát kết quả Phúc cung cấp; không tự chạy code project/notebook/test/training/inference/benchmark/lệnh môi trường hoặc cài gói, trừ khi Phúc yêu cầu rõ ràng khác trong lượt sau. Quy tắc được lưu trong AGENTS.md để giữ qua các session.

## Chuẩn bị Phase 3 và cách kiểm tra T015

- [Notebook02](notebooks/02_Kiem_tra_model_factory_va_training.ipynb): đã sửa để đọc `configs/default.json` thành base_config; data_config và final_config sao chép từ cùng nguồn. Các override chỉ cho smoke được nêu rõ. Forward, smoke training và full training mặc định tắt; kiểm tra CIFAR batch được bật. Source mới chưa chạy lại.
- [Notebook04](notebooks/04_Huan_luyen_chinh_thuc_CIFAR10_resnet18.ipynb): Phúc đã chạy/lưu 4 cell code (counts1–4) không lỗi; preflight T015 PASS. Kiểm tra số mẫu/batch/CUDA/run_name/dung lượng đạt; training=False nên cell cuối SKIP, chưa có epoch hoặc kết quả training chính thức. Khi bật training về sau, từng model có loaders/generator mới và train_model dựng model mới.
- `BÁO CÁO DEEP LEARNING_copy.docx`: đã viết mục3.2 và phần phương pháp4.1, có bảng kích thước, head/factory/fine-tuning, môi trường, cấu hình, loss/optimizer/checkpoint và phạm vi kiểm chứng. [Sơ đồ kiến trúc](reports/figures/model_architectures_cifar10.png) là hình minh họa cấu trúc, không phải kết quả thực nghiệm. Các nguồn torchvision chi tiết [26]/[27] đã bổ sung.

**Bạn thực hiện T015:** mở notebook04, chọn kernel `.venv-1`, giữ `RUN_FULL_TRAINING=False`, chạy Run All, Ctrl+S và gửi output có `T015 PREFLIGHT: PASS — CHƯA TRAINING` hoặc traceback. Kết quả được lưu tại `reports/phase3_preflight_<run_group>.json`; cell cuối chỉ in SKIP training. Không cần chạy lại notebook01–03 khi phần liên quan không đổi. Mức 5GiB là dự phòng dung lượng, không phải số đo kích thước checkpoint. Test loader chỉ được đọc metadata, không đánh giá test.

T013–T017 đã tick đúng phạm vi, tổng18/40; Phase3 hoàn thành. T015 được xác nhận từ [report preflight](reports/phase3_preflight_20261117_174231_303611.json) Phúc lưu lúc17:43:11+08:00 ngày17/11: batches train/validation224/32/workers4 hợp lệ, CUDA đúng, config/code/split khớp, còn94,48GiB tại thời điểm kiểm tra. Notebook02 mới vẫn chưa chạy; backup output cũ ở `artifacts/notebook_smoke/phase3_preparation/20261117/notebook02_before.ipynb`, bản Word trước sửa cũng ở thư mục này. Trường mục lục Word được đặt cập nhật khi mở; nếu vẫn cũ, chọn Update Table → Update entire table.

**Cập nhật sau T019:** ResNet18 đã huấn luyện và kiểm chứng checkpoint hoàn tất. Bước tiếp là ConvNeXt-Tiny T020. Hướng dẫn tạo run sau đây là tham chiếu; không chạy lại ResNet18 chỉ để kiểm tra checkpoint. Muốn bắt đầu run đầu, chọn `MODELS_TO_TRAIN=["resnet18"]` ở cell đầu notebook04; khi chủ động sẵn sàng training thì đổi `RUN_FULL_TRAINING=True` ở cell cuối và chạy từ trên xuống để tạo run group mới/kiểm tra lại trước epoch. Việc rà soát T015 này không tự bật cờ, không thực thi training. Bản Word phương pháp viết lúc chuẩn bị sẽ được đồng bộ ghi chú kiểm chứng khi bổ sung kết quả thật ở T031.


**Rà soát run ResNet18 ngày17/11/2026:** run resnet18_seed42_20261117_175014_404116 đã hoàn tất10 epoch trên45.000 train/5.000 validation, CUDA/AMP, đúng protocol. Best validation accuracy96,46% ở epoch9, loss0,1373025; epoch10 đạt96,42%. CSV hữu hạn, LR cosine/config/best epoch khớp, best.pt/last.pt và ba file logs tồn tại. Tick T018/X2, tổng19/40 task. Chưa strict load hoặc tính validation lại cho checkpoint chính thức: T019 còn mở; chưa chạy ConvNeXt-Tiny chính thức, test hoặc benchmark. Bằng chứng file-only: reports/training_review_resnet18_seed42_20261117_175014_404116.json. Notebook05 đã chuẩn bị8 cell/4 code, chưa chạy; Phúc chọn .venv-1, Run All, Ctrl+S và gửi output/report checkpoint_reload để kiểm tra T019. Assistant chỉ đọc/ghi file, không chạy project, không sửa notebook04 hoặc kết quả gốc.

## Kiểm tra checkpoint chính thức — notebook05

**Vai trò:** [05_Kiem_tra_checkpoint_chinh_thuc.ipynb](notebooks/05_Kiem_tra_checkpoint_chinh_thuc.ipynb) thực hiện T019 (ResNet18), tái sử dụng cho T021 (ConvNeXt-Tiny) bằng cách đổi RUN_NAME. Đọc config/logs/preflight của đúng run; không lấy model mặc định ở JSON để thay model đã train.

**Cách chạy:** mở notebook05 → chọn kernel .venv-1 → dùng RUN_NAME=convnext_tiny_seed42_20261117_191514_964199 (đã đặt sẵn cho T021) → Run All → Ctrl+S. Không cần chạy lại notebook04. Notebook05 dựng model pretrained=False, nạp best.pt strict=True, forward một batch validation FP32 rồi tính lại đủ5.000 ảnh validation với AMP như log gốc. Không huấn luyện hoặc đánh giá test; helper chỉ tạo test loader để kiểm tra số mẫu.

**Kết quả cần gửi:** dòng T019/T021 CHECKPOINT RELOAD: PASS cùng notebook đã lưu. Report mới ở reports/checkpoint_reload_<run_name>_<thời điểm>.json. Nếu lỗi, giữ output/traceback để xem nguyên nhân. Phúc đã chạy/lưu notebook05 PASS cho cả ResNet18 (T019) và ConvNeXt-Tiny (T021). Notebook hiện lưu output ConvNeXt-Tiny; output ResNet18 đã backup, report gốc vẫn giữ. Không cần chạy lại nếu checkpoint/code/dữ liệu không đổi. Tiếp theo T022/T023 để chốt cặp run và chuẩn bị bàn giao Quy.


**Xác nhận T019 ngày 17/11/2026:** Phúc đã chạy và lưu notebook05, 4 cell code counts1–4 không error; T019/T021 CHECKPOINT RELOAD: PASS áp dụng cho ResNet18 (không phải T021 ConvNeXt-Tiny). Run resnet18_seed42_20261117_175014_404116, best.pt epoch9: strict load/metadata/class order/fingerprints/weights finite đạt, forward FP32 batch(32,10) hữu hạn trên CUDA. Tính lại đủ5.000 ảnh validation với AMP đạt accuracy0,9646 (khớp log), loss0,13729811782836915 so với0,13730251471996308; chênh lệch-0,000004396891594 nằm trong tolerance. Report: reports/checkpoint_reload_resnet18_seed42_20261117_175014_404116_20261117_181643_179508.json. Tick T019, tổng20/40 task; tiếp theo T020 ConvNeXt-Tiny rồi T021. Không training/test/benchmark trong notebook05. Assistant chỉ đọc kết quả Phúc lưu và cập nhật tracking, không chạy code hoặc sửa notebook/raw report.

## Hướng dẫn T020 — ConvNeXt-Tiny (đã hoàn tất)

Mở notebook04, chọn kernel .venv-1. Ở cell cấu hình đặt MODELS_TO_TRAIN = ["convnext_tiny"]; ở cell training đặt RUN_FULL_TRAINING = True khi sẵn sàng. Restart Kernel rồi Run All để tạo run group mới, model mới và loaders/generator mới theo seed42; giữ nguyên protocol/config. Notebook tự kiểm tra đầu vào trước training và không ghi đè run ResNet18. Sau khi hoàn tất nhấn Ctrl+S và báo assistant đọc kết quả. Không cần chạy lại notebook01–03 hoặc ResNet18. Sau T020 mới đổi RUN_NAME của notebook05 sang tên run ConvNeXt-Tiny mới để làm T021.


**Cập nhật bổ sung ngày 17/11/2026 — T020:** Phúc đã có output notebook04.1 counts1–4 không lỗi và run convnext_tiny_seed42_20261117_191514_964199 hoàn tất10 epoch trên45.000/5.000 ảnh với CUDA/AMP, đúng baseline. Best epoch10: validation accuracy98,50%, loss0,0583064739. CSV hữu hạn, summary khớp trong sai số đọc số thực và đủ best/last/config/CSV/summary. Tick T020/X3, tổng21/40 task. Notebook05 hiện vẫn lưu PASS ResNet18 T019; chưa có reload ConvNeXt-Tiny, T021 còn mở. Tiếp theo: đổi RUN_NAME trong notebook05 thành convnext_tiny_seed42_20261117_191514_964199, chọn .venv-1, Restart Kernel → Run All → Ctrl+S rồi gửi kết quả. Không chạy lại training hoặc test. Hai notebook training đã được Phúc đặt tên riêng: 04_Huan_luyen_chinh_thuc_CIFAR10_resnet18.ipynb và 04.1_Huan_luyen_chinh_thuc_CIFAR10_convnext_tiny.ipynb. Assistant chỉ đọc/ghi tracking; giữ nguyên notebook và kết quả gốc.


17/11/2026 — Theo yêu cầu Phúc, notebook05 đã đổi RUN_NAME sang convnext_tiny_seed42_20261117_191514_964199 để làm T021. Bản notebook ResNet18 đã PASS (có output) được giữ tại artifacts/notebook_smoke/checkpoint_reload_backups/20261117_194853_4610303/05_ResNet18_da_PASS.ipynb; report ResNet18 gốc giữ nguyên. Notebook05 hiện xóa output/execution_count cũ để không gán PASS ResNet18 cho ConvNeXt-Tiny. Chưa chạy bản vừa sửa; T021 vẫn chưa tick, tổng21/40 task. Phúc chọn .venv-1 → Restart Kernel → Run All → Ctrl+S rồi gửi kết quả. Assistant chỉ sửa/đọc file, không thực thi notebook/model.


**17/11/2026 — T021 hoàn thành:** Phúc đã chạy và lưu notebook05 cho run convnext_tiny_seed42_20261117_191514_964199; 4 cell code counts1–4, không lỗi, CHECKPOINT RELOAD: PASS. Best checkpoint epoch10 nạp strict=True, metadata/class order/fingerprints/weights hữu hạn khớp; forward validation FP32(32,10) hữu hạn trên CUDA RTX5070Ti. Tính lại đủ5.000 ảnh validation với AMP: accuracy98,50% khớp log, loss0,058303282952308653 so với0,058306473898887635; chênh lệch-0,000003190946579 trong tolerance. Bằng chứng: reports/checkpoint_reload_convnext_tiny_seed42_20261117_191514_964199_20261117_195006_056631.json. Tick T021, tổng22/40 task. Cả hai run chính thức đã training/reload đạt; chưa đánh giá test hoặc benchmark. Bước tiếp T022 (ghi nhận baseline chỉ seed42, không chạy thêm seed) và T023 (chốt manifest hai run), sau đó T024 bàn giao artifacts cho Quy. Assistant chỉ đọc output/file và cập nhật tracking; không chạy code hoặc sửa notebook/raw report/checkpoint.


## Danh sách run đã chốt — T022/T023

T022 đã hoàn tất: **không áp dụng chạy thêm seed**. Protocol baseline chỉ dùng seed42, một run/model; đã đủ hai run và cả hai checkpoint đã reload PASS.

[run_manifest.csv](reports/run_manifest.csv) là danh sách checkpoint chính thức được chọn bằng validation trước khi đánh giá test. Mỗi dòng ứng với một model và một seed; đường dẫn tính từ thư mục gốc project, dùng dấu /. Đây là bảng chỉ dẫn tới artifacts, không chứa trọng số.

| Model | Seed | Best epoch | Validation accuracy |
|---|---:|---:|---:|
| ResNet18 | 42 | 9 | 96,46% |
| ConvNeXt-Tiny | 42 | 10 | 98,50% |

- best_checkpoint trỏ tới best.pt; không thay bằng last.pt theo mặc định.
- config_path là config thực tế đã lưu theo run, không phải config dùng chung đang có thể sửa.
- metrics_path, summary_path, reload_report_path cho phép truy ngược kết quả/chứng cứ nạp checkpoint.
- best_val_accuracy lưu theo thang0–1 (0.9646 = 96,46%); best_val_loss lấy từ log epoch được chọn, giữ nguyên sai số gốc.
- status=selected_reload_verified: đã chốt và có reload PASS. test_evaluated=false ghi nhận trạng thái lúc chốt; chưa có test accuracy trong file.
- best_checkpoint_sha256 xác định chính xác file trọng số; split_sha256 dùng chuẩn UTF-8/LF, bỏ BOM như protocol.
- selection_rule=max_val_accuracy_then_min_val_loss áp dụng riêng trong mỗi run. Không dùng validation để bỏ một trong hai kiến trúc khỏi phép so sánh.

**Cách xem:** mở CSV trong VS Code hoặc Excel; không cần chạy notebook/lệnh Python cho T022/T023. Nếu đổi lựa chọn checkpoint hoặc protocol về sau, ghi rõ lý do từ train/validation và lịch sử thay đổi; không chọn lại sau khi xem test.

**Bước tiếp theo:** T024 bàn giao cho Quy manifest kèm hai best.pt, config/logs, class order, split và môi trường; T025 chia sẻ cấu hình cho Khang. CSV không tự chuyển các file lớn: cần bàn giao thật và ghi lại nơi nhận/trạng thái truy cập trước khi tick T024.

**T022–T023 hoàn tất ngày 17/11/2026:** baseline cifar10-baseline-v1 chỉ dùng seed42, một run/model; hai run đã đủ và reload PASS. T022: “không áp dụng chạy thêm seed”; không cần huấn luyện lặp. T023: đã chốt hai best.pt tại reports/run_manifest.csv trước đánh giá test: ResNet18 run resnet18_seed42_20261117_175014_404116, epoch9, validation0,9646; ConvNeXt-Tiny run convnext_tiny_seed42_20261117_191514_964199, epoch10, validation0,985. CSV ghi config/split/metrics/summary/reload report, protocol và SHA256 checkpoint/split; accuracy ở thang0–1. Đã đối chiếu file nguồn/CSV/protocol/fingerprints/class order, đọc lại manifest và kiểm tra đường dẫn/hash. Chỉ xử lý file bằng công cụ tài liệu, không chạy project/model/notebook/test/benchmark hoặc cài gói. Tổng24/40 task; tiếp theo T024 bàn giao artifacts cho Quy và T025 chia sẻ cấu hình cho Khang. Chưa gửi file/tin nhắn, chưa xác nhận thành viên khác truy cập được; T024/T025 còn mở.


## Chuẩn bị bàn giao T024/T025

[handoff_phuc.md](reports/handoff_phuc.md) ghi nội dung từng gói, đường dẫn ZIP local, SHA256, cách gửi, tin nhắn mẫu và bảng để Phúc điền link/thời điểm/xác nhận nhận. [method_summary_khang.md](reports/method_summary_khang.md) ghi cấu hình cuối và các thay đổi Khang cần đồng bộ ở Chương1/mục3.1.

Đã chuẩn bị ZIP Quy (~415MiB, có hai best.pt và artifacts) và ZIP Khang (~0,15MiB, không checkpoint). Các file ZIP nằm trong thư mục ignore artifacts/notebook_smoke/handoff_preparation; không đưa dataset/checkpoint/ZIP lớn lên Git. Hai gói giữ đường dẫn tương đối của manifest; giải nén vào thư mục riêng để đối chiếu code trước khi ghép vào project đang làm.

Phúc tự chọn nơi chia sẻ, tải gói lên, cấp quyền và gửi link cho đúng người. Việc tạo file local hoặc project đang nằm trong OneDrive không chứng minh đã chia sẻ. T024/T025 vẫn mở, tổng24/40 task; chỉ tick sau khi có bằng chứng gửi/nhận đúng phạm vi. Không cần chạy notebook/training để bàn giao.

## Kết quả notebook 06 trên máy Phúc

Lần chạy ngày 23/11 dùng hai checkpoint SHA256 khớp manifest, ResNet18 epoch 9 và ConvNeXt-Tiny epoch 10. Trên 10.000 ảnh test, accuracy lần lượt là 96,06% và 98,30%. Benchmark forward FP32 trên RTX 5070 Ti, input 224, batch 32, warm-up 20 và 100 lần đo cho latency trung bình lần lượt 5,73 và 22,98 ms/batch. Bảng ở [comparison_table.csv](artifacts/comparison_table.csv). Khi đưa vào báo cáo, ghi bộ nhớ bằng MiB và sửa chú thích màu của hình chênh lệch lỗi. Bước bàn giao kết quả cho Quy vẫn ở T029.

Notebook 06 đã đổi bảng màu của khung chênh lệch ConvNeXt trừ ResNet sang RdBu_r. Giá trị âm hiện tương ứng màu xanh khi vẽ. Chạy lại cell xuất confusion_errors_side_by_side.png để cập nhật ảnh từ kết quả đã lưu.
