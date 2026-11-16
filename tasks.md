# Kế hoạch công việc của Phúc

Cập nhật: **16/11/2026**. Thực hiện các việc chưa hoàn thành theo thứ tự từ trên xuống; việc có ghi “có thể làm trong lúc chờ” được phép làm sớm.

Tất cả checkbox trong file này là công việc của **Phúc (bạn)**. Tên Khang và Quy chỉ xuất hiện trong ghi chú về đầu vào cần nhận hoặc đầu ra bạn cần bàn giao. Phạm vi theo [phân công chính thức](PhanCong_DL_CNXtinyVSRN18onCIFAR10.txt) và [checklist hiện tại](checklist.md).

- `[x]`: đã có bằng chứng hoàn thành đúng phạm vi mô tả.
- `[ ]`: còn phải thực hiện hoặc chưa có bằng chứng hoàn thành.
- **Phụ thuộc:** việc của chính bạn phải xong trước.
- **Chờ:** người và đầu ra cụ thể cần nhận. Khi chưa nhận, chuyển sang việc độc lập được chỉ rõ, không tự coi là đã thống nhất.
- File mới nêu trong các việc chưa tick là **đầu ra dự kiến**, chưa được tạo trong lượt lập kế hoạch.
- Sau mỗi thay đổi file, cập nhật `AGENTS.md`, `README.md`, các trạng thái liên quan trong `checklist.md` và file này.

## 1. Những việc đã hoàn thành

Các việc dưới đây được giữ lại để bạn biết điểm bắt đầu; không cần làm lại khi code/cấu hình liên quan chưa thay đổi.

- [x] T001 Xây model factory trong `src/cifar_compare/models.py`: hai pretrained model, đầu phân loại 10 lớp, đếm tham số.
  **Đã có:** forward hai model cho logits hợp lệ trong notebook 02.

- [x] T002 Xây training loop trong `src/cifar_compare/train.py`: train/validation, AdamW, cosine scheduler, AMP, early stopping, logs và checkpoint.
  **Đã có:** loop chạy qua kiểm chứng ngắn; chưa đồng nghĩa đã có kết quả huấn luyện đầy đủ.

- [x] T003 Kiểm chứng DataLoader trên máy của mình bằng `notebooks/01_Kiem_tra_DataLoader_CIFAR10.ipynb`.
  **Đã có:** split 45.000/5.000/10.000; batch cả ba tập hợp lệ với input 224, batch 32, workers 0 và 4 trong phạm vi đã kiểm tra.

- [x] T004 Kiểm chứng backward ConvNeXt-Tiny và training hai pretrained model trên cùng tập con CIFAR-10 trong notebook 03.
  **Đã có:** backward/update đạt; một epoch/model trên 200 train/100 validation. Đây là kiểm chứng pipeline, không phải kết quả so sánh cuối.

- [x] T005 Kiểm chứng nạp lại checkpoint của hai model trong notebook 03.
  **Đã có:** strict load thành công, validation khớp log, logits qua lưu–nạp state_dict khớp hoàn toàn trên batch đã kiểm tra.
  **Bằng chứng T004–T005:** `artifacts/notebook_smoke/cifar_subset/checks/20261109_005858_994753/verification_results.json`.

- [x] T006 Hoàn thiện Chương 2 trong `BÁO CÁO DEEP LEARNING_copy.docx`.
  **Đã có:** mục 2.1 Mô tả bài toán, 2.2 Giới hạn bài toán, 2.3 Bố cục đồ án; mẫu báo cáo đã nhận.

## 2. Chốt đầu vào và cấu hình trước khi chạy chính thức

**Bắt đầu từ T007.** Cấu hình hiện có là bản khởi đầu; chưa có bằng chứng nhóm đã chốt toàn bộ protocol P1.

- [ ] T007 Chia sẻ output notebook 01/03 và cấu hình dữ liệu hiện tại để thống nhất đầu vào.
  **Đầu ra:** bản ghi nội dung đã chia sẻ, gồm split path, seed 42, val_size 5.000, class order, transforms, input 224, batch 32, workers 4.
  **Chờ:** không cần chờ để chuẩn bị và gửi; việc gửi chưa được xác nhận hoàn thành trong session.

- [ ] T008 Ghi nhận xác nhận về pipeline dữ liệu dùng cho hai run chính thức.
  **Phụ thuộc:** T007.
  **Chờ Khang:** xác nhận split, class order, transforms và cấu hình DataLoader dùng chung. Dữ liệu đã kiểm chứng trên máy bạn; đây là bước thống nhất đầu vào, không phải yêu cầu viết lại `data.py`.
  **Xong khi:** có xác nhận hoặc điều chỉnh cụ thể được ghi lại. Trong lúc chờ, làm T009, T012, T013 hoặc bản nháp T016.

- [ ] T009 Soạn bản đề xuất protocol tại `reports/experiment_protocol.md`.
  **Đầu ra:** nêu rõ pretrained `IMAGENET1K_V1`, fine-tune toàn mạng hay freeze, split/seed, preprocessing, optimizer, LR backbone/head, scheduler, AMP, ngân sách epoch, early stopping và cách chọn checkpoint.
  **Điểm xuất phát:** input 224, batch 32, workers 4, LR backbone 1e-4/head 1e-3, weight decay 0.05, cosine, tối đa 10 epoch, patience 5, freeze_backbone=false.
  **Cần quyết định:** một hay nhiều seed; nếu tuning riêng, giới hạn ngân sách cho từng model. Các giá trị này chưa tự động trở thành quyết định đã thống nhất.

- [ ] T010 Ghi nhận yêu cầu đánh giá và điều kiện đo chi phí vào bản đề xuất protocol.
  **Phụ thuộc:** T009.
  **Chờ Quy:** thống nhất metrics, cách đếm FLOPs/MACs, device, dtype/AMP, batch size đo, warm-up, số lần đo và phạm vi có/không tính data loading.
  **Xong khi:** ghi rõ các lựa chọn đã thống nhất; không cần chờ code đánh giá/profiling hoàn thành để chốt các lựa chọn này.

- [ ] T011 Chốt protocol và cập nhật `configs/default.json` theo quyết định cuối.
  **Phụ thuộc:** T008–T010.
  **Đầu ra:** protocol có xác nhận và cấu hình dùng cho từng model; cùng split/preprocessing/ngân sách đã thống nhất, chọn checkpoint bằng validation accuracy, hòa thì xét validation loss.
  **Lưu ý:** test không tham gia tuning hoặc chọn checkpoint. Nếu kết quả validation cho thấy cần thay đổi ngân sách, ghi lại thay đổi trước khi đánh giá test.

- [ ] T012 Lưu thông tin môi trường chạy thực tế.
  **Có thể làm trong lúc chờ T008/T010.**
  **Đầu ra dự kiến:** `reports/environment_phuc.txt` ghi Python, torch, torchvision, CUDA build, GPU, driver, dependencies thực cài và kernel dùng cho training.
  **Xong khi:** notebook dùng đúng môi trường có CUDA; ghi nhận từ máy hiện tại, không chỉ chép phiên bản cũ trong checklist.

## 3. Chuẩn bị notebook và phần phương pháp báo cáo

- [ ] T013 Đồng bộ cấu hình trong `notebooks/02_Kiem_tra_model_factory_va_training.ipynb`.
  **Đầu ra:** đọc JSON làm nguồn cấu hình chung; tạo loaders và training config từ cùng cấu hình, khắc phục nhánh `final_config=dict(DEFAULTS)` đang có khả năng lệch với `data_config`.
  **Xong khi:** đổi input/batch/workers ở JSON không khiến loader và trainer dùng hai bộ giá trị khác nhau. Giữ nhánh chạy training đầy đủ tắt khi chỉ kiểm tra notebook.

- [ ] T014 Chuẩn bị `notebooks/04_Huan_luyen_chinh_thuc_CIFAR10.ipynb` để chạy hai model.
  **Phụ thuộc:** T011–T013.
  **Đầu ra:** các cell có giải thích, đọc protocol/config, chọn từng model, tạo run_name riêng chứa model/seed/thời điểm, khởi tạo model mới và tạo lại loaders/generator với seed đã chốt cho mỗi run.
  **Xong khi:** có thể chạy lần lượt từng model; output vào `artifacts/checkpoints/<run_name>/` và `artifacts/logs/<run_name>/`, không dùng thư mục smoke hoặc ghi đè run cũ. Mặc định chưa tự khởi chạy training dài.

- [ ] T015 Kiểm tra cấu hình cuối ngay trong notebook chạy chính thức.
  **Phụ thuộc:** T014.
  **Đầu ra:** in số mẫu 45.000 train/5.000 validation, class order, cấu hình, device CUDA và một batch theo cấu hình thực dùng; xác nhận run_name mới và đủ dung lượng lưu artifacts.
  **Lưu ý:** nếu giữ cấu hình đã PASS thì không cần chạy lại toàn bộ notebook 01–03. Nếu thay đổi code/cấu hình liên quan, chỉ kiểm tra lại phần bị ảnh hưởng. Không dùng tập test để theo dõi huấn luyện.

- [ ] T016 Viết mục 3.2 “Thiết kế mô hình” trong bản copy báo cáo.
  **Có thể soạn bản nháp trong lúc chờ T008/T010; hoàn thiện sau T011.**
  **Đầu ra:** mô tả hai backbone pretrained, thay classifier thành 10 logits, kích thước qua các stage, input/output, model factory và chính sách fine-tuning thực dùng.
  **Xong khi:** nội dung khớp `models.py` và config; dẫn lại lý thuyết ở Chương 1 thay vì lặp toàn bộ phần lý thuyết.

- [ ] T017 Viết phần thiết lập/phương pháp của mục 4.1 “Các thực nghiệm”.
  **Phụ thuộc:** T011, T012.
  **Đầu ra:** môi trường, loss, optimizer, LR backbone/head, scheduler, AMP, batch size, ngân sách epoch, early stopping, checkpoint selection và tóm tắt các kiểm chứng T003–T005.
  **Xong khi:** phân biệt rõ cấu hình dự kiến, kết quả kiểm chứng ngắn và kết quả thực nghiệm chính thức chưa có.

## 4. Chạy chính thức và kiểm tra artifacts

**Điều kiện bắt đầu training đầy đủ:** T011, T014, T015 đã xong. T016/T017 có thể hoàn thiện xen kẽ, không phải điều kiện kỹ thuật để chạy GPU. Không cần chờ module đánh giá của Quy viết xong.

- [ ] T018 Fine-tune ResNet18 pretrained trên đầy đủ training/validation bằng notebook 04.
  **Phụ thuộc:** T011, T014, T015.
  **Đầu ra:** một run ResNet18 hoàn tất theo ngân sách hoặc early stopping đã chốt; có `best.pt`, `last.pt`, `config.json`, `metrics.csv`, `summary.json`.
  **Xong khi:** dữ liệu thực là 45.000/5.000 ảnh và không phải run tập con; không dùng test để quyết định tiếp tục/dừng/tuning.

- [ ] T019 Kiểm tra run và nạp lại checkpoint ResNet18.
  **Phụ thuộc:** T018.
  **Đầu ra:** xác nhận metrics hữu hạn, config đúng protocol, best epoch khớp CSV, checkpoint tồn tại; dựng model từ metadata và strict load thành công.
  **Xong khi:** forward trên batch validation hữu hạn, đánh giá validation sau reload khớp log trong sai số số học phù hợp. Nếu có lỗi, xử lý lỗi và ghi lại lý do trước khi chốt run.

- [ ] T020 Fine-tune ConvNeXt-Tiny pretrained trên đầy đủ training/validation.
  **Phụ thuộc:** T019; giữ protocol T011.
  **Đầu ra:** run ConvNeXt-Tiny riêng, đầy đủ checkpoint/logs như T018; tạo model và loaders mới theo seed đã chốt.
  **Lưu ý:** chạy tuần tự trên GPU để tránh hai run cạnh tranh bộ nhớ và làm sai lệch phép đo thời gian.

- [ ] T021 Kiểm tra run và nạp lại checkpoint ConvNeXt-Tiny.
  **Phụ thuộc:** T020.
  **Đầu ra:** cùng nội dung kiểm tra như T019, áp dụng cho checkpoint và logs ConvNeXt-Tiny.
  **Xong khi:** checkpoint nạp được, validation khớp log và đủ metadata cho bước bàn giao.

- [ ] T022 Hoàn tất số seed/số run đã thống nhất, nếu protocol yêu cầu nhiều hơn một seed.
  **Phụ thuộc:** T019, T021 và quyết định ở T011.
  **Đầu ra:** đủ các cặp run theo danh sách đã chốt, mỗi run có artifacts riêng; không tự mở rộng số thí nghiệm sau khi xem test.
  **Nếu chỉ dùng một seed:** ghi rõ “không áp dụng chạy thêm seed”; không bắt buộc chạy lặp để đóng task này.

- [ ] T023 Chốt danh sách run/checkpoint dùng cho báo cáo trong `reports/run_manifest.csv`.
  **Phụ thuộc:** T019, T021, T022 đã xử lý theo protocol.
  **Đầu ra:** model, seed, run_name, config path, split path, best checkpoint, best epoch, metrics/summary path và trạng thái hợp lệ.
  **Xong khi:** lựa chọn dựa trên train/validation và được cố định trước đánh giá test. Không gộp các run khác protocol thành một cặp so sánh mà không giải thích.

## 5. Bàn giao và hỗ trợ đánh giá trên máy GPU

Các checkbox ở đây vẫn là việc của bạn: chuẩn bị, nhận bàn giao, tích hợp và hỗ trợ chạy. Không bao gồm việc tự viết các module đánh giá/profiling/trực quan hóa.

- [ ] T024 Chuẩn bị và bàn giao bộ artifacts đã chốt cho bước đánh giá.
  **Phụ thuộc:** T023.
  **Đầu ra:** checkpoint của từng run, config, metrics CSV, summary, class order, split hoặc thông tin tái lập split, môi trường và hướng dẫn load.
  **Nơi bàn giao:** thống nhất nơi lưu checkpoint lớn; không đưa dataset/checkpoint lên Git. Ghi đường dẫn và nội dung bàn giao tại `reports/handoff_phuc.md`.
  **Người nhận:** Quy. Chỉ tick khi đã thực sự bàn giao hoặc xác nhận truy cập được.

- [ ] T025 Chia sẻ cấu hình cuối và các thay đổi ảnh hưởng mô tả phương pháp.
  **Phụ thuộc:** T011; cập nhật lại sau T023 nếu thực nghiệm có thay đổi.
  **Đầu ra:** bản config/protocol được chia sẻ và nội dung cần đồng bộ trong báo cáo.
  **Người nhận:** Khang. Không cần chờ các hình/bảng kết quả để chia sẻ cấu hình.

- [ ] T026 Nhận phần code đánh giá/profiling để tích hợp và xác nhận nạp được checkpoint trên máy mình.
  **Có thể làm sớm sau T011 khi có code bàn giao; chưa cần đợi mọi run training xong.**
  **Chờ Quy:** phiên bản code/interface, dependencies và lệnh sử dụng. Nếu cần đối chiếu đầu vào, **chờ Khang** xác nhận test loader/transforms/class order.
  **Đầu ra:** môi trường/import/lệnh chạy tương thích; có thể dùng checkpoint smoke với validation để kiểm tra interface trước. Bạn xử lý phần tích hợp thuộc model/training của mình và phản hồi lỗi cụ thể, không tự nhận viết thay các module đó.

- [ ] T027 Hỗ trợ đánh giá test của các checkpoint đã chốt trên máy GPU khi cần.
  **Phụ thuộc:** T023, T024, T026.
  **Chờ Quy:** công cụ đánh giá đã kiểm chứng và cách chạy; **chờ Khang** xác nhận pipeline test nếu chưa nhận ở T026.
  **Đầu ra:** chạy công cụ được bàn giao khi cần dùng máy bạn, lưu outputs đúng run và gửi lại; nếu không cần máy bạn, nhận kết quả và đối chiếu checkpoint/run/class order.
  **Xong khi:** mỗi kết quả truy ngược được tới checkpoint đã chốt; không dùng test để quay lại chọn cấu hình.

- [ ] T028 Bố trí GPU rảnh và hỗ trợ chạy công cụ profiling.
  **Phụ thuộc:** T026 và điều kiện đo T010/T011.
  **Chờ Quy:** công cụ đo đã kiểm chứng, hướng dẫn, warm-up, số lần đo, dtype và batch size. Nếu phép đo gắn với checkpoint cuối, dùng đúng T023.
  **Đầu ra:** measurements thô kèm thiết bị, cấu hình và định nghĩa thời gian; không chạy training đồng thời.

- [ ] T029 Bàn giao outputs đánh giá/benchmark đã chạy trên máy mình.
  **Phụ thuộc:** T027/T028 đối với những phép đo cần máy bạn.
  **Đầu ra:** paths và tệp raw outputs có run/model/điều kiện đo; cập nhật `reports/handoff_phuc.md`.
  **Người nhận:** Quy. Việc tổng hợp chỉ số, vẽ hình và phân tích chuyên sâu không được chuyển thành nhiệm vụ viết code của bạn.

- [ ] T030 Nhận bộ hình/bảng kết quả để tích hợp báo cáo.
  **Phụ thuộc:** đã bàn giao đầu vào T024 và outputs máy mình nếu có.
  **Chờ Quy:** learning curves, bảng metrics test, bảng chi phí, confusion matrices, ảnh sai và nhận xét tương ứng từng run.
  **Đầu ra của bạn:** kiểm tra tên run/checkpoint, seed, số mẫu và điều kiện đo khớp manifest; ghi các chỗ cần làm rõ. Trong lúc chờ, hoàn thiện T016, T017, T032 và phần hướng dẫn ở T036.

## 6. Hoàn thiện phần báo cáo và tích hợp

- [ ] T031 Bổ sung kết quả huấn luyện thực tế vào mục 4.1.
  **Phụ thuộc:** T023; phần hình phụ thuộc T030.
  **Đầu ra:** mỗi model có cấu hình thực chạy, số epoch, best epoch, diễn biến train/validation và nhận xét có căn cứ; hình có chú thích run/seed.
  **Chờ Quy:** hình learning curves nếu chưa nhận. Phần mô tả số liệu có thể viết trước từ logs của bạn.

- [ ] T032 Viết phần sử dụng mô hình ở mục 4.3 theo phạm vi nhóm đã chốt.
  **Phụ thuộc:** checkpoint nạp được ở T019/T021.
  **Đầu ra:** cách chọn kernel, nạp checkpoint, tiền xử lý ảnh, dự đoán nhãn và đọc output; nêu công cụ/notebook thực sự sử dụng.
  **Lưu ý:** nếu mẫu yêu cầu giao diện, xác nhận phạm vi với giảng viên/nhóm trước khi lên việc xây giao diện. Không mô tả giao diện chưa tồn tại như sản phẩm đã hoàn thành.

- [ ] T033 Nhận và ghép phần lý thuyết/dữ liệu vào bản báo cáo chung.
  **Chờ Khang:** nội dung Chương 1 và mục 3.1 cùng nguồn tham khảo, hình/bảng tương ứng hoặc xác nhận bản hiện có đã hoàn thiện.
  **Đầu ra của bạn:** giữ nội dung đã được rà soát; đối chiếu preprocessing/split với config thực chạy và tránh trùng lặp với mục 3.2 của mình.

- [ ] T034 Nhận và ghép phần đánh giá/chi phí/phân tích lỗi vào mục 4.2.
  **Phụ thuộc:** T030.
  **Chờ Quy:** nội dung đánh giá hoàn chỉnh, bảng/hình và giải thích điều kiện đo.
  **Đầu ra của bạn:** số liệu trong văn bản, bảng và hình thống nhất; tách rõ kết quả CIFAR-10 của nhóm khỏi kết quả ImageNet của tài liệu tham khảo.

- [ ] T035 Hoàn thiện tóm tắt và kết luận từ kết quả đã xác nhận.
  **Phụ thuộc:** T031, T033, T034.
  **Đầu ra:** vấn đề, phương pháp, kết quả chính, hạn chế và hướng phát triển; mọi con số có nguồn là run/bảng cụ thể.
  **Xong khi:** không đưa accuracy smoke vào kết luận chính và không khẳng định mô hình nào tốt hơn khi chưa có bằng chứng tương ứng.

## 7. Tái lập và bàn giao cuối

- [ ] T036 Hoàn thiện hướng dẫn sử dụng và tái lập trong `README.md`.
  **Phụ thuộc:** có notebook chạy chính thức và các interface đã tích hợp; bổ sung lệnh đánh giá sau T026.
  **Đầu ra:** môi trường, dependencies, config, cách training/nạp checkpoint, vị trí artifacts và liên kết tới notebook. Không cần chép toàn bộ thứ tự công việc trong file này sang README.

- [ ] T037 Kiểm tra hướng dẫn nạp mô hình từ một kernel mới.
  **Phụ thuộc:** T023, T036.
  **Đầu ra:** checkpoint được dựng/nạp và dự đoán một batch validation theo hướng dẫn, không phụ thuộc biến còn sót trong notebook cũ.
  **Lưu ý:** không chạy lại toàn bộ training hoặc đánh giá test chỉ để kiểm tra README.

- [ ] T038 Rà soát bản Word cuối theo mẫu đã nhận.
  **Phụ thuộc:** T016, T017, T031–T035.
  **Đầu ra:** bốn chương thống nhất, mục lục được cập nhật, header/số chương đúng, bảng/hình được đánh số, nguồn trích dẫn đầy đủ; xem bản kết xuất để kiểm tra ngắt trang.
  **Lưu ý:** Chương 2 đã hoàn thành; chỉ điều chỉnh những câu cần đồng bộ nếu phạm vi hoặc kết quả thực tế thay đổi.

- [ ] T039 Rà soát các file chuẩn bị đưa lên Git.
  **Phụ thuộc:** đã hoàn tất các thay đổi dự kiến bàn giao.
  **Đầu ra:** diff chỉ chứa code/config/notebook/tài liệu cần chia sẻ, không có dataset, checkpoint, virtualenv hoặc tài liệu đang bị bỏ qua theo quy tắc repo; notebook không chứa thông tin nhạy cảm.
  **Lưu ý:** đây là bước kiểm tra danh sách file, không phải xác nhận đã commit/push. Không tự thay đổi quy tắc ignore đã được bạn chọn.

- [ ] T040 Chốt bộ bàn giao cuối của riêng mình.
  **Phụ thuộc:** T024, T025, T029 nếu áp dụng, T036–T039.
  **Đầu ra:** đường dẫn code/notebook/config/protocol, manifest run, nơi lưu checkpoint/logs, phần báo cáo của bạn và danh sách giới hạn còn lại; cập nhật tick theo bằng chứng vào `tasks.md`, `checklist.md`, `AGENTS.md`.

## 8. Cách xử lý khi phải chờ

| Đang chờ ở việc nào | Cần nhận gì | Bạn có thể làm trong lúc chờ |
|---|---|---|
| T008 | Khang xác nhận dữ liệu dùng chung | T009, T012, T013, bản nháp T016 |
| T010 | Quy thống nhất metrics/điều kiện đo | T012, T013, bản nháp T016 |
| T026 | Quy bàn giao code/interface; Khang xác nhận test input nếu thiếu | Training theo protocol đã chốt, T016/T017/T031 phần có logs |
| T027/T028 | Quy bàn giao công cụ đã kiểm chứng | Viết báo cáo phương pháp, kiểm tra artifacts, chuẩn bị hướng dẫn |
| T030/T034 | Quy bàn giao bảng/hình/nhận xét | T016, T017, T032, phần hướng dẫn T036 |
| T033 | Khang bàn giao nội dung hoặc xác nhận bản hiện có | Hoàn thiện phần model/training của bạn và các việc độc lập còn lại |

## 9. Các cải tiến chỉ làm khi phát sinh nhu cầu

Các mục này chưa là điều kiện bắt buộc để bắt đầu T018/T020:

- Resume training: chỉ bổ sung khi cần tiếp tục sau gián đoạn. Checkpoint hiện có model/optimizer/scheduler state nhưng chưa có luồng resume hoàn chỉnh hoặc AMP scaler/RNG state để khôi phục đầy đủ.
- Freeze backbone: chỉ kiểm tra thêm gradient và BatchNorm khi quyết định dùng freeze; cấu hình khởi đầu hiện là fine-tune toàn mạng.
- Log LR riêng hoặc đo tổng thời gian toàn run: bổ sung nếu protocol yêu cầu. `epoch_seconds` hiện không phải toàn bộ thời gian từ tải dữ liệu đến lưu xong artifacts.
- Nếu thay đổi hành vi training để thêm các mục trên, kiểm chứng phần bị ảnh hưởng trước run chính thức tiếp theo.

**Ghi chú về công cụ lập kế hoạch:** phiên này không tìm thấy skill `speckit-tasks` trong project hoặc các thư mục skill/plugin đã kiểm tra. File được lập trực tiếp từ phân công, checklist và code hiện có, dùng checkbox, mã công việc, đầu ra và phụ thuộc; không khẳng định đã chạy skill đó. Lượt này chỉ lập kế hoạch, không sửa code/notebook/Word, không chạy training và không gửi tin nhắn cho thành viên khác.
