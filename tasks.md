# Kế hoạch công việc của Phúc

Cập nhật: **17/11/2026**. Thực hiện các việc chưa hoàn thành theo thứ tự từ trên xuống; việc có ghi “có thể làm trong lúc chờ” được phép làm sớm.

Tất cả checkbox trong file này là công việc của **Phúc (bạn)**. Tên Khang và Quy chỉ xuất hiện trong ghi chú về đầu vào cần nhận hoặc đầu ra bạn cần bàn giao. Phạm vi theo [phân công chính thức](PhanCong_DL_CNXtinyVSRN18onCIFAR10.txt) và [checklist hiện tại](checklist.md).

- `[x]`: đã có bằng chứng hoàn thành đúng phạm vi mô tả.
- `[ ]`: còn phải thực hiện hoặc chưa có bằng chứng hoàn thành.
- **Phụ thuộc:** việc của chính bạn phải xong trước.
- **Chờ:** người và đầu ra cụ thể cần nhận. Khi chưa nhận, chuyển sang việc độc lập được chỉ rõ, không tự coi là đã thống nhất.
- File mới trong task chưa tick là đầu ra dự kiến, trừ phần đã ghi rõ “đã chuẩn bị”. Notebook T012 đã chạy và có report; notebook05 đã được Phúc chạy PASS cho checkpoint ResNet18 tại T019.
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

**T018–T023 đã hoàn tất. Không áp dụng chạy thêm seed; manifest hai checkpoint đã chốt. Việc tiếp theo: T024 bàn giao artifacts cho Quy và T025 chia sẻ cấu hình cho Khang.**

- [x] T007 Chia sẻ output notebook 01/03 và cấu hình dữ liệu hiện tại để thống nhất đầu vào.
  **Đầu ra:** bản ghi nội dung đã chia sẻ, gồm split path, seed 42, val_size 5.000, class order, transforms, input 224, batch 32, workers 4.
  **Bằng chứng 16/11:** notebook 01/03 có output và `configs/default.json` đã có trên `origin/main` (`341bf0f`), hiện diện trong bản vừa pull. Đây là chia sẻ qua repo, không khẳng định đã gửi tin nhắn; xác nhận bổ sung của Khang đã được ghi ở T008 bên dưới.

- [x] T008 Ghi nhận xác nhận về pipeline dữ liệu dùng cho hai run chính thức.
  **Bằng chứng 16/11:** Phúc trả lời Khang đã xác nhận đúng split 45.000/5.000/10.000, seed 42, input 224, batch 32, workers 4 và transforms hiện có.
  **Ghi nhận:** `reports/protocol_confirmations.json`; xác nhận do Phúc cung cấp, không phải assistant liên hệ Khang. Không còn chờ xác nhận này.

- [x] T009 Soạn và hoàn thiện protocol tại `reports/experiment_protocol.md`.
  **Đầu ra:** baseline v1 gồm dữ liệu, pretraining/fine-tuning, optimizer/LR, cosine, AMP, early stopping, checkpoint criterion, metrics, timing và giới hạn FLOPs.
  **Lựa chọn baseline:** một seed 42, mỗi model một run hợp lệ; fine-tune toàn mạng; tối đa 10 epoch/patience5; không tuning riêng vòng đầu. Không cam kết đã hội tụ.

- [x] T010 Ghi nhận yêu cầu đánh giá và điều kiện đo chi phí.
  **Bằng chứng 16/11:** Phúc trả lời Quy đã xác nhận accuracy/macro-F1/theo lớp; forward FP32 trên RTX 5070 Ti, input224/batch32, warm-up20/repeats100, không data loading.
  **Ghi nhận:** protocol và confirmations JSON. FLOPs hiện chỉ tham chiếu ImageNet; chính sách ghi đúng nguồn/giới hạn nằm trong protocol. Không tự coi công cụ đếm FLOPs/MACs trực tiếp đã hoàn thành; phần C1 vẫn mở.

- [x] T011 Chốt cấu hình baseline dùng chung cho hai model.
  **Đầu ra:** `configs/default.json` giữ input224/batch32/workers4 và ngân sách baseline; đặt device=cuda, thêm eval_batch_size=32, warmup=20, repeats=100. Dùng cùng config với model tương ứng.
  **Quy tắc:** validation accuracy cao nhất, hòa xét validation loss; không tuning bằng test. Thay đổi protocol về sau phải được ghi phiên bản.
  **Trạng thái sau đó:** T012–T015 đã đạt; ResNet18 T018 đã chạy hoàn tất theo cấu hình này. Checkpoint ResNet18 chính thức đã được xác minh reload tại T019.

- [x] T012 Lưu và xác minh thông tin môi trường thực tế do Phúc tự chạy.
  **Bằng chứng 17/11:** notebook có 4 cell code execution_count 6–9, không error, output PHASE 2: PASS. Thời điểm thu thập: 2026-11-17T17:07:30+08:00.
  **Đã xác nhận:** đúng .venv-1/Python3.14.4, torch2.14.1+cu130, torchvision0.29.1+cu130, CUDA13.0 khả dụng, RTX5070Ti, driver617.42; mọi environment/readiness check đều true. Fingerprint config/data.py/split khớp các file hiện tại.
  **Đã có:** `reports/environment_phuc.txt`, `environment_phuc.json`, `requirements_phuc_snapshot.txt`, `phase2_readiness.json`.
  **Giới hạn:** matplotlib còn NOT_INSTALLED; chỉ ảnh hưởng bước visualize. Assistant chỉ đọc output/file do Phúc lưu, không chạy lại notebook/lệnh môi trường. T013/T014 đã được chuẩn bị sau đó và T015 đã được Phúc tự chạy PASS; training đầy đủ ResNet18 đã hoàn tất tại T018, ConvNeXt-Tiny còn chờ.

## 3. Chuẩn bị notebook và phần phương pháp báo cáo

- [x] T013 Đồng bộ nguồn cấu hình notebook 02.
  **Đã sửa 17/11:** đọc JSON bằng load_config; base_config → data_config → final_config, không còn nhánh full training dùng dict(DEFAULTS) riêng. Loaders training được tạo mới từ final_config.
  **Phạm vi:** source đã được rà soát, chưa thực thi bản mới. Forward/smoke/full training mặc định tắt; cell dữ liệu lấy input/batch/workers từ JSON.
  **Bằng chứng cũ:** output trước sửa được giữ trong `artifacts/notebook_smoke/phase3_preparation/20261117/notebook02_before.ipynb`; outputs của bản sửa đã xóa để tránh gán nhầm kết quả cũ cho code mới.

- [x] T014 Chuẩn bị notebook 04 huấn luyện chính thức.
  **Đã có:** notebook04 gồm 9 cell/4 code; Phúc đã chạy/lưu preflight (counts1–4) không lỗi ngày17/11. Có giải thích từng phần, chọn model, run_name chứa model/seed/thời điểm.
  **Cách hoạt động:** cùng config JSON, tạo lại loaders/generator cho từng run; train_model dựng model mới. Có guard fingerprint, kernel/CUDA, thư mục tồn tại và dung lượng.
  **Đã xác nhận mới nhất:** Phúc chọn MODELS_TO_TRAIN=["resnet18"], RUN_FULL_TRAINING=True và lưu counts5–8 không error, hoàn tất10 epoch. Output preflight/SKIP counts1–4 là lần kiểm tra trước, không còn là trạng thái mới nhất.

- [x] T015 Phúc đã chạy/lưu preflight notebook04 và kết quả đã được rà soát.
  **Bằng chứng:** `reports/phase3_preflight_20261117_174231_303611.json`, thu thập 2026-11-17T17:43:11+08:00; notebook4 code cells counts1–4, không lỗi; T015 PREFLIGHT: PASS — CHƯA TRAINING.
  **Đạt:** samples45.000/5.000/10.000, class order đúng; batch train/validation (32,3,224,224), float32/int64, finite, workers4; đúng .venv-1/CUDA/RTX5070Ti. Dung lượng94,48GiB tại lúc kiểm tra, vượt dự phòng5GiB; tên run chưa tồn tại.
  **Đối chiếu:** config, split và 4 file data/models/train/utils khớp fingerprints hiện tại. Test không được đánh giá. Lần preflight này training=False/SKIP. Lần chạy tiếp theo đã tạo checkpoint/logs ResNet18; xem T018.
  **Cách rà soát:** chỉ đọc notebook/report và đối chiếu file văn bản; assistant không chạy lại code hoặc training.

- [x] T016 Viết mục 3.2 “Thiết kế mô hình” trong báo cáo copy.
  **Đã viết:** input/output, stages của hai mạng, factory/head 10 lớp, số tham số và chính sách fine-tuning; Bảng3.6/3.7 và Hình3.4.
  **Nguồn:** models.py, torchvision, lý thuyết Chương1; thêm tài liệu tham khảo [26]/[27]. Asset sơ đồ: `reports/figures/model_architectures_cifar10.png`.
  **Kiểm tra tài liệu:** đã xem bản Word kết xuất; nội dung thân bài ngoài 3.2/4.1 được giữ, trừ nguồn bổ sung và header Chương3/4 được sửa đúng số chương.

- [x] T017 Viết phần phương pháp của mục 4.1 trong báo cáo copy.
  **Đã viết:** môi trường thực từ T012, loss, AdamW/LR/weight decay, cosine, AMP, early stopping, checkpoint, kiểm chứng đã có và hai run chính thức dự kiến; Bảng4.1–4.3.
  **Phân biệt rõ:** phần phương pháp đã viết theo snapshot lúc chuẩn bị Phase3. Sau đó T015 đã PASS; khi bổ sung kết quả thật ở T031 cần cập nhật ghi chú trạng thái tương ứng, không bịa kết quả training/test/benchmark.
  **Bản gốc:** `artifacts/notebook_smoke/phase3_preparation/20261117/report_before.docx`.

## 4. Chạy chính thức và kiểm tra artifacts

**Điều kiện bắt đầu training đầy đủ:** T011, T014, T015 đã xong. T016/T017 có thể hoàn thiện xen kẽ, không phải điều kiện kỹ thuật để chạy GPU. Không cần chờ module đánh giá của Quy viết xong.

- [x] T018 Fine-tune ResNet18 pretrained trên đầy đủ training/validation bằng notebook 04.
  **Phụ thuộc:** T011, T014, T015.
  **Đầu ra:** một run ResNet18 hoàn tất theo ngân sách hoặc early stopping đã chốt; có `best.pt`, `last.pt`, `config.json`, `metrics.csv`, `summary.json`.
  **Xong khi:** dữ liệu thực là 45.000/5.000 ảnh và không phải run tập con; không dùng test để quyết định tiếp tục/dừng/tuning.
  **Bằng chứng17/11:** run resnet18_seed42_20261117_175014_404116; notebook04 counts5–8 không error;10 epoch, best epoch9/validation accuracy96,46%/loss0,1373025. Đủ best.pt/last.pt và config/CSV/summary. Report rà soát file-only: reports/training_review_resnet18_seed42_20261117_175014_404116.json.

- [x] T019 Kiểm tra run và nạp lại checkpoint ResNet18.
  **Phụ thuộc:** T018 (đã hoàn tất).
  **Đã xác nhận 17/11:** notebook05 counts1–4 không error; strict load/metadata/fingerprints/weights hữu hạn và forward validation batch(32,10) đạt. Validation sau reload đủ5.000 ảnh: accuracy96,46%, loss0,1372981; khớp log trong tolerance. Không đánh giá test.
  **Bằng chứng:** reports/checkpoint_reload_resnet18_seed42_20261117_175014_404116_20261117_181643_179508.json
  **Đầu ra:** xác nhận metrics hữu hạn, config đúng protocol, best epoch khớp CSV, checkpoint tồn tại; dựng model từ metadata và strict load thành công.
  **Xong khi:** forward trên batch validation hữu hạn, đánh giá validation sau reload khớp log trong sai số số học phù hợp. Nếu có lỗi, xử lý lỗi và ghi lại lý do trước khi chốt run.

- [x] T020 Fine-tune ConvNeXt-Tiny pretrained trên đầy đủ training/validation.
  **Phụ thuộc:** T019; giữ protocol T011.
  **Đầu ra:** run ConvNeXt-Tiny riêng, đầy đủ checkpoint/logs như T018; tạo model và loaders mới theo seed đã chốt.
  **Lưu ý:** chạy tuần tự trên GPU để tránh hai run cạnh tranh bộ nhớ và làm sai lệch phép đo thời gian.

- [x] T021 Kiểm tra run và nạp lại checkpoint ConvNeXt-Tiny.
  **Đã xác nhận:** notebook05 counts1–4 không error; strict load/forward/validation PASS. Đủ5.000 ảnh validation: accuracy98,50%, loss0,0583033, khớp log trong tolerance; không đánh giá test.
  **Bằng chứng:** reports/checkpoint_reload_convnext_tiny_seed42_20261117_191514_964199_20261117_195006_056631.json
  **Phụ thuộc:** T020.
  **Đầu ra:** cùng nội dung kiểm tra như T019, áp dụng cho checkpoint và logs ConvNeXt-Tiny.
  **Xong khi:** checkpoint nạp được, validation khớp log và đủ metadata cho bước bàn giao.

- [x] T022 Hoàn tất số seed/số run đã thống nhất, nếu protocol yêu cầu nhiều hơn một seed.
  **Kết luận 17/11:** không áp dụng chạy thêm seed. Protocol T011 chỉ dùng seed42, mỗi model một run; hai run T018/T020 và reload T019/T021 đã đạt. Không cần chạy lặp.
  **Phụ thuộc:** T019, T021 và quyết định ở T011.
  **Đầu ra:** đủ các cặp run theo danh sách đã chốt, mỗi run có artifacts riêng; không tự mở rộng số thí nghiệm sau khi xem test.
  **Nếu chỉ dùng một seed:** ghi rõ “không áp dụng chạy thêm seed”; không bắt buộc chạy lặp để đóng task này.

- [x] T023 Chốt danh sách run/checkpoint dùng cho báo cáo trong `reports/run_manifest.csv`.
  **Đã có 17/11:** CSV gồm2 dòng, ResNet18 seed42/best epoch9 và ConvNeXt-Tiny seed42/best epoch10; status=selected_reload_verified, test_evaluated=false. Lưu config/split/best.pt/metrics/summary/reload report và SHA256; đã đối chiếu nguồn và đường dẫn. Chọn bằng validation accuracy, hòa xét validation loss.
  **Phụ thuộc:** T019, T021, T022 đã xử lý theo protocol.
  **Đầu ra:** model, seed, run_name, config path, split path, best checkpoint, best epoch, metrics/summary path và trạng thái hợp lệ.
  **Xong khi:** lựa chọn dựa trên train/validation và được cố định trước đánh giá test. Không gộp các run khác protocol thành một cặp so sánh mà không giải thích.

## 5. Bàn giao và hỗ trợ đánh giá trên máy GPU

Các checkbox ở đây vẫn là việc của bạn: chuẩn bị, nhận bàn giao, tích hợp và hỗ trợ chạy. Không bao gồm việc tự viết các module đánh giá/profiling/trực quan hóa.

- [ ] T024 Chuẩn bị và bàn giao bộ artifacts đã chốt cho bước đánh giá.
  **Đã chuẩn bị 17/11:** gói ZIP Quy có hai best.pt và đầy đủ metadata/hướng dẫn; xem reports/handoff_phuc.md để lấy đường dẫn. ZIP/sha checkpoint đã kiểm tra. Chưa gửi/chưa có xác nhận Quy truy cập được; giữ task mở.
  **Phụ thuộc:** T023.
  **Đầu ra:** checkpoint của từng run, config, metrics CSV, summary, class order, split hoặc thông tin tái lập split, môi trường và hướng dẫn load.
  **Nơi bàn giao:** thống nhất nơi lưu checkpoint lớn; không đưa dataset/checkpoint lên Git. Ghi đường dẫn và nội dung bàn giao tại `reports/handoff_phuc.md`.
  **Người nhận:** Quy. Chỉ tick khi đã thực sự bàn giao hoặc xác nhận truy cập được.

- [ ] T025 Chia sẻ cấu hình cuối và các thay đổi ảnh hưởng mô tả phương pháp.
  **Đã chuẩn bị 17/11:** reports/method_summary_khang.md và gói ZIP Khang; hướng dẫn/tin nhắn mẫu tại reports/handoff_phuc.md. Chưa chia sẻ thực tế, giữ task mở.
  **Phụ thuộc:** T011; cập nhật lại sau T023 nếu thực nghiệm có thay đổi.
  **Đầu ra:** bản config/protocol được chia sẻ và nội dung cần đồng bộ trong báo cáo.
  **Người nhận:** Khang. Không cần chờ các hình/bảng kết quả để chia sẻ cấu hình.

- [x] T026 Nhận phần code đánh giá/profiling để tích hợp và xác nhận nạp được checkpoint trên máy mình.
  **Đã nhận:** code và CLI từ `origin/main` (`341bf0f`); `evaluate`/`profiling` import và `--help` đều PASS trong `.venv-1`.
  **Đã kiểm chứng 16/11:** dùng `evaluate_checkpoint` với hai checkpoint smoke, thay loader trong harness bằng cùng 10 ảnh validation (mỗi lớp một ảnh), không duyệt test. Cả hai model chạy CUDA, xuất logits/confusion matrix (10,10), metrics/predictions hợp lệ. Profiling mới kiểm tra import/CLI, chưa đo hiệu năng.
  **Bằng chứng:** `artifacts/notebook_smoke/post_pull_review/20261116_c04b0de/review_report.json`.
  **Phạm vi còn lại:** protocol P1 và xác nhận nhóm vẫn ở T008/T010/T011. `visualize.py` đã nhận nhưng chưa import được vì thiếu matplotlib; thư viện này chưa có trong requirements.txt.

- [ ] T027 Hỗ trợ đánh giá test của các checkpoint đã chốt trên máy GPU khi cần.
  **Phụ thuộc:** T023, T024, T026.
  **Đã có:** code đánh giá và kiểm tra tích hợp nhỏ ở T026. Còn cần checkpoint chính thức T023/T024 và protocol cuối; **chờ Khang** xác nhận pipeline dùng chung ở T008, phối hợp **Quy** chốt tùy chọn đánh giá trước khi chạy test cuối.
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
| T008 — đã ghi nhận | Phúc xác nhận Khang đã đồng ý dữ liệu dùng chung ngày 16/11 | Tiếp tục T012, T013 |
| T010 — đã ghi nhận | Phúc xác nhận Quy đã đồng ý metrics/timing ngày 16/11 | Tiếp tục T012, T013; FLOPs trực tiếp vẫn thuộc C1 |
| T026 — đã xong trong phạm vi tích hợp | Đã nhận code; import/CLI và hai checkpoint trên 10 ảnh validation PASS | Tiếp tục T012/T013; protocol baseline đã chốt, không chờ viết lại module đánh giá |
| T027/T028 | Quy bàn giao công cụ đã kiểm chứng | Viết báo cáo phương pháp, kiểm tra artifacts, chuẩn bị hướng dẫn |
| T030/T034 | Quy bàn giao bảng/hình/nhận xét | T016/T017 đã viết phần phương pháp; làm T032, phần hướng dẫn T036 hoặc T015 nếu chưa chạy |
| T033 | Khang bàn giao nội dung hoặc xác nhận bản hiện có | Hoàn thiện phần model/training của bạn và các việc độc lập còn lại |

## 9. Các cải tiến chỉ làm khi phát sinh nhu cầu

Các mục này chưa là điều kiện bắt buộc để bắt đầu T018/T020:

- Resume training: chỉ bổ sung khi cần tiếp tục sau gián đoạn. Checkpoint hiện có model/optimizer/scheduler state nhưng chưa có luồng resume hoàn chỉnh hoặc AMP scaler/RNG state để khôi phục đầy đủ.
- Freeze backbone: chỉ kiểm tra thêm gradient và BatchNorm khi quyết định dùng freeze; cấu hình khởi đầu hiện là fine-tune toàn mạng.
- Log LR riêng hoặc đo tổng thời gian toàn run: bổ sung nếu protocol yêu cầu. `epoch_seconds` hiện không phải toàn bộ thời gian từ tải dữ liệu đến lưu xong artifacts.
- Nếu thay đổi hành vi training để thêm các mục trên, kiểm chứng phần bị ảnh hưởng trước run chính thức tiếp theo.

**Ghi chú về công cụ lập kế hoạch:** lượt tạo file không tìm thấy skill `speckit-tasks`, nên kế hoạch được lập trực tiếp từ phân công, checklist và code; không khẳng định đã chạy skill đó.

**Rà soát sau pull 16/11/2026:** HEAD `c04b0de`, origin/main `341bf0f`. Tick thêm T007 (chia sẻ qua repo) và T026 (nhận/tích hợp code), tổng cộng 8/40 task đã hoàn thành đúng phạm vi. Notebook 01/02/03 vẫn có 10/5/9 cell code đã lưu không lỗi; chưa có notebook 04, protocol/manifest hoặc run training đầy đủ mới. Chỉ kiểm tra cú pháp/import/CLI và inference trên 10 ảnh validation; không sửa code/notebook/Word, không cài thư viện, không training, test chính thức hoặc benchmark.

**Hoàn thiện Phase 2 ngày 16/11/2026:** Phúc trả lời Khang/Quy đã xác nhận các thiết lập được hỏi. T008/T009/T010/T011 đã tick, tổng cộng 12/40 task hoàn thành trong phạm vi ghi rõ. T012 chỉ mới có notebook 4 cell code chưa chạy, chưa có file môi trường mới; chờ bạn chạy/lưu và cung cấp kết quả. Từ nay assistant không tự chạy chương trình sau khi sửa, chỉ hướng dẫn bạn và xem output bạn cung cấp.

**Xác nhận T012 ngày17/11/2026:** đọc notebook 4 cell code counts6–9 và 4 file kết quả do Phúc lưu; mọi readiness check đạt, fingerprints config/data.py/split khớp hiện tại. Tick T012, tổng13/40 task; Phase 2 hoàn tất. Matplotlib còn thiếu, không chạy lại/cài gói; bước tiếp theo T013.

**Chuẩn bị Phase3 ngày17/11/2026:** T013/T014/T016/T017 đã hoàn thiện phần source/tài liệu, tổng17/40 task. Notebook02 và04 đều chưa chạy; T015 còn mở theo quy tắc Phúc tự chạy. Mở notebook04, chọn .venv-1, giữ RUN_FULL_TRAINING=False, Run All, Ctrl+S và gửi output T015 PREFLIGHT hoặc traceback. Không cần chạy lại toàn bộ notebook01–03. Không thực hiện training/test/benchmark hoặc cài gói trong lượt này; chỉ chỉnh file và kết xuất Word để xem bố cục.

**Xác nhận T015 ngày17/11/2026:** chỉ đọc output notebook04 (4 cell counts1–4 không error) và reports/phase3_preflight_20261117_174231_303611.json (17:43:11+08:00). Batches train/validation224/32/workers4, class order, CUDA, số mẫu, tên run và dung lượng94,48GiB đạt; 6 fingerprints khớp file hiện tại. Tick T015, tổng18/40; Phase3 hoàn thành và X1 sẵn sàng. Training=False/SKIP, không tick T018/T020. Tiếp theo Phúc tự chạy ResNet18 T018 rồi kiểm tra T019 trước ConvNeXt-Tiny T020. Không chạy lại notebook/code hoặc cài gói trong lượt rà soát.


**Rà soát run ResNet18 ngày17/11/2026:** run resnet18_seed42_20261117_175014_404116 đã hoàn tất10 epoch trên45.000 train/5.000 validation, CUDA/AMP, đúng protocol. Best validation accuracy96,46% ở epoch9, loss0,1373025; epoch10 đạt96,42%. CSV hữu hạn, LR cosine/config/best epoch khớp, best.pt/last.pt và ba file logs tồn tại. Tick T018/X2, tổng19/40 task. Chưa strict load hoặc tính validation lại cho checkpoint chính thức: T019 còn mở; chưa chạy ConvNeXt-Tiny chính thức, test hoặc benchmark. Bằng chứng file-only: reports/training_review_resnet18_seed42_20261117_175014_404116.json. Notebook05 đã chuẩn bị8 cell/4 code, chưa chạy; Phúc chọn .venv-1, Run All, Ctrl+S và gửi output/report checkpoint_reload để kiểm tra T019. Assistant chỉ đọc/ghi file, không chạy project, không sửa notebook04 hoặc kết quả gốc.


**Xác nhận T019 ngày 17/11/2026:** Phúc đã chạy và lưu notebook05, 4 cell code counts1–4 không error; T019/T021 CHECKPOINT RELOAD: PASS áp dụng cho ResNet18 (không phải T021 ConvNeXt-Tiny). Run resnet18_seed42_20261117_175014_404116, best.pt epoch9: strict load/metadata/class order/fingerprints/weights finite đạt, forward FP32 batch(32,10) hữu hạn trên CUDA. Tính lại đủ5.000 ảnh validation với AMP đạt accuracy0,9646 (khớp log), loss0,13729811782836915 so với0,13730251471996308; chênh lệch-0,000004396891594 nằm trong tolerance. Report: reports/checkpoint_reload_resnet18_seed42_20261117_175014_404116_20261117_181643_179508.json. Tick T019, tổng20/40 task; tiếp theo T020 ConvNeXt-Tiny rồi T021. Không training/test/benchmark trong notebook05. Assistant chỉ đọc kết quả Phúc lưu và cập nhật tracking, không chạy code hoặc sửa notebook/raw report.


**Cập nhật bổ sung ngày 17/11/2026 — T020:** Phúc đã có output notebook04.1 counts1–4 không lỗi và run convnext_tiny_seed42_20261117_191514_964199 hoàn tất10 epoch trên45.000/5.000 ảnh với CUDA/AMP, đúng baseline. Best epoch10: validation accuracy98,50%, loss0,0583064739. CSV hữu hạn, summary khớp trong sai số đọc số thực và đủ best/last/config/CSV/summary. Tick T020/X3, tổng21/40 task. Notebook05 hiện vẫn lưu PASS ResNet18 T019; chưa có reload ConvNeXt-Tiny, T021 còn mở. Tiếp theo: đổi RUN_NAME trong notebook05 thành convnext_tiny_seed42_20261117_191514_964199, chọn .venv-1, Restart Kernel → Run All → Ctrl+S rồi gửi kết quả. Không chạy lại training hoặc test. Hai notebook training đã được Phúc đặt tên riêng: 04_Huan_luyen_chinh_thuc_CIFAR10_resnet18.ipynb và 04.1_Huan_luyen_chinh_thuc_CIFAR10_convnext_tiny.ipynb. Assistant chỉ đọc/ghi tracking; giữ nguyên notebook và kết quả gốc.


17/11/2026 — Theo yêu cầu Phúc, notebook05 đã đổi RUN_NAME sang convnext_tiny_seed42_20261117_191514_964199 để làm T021. Bản notebook ResNet18 đã PASS (có output) được giữ tại artifacts/notebook_smoke/checkpoint_reload_backups/20261117_194853_4610303/05_ResNet18_da_PASS.ipynb; report ResNet18 gốc giữ nguyên. Notebook05 hiện xóa output/execution_count cũ để không gán PASS ResNet18 cho ConvNeXt-Tiny. Chưa chạy bản vừa sửa; T021 vẫn chưa tick, tổng21/40 task. Phúc chọn .venv-1 → Restart Kernel → Run All → Ctrl+S rồi gửi kết quả. Assistant chỉ sửa/đọc file, không thực thi notebook/model.


**17/11/2026 — T021 hoàn thành:** Phúc đã chạy và lưu notebook05 cho run convnext_tiny_seed42_20261117_191514_964199; 4 cell code counts1–4, không lỗi, CHECKPOINT RELOAD: PASS. Best checkpoint epoch10 nạp strict=True, metadata/class order/fingerprints/weights hữu hạn khớp; forward validation FP32(32,10) hữu hạn trên CUDA RTX5070Ti. Tính lại đủ5.000 ảnh validation với AMP: accuracy98,50% khớp log, loss0,058303282952308653 so với0,058306473898887635; chênh lệch-0,000003190946579 trong tolerance. Bằng chứng: reports/checkpoint_reload_convnext_tiny_seed42_20261117_191514_964199_20261117_195006_056631.json. Tick T021, tổng22/40 task. Cả hai run chính thức đã training/reload đạt; chưa đánh giá test hoặc benchmark. Bước tiếp T022 (ghi nhận baseline chỉ seed42, không chạy thêm seed) và T023 (chốt manifest hai run), sau đó T024 bàn giao artifacts cho Quy. Assistant chỉ đọc output/file và cập nhật tracking; không chạy code hoặc sửa notebook/raw report/checkpoint.


**T022–T023 hoàn tất ngày 17/11/2026:** baseline cifar10-baseline-v1 chỉ dùng seed42, một run/model; hai run đã đủ và reload PASS. T022: “không áp dụng chạy thêm seed”; không cần huấn luyện lặp. T023: đã chốt hai best.pt tại reports/run_manifest.csv trước đánh giá test: ResNet18 run resnet18_seed42_20261117_175014_404116, epoch9, validation0,9646; ConvNeXt-Tiny run convnext_tiny_seed42_20261117_191514_964199, epoch10, validation0,985. CSV ghi config/split/metrics/summary/reload report, protocol và SHA256 checkpoint/split; accuracy ở thang0–1. Đã đối chiếu file nguồn/CSV/protocol/fingerprints/class order, đọc lại manifest và kiểm tra đường dẫn/hash. Chỉ xử lý file bằng công cụ tài liệu, không chạy project/model/notebook/test/benchmark hoặc cài gói. Tổng24/40 task; tiếp theo T024 bàn giao artifacts cho Quy và T025 chia sẻ cấu hình cho Khang. Chưa gửi file/tin nhắn, chưa xác nhận thành viên khác truy cập được; T024/T025 còn mở.


17/11/2026 — Chuẩn bị T024/T025: tạo reports/handoff_phuc.md (nội dung, hướng dẫn gửi, bảng trạng thái, tin nhắn mẫu cho Phúc tự gửi) và reports/method_summary_khang.md (config/dữ liệu/thay đổi phương pháp cho Khang). Đã đóng hai ZIP local dưới artifacts/notebook_smoke/handoff_preparation; Quy nhận hai best.pt và logs/split/class order/protocol/môi trường/code snapshot/hướng dẫn load; Khang nhận bộ phương pháp nhỏ không có checkpoint. Đã kiểm tra entries và SHA256 checkpoint trong ZIP khớp manifest. Không gồm dataset/venv/.env/AGENTS/Word, không đổi .gitignore. Chỉ xử lý file, không chạy project/model/test/benchmark, không cài gói, không tải lên dịch vụ hoặc gửi tin nhắn. T024/T025 vẫn chưa tick vì chưa gửi/link/ack thực tế; tổng24/40. Phúc cần tự gửi cho Quy/Khang, ghi nơi gửi/thời điểm và bằng chứng truy cập vào handoff_phuc.md rồi báo kết quả.
