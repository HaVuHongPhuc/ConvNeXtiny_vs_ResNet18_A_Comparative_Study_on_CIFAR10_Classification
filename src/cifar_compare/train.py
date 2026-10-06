# MODULE: train.py
#
# Vai trò: điều phối một training loop dùng chung cho cả hai kiến trúc,
# tính train/validation metrics và lưu checkpoint tốt nhất.
#
# Hướng dẫn triển khai:
# 1. Đọc config, khởi tạo seed/device, DataLoaders và model.
# 2. Mỗi batch: forward -> CrossEntropy loss -> backward -> optimizer step.
# 3. Cuối epoch, đánh giá validation ở eval/no_grad và ghi log.
# 4. Chọn checkpoint chỉ bằng validation; lưu weights, epoch và config run.
# 5. Cho phép chọn model/config từ CLI nhưng giữ chung logic huấn luyện.
# Không dùng test set để chọn epoch, hyperparameter hoặc checkpoint.
