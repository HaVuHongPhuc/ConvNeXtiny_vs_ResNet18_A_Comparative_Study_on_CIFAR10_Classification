# MODULE: evaluate.py
#
# Vai trò: nạp checkpoint validation-best, đánh giá trên test set và lưu
# metrics/dự đoán để lập bảng so sánh và phân tích lỗi.
#
# Hướng dẫn triển khai:
# 1. Dùng config/transforms và class order tương ứng với checkpoint.
# 2. Chạy eval/no_grad; tính accuracy, macro-F1, per-class precision/recall,
#    loss và confusion matrix.
# 3. Lưu true label, predicted label, confidence/logits và index mẫu.
# 4. Báo rõ checkpoint, split và các điều kiện đánh giá.
# Không cập nhật weights hay điều chỉnh hyperparameter bằng test results.
