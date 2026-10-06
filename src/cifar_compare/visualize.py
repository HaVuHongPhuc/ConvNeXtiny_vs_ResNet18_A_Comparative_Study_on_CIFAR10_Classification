# MODULE: visualize.py
#
# Vai trò: tạo hình từ logs/metrics/predictions đã lưu: learning curves,
# confusion matrix và các ảnh phân loại sai.
#
# Hướng dẫn triển khai:
# 1. Đọc artifacts đã lưu thay vì chạy lại training trong module này.
# 2. Ghi tên model, class order, split và metric lên biểu đồ.
# 3. Hiển thị ảnh sai kèm nhãn thật, nhãn dự đoán và confidence.
# 4. Lưu hình vào artifacts/figures/ để báo cáo có thể tái tạo.
# Không dùng hình test để điều chỉnh tham số; nếu cần tuning hãy dùng validation.
