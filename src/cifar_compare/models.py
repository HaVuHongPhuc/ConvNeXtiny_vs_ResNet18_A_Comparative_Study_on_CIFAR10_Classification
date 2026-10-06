# MODULE: models.py
#
# Vai trò: khởi tạo ConvNeXt-Tiny và ResNet18 pretrained, thay classifier
# cuối thành 10 lớp và cung cấp cách đóng băng/mở khóa backbone nếu cần.
#
# Hướng dẫn triển khai:
# 1. Dùng weights enum cụ thể của torchvision và ghi lại tên weights.
# 2. Giữ nguyên pretrained backbone; chỉ thay head phân loại.
# 3. Kiểm tra output là logits có shape [batch_size, 10].
# 4. Cho phép chọn kiến trúc qua tên model, không tạo hai training loop riêng.
# Không đặt dữ liệu, optimizer hoặc vòng lặp huấn luyện trong module này.
