# MODULE: data.py
#
# Vai trò: nạp CIFAR-10, tạo split train/validation có thể tái lập,
# định nghĩa transforms và cung cấp DataLoader cho train/validation/test.
#
# Hướng dẫn triển khai:
# 1. Kiểm tra số mẫu, nhãn, class mapping và shape ảnh [3, 32, 32].
# 2. Chỉ chia tập train gốc thành train/validation; giữ test chuẩn riêng.
# 3. Chỉ dùng augmentation ngẫu nhiên cho train; val/test phải xác định.
# 4. Ghi rõ resize, interpolation và normalization của pretrained weights.
# 5. Trả về loaders cùng metadata cần thiết để tái lập split/transforms.
# Không đặt training loop hoặc logic chọn checkpoint trong module này.
