# MODULE: utils.py
#
# Vai trò: chứa tiện ích nhỏ dùng chung như đặt seed, đọc/kiểm tra config,
# tạo thư mục output, logging và lưu metadata môi trường.
#
# Hướng dẫn triển khai:
# 1. Giữ mỗi helper độc lập, có đầu vào/đầu ra rõ ràng.
# 2. Dùng đường dẫn tương đối từ project/config thay vì hard-code máy cá nhân.
# 3. Không ghi secrets vào logs; không gom logic model/data/training lớn vào đây.
