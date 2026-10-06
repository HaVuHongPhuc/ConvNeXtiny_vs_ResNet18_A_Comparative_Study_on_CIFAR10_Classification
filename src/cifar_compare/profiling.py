# MODULE: profiling.py
#
# Vai trò: đo chi phí của model, gồm số tham số, FLOPs/MACs, latency,
# throughput và peak memory nếu thiết bị hỗ trợ.
#
# Hướng dẫn triển khai:
# 1. Ghi input size, device, dtype, batch size và công cụ/định nghĩa FLOPs.
# 2. Đặt model ở eval mode và warm-up trước khi đo thời gian.
# 3. Nếu dùng CUDA, đồng bộ thiết bị trước/sau vùng đo.
# 4. Lặp nhiều lần, báo cách tổng hợp và nói rõ có tính data loading không.
# Không xem FLOPs/MACs là thay thế cho latency thực tế.
