"""
Module system_manager.py

Chịu trách nhiệm khởi tạo cấu hình, tạo SystemManager, và quản lý vòng đời
(hàm main) của hệ thống theo mô hình đồng bộ (Synchronous + Threading).

Đã loại bỏ hoàn toàn asyncio, await và chuyển các hàm bất đồng bộ
thành đồng bộ, đồng thời đảm bảo tính tương thích với facade.py, event_bus, models, v.v.
"""

import os
import sys
import json
import logging
import uuid
import threading
from pathlib import Path
from typing import Dict, Any

from .facade import SystemFacade
from .logging_config import setup_logging, correlation_id
from .auxiliary_modules.event_bus import EventBus
from .auxiliary_modules.models import ConfigModel

###############################################################################
#                   ĐỊNH NGHĨA ĐƯỜNG DẪN & LOGGER                             #
###############################################################################
CONFIG_DIR = Path(os.getenv('CONFIG_DIR', '/app/mining_environment/config'))
LOGS_DIR = Path(os.getenv('LOGS_DIR', '/app/mining_environment/logs'))
LOGS_DIR.mkdir(parents=True, exist_ok=True)

system_logger = setup_logging('system_manager', LOGS_DIR / 'system_manager.log', 'INFO')
resource_logger = setup_logging('resource_manager', LOGS_DIR / 'resource_manager.log', 'INFO')
anomaly_logger = setup_logging('anomaly_detector', LOGS_DIR / 'anomaly_detector.log', 'INFO')


###############################################################################
#                 HÀM TẢI CẤU HÌNH TỪ TỆP JSON (ĐỒNG BỘ)                     #
###############################################################################
def load_config(config_path: Path) -> ConfigModel:
    """
    Đọc file JSON chứa cấu hình tại config_path, parse thành ConfigModel (đồng bộ).

    :param config_path: Đường dẫn tới tệp JSON.
    :return: Đối tượng ConfigModel chứa dữ liệu cấu hình.
    :raises SystemExit: Nếu file không tồn tại hoặc JSON lỗi.
    """
    try:
        with config_path.open('r') as f:
            content = f.read()
            config_data = json.loads(content)
        system_logger.info(f"Đã tải cấu hình từ {config_path}.")
        config = ConfigModel(**config_data)
        system_logger.info("Cấu hình đã được parse thành công (đồng bộ).")
        return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        system_logger.error(f"Lỗi khi tải cấu hình {config_path}: {e}")
        sys.exit(1)
    except Exception as e:
        system_logger.error(f"Lỗi khi tải cấu hình: {e}")
        sys.exit(1)


class SystemManager:
    """
    Lớp SystemManager chịu trách nhiệm:
      - Khởi tạo Facade (chứa ResourceManager, AnomalyDetector, SafeRestoreEvaluator)
      - Quản lý EventBus
      - Triển khai start/stop đồng bộ
      - Cung cấp cơ chế để dừng hệ thống (shutdown event)
    """

    def __init__(self, config: ConfigModel, logger: logging.Logger):
        """
        Khởi tạo SystemManager.

        :param config: Đối tượng ConfigModel cấu hình hệ thống.
        :param logger: Logger cho SystemManager.
        """
        self.config = config
        self.logger = logger

        # Tạo 1 instance EventBus (đồng bộ)
        self.event_bus = EventBus()

        # Khởi tạo facade (đồng bộ)
        self.facade = SystemFacade(config, self.event_bus, resource_logger, anomaly_logger)

        # Gán correlation_id vào contextvar
        self.correlation_id = str(uuid.uuid4())
        correlation_id.set(self.correlation_id)
        self.logger.info(f"SystemManager khởi tạo với Correlation ID: {self.correlation_id}")

        # Sử dụng threading.Event để chờ dừng
        self.stop_event = threading.Event()

        # Lock để tránh start/stop chồng chéo
        self._start_stop_lock = threading.Lock()

        # Mặc định False => chưa chạy
        self._started = False

    def start(self) -> None:
        """
        Khởi động SystemManager và các module (đồng bộ).
        """
        self.logger.info("Đang khởi động SystemManager (đồng bộ)...")
        with self._start_stop_lock:
            if self._started:
                self.logger.warning("SystemManager đã được khởi động trước đó.")
                return

            # EventBus bắt đầu lắng nghe => synchronous
            self.event_bus.start_listening()
            self.logger.info("EventBus đã bắt đầu lắng nghe (đồng bộ).")

            # Khởi động facade => ResourceManager, AnomalyDetector, SafeRestoreEvaluator
            self.facade.start()
            self.logger.info("SystemManager đã khởi động xong.")
            self._started = True

    def stop(self) -> None:
        """
        Dừng SystemManager (đồng bộ), bao gồm facade và event_bus.
        """
        self.logger.info("Đang dừng SystemManager (đồng bộ)...")
        with self._start_stop_lock:
            if not self._started:
                self.logger.warning("SystemManager chưa được khởi động => bỏ qua.")
                return

            self.facade.stop()
            self.logger.info("SystemFacade đã dừng (đồng bộ).")

            # Dừng EventBus => synchronous
            self.event_bus.stop()
            self.logger.info("EventBus đã dừng (đồng bộ).")

            self._started = False
            self.stop_event.set()

    def run(self) -> None:
        """
        Chạy SystemManager, đợi cho tới khi stop_event được set.
        Thử khởi động (start) tối đa 3 lần nếu lỗi xảy ra.

        :raises RuntimeError: Nếu khởi động thất bại sau 3 lần thử.
        """
        attempts = 3
        for attempt in range(attempts):
            try:
                self.start()
                system_logger.info("SystemManager đang chạy. Chờ tín hiệu stop...")
                # Đợi stop_event set
                self.stop_event.wait()
                break
            except Exception as e:
                system_logger.warning(f"Thử khởi động SystemManager thất bại ({attempt + 1}/{attempts}): {e}")
        else:
            raise RuntimeError("Không thể khởi động SystemManager sau 3 lần thử.")

        # Thoát khi stop_event được set
        system_logger.info("SystemManager đã dừng run().")

    def trigger_shutdown(self) -> None:
        """
        Gửi sự kiện 'shutdown' => EventBus => dẫn đến stop().
        """
        try:
            self.logger.info("Phát sự kiện 'shutdown' => event_bus => callback => stop.")
            self.event_bus.publish('shutdown', None)
        except Exception as e:
            self.logger.error(f"Lỗi khi gửi sự kiện 'shutdown': {e}")

    def handle_shutdown(self) -> None:
        """
        Hàm này được gọi khi nhận sự kiện shutdown => dừng system_manager.
        """
        self.logger.info("Nhận sự kiện shutdown => dừng SystemManager.")
        self.stop()


# Singleton instance
_system_manager_instance = None

def start():
    """
    Hàm xuất cho giao diện => start SystemManager (đồng bộ).
    """
    global _system_manager_instance
    if _system_manager_instance:
        system_logger.warning("SystemManager đã được khởi động => bỏ qua.")
        return

    try:
        resource_config_path = CONFIG_DIR / "resource_config.json"
        config = load_config(resource_config_path)

        _system_manager_instance = SystemManager(config, system_logger)
        _system_manager_instance.run()
    except Exception as e:
        system_logger.error(f"Lỗi khi khởi động SystemManager: {e}")
        sys.exit(1)

def stop():
    """
    Hàm xuất cho giao diện => dừng SystemManager (đồng bộ).
    """
    global _system_manager_instance
    if not _system_manager_instance:
        system_logger.warning("SystemManager chưa được khởi tạo => bỏ qua stop.")
        return
    _system_manager_instance.trigger_shutdown()

def main():
    """
    Điểm vào (entry point) => check quyền root => start => stop nếu có KeyboardInterrupt.
    """
    try:
        if os.geteuid() != 0:
            print("Script phải được chạy với quyền root.")
            sys.exit(1)
        start()
    except KeyboardInterrupt:
        system_logger.info("Dừng SystemManager do KeyboardInterrupt.")
        stop()
    except Exception as e:
        system_logger.error(f"Lỗi trong quá trình chạy SystemManager: {e}")
        stop()
        sys.exit(1)

if __name__ == "__main__":
    main()
