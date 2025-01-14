# system_manager.py

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any
import aiofiles

from .resource_manager import ResourceManager
from .anomaly_detector import AnomalyDetector
from .logging_config import setup_logging

###############################################################################
#                        KHAI BÁO BIẾN VÀ LOGGER CƠ BẢN                       #
###############################################################################

CONFIG_DIR = Path(os.getenv('CONFIG_DIR', '/app/mining_environment/config'))
LOGS_DIR = Path(os.getenv('LOGS_DIR', '/app/mining_environment/logs'))

system_logger = setup_logging('system_manager', LOGS_DIR / 'system_manager.log', 'INFO')
resource_logger = setup_logging('resource_manager', LOGS_DIR / 'resource_manager.log', 'INFO')
anomaly_logger = setup_logging('anomaly_detector', LOGS_DIR / 'anomaly_detector.log', 'INFO')

_system_manager_instance = None  # Singleton cho SystemManager

###############################################################################
#                           LỚP CHÍNH: SystemManager                          #
###############################################################################

class SystemManager:
    """
    Lớp quản lý chính cho hệ thống, bao gồm:
      - Khởi tạo ResourceManager và AnomalyDetector
      - Xử lý luồng khởi động (start) và dừng (stop)
      - Quản lý vòng đời toàn bộ hệ thống
    """
    def __init__(self, config: Dict[str, Any]):
        if not isinstance(config, dict):
            raise ValueError("Cấu hình phải là kiểu dict.")
        self.config = config

        # Khởi tạo các logger cục bộ
        self.system_logger = system_logger
        self.resource_logger = resource_logger
        self.anomaly_logger = anomaly_logger

        # Khởi tạo ResourceManager (event-driven & async) 
        self.resource_manager = ResourceManager(config, resource_logger)

        # Khởi tạo AnomalyDetector (để phát hiện bất thường) với DI
        self.anomaly_detector = AnomalyDetector(config, anomaly_logger, self.resource_manager)

        self.system_logger.info("SystemManager đã được khởi tạo.")

    async def start(self):
        """
        Bắt đầu chạy các thành phần của hệ thống theo kiểu async.
        Tối ưu để tích hợp với kiến trúc event-driven:
          1) Gọi resource_manager.start() để khởi tạo ResourceManager (bao gồm GPUManager, Azure Clients, và watchers).
          2) Gọi anomaly_detector.start() nếu cần giám sát bất thường bổ sung.
        """
        self.system_logger.info("Đang khởi động SystemManager...")

        try:
            # 1. Khởi động ResourceManager
            await self.resource_manager.start()

            # 2. Khởi chạy AnomalyDetector (nếu có hàm start)
            if hasattr(self.anomaly_detector, 'start') and callable(self.anomaly_detector.start):
                await self.anomaly_detector.start()

            self.system_logger.info("SystemManager đã khởi động thành công.")
        except Exception as e:
            self.system_logger.error(f"Lỗi khi khởi động SystemManager: {e}")
            await self.stop()  # Dừng nếu có lỗi khi start
            raise

    async def stop(self):
        """
        Dừng các thành phần của hệ thống theo thứ tự an toàn:
          1) Dừng AnomalyDetector
          2) Dừng ResourceManager (shutdown watchers và restore tài nguyên)
        """
        self.system_logger.info("Đang dừng SystemManager...")
        try:
            # 1. Dừng AnomalyDetector
            if hasattr(self.anomaly_detector, 'stop') and callable(self.anomaly_detector.stop):
                await self.anomaly_detector.stop()

            # 2. Dừng ResourceManager (gọi hàm shutdown() để tắt watchers)
            await self.resource_manager.shutdown()

            self.system_logger.info("SystemManager đã dừng thành công.")
        except Exception as e:
            self.system_logger.error(f"Lỗi khi dừng SystemManager: {e}")
            raise

###############################################################################
#                        HÀM HỖ TRỢ TẢI CẤU HÌNH TỪ JSON                      #
###############################################################################

async def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Tải cấu hình từ tệp JSON (bất đồng bộ).
    """
    try:
        async with aiofiles.open(config_path, 'r') as f:
            content = await f.read()
            config = json.loads(content)
        system_logger.info(f"Đã tải cấu hình từ {config_path}")

        # Xác minh cấu hình phải là dict
        if not isinstance(config, dict):
            system_logger.error("Cấu hình không phải kiểu dict.")
            sys.exit(1)
        return config

    except FileNotFoundError:
        system_logger.error(f"Tệp cấu hình không tìm thấy: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        system_logger.error(f"Lỗi cú pháp JSON trong tệp cấu hình {config_path}: {e}")
        sys.exit(1)
    except Exception as e:
        system_logger.error(f"Lỗi khi tải cấu hình: {e}")
        sys.exit(1)

###############################################################################
#               HÀM main() CHÍNH SỬ DỤNG asyncio.run() ĐỂ KHỞI TẠO           #
###############################################################################

async def main():
    """
    Hàm chính bất đồng bộ:
      - Tải config
      - Khởi tạo và start SystemManager
      - Duy trì vòng lặp cho đến khi nhận tín hiệu dừng
    """
    global _system_manager_instance

    # 1) Tải config
    resource_config_path = CONFIG_DIR / "resource_config.json"
    config = await load_config(resource_config_path)

    # 2) Khởi tạo SystemManager
    _system_manager_instance = SystemManager(config)

    # 3) Bắt đầu chạy SystemManager
    try:
        await _system_manager_instance.start()
        system_logger.info("SystemManager đang chạy. Nhấn Ctrl+C để dừng.")

        # Vòng lặp duy trì vô hạn, chờ tín hiệu
        while True:
            await asyncio.sleep(3600)  # Ngủ 1h, sau đó lặp

    except asyncio.CancelledError:
        system_logger.info("Coroutine chính đã bị hủy.")
    except KeyboardInterrupt:
        system_logger.info("Nhận tín hiệu dừng từ người dùng.")
    finally:
        # Dừng SystemManager khi thoát vòng lặp
        await _system_manager_instance.stop()

###############################################################################
#                     HÀM start() VÀ stop() DÙNG CHO ENTRYPOINT               #
###############################################################################

def start():
    """
    Hàm entrypoint để chạy SystemManager dưới dạng script.
    Đảm bảo chạy với quyền root, sau đó gọi asyncio.run(main()).
    """
    if os.geteuid() != 0:
        print("Script phải được chạy với quyền root.")
        sys.exit(1)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        system_logger.info("Đang dừng SystemManager do KeyboardInterrupt.")
    except Exception as e:
        system_logger.error(f"Lỗi trong quá trình chạy SystemManager: {e}")
        sys.exit(1)

def stop():
    """
    Hàm dừng SystemManager nếu nó đã được khởi tạo.
    """
    global _system_manager_instance
    if _system_manager_instance:
        asyncio.run(_system_manager_instance.stop())
        system_logger.info("SystemManager đã dừng thành công.")
    else:
        system_logger.warning("SystemManager chưa được khởi tạo.")

if __name__ == "__main__":
    start()
