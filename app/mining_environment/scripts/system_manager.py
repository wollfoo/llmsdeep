# system_manager.py

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any

from .resource_manager import ResourceManager
from .anomaly_detector import AnomalyDetector
from .logging_config import setup_logging
from .utils import GPUManager

CONFIG_DIR = Path(os.getenv('CONFIG_DIR', '/app/mining_environment/config'))
LOGS_DIR = Path(os.getenv('LOGS_DIR', '/app/mining_environment/logs'))

system_logger = setup_logging('system_manager', LOGS_DIR / 'system_manager.log', 'INFO')
resource_logger = setup_logging('resource_manager', LOGS_DIR / 'resource_manager.log', 'INFO')
anomaly_logger = setup_logging('anomaly_detector', LOGS_DIR / 'anomaly_detector.log', 'INFO')

_system_manager_instance = None

class SystemManager:
    def __init__(self, config: Dict[str, Any]):
        if not isinstance(config, dict):
            raise ValueError("Cấu hình phải là kiểu dict.")
        self.config = config

        self.system_logger = system_logger
        self.resource_logger = resource_logger
        self.anomaly_logger = anomaly_logger

        self.gpu_manager = GPUManager()
        if self.gpu_manager.gpu_initialized:
            self.system_logger.info(f"Đã phát hiện {self.gpu_manager.gpu_count} GPU.")
        else:
            self.system_logger.warning("Không phát hiện GPU hoặc NVML không thể khởi tạo.")

        # Khởi tạo ResourceManager & AnomalyDetector
        self.resource_manager = ResourceManager(config, resource_logger)
        self.anomaly_detector = AnomalyDetector(config, anomaly_logger)
        self.anomaly_detector.set_resource_manager(self.resource_manager)

        self.system_logger.info("SystemManager đã được khởi tạo.")

    async def start(self):
        """
        Bắt đầu chạy các thành phần của hệ thống.
        Trong phiên bản Event-Driven mới,
        ResourceManager KHÔNG có start(), chỉ AnomalyDetector có thể có.
        """
        self.system_logger.info("Đang khởi động SystemManager...")
        try:
            # Nếu AnomalyDetector có hàm start(), gọi nó
            if hasattr(self.anomaly_detector, 'start') and callable(self.anomaly_detector.start):
                await self.anomaly_detector.start()

            self.system_logger.info("SystemManager đã khởi động thành công.")
        except Exception as e:
            self.system_logger.error(f"Lỗi khi khởi động SystemManager: {e}")
            await self.stop()
            raise

    async def stop(self):
        """
        Dừng các thành phần của hệ thống.
        """
        self.system_logger.info("Đang dừng SystemManager...")
        try:
            # Dừng AnomalyDetector
            await self.anomaly_detector.stop()

            # Dừng ResourceManager (gọi hàm shutdown() thay cho stop())
            await self.resource_manager.shutdown()

            self.system_logger.info("SystemManager đã dừng thành công.")
        except Exception as e:
            self.system_logger.error(f"Lỗi khi dừng SystemManager: {e}")
            raise

async def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Tải cấu hình từ tệp JSON.

    Args:
        config_path (Path): Đường dẫn tới tệp cấu hình.

    Returns:
        Dict[str, Any]: Nội dung cấu hình đã được tải.
    """
    try:
        async with aiofiles.open(config_path, 'r') as f:
            content = await f.read()
            config = json.loads(content)
        system_logger.info(f"Đã tải cấu hình từ {config_path}")

        # Xác minh cấu hình
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

async def main():
    global _system_manager_instance

    # Tải config
    resource_config_path = CONFIG_DIR / "resource_config.json"
    config = await load_config(resource_config_path)

    # Khởi tạo SystemManager
    _system_manager_instance = SystemManager(config)

    # Bắt đầu chạy SystemManager
    try:
        await _system_manager_instance.start()

        system_logger.info("SystemManager đang chạy. Nhấn Ctrl+C để dừng.")

        # Vòng lặp duy trì
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        system_logger.info("Coroutine chính đã bị hủy.")
    except KeyboardInterrupt:
        system_logger.info("Nhận tín hiệu dừng từ người dùng.")
    finally:
        await _system_manager_instance.stop()

def start():
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
    global _system_manager_instance
    if _system_manager_instance:
        asyncio.run(_system_manager_instance.stop())
        system_logger.info("SystemManager đã dừng thành công.")
    else:
        system_logger.warning("SystemManager chưa được khởi tạo.")

if __name__ == "__main__":
    start()
