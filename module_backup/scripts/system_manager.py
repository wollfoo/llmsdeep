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
#                           ĐỊNH NGHĨA ĐƯỜNG DẪN & LOGGER                     #
###############################################################################

CONFIG_DIR = Path(os.getenv('CONFIG_DIR', '/app/mining_environment/config'))
LOGS_DIR = Path(os.getenv('LOGS_DIR', '/app/mining_environment/logs'))

system_logger = setup_logging(
    logger_name='system_manager',
    log_file=LOGS_DIR / 'system_manager.log',
    level='INFO'
)

###############################################################################
#                      HÀM HỖ TRỢ TẢI CẤU HÌNH TỪ TỆP JSON                    #
###############################################################################

async def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Tải cấu hình từ tệp JSON (bất đồng bộ).

    Args:
        config_path (Path): Đường dẫn đến tệp JSON cấu hình.

    Returns:
        Dict[str, Any]: Dữ liệu cấu hình đã tải.

    Raises:
        SystemExit: Nếu có lỗi khi tải cấu hình.
    """
    try:
        async with aiofiles.open(config_path, 'r') as f:
            content = await f.read()
            config = json.loads(content)
        system_logger.info(f"Đã tải cấu hình từ {config_path}")

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
#                              CLASS: SystemManager                            #
###############################################################################

class SystemManager:
    """
    Lớp quản lý chính cho hệ thống:
      - Khởi tạo ResourceManager và AnomalyDetector.
      - Xử lý luồng khởi động (start) và dừng (stop).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Khởi tạo SystemManager.

        Args:
            config (Dict[str, Any]): Cấu hình hệ thống.

        Raises:
            ValueError: Nếu cấu hình không phải là kiểu dict.
        """
        if not isinstance(config, dict):
            raise ValueError("Cấu hình phải là kiểu dict.")

        self.config = config
        self.system_logger = system_logger

        # Khởi tạo các module chính
        self.resource_manager = ResourceManager(config, system_logger)
        self.anomaly_detector = AnomalyDetector(config, system_logger, self.resource_manager)

        self.system_logger.info("SystemManager đã được khởi tạo.")

    async def start(self):
        """
        Bắt đầu chạy các thành phần của hệ thống.
        """
        self.system_logger.info("Đang khởi động SystemManager...")

        try:
            await self.resource_manager.start()
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
            await self.anomaly_detector.stop()
            await self.resource_manager.shutdown()

            self.system_logger.info("SystemManager đã dừng thành công.")
        except Exception as e:
            self.system_logger.error(f"Lỗi khi dừng SystemManager: {e}")
            raise

###############################################################################
#                               HÀM main() CHÍNH                              #
###############################################################################

async def main():
    """
    Hàm chính bất đồng bộ, khởi tạo SystemManager và chạy hệ thống.
    """
    resource_config_path = CONFIG_DIR / "resource_config.json"
    config = await load_config(resource_config_path)

    system_manager = SystemManager(config)

    try:
        await system_manager.start()
        system_logger.info("SystemManager đang chạy. Nhấn Ctrl+C để dừng.")

        # Vòng lặp “chờ” để giữ SystemManager chạy liên tục
        while True:
            await asyncio.sleep(3600)

    except asyncio.CancelledError:
        system_logger.info("Coroutine chính đã bị hủy.")
    except KeyboardInterrupt:
        system_logger.info("Nhận tín hiệu dừng từ người dùng.")
    finally:
        await system_manager.stop()

###############################################################################
#                              ENTRYPOINT SCRIPT                              #
###############################################################################

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        system_logger.info("Đang dừng SystemManager do KeyboardInterrupt.")
    except Exception as e:
        system_logger.error(f"Lỗi trong quá trình chạy SystemManager: {e}")
        sys.exit(1)
