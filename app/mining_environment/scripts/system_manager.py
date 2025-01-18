# system_manager.py

import os
import sys
import json
import asyncio
import logging
import uuid
import aiofiles
from pathlib import Path
from typing import Dict, Any
from contextvars import ContextVar

from .facade import SystemFacade
from .logging_config import setup_logging, correlation_id
from .auxiliary_modules.event_bus import EventBus
from .auxiliary_modules.models import ConfigModel

CONFIG_DIR = Path(os.getenv('CONFIG_DIR', '/app/mining_environment/config'))
LOGS_DIR = Path(os.getenv('LOGS_DIR', '/app/mining_environment/logs'))
LOGS_DIR.mkdir(parents=True, exist_ok=True)

system_logger = setup_logging('system_manager', LOGS_DIR / 'system_manager.log', 'INFO')
resource_logger = setup_logging('resource_manager', LOGS_DIR / 'resource_manager.log', 'INFO')
anomaly_logger = setup_logging('anomaly_detector', LOGS_DIR / 'anomaly_detector.log', 'INFO')

async def load_config(config_path: Path) -> ConfigModel:
    try:
        async with aiofiles.open(config_path, 'r') as f:
            content = await f.read()
            config_data = json.loads(content)
        system_logger.info(f"Đã tải cấu hình từ {config_path}.")

        config = ConfigModel(**config_data)
        system_logger.info("Cấu hình đã được xác thực thành công.")
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

class SystemManager:
    def __init__(self, config: ConfigModel, logger: logging.Logger):
        self.config = config
        self.logger = logger

        self.event_bus = EventBus()
        self.facade = SystemFacade(config, self.event_bus, resource_logger, anomaly_logger)

        self.correlation_id = str(uuid.uuid4())
        correlation_id.set(self.correlation_id)
        self.logger.info(f"SystemManager khởi tạo với Correlation ID: {self.correlation_id}")

        self.stop_event = asyncio.Event()

    async def start_async(self):
        self.logger.info("Đang khởi động SystemManager...")
        try:
            async with asyncio.Lock():
                if not self.facade:
                    raise RuntimeError("SystemFacade chưa được khởi tạo.")

                # Bắt đầu lắng nghe EventBus
                asyncio.create_task(self.event_bus.start_listening())
                self.logger.info("EventBus đã bắt đầu lắng nghe.")

                await self.facade.start()
                self.logger.info("SystemManager đã khởi động thành công.")
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi động SystemManager: {e}")
            raise

    async def stop_async(self):
        self.logger.info("Đang dừng SystemManager...")
        try:
            async with asyncio.Lock():
                if not self.facade:
                    raise RuntimeError("SystemFacade chưa được khởi tạo.")

                await self.facade.stop()
                self.logger.info("SystemFacade đã dừng.")

                await self.event_bus.stop()
                self.logger.info("EventBus đã dừng.")
        except Exception as e:
            self.logger.error(f"Lỗi khi dừng SystemManager: {e}")

async def run_system_manager(system_manager: SystemManager):
    asyncio.create_task(listen_for_shutdown(system_manager))
    try:
        for attempt in range(3):
            try:
                await system_manager.start_async()
                system_logger.info("SystemManager đang chạy. Đợi tín hiệu dừng...")
                await system_manager.stop_event.wait()
                break
            except Exception as e:
                system_logger.warning(f"Thử khởi động SystemManager thất bại ({attempt + 1}/3): {e}")
        else:
            raise RuntimeError("Không thể khởi động SystemManager sau nhiều lần thử.")
    except Exception as e:
        system_logger.error(f"Lỗi khi chạy SystemManager: {e}")
        await system_manager.stop_async()
        sys.exit(1)

async def listen_for_shutdown(system_manager: SystemManager):
    async def shutdown_handler(data):
        try:
            system_manager.logger.info("Nhận sự kiện shutdown từ EventBus.")
            await system_manager.stop_async()
            system_manager.stop_event.set()
        except Exception as e:
            system_manager.logger.error(f"Lỗi khi xử lý sự kiện shutdown: {e}")

    # Vì subscribe là hàm đồng bộ => ta gọi trực tiếp
    system_manager.event_bus.subscribe('shutdown', shutdown_handler)
    system_manager.logger.info("Đã đăng ký sự kiện shutdown.")

_system_manager_instance = None

def start():
    global _system_manager_instance
    if _system_manager_instance:
        system_logger.warning("SystemManager đã được khởi động.")
        return

    try:
        resource_config_path = CONFIG_DIR / "resource_config.json"
        config = asyncio.run(load_config(resource_config_path))

        _system_manager_instance = SystemManager(config, system_logger)
        asyncio.run(run_system_manager(_system_manager_instance))
    except Exception as e:
        system_logger.error(f"Lỗi khi khởi động SystemManager: {e}")
        sys.exit(1)

def stop():
    global _system_manager_instance
    if not _system_manager_instance:
        system_logger.warning("SystemManager chưa được khởi động.")
        return

    try:
        asyncio.run(_system_manager_instance.event_bus.publish('shutdown', None))
        system_logger.info("Đã gửi sự kiện 'shutdown' thành công.")
    except Exception as e:
        system_logger.error(f"Lỗi khi gửi sự kiện 'shutdown': {e}")

def main():
    try:
        if os.geteuid() != 0:
            print("Script phải được chạy với quyền root.")
            sys.exit(1)

        start()
    except KeyboardInterrupt:
        system_logger.info("Đang dừng SystemManager do KeyboardInterrupt.")
        stop()
    except Exception as e:
        system_logger.error(f"Lỗi trong quá trình chạy SystemManager: {e}")
        stop()
        sys.exit(1)

if __name__ == "__main__":
    main()
