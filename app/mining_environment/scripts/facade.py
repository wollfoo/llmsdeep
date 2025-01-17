# facade.py

import logging
import asyncio
from .resource_manager import ResourceManager
from .anomaly_detector import AnomalyDetector
from .anomaly_evaluator import SafeRestoreEvaluator
from .auxiliary_modules.event_bus import EventBus
from .auxiliary_modules.models import ConfigModel

class SystemFacade:
    def __init__(self, config: ConfigModel, event_bus: EventBus, resource_logger: logging.Logger, anomaly_logger: logging.Logger):
        """
        Khởi tạo SystemFacade.

        Args:
            config (ConfigModel): Cấu hình hệ thống.
            event_bus (EventBus): Bus sự kiện dùng để giao tiếp giữa các module.
            resource_logger (logging.Logger): Logger cho ResourceManager.
            anomaly_logger (logging.Logger): Logger cho AnomalyDetector.
        """
        self.config = config
        self.event_bus = event_bus
        self.resource_logger = resource_logger
        self.anomaly_logger = anomaly_logger

        # Khởi tạo các module với logger tương ứng
        try:
            # Khởi tạo ResourceManager
            self.resource_manager = ResourceManager(config, event_bus, self.resource_logger)
            if not self.resource_manager:
                raise RuntimeError("ResourceManager khởi tạo không thành công.")
            self.resource_logger.info("ResourceManager được khởi tạo thành công.")
        except Exception as e:
            self.resource_logger.error(f"Lỗi khi khởi tạo ResourceManager: {e}")
            raise RuntimeError("Không thể khởi tạo ResourceManager.") from e

        try:
            # Khởi tạo AnomalyDetector
            self.anomaly_detector = AnomalyDetector(config, event_bus, self.anomaly_logger, self.resource_manager)
            if not self.anomaly_detector:
                raise RuntimeError("AnomalyDetector khởi tạo không thành công.")
            self.anomaly_logger.info("AnomalyDetector được khởi tạo thành công.")
        except Exception as e:
            self.anomaly_logger.error(f"Lỗi khi khởi tạo AnomalyDetector: {e}")
            raise RuntimeError("Không thể khởi tạo AnomalyDetector.") from e

        try:
            # Khởi tạo SafeRestoreEvaluator
            self.safe_restore_evaluator = SafeRestoreEvaluator(config, self.resource_logger, self.resource_manager)
            if not hasattr(self.safe_restore_evaluator, 'start'):
                self.resource_logger.warning("SafeRestoreEvaluator không có phương thức start().")
            self.resource_logger.info("SafeRestoreEvaluator được khởi tạo thành công.")
        except Exception as e:
            self.resource_logger.error(f"Lỗi khi khởi tạo SafeRestoreEvaluator: {e}")
            raise RuntimeError("Không thể khởi tạo SafeRestoreEvaluator.") from e

    async def start(self):
        """
        Bắt đầu các module trong hệ thống.

        Gọi các phương thức start() của từng module (ResourceManager, AnomalyDetector, SafeRestoreEvaluator).
        """
        self.resource_logger.info("Bắt đầu các module trong SystemFacade...")
        try:
            # Bắt đầu ResourceManager
            await self.resource_manager.start()
            self.resource_logger.info("ResourceManager đã khởi động thành công.")
        except Exception as e:
            self.resource_logger.error(f"Lỗi khi khởi động ResourceManager: {e}")

        try:
            # Bắt đầu AnomalyDetector
            await self.anomaly_detector.start()
            self.anomaly_logger.info("AnomalyDetector đã khởi động thành công.")
        except Exception as e:
            self.anomaly_logger.error(f"Lỗi khi khởi động AnomalyDetector: {e}")

        if hasattr(self.safe_restore_evaluator, 'start'):
            try:
                # Bắt đầu SafeRestoreEvaluator nếu có phương thức start
                await self.safe_restore_evaluator.start()
                self.resource_logger.info("SafeRestoreEvaluator đã khởi động thành công.")
            except Exception as e:
                self.resource_logger.error(f"Lỗi khi khởi động SafeRestoreEvaluator: {e}")
        else:
            self.resource_logger.warning("Bỏ qua SafeRestoreEvaluator do không có phương thức start().")

    async def stop(self):
        """
        Dừng tất cả các module.

        Gọi các phương thức stop() của từng module (ResourceManager, AnomalyDetector, SafeRestoreEvaluator).
        """
        self.resource_logger.info("Dừng các module trong SystemFacade...")

        try:
            # Dừng ResourceManager
            await self.resource_manager.stop()
            self.resource_logger.info("ResourceManager đã được dừng.")
        except Exception as e:
            self.resource_logger.error(f"Lỗi khi dừng ResourceManager: {e}")

        try:
            # Dừng AnomalyDetector
            await self.anomaly_detector.stop()
            self.anomaly_logger.info("AnomalyDetector đã được dừng.")
        except Exception as e:
            self.anomaly_logger.error(f"Lỗi khi dừng AnomalyDetector: {e}")

        try:
            # Dừng SafeRestoreEvaluator
            if hasattr(self.safe_restore_evaluator, 'stop'):
                await self.safe_restore_evaluator.stop()
                self.resource_logger.info("SafeRestoreEvaluator đã được dừng.")
        except Exception as e:
            self.resource_logger.error(f"Lỗi khi dừng SafeRestoreEvaluator: {e}")

    async def handle_shutdown(self):
        """
        Xử lý sự kiện shutdown và dừng tất cả các module.
        """
        self.resource_logger.info("Nhận sự kiện shutdown, đang dừng các module...")
        await self.stop()

    async def restart(self):
        """
        Khởi động lại các module trong hệ thống.
        """
        self.resource_logger.info("Đang khởi động lại các module...")
        await self.stop()
        await self.start()

    def register_shutdown_event(self):
        """
        Đăng ký sự kiện shutdown từ EventBus để dừng tất cả các module khi nhận tín hiệu shutdown.
        """
        async def shutdown_handler(data):
            try:
                await self.handle_shutdown()
            except Exception as e:
                self.resource_logger.error(f"Lỗi khi xử lý sự kiện shutdown: {e}")

        try:
            self.event_bus.subscribe('shutdown', shutdown_handler)
            self.resource_logger.info("Đã đăng ký sự kiện shutdown.")
        except Exception as e:
            self.resource_logger.error(f"Lỗi khi đăng ký sự kiện shutdown: {e}")
