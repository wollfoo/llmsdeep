# facade.py

import logging
import asyncio
from .resource_manager import ResourceManager
from .anomaly_detector import AnomalyDetector
from .anomaly_evaluator import SafeRestoreEvaluator
from .auxiliary_modules.event_bus import EventBus
from .auxiliary_modules.models import ConfigModel




class SystemFacade:
    def __init__(self, config: ConfigModel, event_bus: EventBus, logger: logging.Logger):
        self.config = config
        self.event_bus = event_bus
        self.logger = logger

        try:
            # Khởi tạo ResourceManager
            self.resource_manager = ResourceManager(config, event_bus, logger)
            if not self.resource_manager:
                raise RuntimeError("ResourceManager khởi tạo không thành công.")
            self.logger.info("ResourceManager được khởi tạo thành công.")

            # Khởi tạo AnomalyDetector
            self.anomaly_detector = AnomalyDetector(config, event_bus, logger, self.resource_manager)
            if not self.anomaly_detector:
                raise RuntimeError("AnomalyDetector khởi tạo không thành công.")
            self.logger.info("AnomalyDetector được khởi tạo thành công.")

            # Khởi tạo SafeRestoreEvaluator
            self.safe_restore_evaluator = SafeRestoreEvaluator(config, logger, self.resource_manager)
            if not hasattr(self.safe_restore_evaluator, 'start'):
                self.logger.warning("SafeRestoreEvaluator không có phương thức start().")

            self.logger.info("SafeRestoreEvaluator được khởi tạo thành công.")

        except Exception as e:
            self.logger.error(f"Lỗi khi khởi tạo các module trong SystemFacade: {e}")
            raise

    async def start(self):
        self.logger.info("Bắt đầu các module trong SystemFacade...")

        try:
            await self.resource_manager.start()
            self.logger.info("ResourceManager đã khởi động thành công.")
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi động ResourceManager: {e}")

        try:
            await self.anomaly_detector.start()
            self.logger.info("AnomalyDetector đã khởi động thành công.")
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi động AnomalyDetector: {e}")

        if hasattr(self.safe_restore_evaluator, 'start'):
            try:
                await self.safe_restore_evaluator.start()
                self.logger.info("SafeRestoreEvaluator đã khởi động thành công.")
            except Exception as e:
                self.logger.error(f"Lỗi khi khởi động SafeRestoreEvaluator: {e}")
        else:
            self.logger.warning("Bỏ qua SafeRestoreEvaluator do không có phương thức start().")

    async def stop(self):
        """
        Dừng tất cả các module.
        """
        self.logger.info("Dừng các module trong SystemFacade...")

        try:
            await self.resource_manager.stop()
            self.logger.info("ResourceManager đã được dừng.")
        except Exception as e:
            self.logger.error(f"Lỗi khi dừng ResourceManager: {e}")

        try:
            await self.anomaly_detector.stop()
            self.logger.info("AnomalyDetector đã được dừng.")
        except Exception as e:
            self.logger.error(f"Lỗi khi dừng AnomalyDetector: {e}")

        try:
            await self.safe_restore_evaluator.stop()
            self.logger.info("SafeRestoreEvaluator đã được dừng.")
        except Exception as e:
            self.logger.error(f"Lỗi khi dừng SafeRestoreEvaluator: {e}")
