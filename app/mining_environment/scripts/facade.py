# facade.py


import asyncio
from .resource_manager import ResourceManager
from .anomaly_detector import AnomalyDetector
from .anomaly_evaluator import SafeRestoreEvaluator
from .auxiliary_modules.event_bus import EventBus
from .auxiliary_modules.models import ConfigModel


class SystemFacade:
    """
    Facade để trừu tượng hóa các module ResourceManager và AnomalyDetector.
    Giao tiếp với các module thông qua EventBus.
    """
    def __init__(self, config: ConfigModel, event_bus: EventBus, logger: logging.Logger):
        self.config = config
        self.event_bus = event_bus
        self.logger = logger

        # Khởi tạo ResourceManager
        self.resource_manager = ResourceManager(config, event_bus, logger)

        # Khởi tạo AnomalyDetector với ResourceManager đã được tiêm
        self.anomaly_detector = AnomalyDetector(config, event_bus, logger, self.resource_manager)

        # Khởi tạo SafeRestoreEvaluator với ResourceManager đã được tiêm
        self.safe_restore_evaluator = SafeRestoreEvaluator(config, logger, self.resource_manager)

    async def start(self):
        """
        Bắt đầu tất cả các module.
        """
        await asyncio.gather(
            self.resource_manager.start(),
            self.anomaly_detector.start(),
            self.safe_restore_evaluator.start()
        )

    async def stop(self):
        """
        Dừng tất cả các module.
        """
        await asyncio.gather(
            self.resource_manager.stop(),
            self.anomaly_detector.stop(),
            self.safe_restore_evaluator.stop()
        )
