# facade.py

from app.mining_environment.scripts.auxiliary_modules.event_bus import EventBus
from resource_manager import ResourceManager
from anomaly_detector import AnomalyDetector
from models import ConfigModel
import asyncio

class SystemFacade:
    """
    Facade để trừu tượng hóa các module ResourceManager và AnomalyDetector.
    Giao tiếp với các module thông qua EventBus.
    """
    def __init__(self, config: ConfigModel, event_bus: EventBus, logger: logging.Logger):
        self.config = config
        self.event_bus = event_bus
        self.logger = logger

        # Khởi tạo các module
        self.resource_manager = ResourceManager(config, event_bus, logger)
        self.anomaly_detector = AnomalyDetector(config, event_bus, logger)

    async def start(self):
        """
        Khởi động tất cả các module.
        """
        await asyncio.gather(
            self.resource_manager.start(),
            self.anomaly_detector.start()
        )

    async def stop(self):
        """
        Dừng tất cả các module.
        """
        await asyncio.gather(
            self.resource_manager.stop(),
            self.anomaly_detector.stop()
        )
