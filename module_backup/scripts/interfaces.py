# interfaces.py

from abc import ABC, abstractmethod
from typing import Dict, Any
from .utils import MiningProcess

class IResourceManager(ABC):
    @abstractmethod
    async def enqueue_cloaking(self, process: MiningProcess):
        """
        Thêm tiến trình vào hàng đợi để cloaking.
        """
        pass

    @abstractmethod
    async def enqueue_restoration(self, process: MiningProcess):
        """
        Thêm tiến trình vào hàng đợi để khôi phục tài nguyên.
        """
        pass

    @abstractmethod
    async def collect_metrics(self, process: MiningProcess) -> Dict[str, Any]:
        """
        Thu thập các metrics cho một tiến trình.
        """
        pass

    @abstractmethod
    async def restore_resources(self, process: MiningProcess):
        """
        Khôi phục tài nguyên cho tiến trình đã cloaked.
        """
        pass

    @abstractmethod
    async def is_gpu_initialized(self) -> bool:
        """
        Kiểm tra xem GPU đã được khởi tạo hay chưa.
        """
        pass

    @abstractmethod
    async def shutdown(self):
        """
        Dừng ResourceManager và giải phóng tài nguyên.
        """
        pass
