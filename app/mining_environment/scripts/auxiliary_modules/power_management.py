"""
Module power_management.py

Quản lý năng lượng CPU, GPU theo mô hình đồng bộ (threading), 
đảm bảo tương thích với resource_manager.py refactor đồng bộ.
"""

import os
import sys
import psutil
import logging
import subprocess
import pynvml
import threading
import time
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any


# Thêm đường dẫn tới thư mục chứa `logging_config.py`
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SCRIPT_DIR))

# Thiết lập đường dẫn tới thư mục logs
LOGS_DIR = Path(os.getenv('LOGS_DIR', '/app/mining_environment/logs'))
os.makedirs(LOGS_DIR, exist_ok=True)

from logging_config import setup_logging
logger = setup_logging('power_management', LOGS_DIR / 'power_management.log', 'INFO')



class PowerManager:
    """
    Lớp quản lý năng lượng cho CPU và GPU dưới dạng đồng bộ (threading).
    Cung cấp các phương thức lấy công suất CPU/GPU, giới hạn power limit, v.v.

    Attributes:
        _initialized (bool): Đánh dấu đã init hay chưa.
        _nvml_inited (bool): Đánh dấu NVML (pynvml) đã init hay chưa.
        cpu_base_power_watts (float): Mức công suất CPU tối thiểu (W).
        cpu_max_power_watts (float): Mức công suất CPU tối đa (W).
        gpu_count (int): Số GPU trong hệ thống (sau khi init NVML).
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """
        Khởi tạo PowerManager (đồng bộ).
        """
        if getattr(self, '_initialized', False):
            return

        self._initialized = True
        self._nvml_inited = False

        # Tham số ước lượng CPU
        self.cpu_base_power_watts = 10.0
        self.cpu_max_power_watts = 150.0

        # Khởi tạo NVML (đồng bộ)
        self._init_nvml()

    @classmethod
    def get_instance(cls) -> 'PowerManager':
        """
        Lấy singleton instance PowerManager (đồng bộ).

        :return: Thể hiện PowerManager (singleton).
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def _init_nvml(self) -> None:
        """
        Khởi tạo NVML (pynvml) dạng đồng bộ.
        """
        if not self._nvml_inited:
            try:
                pynvml.nvmlInit()
                self._nvml_inited = True
                logger.info("PowerManager: Đã khởi tạo pynvml thành công.")
            except pynvml.NVMLError as e:
                logger.error(f"PowerManager: Lỗi khi khởi tạo pynvml: {e}")
                self._nvml_inited = False

        self.gpu_count = 0
        if self._nvml_inited:
            try:
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"PowerManager: Tìm thấy {self.gpu_count} GPU trên hệ thống.")
            except pynvml.NVMLError as e:
                logger.error(f"PowerManager: Lỗi khi lấy số lượng GPU: {e}")
                self.gpu_count = 0

    def shutdown(self) -> None:
        """
        Giải phóng NVML khi không còn dùng.
        """
        if self._nvml_inited:
            try:
                pynvml.nvmlShutdown()
                self._nvml_inited = False
                logger.info("PowerManager: Đã shutdown NVML.")
            except pynvml.NVMLError as e:
                logger.error(f"PowerManager: Lỗi khi shutdown NVML: {e}")
        else:
            logger.warning("PowerManager: NVML chưa init hoặc đã shutdown trước đó.")

    def get_cpu_power(self, pid: Optional[int] = None) -> float:
        """
        Ước tính công suất CPU bằng cách đọc cpu_percent(1s),
        nội suy giữa cpu_base_power_watts và cpu_max_power_watts.

        :param pid: (tuỳ chọn) PID tiến trình, hiện không dùng.
        :return: float, công suất CPU ước tính (W).
        """
        try:
            cpu_load = psutil.cpu_percent(interval=1)
            estimated_power = (
                self.cpu_base_power_watts
                + (cpu_load / 100.0) * (self.cpu_max_power_watts - self.cpu_base_power_watts)
            )
            logger.debug(f"PowerManager: CPU Load={cpu_load}%, Estimated Power={estimated_power:.2f}W")
            return estimated_power
        except Exception as e:
            logger.error(f"PowerManager: Lỗi khi ước tính công suất CPU: {e}\n{traceback.format_exc()}")
            return 0.0

    def _get_single_gpu_power(self, gpu_index: int) -> Optional[float]:
        """
        Lấy công suất (W) của 1 GPU (đồng bộ).
        
        :param gpu_index: chỉ số GPU.
        :return: float power (W) hoặc None nếu lỗi.
        """
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            return float(power_mw) / 1000.0  # mW -> W
        except pynvml.NVMLError as e:
            logger.error(f"PowerManager: Lỗi khi đọc power GPU={gpu_index}: {e}")
            return None
        except Exception as ex:
            logger.error(f"PowerManager: Lỗi không xác định GPU={gpu_index}: {ex}\n{traceback.format_exc()}")
            return None

    def get_gpu_power(self, pid: Optional[int] = None) -> List[float]:
        """
        Lấy danh sách công suất GPU (W) (đồng bộ) cho từng GPU.

        :param pid: (tuỳ chọn) PID tiến trình, hiện không sử dụng.
        :return: List[float] công suất cho từng GPU, rỗng nếu không có GPU hoặc NVML chưa init.
        """
        if not self._nvml_inited:
            logger.warning("PowerManager: NVML chưa init => không thể lấy GPU power.")
            return []
        try:
            powers = []
            for i in range(self.gpu_count):
                p = self._get_single_gpu_power(i)
                if p is not None:
                    powers.append(p)
            return powers
        except Exception as e:
            logger.error(f"PowerManager: Lỗi khi lấy công suất GPU: {e}\n{traceback.format_exc()}")
            return []

    def reduce_cpu_power(self, pid: Optional[int] = None, reduction_percentage: float = 20.0) -> None:
        """
        Giảm công suất CPU (giả lập) bằng cách hạ tần số (chưa có logic cụ thể).

        :param pid: (tuỳ chọn) PID tiến trình, nếu cần logic gắn với cgroup.
        :param reduction_percentage: Tỷ lệ giảm (0..100).
        """
        logger.info(f"PowerManager: Giảm CPU power ~{reduction_percentage}%. (chưa có logic chi tiết)")

    def reduce_gpu_power(self, pid: Optional[int] = None, reduction_percentage: float = 20.0) -> None:
        """
        Giảm công suất GPU => set power limit, (giả lập).

        :param pid: (tuỳ chọn) PID tiến trình.
        :param reduction_percentage: Tỷ lệ giảm (0..100).
        """
        if not self._nvml_inited or self.gpu_count == 0:
            logger.warning("PowerManager: Không có GPU hoặc NVML chưa init.")
            return
        if not (0 < reduction_percentage <= 100):
            logger.error(f"PowerManager: Tham số reduction_percentage không hợp lệ: {reduction_percentage}")
            return

        for i in range(self.gpu_count):
            desired = self.calculate_desired_power_limit(i, (100 - reduction_percentage))
            if desired is not None:
                self.set_gpu_power_limit(i, desired)

    def calculate_desired_power_limit(self, gpu_index: int, usage_percentage: float) -> Optional[float]:
        """
        Tính power limit (W) mới dựa trên usage_percentage (0..100).

        :param gpu_index: index GPU.
        :param usage_percentage: % usage (0..100).
        :return: float power_limit (W) hoặc None nếu thất bại.
        """
        if not self._nvml_inited:
            logger.warning("PowerManager: NVML chưa init => không thể tính power limit.")
            return None
        if not (0 <= usage_percentage <= 100):
            logger.error(f"PowerManager: usage_percentage={usage_percentage} không hợp lệ.")
            return None

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            power_max = pynvml.nvmlDeviceGetPowerManagementLimitMax(handle) / 1000.0
            power_min = pynvml.nvmlDeviceGetPowerManagementLimitMin(handle) / 1000.0
            desired_power = power_min + (power_max - power_min) * (usage_percentage / 100.0)
            desired_power = round(desired_power, 2)
            return max(power_min, min(desired_power, power_max))
        except pynvml.NVMLError as e:
            logger.error(f"PowerManager: Lỗi calculate_desired_power_limit GPU={gpu_index}: {e}")
            return None
        except Exception as ex:
            logger.error(f"PowerManager: Lỗi không xác định calculate_desired_power_limit GPU={gpu_index}: {ex}")
            return None

    def set_gpu_power_limit(self, gpu_index: int, power_limit_watts: float) -> bool:
        """
        Thiết lập power limit (W) cho GPU.

        :param gpu_index: index GPU.
        :param power_limit_watts: giá trị power limit (W).
        :return: True nếu thành công, False nếu lỗi.
        """
        if not self._nvml_inited:
            logger.error("PowerManager: NVML chưa init => set GPU power limit thất bại.")
            return False
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            limit_mw = int(power_limit_watts * 1000)
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, limit_mw)
            logger.info(f"PowerManager: Set GPU={gpu_index} limit={power_limit_watts}W.")
            return True
        except pynvml.NVMLError as e:
            logger.error(f"PowerManager: Lỗi khi set GPU power limit {gpu_index}: {e}")
            return False
        except Exception as ex:
            logger.error(f"PowerManager: Lỗi không xác định set GPU power limit GPU={gpu_index}: {ex}")
            return False

    def set_gpu_usage(self, usage_percentages: List[float], pid: Optional[int] = None) -> None:
        """
        Thiết lập giới hạn công suất GPU theo danh sách usage_percentages (0..100 cho mỗi GPU).

        :param usage_percentages: List[float], độ dài bằng số GPU hoặc ít hơn.
        :param pid: (tuỳ chọn) PID tiến trình.
        """
        if not self._nvml_inited or self.gpu_count == 0:
            logger.warning("PowerManager: Không có GPU hoặc NVML chưa init.")
            return
        if len(usage_percentages) != self.gpu_count:
            logger.warning("PowerManager: Số GPU khác với len(usage_percentages) => điều chỉnh.")
            usage_percentages = (usage_percentages + [0]*(self.gpu_count))[:self.gpu_count]

        for i, usage in enumerate(usage_percentages):
            desired = self.calculate_desired_power_limit(i, usage)
            if desired is not None:
                ok = self.set_gpu_power_limit(i, desired)
                if ok:
                    logger.info(f"PowerManager: GPU={i} => limit {desired}W, usage={usage}%")
            else:
                logger.error(f"PowerManager: Tính limit cho GPU={i} usage={usage}% thất bại.")


# --------------- SINGLETON INSTANCE & CÁC HÀM TIỆN ÍCH ----------------

_power_manager_instance: Optional[PowerManager] = None
_power_manager_lock = threading.Lock()

def get_power_manager() -> PowerManager:
    """
    Lấy singleton instance PowerManager (đồng bộ).

    :return: PowerManager instance.
    """
    global _power_manager_instance
    with _power_manager_lock:
        if _power_manager_instance is None:
            _power_manager_instance = PowerManager.get_instance()
    return _power_manager_instance

def get_cpu_power(pid: Optional[int] = None) -> float:
    """
    Trả về công suất CPU hiện tại (W) (đồng bộ).

    :param pid: (tuỳ chọn) PID tiến trình, không sử dụng trong logic giả lập.
    :return: float, công suất CPU ước tính (W).
    """
    pm = get_power_manager()
    return pm.get_cpu_power(pid)

def get_gpu_power(pid: Optional[int] = None) -> List[float]:
    """
    Trả về công suất GPU (W) cho mỗi GPU (đồng bộ).

    :param pid: (tuỳ chọn) PID tiến trình, không sử dụng trong logic hiện tại.
    :return: List[float], công suất GPU (W) cho từng GPU.
    """
    pm = get_power_manager()
    return pm.get_gpu_power(pid)

def reduce_cpu_power(pid: Optional[int] = None, reduction_percentage: float = 20.0) -> None:
    """
    Giảm công suất CPU (đồng bộ).
    
    :param pid: (tuỳ chọn) PID tiến trình.
    :param reduction_percentage: % giảm (0..100).
    """
    pm = get_power_manager()
    pm.reduce_cpu_power(pid, reduction_percentage)

def reduce_gpu_power(pid: Optional[int] = None, reduction_percentage: float = 20.0) -> None:
    """
    Giảm công suất GPU (đồng bộ).

    :param pid: (tuỳ chọn) PID tiến trình.
    :param reduction_percentage: % giảm (0..100).
    """
    pm = get_power_manager()
    pm.reduce_gpu_power(pid, reduction_percentage)

def set_gpu_usage(usage_percentages: List[float], pid: Optional[int] = None) -> None:
    """
    Thiết lập giới hạn công suất cho mỗi GPU qua usage_percentages.

    :param usage_percentages: list[float] (0..100), độ dài bằng số GPU hoặc ngắn hơn => fill 0.
    :param pid: (tuỳ chọn) PID tiến trình.
    """
    pm = get_power_manager()
    pm.set_gpu_usage(usage_percentages, pid)

def shutdown_power_management() -> None:
    """
    Tắt PowerManager => shutdown NVML (đồng bộ).
    """
    pm = get_power_manager()
    pm.shutdown()
