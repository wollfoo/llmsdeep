# power_management.py

import os
import sys
import psutil
import subprocess
import pynvml
from pathlib import Path
from typing import List, Optional, Dict, Any
import threading

# Thêm đường dẫn tới thư mục chứa `logging_config.py`
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SCRIPT_DIR))

# Thiết lập đường dẫn tới thư mục logs
LOGS_DIR = Path(os.getenv('LOGS_DIR', '/app/mining_environment/logs'))
os.makedirs(LOGS_DIR, exist_ok=True)

# Import hàm setup_logging từ logging_config.py
from logging_config import setup_logging

# Thiết lập logging với logging_config.py
logger = setup_logging('power_management', LOGS_DIR / 'power_management.log', 'INFO')

###############################################################################
#                            LỚP PowerManager                                #
###############################################################################

class PowerManager:
    """
    Lớp singleton quản lý năng lượng cho CPU và GPU.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> 'PowerManager':
        """
        Phương thức khởi tạo singleton instance của PowerManager.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(PowerManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Khởi tạo PowerManager. Đảm bảo rằng __init__ chỉ chạy một lần cho singleton.
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        # Tham số ước lượng CPU
        self.cpu_base_power_watts = 10.0
        self.cpu_max_power_watts = 150.0

        # Khởi tạo pynvml
        self.pynvml_initialized = False
        self.gpu_count = 0  # Sẽ được cập nhật sau khi khởi tạo NVML

        self.initialize()

    def initialize(self) -> None:
        """
        Khởi tạo NVML đồng bộ.
        """
        if not self.pynvml_initialized:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.pynvml_initialized = True
                logger.info("PowerManager: Đã khởi tạo pynvml thành công.")
            except pynvml.NVMLError as e:
                logger.error(f"PowerManager: Lỗi khi khởi tạo pynvml: {e}")
                self.gpu_count = 0
        else:
            logger.debug("PowerManager: pynvml đã được khởi tạo trước đó.")

    def shutdown(self) -> None:
        """
        Dừng quản lý năng lượng và giải phóng NVML.
        """
        try:
            if self.pynvml_initialized:
                pynvml.nvmlShutdown()
                self.pynvml_initialized = False
                logger.info("PowerManager: Đã shutdown thành công pynvml.")
            else:
                logger.warning("PowerManager: pynvml chưa được khởi tạo hoặc đã bị shutdown.")
        except pynvml.NVMLError as e:
            logger.error(f"PowerManager: Lỗi khi shutdown pynvml: {e}")
        except Exception as e:
            logger.error(f"PowerManager: Lỗi khi shutdown pynvml: {e}")

    def get_cpu_power(self, pid: Optional[int] = None) -> float:
        """
        Ước tính công suất CPU bằng cách đọc cpu_percent(1s).

        :param pid: PID của tiến trình (không sử dụng trong phiên bản đồng bộ).
        :return: Công suất CPU ước tính (float) tính bằng Watts.
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
            logger.error(f"PowerManager: Lỗi khi ước tính công suất CPU: {e}")
            return 0.0

    def get_gpu_power(self, pid: Optional[int] = None) -> List[float]:
        """
        Lấy công suất tiêu thụ hiện tại của từng GPU bằng pynvml (W).

        :param pid: PID của tiến trình (không sử dụng trong phiên bản đồng bộ).
        :return: Danh sách công suất GPU (float) cho từng GPU.
        """
        if not self.pynvml_initialized:
            logger.warning("PowerManager: NVML chưa được khởi tạo => không thể lấy GPU power.")
            return []
        try:
            powers = []
            for i in range(self.gpu_count):
                power = self._get_single_gpu_power(i)
                if power is not None:
                    powers.append(power)
            return powers
        except Exception as e:
            logger.error(f"PowerManager: Lỗi khi lấy công suất GPU: {e}")
            return []

    def _get_single_gpu_power(self, gpu_index: int) -> Optional[float]:
        """
        Lấy công suất GPU (mW -> W).

        :param gpu_index: Chỉ số GPU.
        :return: Công suất GPU (float) hoặc None nếu lỗi.
        """
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_w = power_mw / 1000.0  # mW -> W
            return power_w
        except pynvml.NVMLError as e:
            logger.error(f"PowerManager: Lỗi khi đọc power GPU={gpu_index}: {e}")
            return None

    def reduce_cpu_power(self, pid: Optional[int] = None, reduction_percentage: float = 20.0) -> None:
        """
        Giảm công suất CPU qua hạ tần số CPU (giả lập).

        :param pid: PID của tiến trình (không sử dụng trong phiên bản đồng bộ).
        :param reduction_percentage: Tỷ lệ giảm tần số CPU (%).
        """
        # Tuỳ logic
        logger.info(f"PowerManager: Giảm CPU power {reduction_percentage}%. (chưa có logic chi tiết)")

    def reduce_gpu_power(self, pid: Optional[int] = None, reduction_percentage: float = 20.0) -> None:
        """
        Giảm công suất GPU bằng cách set power limit dựa trên tỷ lệ giảm.

        :param pid: PID của tiến trình (không sử dụng trong phiên bản đồng bộ).
        :param reduction_percentage: Tỷ lệ giảm công suất GPU (%).
        """
        if not self.pynvml_initialized or self.gpu_count == 0:
            logger.warning("PowerManager: Không có GPU hoặc NVML chưa init.")
            return
        if not (0 < reduction_percentage <= 100):
            logger.error(f"PowerManager: Tham số reduction_percentage không hợp lệ: {reduction_percentage}")
            return
        # Logic mẫu:
        for i in range(self.gpu_count):
            desired = self.calculate_desired_power_limit(i, 100 - reduction_percentage)
            if desired is not None:
                self.set_gpu_power_limit(i, desired)

    def calculate_desired_power_limit(self, gpu_index: int, usage_percentage: float) -> Optional[float]:
        """
        Tính power limit mới dựa trên usage_percentage (0..100).

        :param gpu_index: Chỉ số GPU.
        :param usage_percentage: Tỷ lệ sử dụng (%).
        :return: Power limit mới (float) hoặc None nếu lỗi.
        """
        if not self.pynvml_initialized:
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
            desired_power = max(power_min, min(desired_power, power_max))
            return desired_power
        except pynvml.NVMLError as e:
            logger.error(f"PowerManager: Lỗi calculate_desired_power_limit GPU={gpu_index}: {e}")
            return None

    def set_gpu_power_limit(self, gpu_index: int, power_limit_watts: float) -> bool:
        """
        Thiết lập power limit (W) cho GPU.

        :param gpu_index: Chỉ số GPU.
        :param power_limit_watts: Power limit cần đặt (W).
        :return: True nếu thành công, False nếu thất bại.
        """
        if not self.pynvml_initialized:
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

    def set_gpu_usage(self, usage_percentages: List[float], pid: Optional[int] = None) -> None:
        """
        Thiết lập giới hạn công suất cho từng GPU dựa trên usage_percentages.

        :param usage_percentages: Danh sách % sử dụng cho từng GPU.
        :param pid: PID của tiến trình (không sử dụng trong phiên bản đồng bộ).
        """
        if not self.pynvml_initialized or self.gpu_count == 0:
            logger.warning("PowerManager: Không có GPU hoặc NVML chưa init.")
            return
        if len(usage_percentages) != self.gpu_count:
            logger.warning("PowerManager: Số GPU khác với độ dài usage_percentages => điều chỉnh.")
            usage_percentages = (usage_percentages + [0] * self.gpu_count)[:self.gpu_count]

        for i, usage in enumerate(usage_percentages):
            desired = self.calculate_desired_power_limit(i, usage)
            if desired is not None:
                ok = self.set_gpu_power_limit(i, desired)
                if ok:
                    logger.info(f"PowerManager: GPU={i} => limit {desired}W, usage={usage}%")
            else:
                logger.error(f"PowerManager: Tính limit cho GPU={i} với usage={usage}% thất bại.")

###############################################################################
#                          Singleton Instance                                #
###############################################################################

# Singleton instance của PowerManager sẽ được tạo thông qua factory method
_power_manager_instance: Optional[PowerManager] = None
_power_manager_lock = threading.Lock()

def get_power_manager() -> PowerManager:
    """
    Lấy singleton instance của PowerManager một cách đồng bộ.

    :return: Instance của PowerManager.
    """
    global _power_manager_instance
    if _power_manager_instance is None:
        with _power_manager_lock:
            if _power_manager_instance is None:
                _power_manager_instance = PowerManager()
    return _power_manager_instance

###############################################################################
#                          Các Hàm Ngoài Class                              #
###############################################################################

def get_cpu_power(pid: Optional[int] = None) -> float:
    """
    Trả về công suất CPU hiện tại (W).

    :param pid: PID của tiến trình (không sử dụng trong phiên bản đồng bộ).
    :return: Công suất CPU hiện tại (float) tính bằng Watts.
    """
    power_manager = get_power_manager()
    return power_manager.get_cpu_power(pid)

def get_gpu_power(pid: Optional[int] = None) -> List[float]:
    """
    Trả về danh sách công suất GPU (W) cho từng GPU.

    :param pid: PID của tiến trình (không sử dụng trong phiên bản đồng bộ).
    :return: Danh sách công suất GPU (float) cho từng GPU.
    """
    power_manager = get_power_manager()
    return power_manager.get_gpu_power(pid)

def reduce_cpu_power(pid: Optional[int] = None, reduction_percentage: float = 20.0) -> None:
    """
    Giảm công suất CPU.

    :param pid: PID của tiến trình (không sử dụng trong phiên bản đồng bộ).
    :param reduction_percentage: Tỷ lệ giảm tần số CPU (%).
    """
    power_manager = get_power_manager()
    power_manager.reduce_cpu_power(pid, reduction_percentage)

def reduce_gpu_power(pid: Optional[int] = None, reduction_percentage: float = 20.0) -> None:
    """
    Giảm công suất GPU.

    :param pid: PID của tiến trình (không sử dụng trong phiên bản đồng bộ).
    :param reduction_percentage: Tỷ lệ giảm công suất GPU (%).
    """
    power_manager = get_power_manager()
    power_manager.reduce_gpu_power(pid, reduction_percentage)

def set_gpu_usage(usage_percentages: List[float], pid: Optional[int] = None) -> None:
    """
    Thiết lập mức sử dụng GPU (giới hạn công suất).

    :param usage_percentages: Danh sách % sử dụng cho từng GPU.
    :param pid: PID của tiến trình (không sử dụng trong phiên bản đồng bộ).
    """
    power_manager = get_power_manager()
    power_manager.set_gpu_usage(usage_percentages, pid)

def shutdown_power_management() -> None:
    """
    Dừng quản lý năng lượng và giải phóng tài nguyên GPU (NVML).
    """
    power_manager = get_power_manager()
    power_manager.shutdown()
