# power_management.py

import os
import sys
import psutil
import pynvml
import subprocess
import threading
from pathlib import Path
from typing import List, Optional, Dict

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

class PowerManager:
    """
    PowerManager là một singleton class chịu trách nhiệm quản lý năng lượng cho CPU và GPU.
    Nó cung cấp các phương thức để giám sát công suất tiêu thụ và điều chỉnh công suất khi cần thiết.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PowerManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Initialize NVML for GPU management
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"NVML initialized successfully. Số lượng GPU: {self.gpu_count}")
        except pynvml.NVMLError as e:
            logger.error(f"Không thể khởi tạo NVML: {e}")
            self.gpu_count = 0

        # CPU power parameters (cần điều chỉnh dựa trên hệ thống cụ thể)
        self.cpu_base_power_watts = 10.0  # Công suất cơ bản khi CPU idle (W)
        self.cpu_max_power_watts = 150.0  # Công suất tối đa khi CPU full load (W)


    def get_cpu_power(self, pid: Optional[int] = None) -> float:
        """
        Ước tính công suất tiêu thụ hiện tại của CPU dựa trên tải và tần số.
        Tham số `pid` được thêm vào để tương thích với resource_manager.py, nhưng hiện tại không được sử dụng.

        Args:
            pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).

        Returns:
            float: Công suất CPU hiện tại (W).
        """
        try:
            cpu_load = psutil.cpu_percent(interval=1)
            # Lấy tần số hiện tại có thể sử dụng nếu cần thiết
            # cpu_freq = psutil.cpu_freq().current  # MHz

            # Ước tính công suất: base + (load_percent / 100) * (max_power - base_power)
            estimated_power = self.cpu_base_power_watts + (cpu_load / 100.0) * (self.cpu_max_power_watts - self.cpu_base_power_watts)
            logger.debug(f"CPU Load: {cpu_load}%, Estimated CPU Power: {estimated_power:.2f}W")
            return estimated_power
        except Exception as e:
            logger.error(f"Lỗi khi ước tính công suất CPU: {e}")
            return 0.0

    def get_gpu_power(self, pid: Optional[int] = None) -> List[float]:
        """
        Lấy công suất tiêu thụ hiện tại của từng GPU bằng NVML.
        Tham số `pid` được thêm vào để tương thích với resource_manager.py, nhưng hiện tại không được sử dụng.

        Args:
            pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).

        Returns:
            List[float]: Danh sách công suất GPU hiện tại (W).
        """
        gpu_powers = []
        if self.gpu_count == 0:
            logger.warning("Không có GPU nào được phát hiện để giám sát công suất.")
            return gpu_powers

        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Chuyển từ mW sang W
                gpu_powers.append(power_usage)
                logger.debug(f"GPU {i}: {power_usage}W")
            return gpu_powers
        except pynvml.NVMLError as e:
            logger.error(f"Lỗi khi lấy công suất GPU: {e}")
            return gpu_powers
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi lấy công suất GPU: {e}")
            return gpu_powers

    def reduce_cpu_power(self, pid: Optional[int] = None, reduction_percentage: float = 20.0):
        """
        Giảm công suất CPU bằng cách giảm tần số CPU.
        Tham số `pid` được thêm vào để tương thích với resource_manager.py, nhưng hiện tại không được sử dụng.

        Args:
            pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).
            reduction_percentage (float): Tỷ lệ giảm tần số CPU (%).
        """
        try:
            if not (0 < reduction_percentage < 100):
                logger.error("Reduction percentage phải nằm trong khoảng (0, 100).")
                return

            # Lấy tần số hiện tại và tính toán tần số mới
            cpu_freq = psutil.cpu_freq().current  # MHz
            new_freq = cpu_freq * (1 - reduction_percentage / 100.0)

            # Đảm bảo tần số không thấp hơn mức tối thiểu cấu hình
            min_freq = 1800  # MHz, cần lấy từ cấu hình hoặc xác định động
            new_freq = max(new_freq, min_freq)

            # Thiết lập tần số mới cho từng lõi CPU
            for cpu in range(psutil.cpu_count(logical=True)):
                subprocess.run(['cpufreq-set', '-c', str(cpu), '-f', f"{int(new_freq)}MHz"], check=True)

            logger.info(f"Đã giảm tần số CPU xuống {int(new_freq)}MHz ({reduction_percentage}% giảm).")
        except subprocess.CalledProcessError as e:
            logger.error(f"Lỗi khi giảm tần số CPU: {e}")
        except FileNotFoundError:
            logger.error("cpufreq-set không được cài đặt trên hệ thống.")
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi giảm tần số CPU: {e}")

    def reduce_gpu_power(self, pid: Optional[int] = None, reduction_percentage: float = 20.0):
        """
        Giảm công suất GPU bằng cách giảm giới hạn công suất qua NVML.
        Tham số `pid` được thêm vào để tương thích với resource_manager.py, nhưng hiện tại không được sử dụng.

        Args:
            pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).
            reduction_percentage (float): Tỷ lệ giảm giới hạn công suất GPU (%).
        """
        if self.gpu_count == 0:
            logger.warning("Không có GPU nào để giảm công suất.")
            return

        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                current_power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                new_power_limit = int(current_power_limit * (1 - reduction_percentage / 100.0))

                # Đảm bảo giới hạn công suất không dưới mức tối thiểu
                constraints = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
                min_power_limit = constraints.minPowerLimit
                max_power_limit = constraints.maxPowerLimit
                new_power_limit = max(new_power_limit, min_power_limit)
                new_power_limit = min(new_power_limit, max_power_limit)

                pynvml.nvmlDeviceSetPowerManagementLimit(handle, new_power_limit)
                logger.info(f"Đã giảm giới hạn công suất GPU {i} xuống {new_power_limit}W ({reduction_percentage}% giảm).")
        except pynvml.NVMLError as e:
            logger.error(f"Lỗi khi giảm công suất GPU: {e}")
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi giảm công suất GPU: {e}")

    def set_gpu_usage(self, usage_percentages: List[float], pid: Optional[int] = None):
        """
        Điều chỉnh mức sử dụng GPU bằng cách thiết lập giới hạn công suất dựa trên phần trăm sử dụng mong muốn.
        Tham số `pid` được thêm vào để tương thích với resource_manager.py, nhưng hiện tại không được sử dụng.

        Args:
            usage_percentages (List[float]): Danh sách phần trăm sử dụng cho từng GPU.
            pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).
        """
        if self.gpu_count == 0:
            logger.warning("Không có GPU nào để điều chỉnh mức sử dụng.")
            return

        if len(usage_percentages) != self.gpu_count:
            logger.error("Số lượng phần trăm sử dụng không khớp với số lượng GPU.")
            return

        try:
            for i, usage in enumerate(usage_percentages):
                if not (0 <= usage <= 100):
                    logger.error(f"Phần trăm sử dụng GPU {i} không hợp lệ: {usage}%. Phải trong khoảng [0, 100].")
                    continue

                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                max_power = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                desired_power = int(max_power * (usage / 100.0))

                # Đảm bảo không vượt quá giới hạn tối đa
                constraints = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
                min_power_limit = constraints.minPowerLimit
                max_power_limit = constraints.maxPowerLimit
                desired_power = max(min(desired_power, max_power_limit), min_power_limit)

                pynvml.nvmlDeviceSetPowerManagementLimit(handle, desired_power)
                logger.info(f"Đã thiết lập giới hạn công suất GPU {i} thành {desired_power}W ({usage}%).")
        except pynvml.NVMLError as e:
            logger.error(f"Lỗi khi thiết lập mức sử dụng GPU: {e}")
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi thiết lập mức sử dụng GPU: {e}")

    def setup_power_management(self):
        """
        Thiết lập quản lý năng lượng bằng cách khởi động các tiến trình giám sát nếu cần.
        Được gọi bởi setup_env.py.
        """
        try:
            # Nếu cần thiết, có thể khởi động các tiến trình giám sát bổ sung tại đây
            logger.info("Đã thiết lập quản lý năng lượng.")
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập quản lý năng lượng: {e}")

    def shutdown(self):
        """
        Dừng quản lý năng lượng và giải phóng tài nguyên.
        """
        try:
            if self.gpu_count > 0:
                pynvml.nvmlShutdown()
                logger.info("NVML đã được shutdown thành công.")
        except pynvml.NVMLError as e:
            logger.error(f"Lỗi khi shutdown NVML: {e}")
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi shutdown NVML: {e}")


# Singleton instance của PowerManager
_power_manager_instance = PowerManager()

def setup_power_management():
    """
    Hàm để thiết lập quản lý năng lượng.
    Được gọi bởi setup_env.py.
    """
    _power_manager_instance.setup_power_management()

def get_cpu_power(pid: Optional[int] = None) -> float:
    """
    Hàm để lấy công suất CPU hiện tại.
    Được gọi bởi resource_manager.py.

    Args:
        pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).

    Returns:
        float: Công suất CPU hiện tại (W).
    """
    return _power_manager_instance.get_cpu_power(pid)

def get_gpu_power(pid: Optional[int] = None) -> List[float]:
    """
    Hàm để lấy công suất GPU hiện tại.
    Được gọi bởi resource_manager.py.

    Args:
        pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).

    Returns:
        List[float]: Danh sách công suất GPU hiện tại (W).
    """
    return _power_manager_instance.get_gpu_power(pid)

def reduce_cpu_power(pid: Optional[int] = None, reduction_percentage: float = 20.0):
    """
    Hàm để giảm công suất CPU.
    Được gọi bởi resource_manager.py.

    Args:
        pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).
        reduction_percentage (float): Tỷ lệ giảm tần số CPU (%).
    """
    _power_manager_instance.reduce_cpu_power(pid, reduction_percentage)

def reduce_gpu_power(pid: Optional[int] = None, reduction_percentage: float = 20.0):
    """
    Hàm để giảm công suất GPU.
    Được gọi bởi resource_manager.py.

    Args:
        pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).
        reduction_percentage (float): Tỷ lệ giảm giới hạn công suất GPU (%).
    """
    _power_manager_instance.reduce_gpu_power(pid, reduction_percentage)

def set_gpu_usage(usage_percentages: List[float], pid: Optional[int] = None):
    """
    Hàm để thiết lập mức sử dụng GPU.
    Được gọi bởi resource_manager.py.

    Args:
        usage_percentages (List[float]): Danh sách phần trăm sử dụng cho từng GPU.
        pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).
    """
    _power_manager_instance.set_gpu_usage(usage_percentages, pid)

def shutdown_power_management():
    """
    Hàm để dừng quản lý năng lượng và giải phóng tài nguyên.
    Được gọi khi hệ thống dừng lại.
    """
    _power_manager_instance.shutdown()
