# power_management.py

import os
import sys
import psutil
import pynvml
import subprocess
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

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
    PowerManager là một singleton class chịu trách nhiệm quản lý năng lượng
    cho CPU và GPU. Nó cung cấp các phương thức để giám sát công suất tiêu thụ
    và điều chỉnh công suất khi cần thiết.
    """
    _instance = None
    _lock = asyncio.Lock()

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
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

        # Tham số ước lượng CPU (tùy thuộc hệ thống)
        self.cpu_base_power_watts = 10.0   # Công suất cơ bản khi CPU idle (W)
        self.cpu_max_power_watts = 150.0  # Công suất tối đa khi CPU full load (W)

    @classmethod
    async def create(cls) -> 'PowerManager':
        """
        Async factory method để tạo và khởi tạo instance của PowerManager.

        Returns:
            PowerManager: Instance đã được khởi tạo.
        """
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                await cls._instance.initialize()
            return cls._instance

    async def initialize(self):
        """
        Phương thức async để khởi tạo các thuộc tính bất đồng bộ nếu cần thiết.
        """
        # Trong trường hợp hiện tại, không có tác vụ async nào để khởi tạo thêm.
        # Nếu cần, thêm các tác vụ khởi tạo async ở đây.
        pass

    async def get_gpu_count(self) -> int:
        """
        Lấy số lượng GPU hiện có trên hệ thống.

        Returns:
            int: Số lượng GPU.
        """
        try:
            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(None, pynvml.nvmlDeviceGetCount)
            logger.debug(f"Đã lấy số lượng GPU: {count}")
            return count
        except pynvml.NVMLError as e:
            logger.error(f"Lỗi khi lấy số lượng GPU: {e}")
            return 0
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi lấy số lượng GPU: {e}")
            return 0

    async def get_gpu_usage_percentages(self) -> List[float]:
        """
        Lấy danh sách phần trăm sử dụng của tất cả GPU trên hệ thống.

        Returns:
            List[float]: Danh sách phần trăm sử dụng GPU cho từng GPU.
        """
        usage_percentages = []
        if self.gpu_count == 0:
            logger.warning("Không có GPU nào được phát hiện để giám sát công suất.")
            return usage_percentages

        try:
            loop = asyncio.get_event_loop()
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                utilization = await loop.run_in_executor(None, pynvml.nvmlDeviceGetUtilizationRates, handle)
                usage_percentages.append(utilization.gpu)
                logger.debug(f"GPU {i}: {utilization.gpu}% sử dụng")
            return usage_percentages
        except pynvml.NVMLError as e:
            logger.error(f"Lỗi khi lấy phần trăm sử dụng GPU: {e}")
            return usage_percentages
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi lấy phần trăm sử dụng GPU: {e}")
            return usage_percentages

    async def get_cpu_power(self, pid: Optional[int] = None) -> float:
        """
        Ước tính công suất tiêu thụ hiện tại của CPU dựa trên tải CPU.
        Tham số `pid` chỉ để tương thích, hiện chưa dùng.

        Args:
            pid (Optional[int]): PID của tiến trình (không sử dụng).

        Returns:
            float: Công suất CPU hiện tại (W).
        """
        try:
            loop = asyncio.get_event_loop()
            cpu_load = await loop.run_in_executor(None, psutil.cpu_percent, 1)
            estimated_power = (
                self.cpu_base_power_watts
                + (cpu_load / 100.0) * (self.cpu_max_power_watts - self.cpu_base_power_watts)
            )
            logger.debug(f"CPU Load: {cpu_load}%, Estimated CPU Power: {estimated_power:.2f}W")
            return estimated_power
        except Exception as e:
            logger.error(f"Lỗi khi ước tính công suất CPU: {e}")
            return 0.0

    async def get_gpu_power(self, pid: Optional[int] = None) -> List[float]:
        """
        Lấy công suất tiêu thụ hiện tại của từng GPU bằng NVML.

        Args:
            pid (Optional[int]): PID của tiến trình (không sử dụng).

        Returns:
            List[float]: Danh sách công suất GPU (W) cho mỗi GPU.
        """
        gpu_powers = []
        if self.gpu_count == 0:
            logger.warning("Không có GPU nào được phát hiện để giám sát công suất.")
            return gpu_powers

        try:
            loop = asyncio.get_event_loop()
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                power_usage_mw = await loop.run_in_executor(None, pynvml.nvmlDeviceGetPowerUsage, handle)
                power_usage = power_usage_mw / 1000.0  # mW -> W
                gpu_powers.append(power_usage)
                logger.debug(f"GPU {i}: {power_usage}W")
            return gpu_powers
        except pynvml.NVMLError as e:
            logger.error(f"Lỗi khi lấy công suất GPU: {e}")
            return gpu_powers
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi lấy công suất GPU: {e}")
            return gpu_powers

    async def reduce_cpu_power(self, pid: Optional[int] = None, reduction_percentage: float = 20.0):
        """
        Giảm công suất CPU bằng cách giảm tần số CPU (thông qua cpufreq-set).

        Args:
            pid (Optional[int]): PID của tiến trình (không sử dụng).
            reduction_percentage (float): Tỷ lệ giảm tần số CPU (%).
        """
        try:
            if not (0 < reduction_percentage < 100):
                logger.error("Reduction percentage phải nằm trong khoảng (0, 100).")
                return

            loop = asyncio.get_event_loop()
            cpu_freq = await loop.run_in_executor(None, lambda: psutil.cpu_freq().current)  # MHz
            new_freq = cpu_freq * (1 - reduction_percentage / 100.0)

            min_freq = 1800  # MHz
            new_freq = max(new_freq, min_freq)

            for cpu in range(psutil.cpu_count(logical=True)):
                # Đặt governor thành 'userspace' trước khi thiết lập tần số
                result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(['cpufreq-set', '-c', str(cpu), '-g', 'userspace'],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                )
                if result.returncode != 0:
                    logger.error(f"Lỗi khi đặt governor cho CPU {cpu}: {result.stderr.strip()}")
                    continue  # Tiếp tục với CPU tiếp theo

                logger.info(f"Đặt governor của CPU {cpu} thành 'userspace'.")

                # Thiết lập tần số CPU
                result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(['cpufreq-set', '-c', str(cpu), '-f', f"{int(new_freq)}MHz"],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                )
                if result.returncode != 0:
                    logger.error(f"Lỗi khi thiết lập tần số CPU {cpu}: {result.stderr.strip()}")
                    continue  # Tiếp tục với CPU tiếp theo

            logger.info(f"Đã giảm tần số CPU xuống {int(new_freq)}MHz ({reduction_percentage}% giảm).")
        except subprocess.CalledProcessError as e:
            logger.error(f"Lỗi khi giảm tần số CPU: {e}")
        except FileNotFoundError:
            logger.error("cpufreq-set không được cài đặt trên hệ thống.")
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi giảm tần số CPU: {e}")

    async def reduce_gpu_power(self, pid: Optional[int] = None, reduction_percentage: float = 20.0):
        """
        Giảm công suất GPU bằng cách giảm giới hạn công suất qua NVML.

        Args:
            pid (Optional[int]): PID của tiến trình (không sử dụng).
            reduction_percentage (float): Tỷ lệ giảm công suất (%).
        """
        if self.gpu_count == 0:
            logger.warning("Không có GPU nào để giảm công suất.")
            return

        try:
            loop = asyncio.get_event_loop()
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                current_power_limit = await loop.run_in_executor(None, pynvml.nvmlDeviceGetPowerManagementLimit, handle)
                new_power_limit = int(current_power_limit * (1 - reduction_percentage / 100.0))

                constraints = await loop.run_in_executor(None, pynvml.nvmlDeviceGetPowerManagementLimitConstraints, handle)
                min_power_limit = constraints.minPowerLimit
                max_power_limit = constraints.maxPowerLimit

                new_power_limit = max(new_power_limit, min_power_limit)
                new_power_limit = min(new_power_limit, max_power_limit)

                await loop.run_in_executor(None, pynvml.nvmlDeviceSetPowerManagementLimit, handle, new_power_limit)
                logger.info(
                    f"Đã giảm giới hạn công suất GPU {i} xuống {new_power_limit}W ({reduction_percentage}% giảm)."
                )
        except pynvml.NVMLError as e:
            logger.error(f"Lỗi khi giảm công suất GPU: {e}")
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi giảm công suất GPU: {e}")

    async def set_gpu_usage(self, usage_percentages: List[float], pid: Optional[int] = None):
        """
        Điều chỉnh mức sử dụng GPU bằng cách thiết lập giới hạn công suất.

        Args:
            usage_percentages (List[float]): % sử dụng cho từng GPU.
            pid (Optional[int]): PID của tiến trình (không sử dụng trong ví dụ này).
        """
        if self.gpu_count == 0:
            logger.warning("Không có GPU nào để điều chỉnh mức sử dụng.")
            return

        if not isinstance(usage_percentages, list):
            logger.error(f"usage_percentages không phải là list. Đã nhận: {type(usage_percentages)}")
            return

        if len(usage_percentages) != self.gpu_count:
            logger.error(f"Số lượng phần trăm sử dụng ({len(usage_percentages)}) không khớp với số lượng GPU ({self.gpu_count}).")
            # Tự động điều chỉnh danh sách
            if len(usage_percentages) < self.gpu_count:
                # Bổ sung các phần trăm sử dụng cho GPU còn thiếu
                additional = [0.0] * (self.gpu_count - len(usage_percentages))
                usage_percentages.extend(additional)
                logger.info(f"Bổ sung các phần trăm sử dụng GPU còn thiếu: {additional}")
            else:
                # Cắt bớt các phần trăm sử dụng thừa
                usage_percentages = usage_percentages[:self.gpu_count]
                logger.info(f"Cắt bớt các phần trăm sử dụng GPU thừa. Danh sách mới: {usage_percentages}")

        try:
            loop = asyncio.get_event_loop()
            for i, usage in enumerate(usage_percentages):
                if not (0 <= usage <= 100):
                    logger.error(f"Phần trăm sử dụng GPU {i} không hợp lệ: {usage}%. [0..100].")
                    continue

                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                max_power = await loop.run_in_executor(None, pynvml.nvmlDeviceGetPowerManagementLimit, handle)
                desired_power = int(max_power * (usage / 100.0))

                constraints = await loop.run_in_executor(None, pynvml.nvmlDeviceGetPowerManagementLimitConstraints, handle)

                # Kiểm tra kiểu dữ liệu của constraints
                if not hasattr(constraints, 'minPowerLimit') or not hasattr(constraints, 'maxPowerLimit'):
                    logger.error(f"constraints không có thuộc tính 'minPowerLimit' hoặc 'maxPowerLimit': {constraints}")
                    continue

                min_power_limit = constraints.minPowerLimit
                max_power_limit = constraints.maxPowerLimit

                desired_power = max(min(desired_power, max_power_limit), min_power_limit)

                await loop.run_in_executor(None, pynvml.nvmlDeviceSetPowerManagementLimit, handle, desired_power)
                logger.info(f"Đã thiết lập giới hạn công suất GPU {i} thành {desired_power}W ({usage}%).")
        except pynvml.NVMLError as e:
            logger.error(f"Lỗi khi thiết lập mức sử dụng GPU: {e}")
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi thiết lập mức sử dụng GPU: {e}")

    async def shutdown(self):
        """
        Dừng quản lý năng lượng và giải phóng tài nguyên GPU.
        """
        try:
            if self.gpu_count > 0:
                await asyncio.get_event_loop().run_in_executor(None, pynvml.nvmlShutdown)
                logger.info("NVML đã được shutdown thành công.")
        except pynvml.NVMLError as e:
            logger.error(f"Lỗi khi shutdown NVML: {e}")
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi shutdown NVML: {e}")


# Singleton instance của PowerManager sẽ được tạo thông qua async factory method
_power_manager_instance: Optional[PowerManager] = None
_power_manager_lock = asyncio.Lock()


async def get_power_manager() -> PowerManager:
    """
    Lấy singleton instance của PowerManager một cách bất đồng bộ.

    Returns:
        PowerManager: Instance của PowerManager.
    """
    global _power_manager_instance
    async with _power_manager_lock:
        if _power_manager_instance is None:
            _power_manager_instance = await PowerManager.create()
    return _power_manager_instance


async def get_cpu_power(pid: Optional[int] = None) -> float:
    """
    Trả về công suất CPU hiện tại (W).

    Args:
        pid (Optional[int]): PID của tiến trình (không sử dụng).

    Returns:
        float: Công suất CPU hiện tại (W).
    """
    power_manager = await get_power_manager()
    return await power_manager.get_cpu_power(pid)


async def get_gpu_power(pid: Optional[int] = None) -> List[float]:
    """
    Trả về danh sách công suất GPU (W) cho từng GPU.

    Args:
        pid (Optional[int]): PID của tiến trình (không sử dụng).

    Returns:
        List[float]: Danh sách công suất GPU (W) cho từng GPU.
    """
    power_manager = await get_power_manager()
    return await power_manager.get_gpu_power(pid)


async def reduce_cpu_power(pid: Optional[int] = None, reduction_percentage: float = 20.0):
    """
    Giảm công suất CPU.

    Args:
        pid (Optional[int]): PID của tiến trình (không sử dụng).
        reduction_percentage (float): Tỷ lệ giảm tần số CPU (%).
    """
    power_manager = await get_power_manager()
    await power_manager.reduce_cpu_power(pid, reduction_percentage)


async def reduce_gpu_power(pid: Optional[int] = None, reduction_percentage: float = 20.0):
    """
    Giảm công suất GPU.

    Args:
        pid (Optional[int]): PID của tiến trình (không sử dụng).
        reduction_percentage (float): Tỷ lệ giảm công suất (%).
    """
    power_manager = await get_power_manager()
    await power_manager.reduce_gpu_power(pid, reduction_percentage)


async def set_gpu_usage(usage_percentages: List[float], pid: Optional[int] = None):
    """
    Thiết lập mức sử dụng GPU (giới hạn công suất).

    Args:
        usage_percentages (List[float]): % sử dụng cho từng GPU.
        pid (Optional[int]): PID của tiến trình (không sử dụng trong ví dụ này).
    """
    power_manager = await get_power_manager()
    await power_manager.set_gpu_usage(usage_percentages, pid)


async def shutdown_power_management():
    """
    Được gọi khi hệ thống dừng lại.
    """
    power_manager = await get_power_manager()
    await power_manager.shutdown()
