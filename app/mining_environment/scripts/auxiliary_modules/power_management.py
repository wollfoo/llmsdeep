# power_management.py

import os
import sys
import psutil
import subprocess
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any

# Thêm đường dẫn tới thư mục chứa `logging_config.py`
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SCRIPT_DIR))

# Thiết lập đường dẫn tới thư mục logs
LOGS_DIR = Path(os.getenv('LOGS_DIR', '/app/mining_environment/logs'))
os.makedirs(LOGS_DIR, exist_ok=True)

# Import hàm setup_logging từ logging_config.py
from logging_config import setup_logging

# Import lớp GPUManager từ utils.py đã được cải tiến
from utils import GPUManager

# Thiết lập logging với logging_config.py
logger = setup_logging('power_management', LOGS_DIR / 'power_management.log', 'INFO')

###############################################################################
#                            LỚP PowerManager                                #
###############################################################################
class PowerManager:
    """
    PowerManager là một singleton class chịu trách nhiệm quản lý năng lượng
    cho CPU và GPU. Nó cung cấp các phương thức để giám sát công suất tiêu thụ
    và điều chỉnh công suất khi cần thiết. 
    Trong phiên bản này, GPU được quản lý thông qua lớp GPUManager từ utils.py.
    """
    _instance = None
    _lock = asyncio.Lock()

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        # Khởi tạo GPUManager để quản lý GPU
        self.gpu_manager = GPUManager()
        self.gpu_count = 0  # Số lượng GPU, sẽ được cập nhật sau khi khởi tạo GPUManager

        # Tham số ước lượng CPU (tùy thuộc hệ thống)
        self.cpu_base_power_watts = 10.0   # Công suất cơ bản khi CPU idle (W)
        self.cpu_max_power_watts = 150.0  # Công suất tối đa khi CPU full load (W)

    @classmethod
    async def create(cls) -> 'PowerManager':
        """
        Async factory method để tạo và khởi tạo instance của PowerManager.
        Đảm bảo rằng chỉ có một instance được tạo ra (singleton).
        """
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                await cls._instance.initialize()
            return cls._instance

    async def initialize(self):
        """
        Phương thức async để khởi tạo các thuộc tính bất đồng bộ nếu cần thiết.
        Trong trường hợp này, khởi tạo GPUManager.
        """
        # Khởi tạo GPUManager
        gpu_init_success = await self.gpu_manager.initialize()
        if gpu_init_success:
            self.gpu_count = self.gpu_manager.gpu_count
            logger.info(f"PowerManager: Đã khởi tạo GPUManager thành công với {self.gpu_count} GPU.")
        else:
            self.gpu_count = 0
            logger.warning("PowerManager: GPUManager không được khởi tạo thành công.")

    async def get_gpu_count(self) -> int:
        """
        Lấy số lượng GPU hiện có trên hệ thống thông qua GPUManager.

        Returns:
            int: Số lượng GPU.
        """
        return self.gpu_count

    async def get_gpu_usage_percentages(self) -> List[float]:
        """
        Lấy danh sách phần trăm sử dụng của tất cả GPU trên hệ thống.

        Returns:
            List[float]: Danh sách phần trăm sử dụng GPU cho từng GPU.
        """
        if self.gpu_count == 0:
            logger.warning("PowerManager: Không có GPU nào được phát hiện để giám sát công suất.")
            return []
        try:
            # Sử dụng phương thức từ GPUManager để lấy utilization
            utilization = []
            for i in range(self.gpu_count):
                util = await self.gpu_manager.get_gpu_utilization(i)
                if util and 'gpu_util_percent' in util:
                    utilization.append(util['gpu_util_percent'])
                else:
                    utilization.append(0.0)
            logger.debug(f"PowerManager: Lấy utilization GPU: {utilization}")
            return utilization
        except Exception as e:
            logger.error(f"PowerManager: Lỗi khi lấy utilization GPU: {e}")
            return []

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
            logger.debug(f"PowerManager: CPU Load: {cpu_load}%, Estimated CPU Power: {estimated_power:.2f}W")
            return estimated_power
        except Exception as e:
            logger.error(f"PowerManager: Lỗi khi ước tính công suất CPU: {e}")
            return 0.0

    async def get_gpu_power(self, pid: Optional[int] = None) -> List[float]:
        """
        Lấy công suất tiêu thụ hiện tại của từng GPU bằng GPUManager.

        Args:
            pid (Optional[int]): PID của tiến trình (không sử dụng).

        Returns:
            List[float]: Danh sách công suất GPU (W) cho từng GPU.
        """
        if self.gpu_count == 0:
            logger.warning("PowerManager: Không có GPU nào để giám sát công suất.")
            return []
        try:
            gpu_powers = []
            for i in range(self.gpu_count):
                power = await self.gpu_manager.get_gpu_power_limit(i)
                if power is not None:
                    gpu_powers.append(power)
                else:
                    gpu_powers.append(0.0)
            logger.debug(f"PowerManager: Lấy công suất GPU: {gpu_powers}")
            return gpu_powers
        except Exception as e:
            logger.error(f"PowerManager: Lỗi khi lấy công suất GPU: {e}")
            return []

    async def reduce_cpu_power(self, pid: Optional[int] = None, reduction_percentage: float = 20.0):
        """
        Giảm công suất CPU bằng cách giảm tần số CPU.

        Args:
            pid (Optional[int]): PID của tiến trình (không sử dụng).
            reduction_percentage (float): Tỷ lệ giảm tần số CPU (%).
        """
        try:
            if not (0 < reduction_percentage < 100):
                logger.error("PowerManager: Reduction percentage phải nằm trong khoảng (0, 100).")
                return

            loop = asyncio.get_event_loop()
            cpu_freq = await loop.run_in_executor(None, lambda: psutil.cpu_freq().current)  # MHz
            new_freq = cpu_freq * (1 - reduction_percentage / 100.0)

            min_freq = 1800  # MHz, có thể điều chỉnh tùy hệ thống
            new_freq = max(new_freq, min_freq)

            for cpu in range(psutil.cpu_count(logical=True)):
                # Đặt governor thành 'userspace' trước khi thiết lập tần số
                result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(['cpufreq-set', '-c', str(cpu), '-g', 'userspace'],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                )
                if result.returncode != 0:
                    logger.error(f"PowerManager: Lỗi khi đặt governor cho CPU {cpu}: {result.stderr.strip()}")
                    continue  # Tiếp tục với CPU tiếp theo

                logger.info(f"PowerManager: Đặt governor của CPU {cpu} thành 'userspace'.")

                # Thiết lập tần số CPU
                result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(['cpufreq-set', '-c', str(cpu), '-f', f"{int(new_freq)}MHz"],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                )
                if result.returncode != 0:
                    logger.error(f"PowerManager: Lỗi khi thiết lập tần số CPU {cpu}: {result.stderr.strip()}")
                    continue  # Tiếp tục với CPU tiếp theo

            logger.info(f"PowerManager: Đã giảm tần số CPU xuống {int(new_freq)}MHz ({reduction_percentage}% giảm).")
        except subprocess.CalledProcessError as e:
            logger.error(f"PowerManager: Lỗi khi giảm tần số CPU: {e}")
        except FileNotFoundError:
            logger.error("PowerManager: cpufreq-set không được cài đặt trên hệ thống.")
        except Exception as e:
            logger.error(f"PowerManager: Lỗi không mong muốn khi giảm tần số CPU: {e}")

    async def reduce_gpu_power(self, pid: Optional[int] = None, reduction_percentage: float = 20.0):
        """
        Giảm công suất GPU bằng cách giảm giới hạn công suất qua GPUManager.

        Args:
            pid (Optional[int]): PID của tiến trình (không sử dụng).
            reduction_percentage (float): Tỷ lệ giảm công suất (%).
        """
        if self.gpu_count == 0:
            logger.warning("PowerManager: Không có GPU nào để giảm công suất.")
            return

        try:
            for i in range(self.gpu_count):
                # Tính toán power limit mới dựa trên reduction_percentage
                desired_power = await self.gpu_manager.calculate_desired_power_limit(i, reduction_percentage)
                if desired_power is not None:
                    success = await self.gpu_manager.set_gpu_power_limit(i, desired_power)
                    if success:
                        logger.info(f"PowerManager: Đã giảm giới hạn công suất GPU {i} xuống {desired_power}W ({reduction_percentage}% giảm).")
                    else:
                        logger.error(f"PowerManager: Không thể giảm giới hạn công suất GPU {i}.")
                else:
                    logger.error(f"PowerManager: Không thể tính toán power limit mới cho GPU {i}.")
        except Exception as e:
            logger.error(f"PowerManager: Lỗi không mong muốn khi giảm công suất GPU: {e}")

    async def set_gpu_usage(self, usage_percentages: List[float], pid: Optional[int] = None):
        """
        Điều chỉnh mức sử dụng GPU bằng cách thiết lập giới hạn công suất thông qua GPUManager.

        Args:
            usage_percentages (List[float]): % sử dụng cho từng GPU.
            pid (Optional[int]): PID của tiến trình (không sử dụng trong ví dụ này).
        """
        if self.gpu_count == 0:
            logger.warning("PowerManager: Không có GPU nào để điều chỉnh mức sử dụng.")
            return

        if not isinstance(usage_percentages, list):
            logger.error(f"PowerManager: usage_percentages không phải là list. Đã nhận: {type(usage_percentages)}")
            return

        if len(usage_percentages) != self.gpu_count:
            logger.error(f"PowerManager: Số lượng phần trăm sử dụng ({len(usage_percentages)}) không khớp với số lượng GPU ({self.gpu_count}).")
            # Tự động điều chỉnh danh sách
            if len(usage_percentages) < self.gpu_count:
                # Bổ sung các phần trăm sử dụng cho GPU còn thiếu
                additional = [0.0] * (self.gpu_count - len(usage_percentages))
                usage_percentages.extend(additional)
                logger.info(f"PowerManager: Bổ sung các phần trăm sử dụng GPU còn thiếu: {additional}")
            else:
                # Cắt bớt các phần trăm sử dụng thừa
                usage_percentages = usage_percentages[:self.gpu_count]
                logger.info(f"PowerManager: Cắt bớt các phần trăm sử dụng GPU thừa. Danh sách mới: {usage_percentages}")

        try:
            for i, usage in enumerate(usage_percentages):
                if not (0 <= usage <= 100):
                    logger.error(f"PowerManager: Phần trăm sử dụng GPU {i} không hợp lệ: {usage}%. [0..100].")
                    continue

                desired_power = await self.gpu_manager.calculate_desired_power_limit(i, usage)
                if desired_power is not None:
                    success = await self.gpu_manager.set_gpu_power_limit(i, desired_power)
                    if success:
                        logger.info(f"PowerManager: Đã thiết lập giới hạn công suất GPU {i} thành {desired_power}W ({usage}%).")
                    else:
                        logger.error(f"PowerManager: Không thể thiết lập giới hạn công suất GPU {i}.")
                else:
                    logger.error(f"PowerManager: Không thể tính toán power limit mới cho GPU {i}.")
        except Exception as e:
            logger.error(f"PowerManager: Lỗi khi thiết lập mức sử dụng GPU: {e}")

    async def shutdown(self):
        """
        Dừng quản lý năng lượng và giải phóng tài nguyên GPU thông qua GPUManager.
        """
        try:
            await self.gpu_manager.shutdown_nvml()
            logger.info("PowerManager: Đã shutdown thành công GPUManager.")
        except Exception as e:
            logger.error(f"PowerManager: Lỗi khi shutdown GPUManager: {e}")

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

###############################################################################
#                          Các Hàm Ngoài Class                              #
###############################################################################
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
    Được gọi khi hệ thống dừng lại để giải phóng tài nguyên.
    """
    power_manager = await get_power_manager()
    await power_manager.shutdown()
