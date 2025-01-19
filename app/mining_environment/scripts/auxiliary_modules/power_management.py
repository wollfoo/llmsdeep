# power_management.py

import os
import sys
import psutil
import subprocess
import asyncio
import pynvml
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
    _lock = asyncio.Lock()

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        # Tham số ước lượng CPU
        self.cpu_base_power_watts = 10.0
        self.cpu_max_power_watts = 150.0

        # Khởi tạo pynvml (đồng bộ) ngay trong init => handle trong threadpool
        self.pynvml_initialized = False

    @classmethod
    async def create(cls) -> 'PowerManager':
        """
        Async factory method => bảo đảm chỉ tạo 1 instance (singleton).
        """
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                await cls._instance.initialize()
            return cls._instance

    async def initialize(self):
        """
        Khởi tạo NVML ở dạng async (trong threadpool).
        """
        if not self.pynvml_initialized:
            try:
                await asyncio.to_thread(pynvml.nvmlInit)
                self.pynvml_initialized = True
                logger.info("PowerManager: Đã khởi tạo pynvml thành công.")
            except pynvml.NVMLError as e:
                logger.error(f"PowerManager: Lỗi khi khởi tạo pynvml: {e}")
                self.pynvml_initialized = False

        self.gpu_count = 0
        if self.pynvml_initialized:
            try:
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"PowerManager: Tìm thấy {self.gpu_count} GPU trên hệ thống.")
            except pynvml.NVMLError as e:
                logger.error(f"PowerManager: Lỗi khi lấy số lượng GPU: {e}")
                self.gpu_count = 0

    async def shutdown(self):
        """
        Dừng quản lý năng lượng và giải phóng NVML.
        """
        try:
            if self.pynvml_initialized:
                await asyncio.to_thread(pynvml.nvmlShutdown)
                self.pynvml_initialized = False
                logger.info("PowerManager: Đã shutdown thành công pynvml.")
            else:
                logger.warning("PowerManager: pynvml chưa được khởi tạo hoặc đã bị shutdown.")
        except pynvml.NVMLError as e:
            logger.error(f"PowerManager: Lỗi khi shutdown pynvml: {e}")
        except Exception as e:
            logger.error(f"PowerManager: Lỗi khi shutdown pynvml: {e}")

    async def get_cpu_power(self, pid: Optional[int] = None) -> float:
        """
        Ước tính công suất CPU bằng cách đọc cpu_percent(1s).
        """
        try:
            cpu_load = await asyncio.to_thread(psutil.cpu_percent, 1)
            estimated_power = (
                self.cpu_base_power_watts
                + (cpu_load / 100.0) * (self.cpu_max_power_watts - self.cpu_base_power_watts)
            )
            logger.debug(f"PowerManager: CPU Load={cpu_load}%, Estimated Power={estimated_power:.2f}W")
            return estimated_power
        except Exception as e:
            logger.error(f"PowerManager: Lỗi khi ước tính công suất CPU: {e}")
            return 0.0

    async def get_gpu_power_async(self, pid: Optional[int] = None) -> List[float]:
        """
        Lấy công suất tiêu thụ hiện tại của từng GPU bằng pynvml (W).
        """
        if not self.pynvml_initialized:
            logger.warning("PowerManager: NVML chưa khởi tạo => không thể lấy GPU power.")
            return []
        try:
            powers = []
            for i in range(self.gpu_count):
                p = await asyncio.to_thread(self._get_single_gpu_power, i)
                if p is not None:
                    powers.append(p)
            return powers
        except Exception as e:
            logger.error(f"PowerManager: Lỗi khi lấy công suất GPU: {e}")
            return []

    def _get_single_gpu_power(self, gpu_index: int) -> Optional[float]:
        """
        Lấy công suất GPU (mW -> W) chạy trong threadpool.
        """
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            return power_mw / 1000.0  # mW -> W
        except pynvml.NVMLError as e:
            logger.error(f"PowerManager: Lỗi khi đọc power GPU={gpu_index}: {e}")
            return None

    async def reduce_cpu_power(self, pid: Optional[int] = None, reduction_percentage: float = 20.0):
        """
        Giảm công suất CPU qua hạ tần số CPU (giả lập).
        """
        # Tuỳ logic
        logger.info(f"PowerManager: Giảm CPU power {reduction_percentage}%. (chưa có logic chi tiết)")

    async def reduce_gpu_power(self, pid: Optional[int] = None, reduction_percentage: float = 20.0):
        """
        Giảm công suất GPU => set power limit (tuỳ logic).
        """
        if not self.pynvml_initialized or self.gpu_count == 0:
            logger.warning("PowerManager: Không có GPU hoặc NVML chưa init.")
            return
        if not (0 < reduction_percentage <= 100):
            logger.error(f"PowerManager: Tham số reduction_percentage không hợp lệ: {reduction_percentage}")
            return
        # Demo logic:
        for i in range(self.gpu_count):
            desired = await self.calculate_desired_power_limit(i, (100 - reduction_percentage))
            if desired is not None:
                await self.set_gpu_power_limit(i, desired)

    async def calculate_desired_power_limit(self, gpu_index: int, usage_percentage: float) -> Optional[float]:
        """
        Tính power limit mới dựa trên usage_percentage (0..100).
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
            return max(power_min, min(desired_power, power_max))
        except pynvml.NVMLError as e:
            logger.error(f"PowerManager: Lỗi calculate_desired_power_limit GPU={gpu_index}: {e}")
            return None

    async def set_gpu_power_limit(self, gpu_index: int, power_limit_watts: float) -> bool:
        """
        Thiết lập power limit (W) cho GPU.
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

    async def set_gpu_usage(self, usage_percentages: List[float], pid: Optional[int] = None):
        """
        Thiết lập giới hạn công suất cho từng GPU dựa trên usage_percentages.
        """
        if not self.pynvml_initialized or self.gpu_count == 0:
            logger.warning("PowerManager: Không có GPU hoặc NVML chưa init.")
            return
        if len(usage_percentages) != self.gpu_count:
            logger.warning("PowerManager: Số GPU khác với độ dài usage_percentages => điều chỉnh.")
            usage_percentages = (usage_percentages + [0]*(self.gpu_count))[:self.gpu_count]

        for i, usage in enumerate(usage_percentages):
            desired = await self.calculate_desired_power_limit(i, usage)
            if desired is not None:
                ok = await self.set_gpu_power_limit(i, desired)
                if ok:
                    logger.info(f"PowerManager: GPU={i} => limit {desired}W, usage={usage}%")
            else:
                logger.error(f"PowerManager: Tính limit cho GPU={i} với usage={usage}% thất bại.")


###############################################################################
#                          Singleton Instance                                #
###############################################################################

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
    return await power_manager.get_gpu_power_async(pid)

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
