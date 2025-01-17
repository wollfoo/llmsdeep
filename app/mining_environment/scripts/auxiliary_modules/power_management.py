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
    PowerManager là một singleton class chịu trách nhiệm quản lý năng lượng
    cho CPU và GPU. Nó cung cấp các phương thức để giám sát công suất tiêu thụ
    và điều chỉnh công suất khi cần thiết. 
    Trong phiên bản này, GPU được quản lý trực tiếp thông qua pynvml.
    """
    _instance = None
    _lock = asyncio.Lock()

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        # Khởi tạo pynvml nếu chưa được khởi tạo
        self._initialize_pynvml()

        # Lấy số lượng GPU hiện có
        self.gpu_count = self._get_gpu_count()

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

    def _initialize_pynvml(self):
        """
        Khởi tạo pynvml nếu chưa được khởi tạo.
        """
        try:
            pynvml.nvmlInit()
            logger.info("PowerManager: Đã khởi tạo pynvml thành công.")
        except pynvml.NVMLError as e:
            logger.error(f"PowerManager: Lỗi khi khởi tạo pynvml: {e}")
            self.pynvml_initialized = False
            return
        self.pynvml_initialized = True

    def _check_pynvml_initialized(self) -> bool:
        """
        Kiểm tra xem pynvml đã được khởi tạo chưa.
        
        Returns:
            bool: True nếu đã khởi tạo, False ngược lại.
        """
        try:
            state = pynvml.nvmlSystemGetDriverVersion()
            logger.debug(f"PowerManager: pynvml đã được khởi tạo. Phiên bản driver: {state}")
            return True
        except pynvml.NVMLError_NotInitialized:
            logger.warning("PowerManager: pynvml chưa được khởi tạo. Đang khởi tạo lại...")
            try:
                pynvml.nvmlInit()
                logger.info("PowerManager: Đã khởi tạo pynvml thành công.")
                return True
            except pynvml.NVMLError as e:
                logger.error(f"PowerManager: Lỗi khi khởi tạo pynvml: {e}")
                return False

    def _get_gpu_count(self) -> int:
        """
        Lấy số lượng GPU hiện có trên hệ thống.
        
        Returns:
            int: Số lượng GPU.
        """
        if not self._check_pynvml_initialized():
            return 0
        try:
            count = pynvml.nvmlDeviceGetCount()
            logger.info(f"PowerManager: Tìm thấy {count} GPU trên hệ thống.")
            return count
        except pynvml.NVMLError as e:
            logger.error(f"PowerManager: Lỗi khi lấy số lượng GPU: {e}")
            return 0

    async def initialize(self):
        """
        Phương thức async để khởi tạo các thuộc tính bất đồng bộ nếu cần thiết.
        """
        # Có thể thêm các khởi tạo khác nếu cần
        pass

    async def get_gpu_count_async(self) -> int:
        """
        Async method để lấy số lượng GPU hiện có trên hệ thống.
        
        Returns:
            int: Số lượng GPU.
        """
        return self.gpu_count

    async def get_gpu_utilization_percentages(self) -> List[float]:
        if not self.pynvml_initialized:
            logger.warning("PowerManager: NVML chưa được khởi tạo, không thể lấy utilization GPU.")
            return []
        try:
            loop = asyncio.get_event_loop()
            utilization = await asyncio.gather(
                *(loop.run_in_executor(None, self._get_single_gpu_utilization, i) for i in range(self.gpu_count))
            )
            logger.debug(f"PowerManager: GPU utilizations: {utilization}")
            return [u for u in utilization if u is not None]
        except Exception as e:
            logger.error(f"PowerManager: Lỗi khi lấy utilization GPU: {e}")
            return []

    def _get_single_gpu_utilization(self, gpu_index: int) -> Optional[float]:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except pynvml.NVMLError as e:
            logger.error(f"PowerManager: Lỗi khi lấy utilization GPU {gpu_index}: {e}")
            return None

    async def get_gpu_power_limits(self) -> List[float]:
        """
        Lấy danh sách giới hạn công suất của tất cả GPU trên hệ thống.
        
        Returns:
            List[float]: Danh sách giới hạn công suất GPU (W) cho từng GPU.
        """
        if self.gpu_count == 0:
            logger.warning("PowerManager: Không có GPU nào được phát hiện để giám sát công suất.")
            return []
        try:
            power_limits = []
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0  # chuyển từ mW sang W
                power_limits.append(power_limit)
            logger.debug(f"PowerManager: Lấy giới hạn công suất GPU: {power_limits}")
            return power_limits
        except pynvml.NVMLError as e:
            logger.error(f"PowerManager: Lỗi khi lấy giới hạn công suất GPU: {e}")
            return []

    async def set_gpu_power_limit(self, gpu_index: int, power_limit_watts: float) -> bool:
        """
        Thiết lập giới hạn công suất cho GPU cụ thể.
        
        Args:
            gpu_index (int): Chỉ số GPU (0-based).
            power_limit_watts (float): Giới hạn công suất mới (W).
        
        Returns:
            bool: True nếu thành công, False ngược lại.
        """
        if not self._check_pynvml_initialized():
            logger.error("PowerManager: Không thể thiết lập giới hạn công suất vì pynvml chưa được khởi tạo.")
            return False
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            power_limit_mw = int(power_limit_watts * 1000)  # chuyển từ W sang mW
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, power_limit_mw)
            logger.info(f"PowerManager: Đã thiết lập giới hạn công suất GPU {gpu_index} thành {power_limit_watts}W.")
            return True
        except pynvml.NVMLError as e:
            logger.error(f"PowerManager: Lỗi khi thiết lập giới hạn công suất GPU {gpu_index}: {e}")
            return False

    async def calculate_desired_power_limit(self, gpu_index: int, usage_percentage: float) -> Optional[float]:
        """
        Tính toán giới hạn công suất mới dựa trên phần trăm sử dụng mong muốn.
        
        Args:
            gpu_index (int): Chỉ số GPU (0-based).
            usage_percentage (float): Phần trăm sử dụng mong muốn (0-100).
        
        Returns:
            Optional[float]: Giới hạn công suất mới (W) nếu thành công, None ngược lại.
        """
        if not (0 <= usage_percentage <= 100):
            logger.error(f"PowerManager: usage_percentage phải nằm trong khoảng [0, 100], nhận được: {usage_percentage}")
            return None
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            power_max = pynvml.nvmlDeviceGetPowerManagementLimitMax(handle) / 1000.0  # W
            power_min = pynvml.nvmlDeviceGetPowerManagementLimitMin(handle) / 1000.0  # W

            # Giới hạn mong muốn dựa trên usage_percentage
            desired_power = power_min + (power_max - power_min) * (usage_percentage / 100.0)
            desired_power = round(desired_power, 2)  # làm tròn đến 2 chữ số thập phân

            # Đảm bảo giới hạn nằm trong khoảng [power_min, power_max]
            desired_power = max(power_min, min(desired_power, power_max))

            logger.debug(f"PowerManager: Tính toán giới hạn công suất mới cho GPU {gpu_index}: {desired_power}W dựa trên {usage_percentage}% sử dụng.")
            return desired_power
        except pynvml.NVMLError as e:
            logger.error(f"PowerManager: Lỗi khi tính toán giới hạn công suất GPU {gpu_index}: {e}")
            return None

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

    async def get_gpu_power_async(self, pid: Optional[int] = None) -> List[float]:
        """
        Lấy công suất tiêu thụ hiện tại của từng GPU bằng pynvml.
    
        Args:
            pid (Optional[int]): PID của tiến trình (không sử dụng).
    
        Returns:
            List[float]: Danh sách công suất GPU (W) cho từng GPU.
        """
        return await self.get_gpu_power_limits()

    async def reduce_cpu_power(self, pid: Optional[int] = None, reduction_percentage: float = 20.0):
        try:
            if not (0 < reduction_percentage < 100):
                logger.error("PowerManager: Reduction percentage phải nằm trong khoảng (0, 100).")
                return
            if not shutil.which("cpufreq-set"):
                logger.error("PowerManager: Lệnh 'cpufreq-set' không được cài đặt trên hệ thống.")
                return

            loop = asyncio.get_event_loop()
            cpu_freq = await loop.run_in_executor(None, lambda: psutil.cpu_freq().current)  # MHz
            new_freq = max(cpu_freq * (1 - reduction_percentage / 100.0), 1800)  # Đảm bảo không thấp hơn min_freq

            for cpu in range(psutil.cpu_count(logical=True)):
                await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(['cpufreq-set', '-c', str(cpu), '-f', f"{int(new_freq)}MHz"],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                )
            logger.info(f"PowerManager: Đã giảm tần số CPU xuống {int(new_freq)}MHz ({reduction_percentage}% giảm).")
        except Exception as e:
            logger.error(f"PowerManager: Lỗi khi giảm tần số CPU: {e}")

    async def reduce_gpu_power(self, pid: Optional[int] = None, reduction_percentage: float = 20.0):
        if self.gpu_count == 0:
            logger.warning("PowerManager: Không có GPU nào để giảm công suất.")
            return
        if not (0 < reduction_percentage <= 100):
            logger.error(f"PowerManager: reduction_percentage phải trong khoảng (0, 100], nhận: {reduction_percentage}.")
            return
        try:
            for i in range(self.gpu_count):
                desired_power = await self.calculate_desired_power_limit(i, 100 - reduction_percentage)
                if desired_power is not None:
                    success = await self.set_gpu_power_limit(i, desired_power)
                    if success:
                        logger.info(f"PowerManager: Đã giảm công suất GPU {i} xuống {desired_power}W.")
                    else:
                        logger.error(f"PowerManager: Không thể giảm công suất GPU {i}.")
        except Exception as e:
            logger.error(f"PowerManager: Lỗi khi giảm công suất GPU: {e}")

    async def set_gpu_usage(self, usage_percentages: List[float], pid: Optional[int] = None):
        """
        Điều chỉnh mức sử dụng GPU bằng cách thiết lập giới hạn công suất thông qua pynvml.
    
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

                desired_power = await self.calculate_desired_power_limit(i, usage)
                if desired_power is not None:
                    success = await self.set_gpu_power_limit(i, desired_power)
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
        Dừng quản lý năng lượng và giải phóng tài nguyên GPU thông qua pynvml.
        """
        try:
            if self._check_pynvml_initialized():
                pynvml.nvmlShutdown()
                logger.info("PowerManager: Đã shutdown thành công pynvml.")
            else:
                logger.warning("PowerManager: pynvml chưa được khởi tạo hoặc đã bị shutdown.")
        except pynvml.NVMLError as e:
            logger.error(f"PowerManager: Lỗi khi shutdown pynvml: {e}")
        except Exception as e:
            logger.error(f"PowerManager: Lỗi khi shutdown pynvml: {e}")

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
