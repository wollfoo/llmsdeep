import logging
import subprocess
import functools
import asyncio
import psutil
import pynvml
from typing import Any, Dict, Optional
from asyncio import Lock

def async_retry(exception_to_check: Any, tries: int = 4, delay: float = 3.0, backoff: float = 2.0):
    """
    Decorator async để retry một coroutine nếu gặp exception cụ thể.
    """

    def decorator_retry(func):
        @functools.wraps(func)
        async def wrapper_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return await func(*args, **kwargs)
                except exception_to_check as e:
                    logging.getLogger(__name__).warning(
                        f"Lỗi '{e}' xảy ra trong '{func.__name__}'. Thử lại sau {mdelay} giây..."
                    )
                    await asyncio.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return await func(*args, **kwargs)
        return wrapper_retry
    return decorator_retry


class GPUManager:
    """
    Singleton quản lý GPU, sử dụng NVML để lấy/điều chỉnh GPU.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GPUManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True
        self.gpu_initialized = False
        self.logger = logging.getLogger(__name__)
        self.gpu_count = 0

    async def initialize(self) -> bool:
        try:
            await asyncio.to_thread(pynvml.nvmlInit)
            self.gpu_count = await asyncio.to_thread(pynvml.nvmlDeviceGetCount)
            self.gpu_initialized = True
            self.logger.info(f"NVML khởi tạo thành công. Phát hiện {self.gpu_count} GPU.")
            return True
        except pynvml.NVMLError as e:
            self.gpu_initialized = False
            self.logger.warning(f"Không thể khởi tạo NVML: {e}. GPUManager sẽ vô hiệu.")
            return False
        except Exception as e:
            self.gpu_initialized = False
            self.logger.error(f"Lỗi không xác định khi khởi tạo GPUManager: {e}")
            return False

    async def shutdown_nvml(self):
        if self.gpu_initialized:
            try:
                await asyncio.to_thread(pynvml.nvmlShutdown)
                self.logger.info("NVML đã được đóng thành công.")
                self.gpu_initialized = False
            except pynvml.NVMLError as e:
                self.logger.error(f"Lỗi khi đóng NVML: {e}")

    async def get_total_gpu_memory(self) -> float:
        if not self.gpu_initialized:
            return 0.0
        total_memory = 0.0
        try:
            for i in range(self.gpu_count):
                handle = await asyncio.to_thread(pynvml.nvmlDeviceGetHandleByIndex, i)
                mem_info = await asyncio.to_thread(pynvml.nvmlDeviceGetMemoryInfo, handle)
                total_memory += mem_info.total / (1024**2)
            return total_memory
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi khi lấy tổng bộ nhớ GPU: {e}")
            return 0.0

    async def get_used_gpu_memory(self) -> float:
        if not self.gpu_initialized:
            return 0.0
        used_memory = 0.0
        try:
            for i in range(self.gpu_count):
                handle = await asyncio.to_thread(pynvml.nvmlDeviceGetHandleByIndex, i)
                mem_info = await asyncio.to_thread(pynvml.nvmlDeviceGetMemoryInfo, handle)
                used_memory += mem_info.used / (1024**2)
            return used_memory
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi khi lấy bộ nhớ GPU đã sử dụng: {e}")
            return 0.0

    @async_retry(pynvml.NVMLError, tries=3, delay=2, backoff=2)
    async def set_gpu_power_limit(self, gpu_index: int, power_limit_w: int) -> bool:
        try:
            handle = await asyncio.to_thread(pynvml.nvmlDeviceGetHandleByIndex, gpu_index)
            power_limit_mw = power_limit_w * 1000
            await asyncio.to_thread(pynvml.nvmlDeviceSetPowerManagementLimit, handle, power_limit_mw)
            self.logger.info(f"Đặt power limit GPU {gpu_index} = {power_limit_w}W.")
            return True
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi khi đặt power limit GPU {gpu_index}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Lỗi bất ngờ set power limit GPU {gpu_index}: {e}")
            raise

    async def get_gpu_power_limit(self, gpu_index: int) -> Optional[float]:
        if not self.gpu_initialized:
            self.logger.error("GPU chưa init. Không thể lấy power limit.")
            return None
        if gpu_index < 0 or gpu_index >= self.gpu_count:
            self.logger.error(f"GPU index {gpu_index} không hợp lệ, có {self.gpu_count} GPU.")
            return None
        try:
            handle = await asyncio.to_thread(pynvml.nvmlDeviceGetHandleByIndex, gpu_index)
            power_limit_mw = await asyncio.to_thread(pynvml.nvmlDeviceGetPowerManagementLimit, handle)
            power_limit_w = power_limit_mw / 1000
            self.logger.debug(f"GPU {gpu_index} power limit = {power_limit_w}W.")
            return power_limit_w
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi NVML get power limit GPU {gpu_index}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi get power limit GPU {gpu_index}: {e}")
            return None

    async def get_gpu_temperature(self, gpu_index: int) -> Optional[float]:
        if not self.gpu_initialized:
            self.logger.error("Chưa init NVML. Không thể lấy nhiệt độ.")
            return None
        if gpu_index < 0 or gpu_index >= self.gpu_count:
            self.logger.error(f"GPU index {gpu_index} không hợp lệ.")
            return None
        try:
            handle = await asyncio.to_thread(pynvml.nvmlDeviceGetHandleByIndex, gpu_index)
            temperature = await asyncio.to_thread(pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU)
            self.logger.debug(f"Nhiệt độ GPU {gpu_index} = {temperature}°C.")
            return temperature
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi NVML khi lấy nhiệt độ GPU {gpu_index}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy nhiệt độ GPU {gpu_index}: {e}")
            return None

    async def get_gpu_utilization(self, gpu_index: int) -> Optional[Dict[str, float]]:
        if not self.gpu_initialized:
            self.logger.error("Chưa init GPU. Không thể lấy utilization.")
            return None
        if gpu_index < 0 or gpu_index >= self.gpu_count:
            self.logger.error(f"GPU index {gpu_index} không hợp lệ.")
            return None
        try:
            handle = await asyncio.to_thread(pynvml.nvmlDeviceGetHandleByIndex, gpu_index)
            utilization = await asyncio.to_thread(pynvml.nvmlDeviceGetUtilizationRates, handle)
            return {
                'gpu_util_percent': utilization.gpu,
                'memory_util_percent': utilization.memory
            }
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi NVML get utilization GPU {gpu_index}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi khi get utilization GPU {gpu_index}: {e}")
            return None

    async def set_gpu_clocks(self, gpu_index: int, sm_clock: int, mem_clock: int) -> bool:
        if not self.gpu_initialized:
            self.logger.error("Chưa init GPU. Không thể set xung nhịp.")
            return False
        if gpu_index < 0 or gpu_index >= self.gpu_count:
            self.logger.error(f"GPU index {gpu_index} không hợp lệ.")
            return False
        try:
            cmd_sm = ['nvidia-smi', '-i', str(gpu_index), f'--lock-gpu-clocks={sm_clock}']
            await asyncio.to_thread(subprocess.run, cmd_sm, {'check': True})

            cmd_mem = ['nvidia-smi', '-i', str(gpu_index), f'--lock-memory-clocks={mem_clock}']
            await asyncio.to_thread(subprocess.run, cmd_mem, {'check': True})
            self.logger.debug(f"Đặt SM={sm_clock}MHz, MEM={mem_clock}MHz cho GPU={gpu_index}.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi nvidia-smi khi set xung nhịp GPU {gpu_index}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi set xung nhịp GPU {gpu_index}: {e}")
            return False

    async def control_fan_speed(self, gpu_index: int, increase_percentage: float) -> bool:
        try:
            cmd = [
                'nvidia-settings',
                '-a', f'[fan:{gpu_index}]/GPUFanControlState=1',
                '-a', f'[fan:{gpu_index}]/GPUTargetFanSpeed={int(increase_percentage)}'
            ]
            await asyncio.to_thread(subprocess.run, cmd, {'check': True})
            self.logger.debug(f"Tăng fan GPU {gpu_index}={increase_percentage}%.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi điều chỉnh quạt GPU {gpu_index}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi điều chỉnh quạt GPU {gpu_index}: {e}")
            return False

    async def calculate_desired_power_limit(self, gpu_index: int, throttle_percentage: float) -> Optional[int]:
        try:
            current_power_limit = await self.get_gpu_power_limit(gpu_index)
            if current_power_limit is None:
                self.logger.warning(f"Không lấy được power limit GPU {gpu_index}, mặc định=100W.")
                current_power_limit = 100
            desired_limit = int(round(current_power_limit * (1 - throttle_percentage / 100)))
            self.logger.debug(f"Power limit mới GPU={gpu_index}={desired_limit}W (throttle={throttle_percentage}%).")
            return desired_limit
        except Exception as e:
            self.logger.error(f"Lỗi calculate_desired_power_limit GPU={gpu_index}: {e}")
            return None

    async def restore_resources(self, pid: int, gpu_settings: Dict[int, Dict[str, Any]]) -> bool:
        try:
            restored_all = True
            for gpu_index, settings in gpu_settings.items():
                original_power_limit_w = settings.get('power_limit_w')
                if original_power_limit_w is not None:
                    ok_power = await self.set_gpu_power_limit(gpu_index, int(original_power_limit_w))
                    if ok_power:
                        self.logger.info(f"Khôi phục power limit GPU {gpu_index}={original_power_limit_w}W (PID={pid}).")
                    else:
                        self.logger.error(f"Không thể khôi phục power limit GPU {gpu_index} (PID={pid}).")
                        restored_all = False

                original_sm = settings.get('sm_clock_mhz')
                original_mem = settings.get('mem_clock_mhz')
                if original_sm and original_mem:
                    ok_clocks = await self.set_gpu_clocks(gpu_index, int(original_sm), int(original_mem))
                    if ok_clocks:
                        self.logger.info(f"Khôi phục xung nhịp GPU {gpu_index}, SM={original_sm}, MEM={original_mem} (PID={pid}).")
                    else:
                        self.logger.error(f"Không thể khôi phục xung nhịp GPU {gpu_index} (PID={pid}).")
                        restored_all = False

            self.logger.info(f"Khôi phục GPU cho PID={pid} hoàn tất.")
            return restored_all
        except Exception as e:
            self.logger.error(f"Lỗi khôi phục GPU PID={pid}: {e}")
            return False


class MiningProcess:
    """
    Đại diện cho một tiến trình khai thác.
    """
    def __init__(self, pid: int, name: str, priority: int = 1, network_interface: str = 'eth0',
                 logger: Optional[logging.Logger] = None):
        self.pid = pid
        self.name = name
        self.priority = priority
        self.cpu_usage = 0.0
        self.gpu_usage = 0.0
        self.memory_usage = 0.0
        self.disk_io = 0.0
        self.network_io = 0.0
        self.mark = pid % 65535
        self.network_interface = network_interface
        self._prev_bytes_sent: Optional[int] = None
        self._prev_bytes_recv: Optional[int] = None
        self.is_cloaked = False
        self.logger = logger or logging.getLogger(__name__)

        self.gpu_manager = GPUManager()
        self.gpu_initialized = self.gpu_manager.gpu_initialized

    def is_gpu_process(self) -> bool:
        gpu_process_keywords = ['inference-cuda']
        return any(keyword in self.name.lower() for keyword in gpu_process_keywords)

    async def get_gpu_usage(self) -> float:
        if not self.gpu_manager.gpu_initialized:
            return 0.0
        try:
            total_gpu_memory = await self.gpu_manager.get_total_gpu_memory()
            used_gpu_memory = await self.gpu_manager.get_used_gpu_memory()
            if total_gpu_memory > 0:
                return (used_gpu_memory / total_gpu_memory) * 100
            else:
                self.logger.warning("Tổng bộ nhớ GPU không hợp lệ (=0).")
                return 0.0
        except Exception as e:
            self.logger.error(f"Lỗi khi get_gpu_usage: {e}")
            return 0.0

    async def update_resource_usage(self):
        try:
            proc = psutil.Process(self.pid)
            self.cpu_usage = proc.cpu_percent(interval=0.1)
            self.memory_usage = proc.memory_percent()

            io_counters = await asyncio.to_thread(proc.io_counters)
            self.disk_io = max((io_counters.read_bytes + io_counters.write_bytes) / (1024 * 1024), 0.0)

            net_io_all = await asyncio.to_thread(psutil.net_io_counters, True)
            if self.network_interface in net_io_all:
                current_bytes_sent = net_io_all[self.network_interface].bytes_sent
                current_bytes_recv = net_io_all[self.network_interface].bytes_recv

                if self._prev_bytes_sent is not None and self._prev_bytes_recv is not None:
                    sent_diff = max(current_bytes_sent - self._prev_bytes_sent, 0)
                    recv_diff = max(current_bytes_recv - self._prev_bytes_recv, 0)
                    self.network_io = (sent_diff + recv_diff) / (1024 * 1024)
                else:
                    self.network_io = 0.0

                self._prev_bytes_sent = current_bytes_sent
                self._prev_bytes_recv = current_bytes_recv
            else:
                self.logger.warning(
                    f"Giao diện mạng '{self.network_interface}' không tìm thấy cho PID={self.pid}."
                )
                self.network_io = 0.0

            if self.gpu_initialized and self.is_gpu_process():
                self.gpu_usage = await self.get_gpu_usage()
            else:
                self.gpu_usage = 0.0

            self.logger.debug(
                f"[MiningProcess update] {self.name} (PID={self.pid}): "
                f"CPU={self.cpu_usage}%, GPU={self.gpu_usage}%, RAM={self.memory_usage}%, "
                f"DiskIO={self.disk_io}MB, NetIO={self.network_io}MB."
            )
        except psutil.NoSuchProcess:
            self.logger.error(f"Tiến trình {self.name} (PID={self.pid}) không tồn tại.")
            self.cpu_usage = self.memory_usage = self.disk_io = self.network_io = self.gpu_usage = 0.0
        except Exception as e:
            self.logger.error(f"Lỗi update_resource_usage PID={self.pid}: {e}")
            self.cpu_usage = self.memory_usage = self.disk_io = self.network_io = self.gpu_usage = 0.0

    async def reset_network_io(self):
        self._prev_bytes_sent = None
        self._prev_bytes_recv = None
        self.network_io = 0.0

    def to_dict(self) -> Dict[str, Any]:
        try:
            return {
                'pid': self.pid,
                'name': self.name,
                'priority': int(self.priority) if isinstance(self.priority, int) else 1,
                'cpu_usage_percent': float(self.cpu_usage),
                'memory_usage_percent': float(self.memory_usage),
                'gpu_usage_percent': float(self.gpu_usage),
                'disk_io_mb': float(self.disk_io),
                'network_bandwidth_mb': float(self.network_io),
                'mark': self.mark,
                'network_interface': self.network_interface,
                'is_cloaked': self.is_cloaked
            }
        except Exception as e:
            self.logger.error(f"Lỗi to_dict PID={self.pid}: {e}")
            return {}
