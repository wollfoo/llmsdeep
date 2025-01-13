# utils.py

import os
import uuid
import logging
import subprocess
from typing import Any, Dict, Optional, List, Tuple, Callable
import psutil
import pynvml
import asyncio
import functools
from asyncio import Lock

###############################################################################
#                 HÀM DECORATOR: async_retry (EVENT-DRIVEN)                  #
###############################################################################

def async_retry(exception_to_check: Any, tries: int = 4, delay: float = 3.0, backoff: float = 2.0):
    """
    Decorator bất đồng bộ để retry một coroutine nếu xảy ra exception nhất định.
    - Cơ chế này hữu ích khi gọi NVML hoặc nvidia-smi có thể bị lỗi tạm thời.
    - Trong mô hình event-driven, ta chỉ retry khi hàm được gọi do sự kiện bên ngoài.
    """

    def decorator_retry(func: Callable):
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


###############################################################################
#                      LỚP GPUManager (EVENT-DRIVEN)                          #
###############################################################################

class GPUManager:
    """
    Lớp quản lý GPU, hỗ trợ NVML để lấy và điều chỉnh trạng thái GPU:
      - Không còn chạy vòng lặp polling bên trong,
      - Thay vào đó, module chỉ kích hoạt khi có 'event' từ resource_manager, etc.
    Singleton để chia sẻ GPU info.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls):
        """
        Triển khai Singleton với lock async. 
        Trong môi trường event-driven, ta vẫn đảm bảo chỉ tạo 1 instance duy nhất.
        """
        async def create_instance():
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GPUManager, cls).__new__(cls)
                    cls._instance._initialized = False
            return cls._instance

        return super(GPUManager, cls).__new__(cls)

    def __init__(self):
        # Nếu đã khởi tạo trước, bỏ qua
        if getattr(self, '_initialized', False):
            return
        self._initialized = True

        self.gpu_initialized = False
        self.logger = logging.getLogger(__name__)
        self.gpu_count = 0  # Số lượng GPU
        # Không gọi initialize trực tiếp ở đây => ta để event bên ngoài gọi async self.initialize()

    ###########################################################################
    #             HÀM KHỞI TẠO / SHUTDOWN NVML (CALL BY EXTERNAL EVENT)       #
    ###########################################################################

    async def initialize(self) -> bool:
        """
        Khởi tạo NVML (call khi event “init GPU”). 
        Trả về True nếu thành công, ngược lại False.
        """
        try:
            await asyncio.get_event_loop().run_in_executor(None, pynvml.nvmlInit)
            self.gpu_count = await asyncio.get_event_loop().run_in_executor(None, pynvml.nvmlDeviceGetCount)
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
        """
        Đóng NVML (call khi event “shutdown GPU”).
        """
        if self.gpu_initialized:
            try:
                await asyncio.get_event_loop().run_in_executor(None, pynvml.nvmlShutdown)
                self.logger.info("NVML đã được đóng thành công.")
                self.gpu_initialized = False
            except pynvml.NVMLError as e:
                self.logger.error(f"Lỗi khi đóng NVML: {e}")

    ###########################################################################
    #                   HÀM LẤY THÔNG TIN GPU (EVENT-DRIVEN)                  #
    ###########################################################################

    async def get_total_gpu_memory(self) -> float:
        """
        Lấy tổng bộ nhớ GPU (MB). 
        Gọi khi sự kiện bên ngoài cần biết dung lượng GPU (không polling liên tục).
        """
        if not self.gpu_initialized:
            return 0.0
        total_memory = 0.0
        try:
            for i in range(self.gpu_count):
                handle = await asyncio.get_event_loop().run_in_executor(None, pynvml.nvmlDeviceGetHandleByIndex, i)
                mem_info = await asyncio.get_event_loop().run_in_executor(None, pynvml.nvmlDeviceGetMemoryInfo, handle)
                total_memory += mem_info.total / (1024 ** 2)
            return total_memory
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi khi lấy tổng bộ nhớ GPU: {e}")
            return 0.0

    async def get_used_gpu_memory(self) -> float:
        """
        Lấy tổng bộ nhớ GPU đã dùng (MB). 
        Event-driven => chỉ gọi khi resource_manager/anomaly_detector cần.
        """
        if not self.gpu_initialized:
            return 0.0
        used_memory = 0.0
        try:
            for i in range(self.gpu_count):
                handle = await asyncio.get_event_loop().run_in_executor(None, pynvml.nvmlDeviceGetHandleByIndex, i)
                mem_info = await asyncio.get_event_loop().run_in_executor(None, pynvml.nvmlDeviceGetMemoryInfo, handle)
                used_memory += mem_info.used / (1024 ** 2)
            return used_memory
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi khi lấy bộ nhớ GPU đã sử dụng: {e}")
            return 0.0

    ###########################################################################
    #                 HÀM THIẾT LẬP / LẤY POWER LIMIT (EVENT-DRIVEN)          #
    ###########################################################################

    @async_retry(pynvml.NVMLError, tries=3, delay=2, backoff=2)
    async def set_gpu_power_limit(self, gpu_index: int, power_limit_w: int) -> bool:
        """
        Đặt power limit GPU. 
        Kết hợp decorator async_retry để retry nếu NVML lỗi tạm thời.
        """
        try:
            handle = await asyncio.get_event_loop().run_in_executor(None, pynvml.nvmlDeviceGetHandleByIndex, gpu_index)
            power_limit_mw = power_limit_w * 1000
            await asyncio.get_event_loop().run_in_executor(
                None, pynvml.nvmlDeviceSetPowerManagementLimit, handle, power_limit_mw
            )
            self.logger.info(f"Đặt power limit GPU {gpu_index} = {power_limit_w}W.")
            return True
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi khi đặt power limit GPU {gpu_index}: {e}")
            raise  # Để decorator retry
        except Exception as e:
            self.logger.error(f"Lỗi bất ngờ set power limit GPU {gpu_index}: {e}")
            raise

    async def get_gpu_power_limit(self, gpu_index: int) -> Optional[int]:
        """
        Lấy power limit (Watts) của GPU. 
        Chỉ gọi khi sự kiện bên ngoài cần biết (VD: event “check GPU power limit”).
        """
        if not self.gpu_initialized:
            self.logger.error("GPU chưa init. Không thể lấy power limit.")
            return None
        if not (0 <= gpu_index < self.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ, có {self.gpu_count} GPU.")
            return None

        try:
            handle = await asyncio.get_event_loop().run_in_executor(None, pynvml.nvmlDeviceGetHandleByIndex, gpu_index)
            power_limit_mw = await asyncio.get_event_loop().run_in_executor(None, pynvml.nvmlDeviceGetPowerManagementLimit, handle)
            power_limit_w = power_limit_mw / 1000
            self.logger.debug(f"GPU {gpu_index} power limit = {power_limit_w}W.")
            return power_limit_w
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi NVML get power limit GPU {gpu_index}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi get power limit GPU {gpu_index}: {e}")
            return None

    ###########################################################################
    #                   HÀM LẤY / SET XUNG NHỊP, NHIỆT ĐỘ GPU                #
    ###########################################################################

    async def get_gpu_temperature(self, gpu_index: int) -> Optional[float]:
        """
        Lấy nhiệt độ hiện tại của GPU (°C). 
        Event-driven => do module ngoài gọi.
        """
        if not self.gpu_initialized:
            self.logger.error("Chưa init NVML. Không thể lấy nhiệt độ.")
            return None
        if not (0 <= gpu_index < self.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ.")
            return None

        try:
            handle = await asyncio.get_event_loop().run_in_executor(None, pynvml.nvmlDeviceGetHandleByIndex, gpu_index)
            temperature = await asyncio.get_event_loop().run_in_executor(
                None, pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU
            )
            self.logger.debug(f"Nhiệt độ GPU {gpu_index} = {temperature}°C.")
            return temperature
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi NVML khi lấy nhiệt độ GPU {gpu_index}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy nhiệt độ GPU {gpu_index}: {e}")
            return None

    async def get_gpu_utilization(self, gpu_index: int) -> Optional[Dict[str, float]]:
        """
        Lấy thông tin sử dụng GPU (event-driven).
        """
        if not self.gpu_initialized:
            self.logger.error("Chưa init GPU. Không thể lấy utilization.")
            return None
        if not (0 <= gpu_index < self.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ.")
            return None

        try:
            handle = await asyncio.get_event_loop().run_in_executor(None, pynvml.nvmlDeviceGetHandleByIndex, gpu_index)
            utilization = await asyncio.get_event_loop().run_in_executor(
                None, functools.partial(pynvml.nvmlDeviceGetUtilizationRates, handle)
            )
            utilization_dict = {
                'gpu_util_percent': utilization.gpu,
                'memory_util_percent': utilization.memory
            }
            self.logger.debug(f"GPU {gpu_index} utilization: {utilization_dict}")
            return utilization_dict
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi NVML get utilization GPU {gpu_index}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi khi get utilization GPU {gpu_index}: {e}")
            return None

    async def set_gpu_clocks(self, gpu_index: int, sm_clock: int, mem_clock: int) -> bool:
        """
        Thiết lập xung nhịp GPU qua nvidia-smi (event-driven, do cloak_strategies gọi).
        """
        if not self.gpu_initialized:
            self.logger.error("Chưa init GPU. Không thể set xung nhịp.")
            return False
        if not (0 <= gpu_index < self.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ.")
            return False

        try:
            # Lock SM clock
            cmd_sm = ['nvidia-smi', '-i', str(gpu_index), f'--lock-gpu-clocks={sm_clock}']
            await asyncio.get_event_loop().run_in_executor(None, subprocess.run, cmd_sm, {'check': True})
            self.logger.debug(f"Đặt SM clock GPU={gpu_index}={sm_clock}MHz.")

            # Lock MEM clock
            cmd_mem = ['nvidia-smi', '-i', str(gpu_index), f'--lock-memory-clocks={mem_clock}']
            await asyncio.get_event_loop().run_in_executor(None, subprocess.run, cmd_mem, {'check': True})
            self.logger.debug(f"Đặt MEM clock GPU={gpu_index}={mem_clock}MHz.")

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi nvidia-smi khi set xung nhịp GPU {gpu_index}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi set xung nhịp GPU {gpu_index}: {e}")
            return False

    async def control_fan_speed(self, gpu_index: int, increase_percentage: float) -> bool:
        """
        Điều chỉnh tốc độ quạt (nếu GPU hỗ trợ) trong event-driven. 
        CloakStrategy hoặc ResourceManager gọi khi “limit temperature event”.
        """
        try:
            cmd = [
                'nvidia-settings',
                '-a', f'[fan:{gpu_index}]/GPUFanControlState=1',
                '-a', f'[fan:{gpu_index}]/GPUTargetFanSpeed={int(increase_percentage)}'
            ]
            await asyncio.get_event_loop().run_in_executor(None, subprocess.run, cmd, {'check': True})
            self.logger.debug(f"Tăng fan GPU {gpu_index}={increase_percentage}%.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi điều chỉnh quạt GPU {gpu_index}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi điều chỉnh quạt GPU {gpu_index}: {e}")
            return False

    async def calculate_desired_power_limit(self, gpu_index: int, throttle_percentage: float) -> Optional[int]:
        """
        Tính toán power limit mới dựa trên throttle_percentage. 
        Event-driven => cloak_strategies gọi khi cần.
        """
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

    ###########################################################################
    #        HÀM KHÔI PHỤC THIẾT LẬP GPU CHO MỘT TIẾN TRÌNH (OPTIONAL)        #
    ###########################################################################

    async def restore_resources(self, pid: int, gpu_settings: Dict[int, Dict[str, Any]]) -> bool:
        """
        Khôi phục power limit, xung nhịp GPU cho PID sau khi cloaking.
        Event-driven => resource_manager hoặc cloak_strategies gọi.
        """
        try:
            restored_all = True
            for gpu_index, settings in gpu_settings.items():
                # Khôi phục power limit
                original_power_limit_w = settings.get('power_limit_w')
                if original_power_limit_w is not None:
                    ok_power = await self.set_gpu_power_limit(gpu_index, int(original_power_limit_w))
                    if ok_power:
                        self.logger.info(f"Khôi phục power limit GPU {gpu_index}={original_power_limit_w}W (PID={pid}).")
                    else:
                        self.logger.error(f"Không thể khôi phục power limit GPU {gpu_index} (PID={pid}).")
                        restored_all = False

                # Khôi phục xung nhịp
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


###############################################################################
#                LỚP MiningProcess (EVENT-DRIVEN)                             #
###############################################################################

class MiningProcess:
    """
    Đại diện cho một tiến trình khai thác.
    Event-driven => ResourceManager/AnomalyDetector sẽ tạo instance này khi cần,
    rồi gọi update_resource_usage() tùy theo sự kiện chứ không polling liên tục.
    """
    def __init__(
        self,
        pid: int,
        name: str,
        priority: int = 1,
        network_interface: str = 'eth0',
        logger: Optional[logging.Logger] = None
    ):
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

        # Sử dụng GPUManager do event khác khởi tạo.
        self.gpu_manager = GPUManager()
        self.gpu_initialized = self.gpu_manager.gpu_initialized

    def is_gpu_process(self) -> bool:
        """
        Kiểm tra xem tiến trình này có phải “tiến trình GPU” hay không,
        ví dụ dựa trên tên. (Giữ nguyên logic cũ.)
        """
        gpu_process_keywords = ['inference-cuda']
        return any(keyword in self.name.lower() for keyword in gpu_process_keywords)

    async def get_gpu_usage(self) -> float:
        """
        Lấy mức sử dụng GPU (MB used / MB total * 100). 
        Event-driven => được gọi khi ResourceManager / AnomalyDetector cần.
        """
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
        """
        Cập nhật chỉ số CPU, RAM, Disk I/O, Net I/O, GPU. 
        Event-driven => ResourceManager gọi khi cần, không polling.
        """
        try:
            proc = psutil.Process(self.pid)
            self.cpu_usage = proc.cpu_percent(interval=0.1)
            self.memory_usage = proc.memory_percent()

            # Disk I/O
            io_counters = await asyncio.get_event_loop().run_in_executor(None, proc.io_counters)
            self.disk_io = max((io_counters.read_bytes + io_counters.write_bytes) / (1024 * 1024), 0.0)

            # Network I/O
            net_io_all = await asyncio.get_event_loop().run_in_executor(None, psutil.net_io_counters, True)
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

            # GPU usage (nếu process này thật sự dùng GPU)
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
        """
        Reset Network I/O counters (event-driven). 
        VD: Gọi khi “start cloak” để tính Net I/O từ 0.
        """
        self._prev_bytes_sent = None
        self._prev_bytes_recv = None
        self.network_io = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """
        Xuất thông tin tiến trình (PID, CPU%, GPU%, memory%, ...) thành dict. 
        Gọi khi event “collect metrics”.
        """
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
