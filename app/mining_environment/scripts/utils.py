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

def async_retry(exception_to_check: Any, tries: int = 4, delay: float = 3.0, backoff: float = 2.0):
    """
    Async decorator to retry a coroutine function if specified exceptions occur.

    Args:
        exception_to_check (Exception or tuple): The exception(s) to check.
        tries (int): Number of attempts. Must be at least 1.
        delay (float): Initial delay between retries in seconds.
        backoff (float): Backoff multiplier for delay between retries.

    Returns:
        Callable: The decorated coroutine.
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


class GPUManager:
    """
    Lớp quản lý GPU, bao gồm việc khởi tạo NVML và thu thập thông tin GPU.
    Singleton pattern để chia sẻ GPU info toàn cục.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls):
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
        self.gpu_count = 0  # Đảm bảo có thuộc tính gpu_count
        # Không gọi initialize tại đây vì chúng ta sẽ gọi nó từ bên ngoài một cách bất đồng bộ

    async def initialize(self) -> bool:
        """
        Khởi tạo NVML để quản lý GPU.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            await asyncio.get_event_loop().run_in_executor(None, pynvml.nvmlInit)
            self.gpu_count = await asyncio.get_event_loop().run_in_executor(
                None, pynvml.nvmlDeviceGetCount
            )
            self.gpu_initialized = True
            self.logger.info(
                f"NVML đã được khởi tạo thành công. Đã phát hiện {self.gpu_count} GPU."
            )
            return True
        except pynvml.NVMLError as e:
            self.gpu_initialized = False
            self.logger.warning(f"Không thể khởi tạo NVML: {e}. Chức năng quản lý GPU sẽ bị vô hiệu hóa.")
            return False
        except Exception as e:
            self.gpu_initialized = False
            self.logger.error(f"Lỗi không xác định khi khởi tạo GPUManager: {e}")
            return False

    async def shutdown_nvml(self):
        """
        Đóng NVML.
        """
        if self.gpu_initialized:
            try:
                await asyncio.get_event_loop().run_in_executor(None, pynvml.nvmlShutdown)
                self.logger.info("NVML đã được đóng thành công.")
                self.gpu_initialized = False
            except pynvml.NVMLError as e:
                self.logger.error(f"Lỗi khi đóng NVML: {e}")

    async def get_total_gpu_memory(self) -> float:
        """
        Lấy tổng bộ nhớ GPU (MB).

        Returns:
            float: Tổng bộ nhớ GPU tính bằng MB.
        """
        if not self.gpu_initialized:
            return 0.0
        total_memory = 0.0
        try:
            for i in range(self.gpu_count):
                handle = await asyncio.get_event_loop().run_in_executor(
                    None, pynvml.nvmlDeviceGetHandleByIndex, i
                )
                mem_info = await asyncio.get_event_loop().run_in_executor(
                    None, pynvml.nvmlDeviceGetMemoryInfo, handle
                )
                total_memory += mem_info.total / (1024 ** 2)  # Chuyển sang MB
            return total_memory
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi khi lấy tổng bộ nhớ GPU: {e}")
            return 0.0

    async def get_used_gpu_memory(self) -> float:
        """
        Lấy bộ nhớ GPU đã sử dụng (MB).

        Returns:
            float: Bộ nhớ GPU đã sử dụng tính bằng MB.
        """
        if not self.gpu_initialized:
            return 0.0
        used_memory = 0.0
        try:
            for i in range(self.gpu_count):
                handle = await asyncio.get_event_loop().run_in_executor(
                    None, pynvml.nvmlDeviceGetHandleByIndex, i
                )
                mem_info = await asyncio.get_event_loop().run_in_executor(
                    None, pynvml.nvmlDeviceGetMemoryInfo, handle
                )
                used_memory += mem_info.used / (1024 ** 2)  # Chuyển sang MB
            return used_memory
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi khi lấy bộ nhớ GPU đã sử dụng: {e}")
            return 0.0

    @async_retry(pynvml.NVMLError, tries=3, delay=2, backoff=2)
    async def set_gpu_power_limit(self, gpu_index: int, power_limit_w: int) -> bool:
        """
        Đặt power limit cho GPU cụ thể.

        Args:
            gpu_index (int): Chỉ số GPU.
            power_limit_w (int): Power limit mới (W).

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            handle = await asyncio.get_event_loop().run_in_executor(
                None, pynvml.nvmlDeviceGetHandleByIndex, gpu_index
            )
            power_limit_mw = power_limit_w * 1000
            await asyncio.get_event_loop().run_in_executor(
                None, pynvml.nvmlDeviceSetPowerManagementLimit, handle, power_limit_mw
            )
            self.logger.info(f"Đặt power limit GPU {gpu_index} thành {power_limit_w}W.")
            return True
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi khi đặt power limit GPU {gpu_index}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Lỗi bất ngờ khi đặt power limit GPU {gpu_index}: {e}")
            raise

    async def get_gpu_power_limit(self, gpu_index: int) -> Optional[int]:
        """
        Lấy giới hạn power hiện tại của GPU.

        Args:
            gpu_index (int): Chỉ số GPU (bắt đầu từ 0).

        Returns:
            Optional[int]: Giới hạn power hiện tại tính bằng Watts hoặc None nếu lỗi.
        """
        if not self.gpu_initialized:
            self.logger.error("GPU chưa được khởi tạo hoặc không có GPU nào trên hệ thống. Không thể lấy power limit.")
            return None

        if not (0 <= gpu_index < self.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ. Có tổng cộng {self.gpu_count} GPU.")
            return None

        try:
            handle = await asyncio.get_event_loop().run_in_executor(
                None, pynvml.nvmlDeviceGetHandleByIndex, gpu_index
            )
            power_limit_mw = await asyncio.get_event_loop().run_in_executor(
                None, pynvml.nvmlDeviceGetPowerManagementLimit, handle
            )
            power_limit_w = power_limit_mw / 1000  # Chuyển từ milliWatts sang Watts
            self.logger.debug(f"Giới hạn power cho GPU {gpu_index} là {power_limit_w}W.")
            return power_limit_w
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi NVML khi lấy power limit cho GPU {gpu_index}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi lấy power limit cho GPU {gpu_index}: {e}")
            return None

    async def get_gpu_temperature(self, gpu_index: int) -> Optional[float]:
        """
        Lấy nhiệt độ hiện tại của GPU.

        Args:
            gpu_index (int): Chỉ số GPU (bắt đầu từ 0).

        Returns:
            Optional[float]: Nhiệt độ GPU tính bằng °C hoặc None nếu lỗi.
        """
        if not self.gpu_initialized:
            self.logger.error("GPU chưa được khởi tạo hoặc không có GPU nào trên hệ thống. Không thể lấy nhiệt độ.")
            return None

        if not (0 <= gpu_index < self.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ. Có tổng cộng {self.gpu_count} GPU.")
            return None

        try:
            handle = await asyncio.get_event_loop().run_in_executor(
                None, pynvml.nvmlDeviceGetHandleByIndex, gpu_index
            )
            temperature = await asyncio.get_event_loop().run_in_executor(
                None, pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU
            )
            self.logger.debug(f"Nhiệt độ GPU {gpu_index} là {temperature}°C.")
            return temperature
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi NVML khi lấy nhiệt độ cho GPU {gpu_index}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi lấy nhiệt độ cho GPU {gpu_index}: {e}")
            return None

    async def get_gpu_utilization(self, gpu_index: int) -> Optional[Dict[str, float]]:
        """
        Lấy thông tin sử dụng GPU hiện tại.

        Args:
            gpu_index (int): Chỉ số GPU (bắt đầu từ 0).

        Returns:
            Optional[Dict[str, float]]: Dictionary chứa thông tin sử dụng GPU hoặc None nếu lỗi.
        """
        if not self.gpu_initialized:
            self.logger.error("GPU chưa được khởi tạo hoặc không có GPU nào trên hệ thống. Không thể lấy thông tin sử dụng GPU.")
            return None

        if not (0 <= gpu_index < self.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ. Có tổng cộng {self.gpu_count} GPU.")
            return None

        try:
            handle = await asyncio.get_event_loop().run_in_executor(
                None, pynvml.nvmlDeviceGetHandleByIndex, gpu_index
            )
            utilization = await asyncio.get_event_loop().run_in_executor(
                None, functools.partial(pynvml.nvmlDeviceGetUtilizationRates, handle)
            )
            utilization_dict = {
                'gpu_util_percent': utilization.gpu,
                'memory_util_percent': utilization.memory
            }
            self.logger.debug(f"Sử dụng GPU {gpu_index}: {utilization_dict}")
            return utilization_dict
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi NVML khi lấy sử dụng GPU cho GPU {gpu_index}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi lấy sử dụng GPU cho GPU {gpu_index}: {e}")
            return None

    async def set_gpu_clocks(self, gpu_index: int, sm_clock: int, mem_clock: int) -> bool:
        """
        Thiết lập xung nhịp GPU thông qua nvidia-smi.

        Args:
            gpu_index (int): Chỉ số GPU (bắt đầu từ 0).
            sm_clock (int): Xung nhịp SM GPU tính bằng MHz.
            mem_clock (int): Xung nhịp bộ nhớ GPU tính bằng MHz.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        if not self.gpu_initialized:
            self.logger.error("GPU chưa được khởi tạo hoặc không có GPU nào trên hệ thống. Không thể đặt xung nhịp.")
            return False

        if not (0 <= gpu_index < self.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ. Có tổng cộng {self.gpu_count} GPU.")
            return False

        if mem_clock <= 0 or sm_clock <= 0:
            self.logger.error("Giá trị xung nhịp phải lớn hơn 0.")
            return False

        try:
            # Thiết lập xung nhịp SM
            cmd_sm = [
                'nvidia-smi',
                '-i', str(gpu_index),
                '--lock-gpu-clocks=' + str(sm_clock)
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, subprocess.run, cmd_sm, {'check': True}
            )
            self.logger.debug(f"Đã thiết lập xung nhịp SM cho GPU {gpu_index} là {sm_clock}MHz.")

            # Thiết lập xung nhịp bộ nhớ
            cmd_mem = [
                'nvidia-smi',
                '-i', str(gpu_index),
                '--lock-memory-clocks=' + str(mem_clock)
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, subprocess.run, cmd_mem, {'check': True}
            )
            self.logger.debug(f"Đã thiết lập xung nhịp bộ nhớ cho GPU {gpu_index} là {mem_clock}MHz.")

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi thiết lập xung nhịp GPU {gpu_index} bằng nvidia-smi: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi thiết lập xung nhịp GPU {gpu_index}: {e}")
            return False

    async def control_fan_speed(self, gpu_index: int, increase_percentage: float) -> bool:
        """
        Điều chỉnh tốc độ quạt GPU (nếu hỗ trợ).

        Args:
            gpu_index (int): Chỉ số GPU.
            increase_percentage (float): Tỷ lệ tăng tốc độ quạt (%).

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            # Sử dụng nvidia-settings để điều chỉnh tốc độ quạt
            cmd = [
                'nvidia-settings',
                '-a', f'[fan:{gpu_index}]/GPUFanControlState=1',
                '-a', f'[fan:{gpu_index}]/GPUTargetFanSpeed={int(increase_percentage)}'
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, subprocess.run, cmd, {'check': True}
            )
            self.logger.debug(f"Đã tăng tốc độ quạt GPU {gpu_index} lên {increase_percentage}%.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi điều chỉnh tốc độ quạt GPU {gpu_index}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi điều chỉnh tốc độ quạt GPU {gpu_index}: {e}")
            return False

    async def calculate_desired_power_limit(self, gpu_index: int, throttle_percentage: float) -> Optional[int]:
        """
        Tính toán power limit mới dựa trên throttle_percentage.

        Args:
            gpu_index (int): Chỉ số GPU.
            throttle_percentage (float): Phần trăm throttle (0-100).

        Returns:
            Optional[int]: Power limit mới tính bằng Watts hoặc None nếu lỗi.
        """
        try:
            current_power_limit = await self.get_gpu_power_limit(gpu_index)
            if current_power_limit is None:
                self.logger.warning(f"Không thể lấy power limit hiện tại cho GPU {gpu_index}. Sử dụng giá trị mặc định.")
                current_power_limit = 100  # Giá trị mặc định nếu không thể lấy

            desired_power_limit_w = int(round(current_power_limit * (1 - throttle_percentage / 100)))
            self.logger.debug(f"Tính toán power limit mới cho GPU {gpu_index}: {desired_power_limit_w}W.")
            return desired_power_limit_w
        except Exception as e:
            self.logger.error(f"Lỗi khi tính toán power limit mới cho GPU {gpu_index}: {e}")
            return None

    async def get_current_sm_clock(self, handle: Any) -> Optional[int]:
        """
        Lấy xung nhịp SM hiện tại của GPU.

        Args:
            handle (Any): Handle GPU từ NVML.

        Returns:
            Optional[int]: Xung nhịp SM hiện tại tính bằng MHz hoặc None nếu lỗi.
        """
        try:
            sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            return sm_clock
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi NVML khi lấy xung nhịp SM: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi lấy xung nhịp SM: {e}")
            return None

    async def get_current_mem_clock(self, handle: Any) -> Optional[int]:
        """
        Lấy xung nhịp bộ nhớ hiện tại của GPU.

        Args:
            handle (Any): Handle GPU từ NVML.

        Returns:
            Optional[int]: Xung nhịp bộ nhớ hiện tại tính bằng MHz hoặc None nếu lỗi.
        """
        try:
            mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            return mem_clock
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi NVML khi lấy xung nhịp MEM: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi lấy xung nhịp MEM: {e}")
            return None

    async def restore_resources(self, pid: int, gpu_settings: Dict[int, Dict[str, Any]]) -> bool:
        """
        Khôi phục các thiết lập GPU cho tiến trình bằng cách đặt lại power limit và xung nhịp cho tất cả các GPU mà tiến trình đang sử dụng.

        Args:
            pid (int): PID của tiến trình.
            gpu_settings (Dict[int, Dict[str, Any]]): Thiết lập GPU ban đầu cho mỗi GPU index.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            restored_all = True

            for gpu_index, settings in gpu_settings.items():
                # Khôi phục power limit nếu có
                original_power_limit_w = settings.get('power_limit_w')
                if original_power_limit_w is not None:
                    success_power = await self.set_gpu_power_limit(gpu_index, int(original_power_limit_w))
                    if success_power:
                        self.logger.info(f"Đã khôi phục power limit GPU {gpu_index} về {original_power_limit_w}W cho PID={pid}.")
                    else:
                        self.logger.error(f"Không thể khôi phục power limit GPU {gpu_index} cho PID={pid}.")
                        restored_all = False

                # Khôi phục xung nhịp nếu có
                original_sm_clock = settings.get('sm_clock_mhz')
                original_mem_clock = settings.get('mem_clock_mhz')
                if original_sm_clock and original_mem_clock:
                    success_clocks = await self.set_gpu_clocks(
                        gpu_index, int(original_sm_clock), int(original_mem_clock)
                    )
                    if success_clocks:
                        self.logger.info(
                            f"Đã khôi phục xung nhịp GPU {gpu_index} về SM={original_sm_clock}MHz, MEM={original_mem_clock}MHz cho PID={pid}."
                        )
                    else:
                        self.logger.error(f"Không thể khôi phục xung nhịp GPU {gpu_index} cho PID={pid}.")
                        restored_all = False

            self.logger.info(f"Đã khôi phục tất cả các thiết lập GPU cho PID={pid}.")
            return restored_all
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục GPU settings cho PID={pid}: {e}")
            return False


class MiningProcess:
    """
    Đại diện cho một tiến trình khai thác với các chỉ số sử dụng tài nguyên.
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
        self.priority = priority  # Giá trị ưu tiên (1 là thấp nhất)
        self.cpu_usage = 0.0      # Theo phần trăm
        self.gpu_usage = 0.0      # Theo phần trăm
        self.memory_usage = 0.0   # Theo phần trăm
        self.disk_io = 0.0        # MB
        self.network_io = 0.0     # MB kể từ lần cập nhật cuối cùng
        self.mark = pid % 65535   # Dấu hiệu duy nhất cho mạng, giới hạn 16 bit
        self.network_interface = network_interface
        self._prev_bytes_sent: Optional[int] = None
        self._prev_bytes_recv: Optional[int] = None
        self.is_cloaked = False
        self.logger = logger or logging.getLogger(__name__)

        # Sử dụng GPUManager để kiểm tra GPU
        self.gpu_manager = GPUManager()
        self.gpu_initialized = self.gpu_manager.gpu_initialized

    async def get_gpu_usage(self) -> float:
        """
        Lấy mức sử dụng GPU.

        Returns:
            float: Mức sử dụng GPU theo phần trăm.
        """
        if not self.gpu_manager.gpu_initialized:
            return 0.0
        try:
            total_gpu_memory = await self.gpu_manager.get_total_gpu_memory()
            used_gpu_memory = await self.gpu_manager.get_used_gpu_memory()
            if total_gpu_memory > 0:
                gpu_usage_percent = (used_gpu_memory / total_gpu_memory) * 100
                return gpu_usage_percent
            else:
                self.logger.warning("Tổng bộ nhớ GPU không hợp lệ.")
                return 0.0
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy mức sử dụng GPU: {e}")
            return 0.0

    def is_gpu_process(self) -> bool:
        """
        Xác định xem tiến trình có sử dụng GPU hay không.

        Returns:
            bool: True nếu tiến trình sử dụng GPU, False ngược lại.
        """
        gpu_process_keywords = ['llmsengen', 'gpu_miner']
        return any(keyword in self.name.lower() for keyword in gpu_process_keywords)

    async def update_resource_usage(self):
        """
        Cập nhật các chỉ số sử dụng tài nguyên của tiến trình khai thác.
        """
        try:
            proc = psutil.Process(self.pid)
            self.cpu_usage = proc.cpu_percent(interval=0.1)
            self.memory_usage = proc.memory_percent()

            # Disk I/O
            io_counters = await asyncio.get_event_loop().run_in_executor(None, proc.io_counters)
            self.disk_io = max(
                (io_counters.read_bytes + io_counters.write_bytes) / (1024 * 1024), 0.0
            )

            # Network I/O
            net_io = await asyncio.get_event_loop().run_in_executor(
                None, psutil.net_io_counters, True
            )
            if self.network_interface in net_io:
                current_bytes_sent = net_io[self.network_interface].bytes_sent
                current_bytes_recv = net_io[self.network_interface].bytes_recv

                if self._prev_bytes_sent is not None and self._prev_bytes_recv is not None:
                    sent_diff = max(current_bytes_sent - self._prev_bytes_sent, 0)
                    recv_diff = max(current_bytes_recv - self._prev_bytes_recv, 0)
                    self.network_io = (sent_diff + recv_diff) / (1024 * 1024)  # MB
                else:
                    self.network_io = 0.0

                self._prev_bytes_sent = current_bytes_sent
                self._prev_bytes_recv = current_bytes_recv
            else:
                self.logger.warning(
                    f"Giao diện mạng '{self.network_interface}' không tìm thấy cho tiến trình {self.name} (PID: {self.pid})."
                )
                self.network_io = 0.0

            # GPU Usage
            if self.gpu_initialized and self.is_gpu_process():
                self.gpu_usage = max(await self.get_gpu_usage(), 0.0)
            else:
                self.gpu_usage = 0.0

            self.logger.debug(
                f"Cập nhật usage cho {self.name} (PID: {self.pid}): "
                f"CPU={self.cpu_usage}%, GPU={self.gpu_usage}%, RAM={self.memory_usage}%, "
                f"Disk I/O={self.disk_io}MB, Net I/O={self.network_io}MB."
            )

        except psutil.NoSuchProcess:
            self.logger.error(f"Tiến trình {self.name} (PID: {self.pid}) không tồn tại.")
            self.cpu_usage = self.memory_usage = self.disk_io = self.network_io = self.gpu_usage = 0.0
        except Exception as e:
            self.logger.error(f"Lỗi khi cập nhật usage cho {self.name} (PID: {self.pid}): {e}")
            self.cpu_usage = self.memory_usage = self.disk_io = self.network_io = self.gpu_usage = 0.0

    async def reset_network_io(self):
        """
        Reset các biến liên quan đến Network I/O.
        """
        self._prev_bytes_sent = None
        self._prev_bytes_recv = None
        self.network_io = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi các thuộc tính của MiningProcess thành dictionary.

        Returns:
            Dict[str, Any]: Dictionary chứa các thông tin của tiến trình.
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
            self.logger.error(f"Lỗi khi chuyển đổi {self.name} (PID: {self.pid}) sang dictionary: {e}")
            return {}
