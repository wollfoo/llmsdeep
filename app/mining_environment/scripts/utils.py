# utils.py

import psutil
import functools
import logging
from time import sleep
from typing import Any, Dict, Optional, List
import pynvml
from threading import Lock

def retry(ExceptionToCheck, tries=4, delay=3, backoff=2):
    """
    Decorator để tự động thử lại một hàm nếu xảy ra ngoại lệ.

    :param ExceptionToCheck: Ngoại lệ cần kiểm tra (có thể là tuple các ngoại lệ).
    :param tries: Số lần thử (tính cả lần đầu) trước khi bỏ cuộc.
    :param delay: Thời gian trì hoãn giữa các lần thử (giây).
    :param backoff: Hệ số nhân áp dụng cho thời gian trì hoãn giữa các lần thử.
    """
    def deco_retry(f):
        @functools.wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToCheck as e:
                    logging.getLogger(__name__).warning(
                        f"Lỗi '{e}' xảy ra trong '{f.__name__}'. Thử lại sau {mdelay} giây..."
                    )
                    sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return f_retry
    return deco_retry


class GPUManager:
    """
    Lớp quản lý GPU, bao gồm việc khởi tạo NVML và thu thập thông tin GPU.
    Singleton pattern để chia sẻ GPU info toàn cục.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GPUManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # Nếu đã khởi tạo trước, bỏ qua
        if self._initialized:
            return
        self._initialized = True

        self.gpu_initialized = False
        self.logger = logging.getLogger(__name__)
        self.gpu_count = 0  # [CHANGES] Đảm bảo có thuộc tính gpu_count
        self.initialize_nvml()

    def initialize_nvml(self):
        """Khởi tạo NVML để quản lý GPU."""
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            self.gpu_initialized = True
            self.logger.info(
                "NVML đã được khởi tạo thành công. Đã phát hiện {0} GPU.".format(self.gpu_count)
            )
        except pynvml.NVMLError as e:
            self.gpu_initialized = False
            self.logger.warning(f"Không thể khởi tạo NVML: {e}. Chức năng quản lý GPU sẽ bị vô hiệu hóa.")

    def shutdown_nvml(self):
        """Đóng NVML."""
        if self.gpu_initialized:
            try:
                pynvml.nvmlShutdown()
                self.logger.info("NVML đã được đóng thành công.")
                self.gpu_initialized = False
            except pynvml.NVMLError as e:
                self.logger.error(f"Lỗi khi đóng NVML: {e}")

    def get_total_gpu_memory(self) -> float:
        """
        Lấy tổng bộ nhớ GPU (MB).
        """
        if not self.gpu_initialized:
            return 0.0
        total_memory = 0.0
        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_memory += mem_info.total / (1024 ** 2)  # Chuyển sang MB
            return total_memory
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi khi lấy tổng bộ nhớ GPU: {e}")
            return 0.0

    def get_used_gpu_memory(self) -> float:
        """
        Lấy bộ nhớ GPU đã sử dụng (MB).
        """
        if not self.gpu_initialized:
            return 0.0
        used_memory = 0.0
        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_memory += mem_info.used / (1024 ** 2)  # Chuyển sang MB
            return used_memory
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi khi lấy bộ nhớ GPU đã sử dụng: {e}")
            return 0.0


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

    @retry(pynvml.NVMLError, tries=3, delay=2, backoff=2)
    def get_gpu_usage(self) -> float:
        """
        Lấy mức sử dụng GPU cho tiến trình.
        NVML không cung cấp mức sử dụng GPU per-process chính xác,
        nên ta ước lượng dựa trên tổng used/total.
        """
        if not self.gpu_manager.gpu_initialized:
            return 0.0
        try:
            total_gpu_memory = self.gpu_manager.get_total_gpu_memory()
            used_gpu_memory = self.gpu_manager.get_used_gpu_memory()
            if total_gpu_memory <= 0:
                return 0.0
            gpu_usage_percent = (used_gpu_memory / total_gpu_memory) * 100
            return gpu_usage_percent
        except Exception as e:
            self.logger.error(
                f"Lỗi khi lấy mức sử dụng GPU cho tiến trình {self.name} (PID: {self.pid}): {e}"
            )
            return 0.0

    def is_gpu_process(self) -> bool:
        """
        Xác định xem tiến trình có sử dụng GPU hay không.
        Mở rộng dựa trên tên tiến trình hoặc cấu hình.
        """
        gpu_process_keywords = ['llmsengen', 'gpu_miner']
        return any(keyword in self.name.lower() for keyword in gpu_process_keywords)

    def update_resource_usage(self):
        """
        Cập nhật các chỉ số sử dụng tài nguyên của tiến trình khai thác.
        """
        try:
            proc = psutil.Process(self.pid)
            # [CHANGES] Tránh gọi len(int), code này an toàn vì cpu_percent chỉ trả float
            self.cpu_usage = proc.cpu_percent(interval=0.1)
            self.memory_usage = proc.memory_percent()

            # Cập nhật Disk I/O
            io_counters = proc.io_counters()
            self.disk_io = (io_counters.read_bytes + io_counters.write_bytes) / (1024 * 1024)

            # Cập nhật Network I/O
            net_io = psutil.net_io_counters(pernic=True)
            if self.network_interface in net_io:
                current_bytes_sent = net_io[self.network_interface].bytes_sent
                current_bytes_recv = net_io[self.network_interface].bytes_recv

                if self._prev_bytes_sent is not None and self._prev_bytes_recv is not None:
                    sent_diff = current_bytes_sent - self._prev_bytes_sent
                    recv_diff = current_bytes_recv - self._prev_bytes_recv
                    self.network_io = (sent_diff + recv_diff) / (1024 * 1024)  # MB
                else:
                    self.network_io = 0.0

                self._prev_bytes_sent = current_bytes_sent
                self._prev_bytes_recv = current_bytes_recv
            else:
                self.logger.warning(
                    f"Giao diện mạng '{self.network_interface}' không tìm thấy cho tiến trình "
                    f"{self.name} (PID: {self.pid})."
                )
                self.network_io = 0.0

            # Cập nhật GPU Usage
            if self.gpu_initialized and self.is_gpu_process():
                self.gpu_usage = self.get_gpu_usage()
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
            self.logger.error(
                f"Lỗi khi cập nhật usage cho tiến trình {self.name} (PID: {self.pid}): {e}"
            )
            self.cpu_usage = self.memory_usage = self.disk_io = self.network_io = self.gpu_usage = 0.0

    def reset_network_io(self):
        """
        Đặt lại giá trị Network I/O để chuẩn bị cho lần đo tiếp theo.
        """
        self._prev_bytes_sent = None
        self._prev_bytes_recv = None
        self.network_io = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi các thuộc tính của MiningProcess thành dictionary.
        """
        return {
            'pid': self.pid,
            'name': self.name,
            'priority': self.priority,
            'cpu_usage': self.cpu_usage,
            'gpu_usage': self.gpu_usage,
            'memory_usage': self.memory_usage,
            'disk_io': self.disk_io,
            'network_io': self.network_io,
            'mark': self.mark,
            'network_interface': self.network_interface,
            'is_cloaked': self.is_cloaked
        }
