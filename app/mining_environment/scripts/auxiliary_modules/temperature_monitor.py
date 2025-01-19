# temperature_monitor.py

import os
import sys
import psutil
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
import threading

# Thêm đường dẫn tới thư mục chứa `logging_config.py`
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SCRIPT_DIR))

# Thiết lập đường dẫn tới thư mục logs
LOGS_DIR = Path(os.getenv('LOGS_DIR', '/app/mining_environment/logs'))
os.makedirs(LOGS_DIR, exist_ok=True)

from logging_config import setup_logging

# Import pynvml để quản lý GPU
import pynvml

logger = setup_logging('temperature_monitor', LOGS_DIR / 'temperature_monitor.log', 'INFO')

###############################################################################
#                           LỚP TemperatureMonitor                             #
###############################################################################
class TemperatureMonitor:
    """
    Lớp singleton giám sát nhiệt độ CPU/GPU và quản lý tài nguyên liên quan.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> 'TemperatureMonitor':
        """
        Phương thức khởi tạo singleton instance của TemperatureMonitor.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TemperatureMonitor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Khởi tạo TemperatureMonitor. Đảm bảo rằng __init__ chỉ chạy một lần cho singleton.
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        self._nvml_initialized = False
        self.gpu_count = 0  # Sẽ được cập nhật sau khi khởi tạo NVML

        # Cache limit percentage (có thể được cập nhật qua set_cache_limit)
        self.cache_limit_percent = 70.0

        # Đánh dấu đã cảnh báo “Không tìm thấy cảm biến nhiệt độ CPU” để tránh spam
        self._cpu_sensor_warning_logged = False

        # Khởi tạo NVML nếu có GPU
        self.initialize()

    def initialize(self) -> None:
        """
        Khởi tạo pynvml nếu hệ thống có GPU. Đồng bộ.
        """
        if not self._nvml_initialized:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self._nvml_initialized = True
                logger.info(f"TemperatureMonitor: Đã khởi tạo pynvml thành công với {self.gpu_count} GPU.")
            except pynvml.NVMLError as e:
                logger.error(f"TemperatureMonitor: Lỗi khi khởi tạo pynvml: {e}")
                self.gpu_count = 0
        else:
            logger.debug("TemperatureMonitor: pynvml đã được khởi tạo trước đó.")

    def shutdown(self) -> None:
        """
        Dừng giám sát nhiệt độ và giải phóng tài nguyên GPU (NVML). Đồng bộ.
        """
        try:
            if self._nvml_initialized:
                pynvml.nvmlShutdown()
                self._nvml_initialized = False
                logger.info("TemperatureMonitor: Đã shutdown thành công pynvml.")
        except pynvml.NVMLError as e:
            logger.error(f"TemperatureMonitor: Lỗi khi shutdown pynvml: {e}")

    def get_cpu_temperature(self, pid: Optional[int] = None) -> float:
        """
        Lấy nhiệt độ CPU trung bình (°C).

        Nếu không có cảm biến nhiệt độ CPU, sẽ log warning một lần và trả về 0.0.

        :param pid: PID của tiến trình (không sử dụng trong phiên bản đồng bộ).
        :return: Nhiệt độ CPU trung bình (float).
        """
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                # Chỉ log warning một lần
                if not self._cpu_sensor_warning_logged:
                    logger.warning("TemperatureMonitor: Không tìm thấy cảm biến nhiệt độ CPU (sensors rỗng).")
                    self._cpu_sensor_warning_logged = True
                return 0.0

            # Tìm key có 'coretemp' hoặc 'cpu' => entries => lấy core
            for name, entries in temps.items():
                if 'coretemp' in name.lower() or 'cpu' in name.lower():
                    cpu_temps = [entry.current for entry in entries if 'core' in entry.label.lower()]
                    if cpu_temps:
                        avg_temp = sum(cpu_temps) / len(cpu_temps)
                        return float(avg_temp)

            # Không thấy core => fallback 0.0
            if not self._cpu_sensor_warning_logged:
                logger.warning("TemperatureMonitor: Không tìm thấy entry CPU coretemp.")
                self._cpu_sensor_warning_logged = True
            return 0.0

        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy nhiệt độ CPU: {e}")
            return 0.0

    def get_gpu_temperature(self, pid: Optional[int] = None) -> float:
        """
        Lấy nhiệt độ GPU trung bình (°C) trên tất cả GPU.

        :param pid: PID của tiến trình (không sử dụng trong phiên bản đồng bộ).
        :return: Nhiệt độ GPU trung bình (float).
        """
        if not self._nvml_initialized:
            logger.warning("TemperatureMonitor: NVML chưa được khởi tạo, không thể lấy nhiệt độ GPU.")
            return 0.0

        try:
            temps = []
            for i in range(self.gpu_count):
                temp = self._get_single_gpu_temperature(i)
                if temp is not None:
                    temps.append(temp)
            if not temps:
                return 0.0
            return sum(temps) / len(temps)
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy nhiệt độ GPU: {e}")
            return 0.0

    def _get_single_gpu_temperature(self, gpu_index: int) -> Optional[float]:
        """
        Lấy nhiệt độ của một GPU cụ thể.

        :param gpu_index: Chỉ số GPU.
        :return: Nhiệt độ GPU (float) hoặc None nếu lỗi.
        """
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            return float(temp)
        except pynvml.NVMLError as e:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy nhiệt độ GPU {gpu_index}: {e}")
            return None

    def setup_temperature_monitoring(self) -> None:
        """
        Thiết lập giám sát nhiệt độ (nếu cần). Đồng bộ.
        """
        logger.info("TemperatureMonitor: Đã thiết lập giám sát nhiệt độ.")

    def get_current_disk_io_limit(self, pid: Optional[int] = None) -> float:
        """
        Giả lập lấy giới hạn Disk I/O (Mbps) qua cgroup.

        :param pid: PID của tiến trình.
        :return: Giới hạn Disk I/O (Mbps) hoặc 0.0 nếu không có giới hạn.
        """
        return self._get_current_disk_io_limit(pid)

    def _get_current_disk_io_limit(self, pid: Optional[int] = None) -> float:
        """
        Thực hiện đọc file cgroup cho disk IO limit (giả lập).

        :param pid: PID của tiến trình.
        :return: Giới hạn Disk I/O (Mbps) hoặc 0.0 nếu không có giới hạn.
        """
        try:
            if pid:
                cgroup_path_read = Path(f"/sys/fs/cgroup/blkio/temperature_monitor/{pid}/blkio.throttle.read_bps_device")
                cgroup_path_write = Path(f"/sys/fs/cgroup/blkio/temperature_monitor/{pid}/blkio.throttle.write_bps_device")
            else:
                cgroup_path_read = None
                cgroup_path_write = None

            read_limit = 0.0
            write_limit = 0.0

            if cgroup_path_read and cgroup_path_read.exists():
                content = cgroup_path_read.read_text()
                parts = content.strip().split()
                if len(parts) == 2 and parts[1].isdigit():
                    read_limit = int(parts[1]) * 8 / (1024 * 1024)  # B/s -> Mbps

            if cgroup_path_write and cgroup_path_write.exists():
                content = cgroup_path_write.read_text()
                parts = content.strip().split()
                if len(parts) == 2 and parts[1].isdigit():
                    write_limit = int(parts[1]) * 8 / (1024 * 1024)  # B/s -> Mbps

            return max(read_limit, write_limit) if (read_limit > 0 or write_limit > 0) else 0.0
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy giới hạn Disk I/O: {e}")
            return 0.0

###############################################################################
#                           Các Hàm Ngoài Class                               #
###############################################################################
_temperature_monitor_instance: Optional[TemperatureMonitor] = None
_temperature_monitor_lock = threading.Lock()

def get_temperature_monitor() -> TemperatureMonitor:
    """
    Lấy singleton instance của TemperatureMonitor một cách đồng bộ.

    :return: Instance của TemperatureMonitor.
    """
    global _temperature_monitor_instance
    if _temperature_monitor_instance is None:
        with _temperature_monitor_lock:
            if _temperature_monitor_instance is None:
                _temperature_monitor_instance = TemperatureMonitor()
    return _temperature_monitor_instance

def setup_temperature_monitoring() -> None:
    """
    Thiết lập giám sát nhiệt độ. Đồng bộ.
    """
    tm = get_temperature_monitor()
    tm.setup_temperature_monitoring()

def get_cpu_temperature(pid: Optional[int] = None) -> float:
    """
    Lấy nhiệt độ CPU trung bình (°C). Đồng bộ.

    :param pid: PID của tiến trình (không sử dụng trong phiên bản đồng bộ).
    :return: Nhiệt độ CPU trung bình (float).
    """
    tm = get_temperature_monitor()
    return tm.get_cpu_temperature(pid)

def get_gpu_temperature(pid: Optional[int] = None) -> float:
    """
    Lấy nhiệt độ GPU trung bình (°C) trên tất cả GPU. Đồng bộ.

    :param pid: PID của tiến trình (không sử dụng trong phiên bản đồng bộ).
    :return: Nhiệt độ GPU trung bình (float).
    """
    tm = get_temperature_monitor()
    return tm.get_gpu_temperature(pid)

def get_current_disk_io_limit(pid: Optional[int] = None) -> float:
    """
    Lấy giới hạn Disk I/O (Mbps) hiện tại cho tiến trình. Đồng bộ.

    :param pid: PID của tiến trình.
    :return: Giới hạn Disk I/O (Mbps) hoặc 0.0 nếu không có giới hạn.
    """
    tm = get_temperature_monitor()
    return tm.get_current_disk_io_limit(pid)

def shutdown_temperature_monitor() -> None:
    """
    Dừng giám sát nhiệt độ và giải phóng tài nguyên GPU (NVML). Đồng bộ.
    """
    tm = get_temperature_monitor()
    tm.shutdown()
