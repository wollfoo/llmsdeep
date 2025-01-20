"""
Module temperature_monitor.py

Quản lý giám sát nhiệt độ CPU, GPU, và giới hạn Disk I/O (theo mô hình đồng bộ).
Đã refactor loại bỏ hoàn toàn asyncio/await, chuyển sang synchronous (threading).

Đảm bảo tương thích với resource_manager.py (cũng hoạt động đồng bộ).
"""
import logging
import os
import sys
import psutil
import pynvml
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any

# Thêm đường dẫn tới thư mục chứa `logging_config.py`
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SCRIPT_DIR))

# Thiết lập đường dẫn tới thư mục logs
LOGS_DIR = Path(os.getenv('LOGS_DIR', '/app/mining_environment/logs'))
os.makedirs(LOGS_DIR, exist_ok=True)

from logging_config import setup_logging
logger = setup_logging('power_management', LOGS_DIR / 'power_management.log', 'INFO')


class TemperatureMonitor:
    """
    Lớp đồng bộ giám sát nhiệt độ CPU/GPU và giới hạn Disk I/O.

    Attributes:
        _initialized (bool): Đánh dấu đã init hay chưa.
        _nvml_initialized (bool): Đánh dấu đã init pynvml hay chưa.
        gpu_count (int): Số GPU trên hệ thống.
        cache_limit_percent (float): Phần trăm cache limit (nếu cần).
        _cpu_sensor_warning_logged (bool): Đánh dấu đã log warning “không tìm thấy sensor CPU” chưa.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        """
        Khởi tạo TemperatureMonitor (đồng bộ).
        """
        if getattr(self, '_initialized', False):
            return

        self._initialized = True
        self._nvml_initialized = False
        self.gpu_count = 0
        self.cache_limit_percent = 70.0
        self._cpu_sensor_warning_logged = False

        # Cố gắng init NVML ngay
        self._initialize_nvml()

    @classmethod
    def get_instance(cls) -> 'TemperatureMonitor':
        """
        Lấy singleton instance của TemperatureMonitor (đồng bộ).

        :return: Thể hiện TemperatureMonitor (singleton).
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def _initialize_nvml(self) -> None:
        """
        Khởi tạo pynvml (NVML) theo dạng đồng bộ.
        """
        if not self._nvml_initialized:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self._nvml_initialized = True
                logger.info(f"TemperatureMonitor: Đã khởi tạo pynvml với {self.gpu_count} GPU.")
            except pynvml.NVMLError as e:
                logger.error(f"TemperatureMonitor: Lỗi khi khởi tạo pynvml: {e}")
                self.gpu_count = 0
                self._nvml_initialized = False

    def shutdown(self) -> None:
        """
        Dừng giám sát nhiệt độ và giải phóng NVML (đồng bộ).
        """
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                self._nvml_initialized = False
                logger.info("TemperatureMonitor: Đã shutdown pynvml.")
            except pynvml.NVMLError as e:
                logger.error(f"TemperatureMonitor: Lỗi khi shutdown pynvml: {e}")

    def get_cpu_temperature(self, pid: Optional[int] = None) -> float:
        """
        Lấy nhiệt độ CPU trung bình (°C) (đồng bộ).

        :param pid: (Không bắt buộc) PID tiến trình, nếu cần logic riêng (hiện chưa dùng).
        :return: Nhiệt độ CPU trung bình (float). Trả về 0.0 nếu không tìm thấy sensor.
        """
        try:
            # psutil.sensors_temperatures() => { 'coretemp': [entries], ... }
            temps_info = psutil.sensors_temperatures()
            if not temps_info:
                if not self._cpu_sensor_warning_logged:
                    logger.warning("TemperatureMonitor: Không tìm thấy cảm biến nhiệt độ CPU (sensors rỗng).")
                    self._cpu_sensor_warning_logged = True
                return 0.0

            # Tìm key có 'coretemp' hoặc 'cpu'
            for name, entries in temps_info.items():
                if 'coretemp' in name.lower() or 'cpu' in name.lower():
                    cpu_temps = []
                    for entry in entries:
                        # entry.label có thể là 'Core 0', 'Core 1'
                        if 'core' in entry.label.lower() or entry.label == '':
                            cpu_temps.append(entry.current)
                    if cpu_temps:
                        avg_temp = sum(cpu_temps) / len(cpu_temps)
                        return float(avg_temp)

            # Không tìm thấy core => fallback 0.0
            if not self._cpu_sensor_warning_logged:
                logger.warning("TemperatureMonitor: Không tìm thấy entry CPU coretemp.")
                self._cpu_sensor_warning_logged = True
            return 0.0

        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy nhiệt độ CPU: {e}")
            return 0.0

    def get_gpu_temperature(self, pid: Optional[int] = None) -> float or List[float]:
        """
        Lấy nhiệt độ GPU (°C). Có thể trả về giá trị trung bình (float) 
        hoặc danh sách tất cả GPU (tuỳ nhu cầu).

        :param pid: (Không bắt buộc) PID tiến trình, nếu cần logic riêng (hiện chưa dùng).
        :return: float trung bình, hoặc List[float] nếu muốn tách cho từng GPU. 
                 0.0 (hoặc []) nếu lỗi hoặc không có GPU.
        """
        if not self._nvml_initialized:
            logger.warning("TemperatureMonitor: NVML chưa được khởi tạo, không thể lấy nhiệt độ GPU.")
            return 0.0

        try:
            temps = []
            for i in range(self.gpu_count):
                t = self._get_single_gpu_temperature(i)
                if t is not None:
                    temps.append(t)
            if not temps:
                return 0.0
            # Tuỳ nhu cầu: trả về danh sách hay trung bình:
            # Ví dụ: trả về danh sách (nếu ResourceManager muốn check từng GPU)
            return temps
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy nhiệt độ GPU: {e}")
            return 0.0

    def _get_single_gpu_temperature(self, gpu_index: int) -> Optional[float]:
        """
        Lấy nhiệt độ 1 GPU cụ thể (đồng bộ).

        :param gpu_index: index GPU.
        :return: float nhiệt độ, hoặc None nếu lỗi.
        """
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            return float(temp)
        except pynvml.NVMLError as e:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy nhiệt độ GPU {gpu_index}: {e}")
            return None

    def get_current_disk_io_limit(self, pid: Optional[int] = None) -> float:
        """
        Giả lập lấy giới hạn Disk I/O (Mbps) qua cgroup (đồng bộ).

        :param pid: (Không bắt buộc) PID tiến trình, nếu dùng cgroup riêng.
        :return: Giá trị Disk IO limit (Mbps) hoặc 0.0 nếu không có hoặc lỗi.
        """
        try:
            return self._get_current_disk_io_limit(pid)
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi get_current_disk_io_limit: {e}")
            return 0.0

    def _get_current_disk_io_limit(self, pid: Optional[int] = None) -> float:
        """
        Triển khai logic đọc file cgroup (giả lập) cho disk IO limit (đồng bộ).

        :param pid: (Không bắt buộc) PID tiến trình.
        :return: float Mbps, 0.0 nếu không xác định.
        """
        try:
            # Giả sử file cgroup path
            if pid:
                cgroup_path_read = Path(f"/sys/fs/cgroup/blkio/temperature_monitor/{pid}/blkio.throttle.read_bps_device")
                cgroup_path_write = Path(f"/sys/fs/cgroup/blkio/temperature_monitor/{pid}/blkio.throttle.write_bps_device")
            else:
                cgroup_path_read = None
                cgroup_path_write = None

            read_limit = 0.0
            write_limit = 0.0

            if cgroup_path_read and cgroup_path_read.exists():
                content = cgroup_path_read.read_text().strip()
                parts = content.split()
                if len(parts) == 2 and parts[1].isdigit():
                    # parts[1] = B/s => chuyển -> Mbps
                    read_bps = int(parts[1])
                    read_limit = read_bps * 8 / (1024 * 1024)

            if cgroup_path_write and cgroup_path_write.exists():
                content = cgroup_path_write.read_text().strip()
                parts = content.split()
                if len(parts) == 2 and parts[1].isdigit():
                    write_bps = int(parts[1])
                    write_limit = write_bps * 8 / (1024 * 1024)

            return max(read_limit, write_limit) if (read_limit > 0 or write_limit > 0) else 0.0
        except Exception as ex:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy giới hạn Disk I/O: {ex}")
            return 0.0


# ----------------------- CÁC HÀM NGOÀI CLASS (ĐỒNG BỘ) -----------------------

_temperature_monitor_instance: Optional[TemperatureMonitor] = None
_temperature_monitor_lock = threading.Lock()

def get_temperature_monitor() -> TemperatureMonitor:
    """
    Lấy singleton instance của TemperatureMonitor (đồng bộ).
    
    :return: Thể hiện duy nhất (singleton) của TemperatureMonitor.
    """
    global _temperature_monitor_instance
    with _temperature_monitor_lock:
        if _temperature_monitor_instance is None:
            _temperature_monitor_instance = TemperatureMonitor.get_instance()
    return _temperature_monitor_instance

def setup_temperature_monitoring() -> None:
    """
    Thiết lập giám sát nhiệt độ (đồng bộ). 
    Ở đây mô phỏng, có thể mở thread hoặc config.
    """
    tm = get_temperature_monitor()
    logger.info("TemperatureMonitor: Đã thiết lập giám sát nhiệt độ (đồng bộ).")

def get_cpu_temperature(pid: Optional[int] = None) -> float:
    """
    Lấy nhiệt độ CPU (°C) (đồng bộ), gọi từ singleton TemperatureMonitor.

    :param pid: (Không bắt buộc) PID tiến trình.
    :return: float nhiệt độ CPU, 0.0 nếu không có sensor.
    """
    tm = get_temperature_monitor()
    return tm.get_cpu_temperature(pid)

def get_gpu_temperature(pid: Optional[int] = None) -> float or List[float]:
    """
    Lấy nhiệt độ GPU (°C) (đồng bộ) từ singleton TemperatureMonitor.
    Có thể trả float hoặc list, tuỳ logic trong ham get_gpu_temperature.

    :param pid: (Không bắt buộc) PID tiến trình.
    :return: float (nhiệt độ trung bình) hoặc list[float] (nhiệt độ từng GPU).
    """
    tm = get_temperature_monitor()
    return tm.get_gpu_temperature(pid)

def get_current_disk_io_limit(pid: Optional[int] = None) -> float:
    """
    Lấy giới hạn Disk I/O (Mbps) (đồng bộ).

    :param pid: (Không bắt buộc) PID tiến trình.
    :return: float Mbps, 0.0 nếu không có hoặc lỗi.
    """
    tm = get_temperature_monitor()
    return tm.get_current_disk_io_limit(pid)

def shutdown_temperature_monitor() -> None:
    """
    Tắt TemperatureMonitor => free NVML (nếu init).
    """
    tm = get_temperature_monitor()
    tm.shutdown()
