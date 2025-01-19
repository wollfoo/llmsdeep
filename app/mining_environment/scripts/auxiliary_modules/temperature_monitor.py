# temperature_monitor.py

import os
import sys
import psutil
import subprocess
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

# Thêm đường dẫn tới thư mục chứa `logging_config.py`
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SCRIPT_DIR))

# Thiết lập đường dẫn tới thư mục logs
LOGS_DIR = Path(os.getenv('LOGS_DIR', '/app/mining_environment/logs'))
os.makedirs(LOGS_DIR, exist_ok=True)

from logging_config import setup_logging

# Import pynvml để quản lý GPU
import pynvml

# Thiết lập logging với logging_config.py
logger = setup_logging('temperature_monitor', LOGS_DIR / 'temperature_monitor.log', 'INFO')

###############################################################################
#                            LỚP TemperatureMonitor                         #
###############################################################################

class TemperatureMonitor:
    """
    Lớp singleton giám sát nhiệt độ CPU/GPU và quản lý tài nguyên liên quan.
    """
    _instance = None
    _lock = asyncio.Lock()

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        self._nvml_initialized = False
        self.gpu_count = 0  # sẽ được cập nhật sau khi khởi tạo NVML

        # Cache limit percentage (có thể được cập nhật qua set_cache_limit)
        self.cache_limit_percent = 70.0

    @classmethod
    async def create(cls) -> 'TemperatureMonitor':
        """
        Async factory method để tạo và khởi tạo instance (singleton).
        """
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                await cls._instance.initialize()
            return cls._instance

    async def initialize(self):
        """
        Khởi tạo pynvml ở chế độ async (chạy trong threadpool để tránh block event loop).
        """
        if not self._nvml_initialized:
            try:
                await asyncio.to_thread(pynvml.nvmlInit)
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self._nvml_initialized = True
                logger.info(f"TemperatureMonitor: Đã khởi tạo pynvml thành công với {self.gpu_count} GPU.")
            except pynvml.NVMLError as e:
                logger.error(f"TemperatureMonitor: Lỗi khi khởi tạo pynvml: {e}")
                self.gpu_count = 0
        else:
            logger.debug("TemperatureMonitor: pynvml đã được khởi tạo trước đó.")

    async def shutdown(self):
        """
        Dừng giám sát nhiệt độ và giải phóng tài nguyên GPU (NVML).
        """
        try:
            if self._nvml_initialized:
                await asyncio.to_thread(pynvml.nvmlShutdown)
                self._nvml_initialized = False
                logger.info("TemperatureMonitor: Đã shutdown thành công pynvml.")
        except pynvml.NVMLError as e:
            logger.error(f"TemperatureMonitor: Lỗi khi shutdown pynvml: {e}")

    async def get_cpu_temperature(self, pid: Optional[int] = None) -> float:
        """
        Lấy nhiệt độ CPU trung bình (°C).
        """
        try:
            temps = await asyncio.to_thread(psutil.sensors_temperatures)
            if not temps:
                logger.warning("TemperatureMonitor: Không tìm thấy cảm biến nhiệt độ CPU.")
                return 0.0

            for name, entries in temps.items():
                if 'coretemp' in name.lower() or 'cpu' in name.lower():
                    cpu_temps = [entry.current for entry in entries if 'core' in entry.label.lower()]
                    if cpu_temps:
                        avg_temp = sum(cpu_temps) / len(cpu_temps)
                        return float(avg_temp)
            return 0.0
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy nhiệt độ CPU: {e}")
            return 0.0

    async def get_gpu_temperature(self, pid: Optional[int] = None) -> float:
        """
        Lấy nhiệt độ GPU trung bình (°C) trên tất cả GPU.
        """
        if not self._nvml_initialized:
            logger.warning("TemperatureMonitor: NVML chưa được khởi tạo, không thể lấy nhiệt độ GPU.")
            return 0.0

        try:
            temps = []
            for i in range(self.gpu_count):
                temp = await asyncio.to_thread(self._get_single_gpu_temperature, i)
                if temp is not None:
                    temps.append(temp)
            return (sum(temps) / len(temps)) if temps else 0.0
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy nhiệt độ GPU: {e}")
            return 0.0

    def _get_single_gpu_temperature(self, gpu_index: int) -> Optional[float]:
        """
        Lấy nhiệt độ của một GPU cụ thể (chạy trong threadpool).
        """
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            return float(temp)
        except pynvml.NVMLError as e:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy nhiệt độ GPU {gpu_index}: {e}")
            return None

    async def setup_temperature_monitoring(self):
        """
        Thiết lập giám sát nhiệt độ (nếu cần luồng riêng).
        """
        logger.info("TemperatureMonitor: Đã thiết lập giám sát nhiệt độ.")

    async def get_current_disk_io_limit(self, pid: Optional[int] = None) -> float:
        """
        Giả lập lấy giới hạn Disk I/O (Mbps) qua cgroup.
        """
        return await self._get_current_disk_io_limit(pid)

    async def _get_current_disk_io_limit(self, pid: Optional[int] = None) -> float:
        """
        Thực hiện đọc file cgroup cho disk IO limit (giả lập).
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
                content = await asyncio.to_thread(cgroup_path_read.read_text)
                parts = content.strip().split()
                if len(parts) == 2 and parts[1].isdigit():
                    read_limit = int(parts[1]) * 8 / (1024 * 1024)  # B/s -> Mbps

            if cgroup_path_write and cgroup_path_write.exists():
                content = await asyncio.to_thread(cgroup_path_write.read_text)
                parts = content.strip().split()
                if len(parts) == 2 and parts[1].isdigit():
                    write_limit = int(parts[1]) * 8 / (1024 * 1024)  # B/s -> Mbps

            return max(read_limit, write_limit) if (read_limit > 0 or write_limit > 0) else 0.0
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy giới hạn Disk I/O: {e}")
            return 0.0

    # ... (các hàm khác: set_cpu_threads, set_ram_allocation, set_cache_limit, etc.)
    # Lược bớt do nội dung logic tương tự => chuyển sang await asyncio.to_thread(...) thay cho run_in_executor(...).

    # Demo: cắt gọn 1 ví dụ _find_mining_process => to_thread:
    def _find_mining_process(self) -> Optional[psutil.Process]:
        """Tìm tiến trình khai thác."""
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] in ['ml-inference', 'inference-cuda']:
                    logger.debug(f"TemperatureMonitor: Tìm thấy tiến trình {proc.info['name']} PID={proc.pid}")
                    return proc
            return None
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi tìm tiến trình khai thác: {e}")
            return None



# Singleton instance của TemperatureMonitor sẽ được tạo thông qua async factory method
_temperature_monitor_instance: Optional[TemperatureMonitor] = None
_temperature_monitor_lock = asyncio.Lock()

async def get_temperature_monitor() -> TemperatureMonitor:
    """
    Lấy singleton instance của TemperatureMonitor một cách bất đồng bộ.

    Returns:
        TemperatureMonitor: Instance của TemperatureMonitor.
    """
    global _temperature_monitor_instance
    async with _temperature_monitor_lock:
        if _temperature_monitor_instance is None:
            _temperature_monitor_instance = await TemperatureMonitor.create()
    return _temperature_monitor_instance

###############################################################################
#                          Các Hàm Ngoài Class                              #
###############################################################################
async def setup_temperature_monitoring():
    """
    Thiết lập giám sát nhiệt độ.
    """
    temperature_monitor = await get_temperature_monitor()
    await temperature_monitor.setup_temperature_monitoring()

async def get_cpu_temperature(pid: Optional[int] = None) -> float:
    """
    Trả về nhiệt độ CPU hiện tại (°C).

    Args:
        pid (Optional[int]): PID của tiến trình (không sử dụng).

    Returns:
        float: Nhiệt độ CPU trung bình (°C).
    """
    temperature_monitor = await get_temperature_monitor()
    return await temperature_monitor.get_cpu_temperature(pid)

async def get_gpu_temperature(pid: Optional[int] = None) -> float:
    """
    Trả về nhiệt độ GPU hiện tại (°C).

    Args:
        pid (Optional[int]): PID của tiến trình (không sử dụng).

    Returns:
        float: Nhiệt độ GPU trung bình (°C).
    """
    temperature_monitor = await get_temperature_monitor()
    return await temperature_monitor.get_gpu_temperature(pid)

async def get_current_cpu_threads(pid: Optional[int] = None) -> int:
    """
    Trả về số lượng CPU threads hiện tại.

    Args:
        pid (Optional[int]): PID của tiến trình (không sử dụng).

    Returns:
        int: Số lượng CPU threads.
    """
    temperature_monitor = await get_temperature_monitor()
    return await temperature_monitor.get_current_cpu_threads(pid)

async def set_cpu_threads(new_threads: int, pid: Optional[int] = None):
    """
    Giảm công suất CPU bằng cách giảm số lượng threads.

    Args:
        new_threads (int): Số lượng threads mới.
        pid (Optional[int]): PID của tiến trình (không sử dụng).
    """
    temperature_monitor = await get_temperature_monitor()
    await temperature_monitor.set_cpu_threads(new_threads, pid)

async def get_current_ram_allocation(pid: Optional[int] = None) -> float:
    """
    Trả về lượng RAM hiện tại (MB).

    Args:
        pid (Optional[int]): PID của tiến trình (không sử dụng).

    Returns:
        float: Lượng RAM hiện tại (MB).
    """
    temperature_monitor = await get_temperature_monitor()
    return await temperature_monitor.get_current_ram_allocation(pid)

async def set_ram_allocation(new_ram_mb: int, pid: Optional[int] = None):
    """
    Thiết lập giới hạn RAM (MB).

    Args:
        new_ram_mb (int): Lượng RAM mới (MB).
        pid (Optional[int]): PID của tiến trình (không sử dụng).
    """
    temperature_monitor = await get_temperature_monitor()
    await temperature_monitor.set_ram_allocation(new_ram_mb, pid)

async def get_current_gpu_usage(pid: Optional[int] = None) -> float:
    """
    Trả về mức sử dụng GPU hiện tại (%).

    Args:
        pid (Optional[int]): PID của tiến trình (không sử dụng).

    Returns:
        float: Mức sử dụng GPU trung bình (%).
    """
    temperature_monitor = await get_temperature_monitor()
    return await temperature_monitor.get_current_gpu_usage(pid)

async def get_current_disk_io_limit(pid: Optional[int] = None) -> float:
    """
    Trả về giới hạn Disk I/O hiện tại (Mbps).

    Args:
        pid (Optional[int]): PID của tiến trình (không sử dụng).

    Returns:
        float: Giới hạn Disk I/O (Mbps).
    """
    temperature_monitor = await get_temperature_monitor()
    return await temperature_monitor.get_current_disk_io_limit(pid)

async def set_disk_io_limit(new_disk_io_mbps: float, pid: Optional[int] = None):
    """
    Thiết lập giới hạn Disk I/O (Mbps).

    Args:
        new_disk_io_mbps (float): Giới hạn Disk I/O mới (Mbps).
        pid (Optional[int]): PID của tiến trình (không sử dụng).
    """
    temperature_monitor = await get_temperature_monitor()
    await temperature_monitor.set_disk_io_limit(new_disk_io_mbps, pid)

async def get_current_network_bandwidth_limit(pid: Optional[int] = None) -> float:
    """
    Trả về giới hạn băng thông mạng hiện tại (Mbps).

    Args:
        pid (Optional[int]): PID của tiến trình (không sử dụng).

    Returns:
        float: Giới hạn băng thông mạng (Mbps).
    """
    temperature_monitor = await get_temperature_monitor()
    return await temperature_monitor.get_current_network_bandwidth_limit(pid)

async def set_network_bandwidth_limit(new_network_bw_mbps: float, pid: Optional[int] = None):
    """
    Thiết lập giới hạn băng thông mạng (Mbps).

    Args:
        new_network_bw_mbps (float): Giới hạn băng thông mạng mới (Mbps).
        pid (Optional[int]): PID của tiến trình (không sử dụng).
    """
    temperature_monitor = await get_temperature_monitor()
    await temperature_monitor.set_network_bandwidth_limit(new_network_bw_mbps, pid)

async def get_current_cache_limit(pid: Optional[int] = None) -> float:
    """
    Trả về giới hạn Cache hiện tại (%).

    Args:
        pid (Optional[int]): PID của tiến trình (không sử dụng).

    Returns:
        float: Giới hạn Cache (%).
    """
    temperature_monitor = await get_temperature_monitor()
    return await temperature_monitor.get_current_cache_limit(pid)

async def set_cache_limit(new_cache_limit_percent: float, pid: Optional[int] = None):
    """
    Thiết lập giới hạn Cache (%).

    Args:
        new_cache_limit_percent (float): Giới hạn Cache mới (%).
        pid (Optional[int]): PID của tiến trình (không sử dụng).
    """
    temperature_monitor = await get_temperature_monitor()
    await temperature_monitor.set_cache_limit(new_cache_limit_percent, pid)

async def shutdown_temperature_monitor():
    """
    Dừng giám sát nhiệt độ và giải phóng tài nguyên.
    """
    temperature_monitor = await get_temperature_monitor()
    await temperature_monitor.shutdown()

# Utility functions
def _write_to_file(path: Path, content: str):
    """
    Ghi nội dung vào file.

    Args:
        path (Path): Đường dẫn tới file.
        content (str): Nội dung cần ghi.
    """
    try:
        with open(path, 'w') as f:
            f.write(content)
        logger.debug(f"TemperatureMonitor: Đã ghi nội dung vào file {path}: {content.strip()}")
    except Exception as e:
        logger.error(f"TemperatureMonitor: Lỗi khi ghi vào file {path}: {e}")
