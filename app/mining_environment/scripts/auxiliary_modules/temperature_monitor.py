# temperature_monitor.py

import os
import sys
import psutil
import pynvml
import subprocess
import threading
import logging
from pathlib import Path
from typing import Optional

# Thêm đường dẫn tới thư mục chứa `logging_config.py`
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SCRIPT_DIR))

# Thiết lập đường dẫn tới thư mục logs
LOGS_DIR = Path(os.getenv('LOGS_DIR', '/app/mining_environment/logs'))
os.makedirs(LOGS_DIR, exist_ok=True)

from logging_config import setup_logging

# Thiết lập logging
logger = setup_logging('temperature_monitor', LOGS_DIR / 'temperature_monitor.log', 'INFO')


class TemperatureMonitor:
    """
    TemperatureMonitor là một singleton class chịu trách nhiệm giám sát nhiệt độ của CPU và GPU,
    cũng như quản lý các tài nguyên liên quan đến tiến trình khai thác.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TemperatureMonitor, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Initialize NVML for GPU temperature monitoring
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"NVML initialized successfully. Số lượng GPU: {self.gpu_count}")
        except pynvml.NVMLError as e:
            logger.error(f"Không thể khởi tạo NVML: {e}")
            self.gpu_count = 0

        # Cache limit percentage (có thể được cập nhật qua set_cache_limit)
        self.cache_limit_percent = 70.0

    def get_cpu_temperature(self, pid: Optional[int] = None) -> float:
        """
        Lấy nhiệt độ hiện tại của CPU. 
        Trả về 0.0 nếu không thể lấy.
        """
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                logger.warning("Không tìm thấy cảm biến nhiệt độ CPU.")
                return 0.0

            # Tìm kiếm cảm biến CPU
            for name, entries in temps.items():
                if 'coretemp' in name.lower() or 'cpu' in name.lower():
                    cpu_temps = [entry.current for entry in entries if 'core' in entry.label.lower()]
                    if cpu_temps:
                        avg_temp = sum(cpu_temps) / len(cpu_temps)
                        logger.debug(f"Nhiệt độ CPU trung bình: {avg_temp}°C")
                        return float(avg_temp)
            logger.warning("Không tìm thấy nhãn nhiệt độ CPU phù hợp.")
            return 0.0
        except Exception as e:
            logger.error(f"Lỗi khi lấy nhiệt độ CPU: {e}")
            return 0.0

    def get_gpu_temperature(self, pid: Optional[int] = None) -> float:
        """
        Lấy nhiệt độ hiện tại của từng GPU và trả về nhiệt độ trung bình.
        Trả về 0.0 nếu không có GPU hoặc gặp lỗi.
        """
        if self.gpu_count == 0:
            logger.warning("Không có GPU nào được phát hiện để giám sát nhiệt độ.")
            return 0.0

        gpu_temps = []
        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_temps.append(temp)
                logger.debug(f"GPU {i} Temperature: {temp}°C")
            if gpu_temps:
                avg_temp = sum(gpu_temps) / len(gpu_temps)
                logger.debug(f"Nhiệt độ GPU trung bình: {avg_temp}°C")
                return float(avg_temp)
            else:
                logger.warning("Không có dữ liệu nhiệt độ GPU để tính trung bình.")
                return 0.0
        except pynvml.NVMLError as e:
            logger.error(f"Lỗi khi lấy nhiệt độ GPU: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi lấy nhiệt độ GPU: {e}")
            return 0.0

    def get_current_cpu_threads(self, pid: Optional[int] = None) -> int:
        """
        Lấy số lượng CPU threads hiện tại được gán cho tiến trình (nếu pid=None, tìm mining process).
        """
        try:
            if pid:
                proc = psutil.Process(pid)
            else:
                proc = self._find_mining_process()

            if proc:
                affinity = proc.cpu_affinity()
                num_threads = len(affinity)
                logger.debug(
                    f"Số lượng CPU threads cho tiến trình '{proc.name()}' (PID {proc.pid}): {num_threads}"
                )
                return num_threads
            else:
                logger.warning("Không tìm thấy tiến trình khai thác.")
                return 0
        except Exception as e:
            logger.error(f"Lỗi khi lấy số lượng CPU threads: {e}")
            return 0

    def set_cpu_threads(self, new_threads: int, pid: Optional[int] = None):
        """
        Gán số lượng CPU threads cho tiến trình (nếu pid=None, tìm mining process).
        """
        try:
            if pid:
                proc = psutil.Process(pid)
            else:
                proc = self._find_mining_process()
            if proc:
                total_cores = psutil.cpu_count(logical=True)
                if new_threads > total_cores:
                    logger.warning(
                        f"Số threads mới ({new_threads}) vượt quá số lõi CPU ({total_cores}). "
                        f"Đặt lại thành {total_cores}."
                    )
                    new_threads = total_cores

                new_affinity = list(range(new_threads))
                proc.cpu_affinity(new_affinity)
                logger.info(
                    f"Đã gán tiến trình '{proc.name()}' (PID {proc.pid}) vào CPU cores: {new_affinity}"
                )
            else:
                logger.warning("Không tìm thấy tiến trình khai thác để gán CPU threads.")
        except Exception as e:
            logger.error(f"Lỗi khi gán CPU threads: {e}")

    def get_current_ram_allocation(self, pid: Optional[int] = None) -> float:
        """
        Lấy lượng RAM hiện tại (MB) cho tiến trình.
        Trả về 0.0 nếu không thể lấy.
        """
        try:
            if pid:
                proc = psutil.Process(pid)
            else:
                proc = self._find_mining_process()
            if proc:
                mem_info = proc.memory_info()
                ram_mb = mem_info.rss / (1024 * 1024)
                logger.debug(f"Lượng RAM hiện tại cho tiến trình '{proc.name()}': {ram_mb} MB")
                return float(ram_mb)
            else:
                logger.warning("Không tìm thấy tiến trình khai thác để lấy RAM allocation.")
                return 0.0
        except Exception as e:
            logger.error(f"Lỗi khi lấy RAM allocation: {e}")
            return 0.0

    def set_ram_allocation(self, new_ram_mb: int, pid: Optional[int] = None):
        """
        Thiết lập giới hạn RAM (MB) cho tiến trình, giả lập bằng cgroups (nếu có quyền).
        """
        try:
            if pid:
                proc = psutil.Process(pid)
            else:
                proc = self._find_mining_process()
            if proc:
                cgroup_dir = Path(f"/sys/fs/cgroup/memory/temperature_monitor/{proc.pid}")
                cgroup_dir.mkdir(parents=True, exist_ok=True)
                cgroup_path = cgroup_dir / 'memory.limit_in_bytes'
                new_limit_bytes = new_ram_mb * 1024 * 1024
                with open(cgroup_path, 'w') as f:
                    f.write(str(new_limit_bytes))
                logger.info(
                    f"Đã thiết lập giới hạn RAM cho tiến trình '{proc.name()}' thành {new_ram_mb} MB."
                )
            else:
                logger.warning("Không tìm thấy tiến trình khai thác để thiết lập RAM allocation.")
        except PermissionError:
            logger.error("Không có quyền để thiết lập RAM allocation qua cgroup.")
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập RAM allocation: {e}")

    def get_current_gpu_usage(self, pid: Optional[int] = None) -> float:
        """
        Lấy mức sử dụng GPU hiện tại (%) cho tiến trình pid hoặc tổng thể.
        Trả về 0.0 nếu không có GPU hoặc gặp lỗi.
        """
        return self.get_gpu_temperature(pid)

    def get_current_disk_io_limit(self, pid: Optional[int] = None) -> float:
        """
        Lấy giới hạn Disk I/O hiện tại (Mbps) giả lập thông qua cgroup.
        Trả về 0.0 nếu không thể lấy.
        """
        return self._get_current_disk_io_limit(pid)

    def _get_current_disk_io_limit(self, pid: Optional[int] = None) -> float:
        """
        Hàm nội bộ để lấy giới hạn Disk I/O, đảm bảo luôn trả về float.
        """
        try:
            if pid:
                cgroup_path_read = f"/sys/fs/cgroup/blkio/temperature_monitor/{pid}/blkio.throttle.read_bps_device"
                cgroup_path_write = f"/sys/fs/cgroup/blkio/temperature_monitor/{pid}/blkio.throttle.write_bps_device"
            else:
                cgroup_path_read = None
                cgroup_path_write = None

            read_limit = 0.0
            write_limit = 0.0

            if cgroup_path_read and Path(cgroup_path_read).exists():
                content = Path(cgroup_path_read).read_text().strip()
                parts = content.split()
                # parts = ["8:0", "1048576"]
                if len(parts) == 2 and parts[1].isdigit():
                    read_limit = int(parts[1]) * 8 / (1024 * 1024)  # B/s -> Mbps
                    logger.debug(f"Disk I/O limit Read: {read_limit} Mbps")

            if cgroup_path_write and Path(cgroup_path_write).exists():
                content = Path(cgroup_path_write).read_text().strip()
                parts = content.split()
                if len(parts) == 2 and parts[1].isdigit():
                    write_limit = int(parts[1]) * 8 / (1024 * 1024)  # B/s -> Mbps
                    logger.debug(f"Disk I/O limit Write: {write_limit} Mbps")

            if read_limit > 0.0 and write_limit > 0.0:
                disk_io_limit = max(read_limit, write_limit)
                logger.debug(f"Disk I/O limit (cgroup): Read={read_limit}Mbps, Write={write_limit}Mbps. Sử dụng max: {disk_io_limit}Mbps")
                return disk_io_limit
            elif read_limit > 0.0:
                logger.debug(f"Disk I/O limit (cgroup) chỉ giới hạn Read={read_limit}Mbps")
                return read_limit
            elif write_limit > 0.0:
                logger.debug(f"Disk I/O limit (cgroup) chỉ giới hạn Write={write_limit}Mbps")
                return write_limit
            else:
                logger.warning("Không thể lấy giới hạn Disk I/O thông qua cgroup blkio. Gán giá trị mặc định 0.0 Mbps.")
                return 0.0
        except Exception as e:
            logger.error(f"Lỗi khi lấy giới hạn Disk I/O: {e}")
            return 0.0

    def set_disk_io_limit(self, new_disk_io_mbps: float, pid: Optional[int] = None):
        """
        Thiết lập giới hạn Disk I/O (Mbps) cho tiến trình qua cgroup blkio (giả lập).
        """
        try:
            if pid:
                proc = psutil.Process(pid)
            else:
                proc = self._find_mining_process()
            if proc:
                cgroup_dir = Path(f"/sys/fs/cgroup/blkio/temperature_monitor/{proc.pid}")
                cgroup_dir.mkdir(parents=True, exist_ok=True)

                device_major = 8
                device_minor = 0
                bps_limit = int(new_disk_io_mbps * 1024 * 1024 / 8)

                cgroup_path_read = cgroup_dir / 'blkio.throttle.read_bps_device'
                cgroup_path_write = cgroup_dir / 'blkio.throttle.write_bps_device'

                cgroup_path_read.write_text(f"{device_major}:{device_minor} {bps_limit}\n")
                cgroup_path_write.write_text(f"{device_major}:{device_minor} {bps_limit}\n")

                logger.info(
                    f"Đã thiết lập giới hạn Disk I/O cho tiến trình '{proc.name()}' thành {new_disk_io_mbps} Mbps."
                )
            else:
                logger.warning("Không tìm thấy tiến trình khai thác để thiết lập Disk I/O limit.")
        except PermissionError:
            logger.error("Không có quyền cgroup để thiết lập Disk I/O limit.")
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập Disk I/O limit: {e}")

    def get_current_network_bandwidth_limit(self, pid: Optional[int] = None) -> float:
        """
        Lấy giới hạn băng thông mạng (Mbps) giả lập qua tc.
        Trả về 0.0 nếu không thể lấy.
        """
        try:
            network_interface = 'eth0'
            result = subprocess.check_output(
                ['tc', 'class', 'show', 'dev', network_interface],
                stderr=subprocess.STDOUT
            )
            output = result.decode()
            for line in output.splitlines():
                if '1:1' in line and 'rate' in line:
                    tokens = line.split()
                    if 'rate' in tokens:
                        rate_index = tokens.index('rate')
                        rate = tokens[rate_index + 1]
                        if 'Kbit' in rate:
                            bw = float(rate.replace('Kbit', '')) / 1024
                        elif 'Mbit' in rate:
                            bw = float(rate.replace('Mbit', ''))
                        else:
                            bw = 0.0
                        logger.debug(f"Network bandwidth limit current: {bw} Mbps")
                        return bw
            logger.warning("Không tìm thấy giới hạn băng thông mạng. Gán giá trị mặc định 0.0 Mbps.")
            return 0.0
        except Exception as e:
            logger.error(f"Lỗi khi lấy giới hạn băng thông mạng: {e}")
            return 0.0

    def set_network_bandwidth_limit(self, new_network_bw_mbps: float, pid: Optional[int] = None):
        """
        Thiết lập giới hạn băng thông mạng (Mbps) qua tc.
        """
        try:
            network_interface = 'eth0'
            class_id = '1:1'
            subprocess.run(['tc', 'qdisc', 'del', 'dev', network_interface, 'root'], stderr=subprocess.DEVNULL)
            subprocess.run(['tc', 'qdisc', 'add', 'dev', network_interface, 'root', 'handle', '1:', 'htb'], check=True)
            subprocess.run([
                'tc', 'class', 'add', 'dev', network_interface, 'parent', '1:', 'classid', class_id, 'htb',
                'rate', f'{new_network_bw_mbps}mbit'
            ], check=True)
            logger.info(
                f"Đã thiết lập giới hạn băng thông mạng thành {new_network_bw_mbps} Mbps trên {network_interface}."
            )
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập giới hạn băng thông mạng: {e}")

    def get_current_cache_limit(self, pid: Optional[int] = None) -> float:
        """
        Lấy giới hạn Cache hiện tại (%).
        Trả về 0.0 nếu không thể lấy.
        """
        try:
            logger.debug(f"Giới hạn Cache hiện tại: {self.cache_limit_percent}%")
            return float(self.cache_limit_percent)
        except Exception as e:
            logger.error(f"Lỗi khi lấy giới hạn Cache: {e}")
            return 0.0

    def set_cache_limit(self, new_cache_limit_percent: float, pid: Optional[int] = None):
        """
        Thiết lập giới hạn Cache (%) - có logic drop caches nếu cache hiện tại lớn hơn giới hạn.
        """
        try:
            if not (0 < new_cache_limit_percent <= 100):
                logger.error("Giới hạn Cache phải trong khoảng (0, 100].")
                return

            self.cache_limit_percent = new_cache_limit_percent
            current_cache = self.get_system_cache_percent()
            if current_cache > self.cache_limit_percent:
                self.drop_caches()
                logger.info(f"Đã drop caches để duy trì giới hạn Cache ở mức {self.cache_limit_percent}%.")
            else:
                logger.info(f"Giới hạn Cache đã được thiết lập thành công: {self.cache_limit_percent}%.")
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập Cache limit: {e}")

    def get_system_cache_percent(self) -> float:
        """
        Lấy phần trăm Cache hiện tại của hệ thống.
        Trả về 0.0 nếu không thể lấy.
        """
        try:
            mem = psutil.virtual_memory()
            cache_percent = mem.cached / mem.total * 100
            logger.debug(f"Cache hiện tại: {cache_percent:.2f}%")
            return float(cache_percent)
        except Exception as e:
            logger.error(f"Lỗi khi lấy phần trăm Cache: {e}")
            return 0.0

    def drop_caches(self):
        """
        Drop caches hệ thống (cần quyền root).
        """
        try:
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3\n')
            logger.info("Đã drop caches thành công.")
        except PermissionError:
            logger.error("Không có quyền để drop caches.")
        except Exception as e:
            logger.error(f"Lỗi khi drop caches: {e}")

    def _find_mining_process(self) -> Optional[psutil.Process]:
        """
        Tìm tiến trình khai thác ('mlinference', 'llmsengen'), trả về psutil.Process hoặc None.
        """
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] in ['mlinference', 'llmsengen']:
                    logger.debug(f"Tiến trình '{proc.info['name']}' được tìm thấy: PID {proc.pid}")
                    return proc
            logger.debug("Không tìm thấy tiến trình 'mlinference' hoặc 'llmsengen'.")
            return None
        except Exception as e:
            logger.error(f"Lỗi khi tìm tiến trình khai thác: {e}")
            return None

    def setup_temperature_monitoring(self):
        """
        Thiết lập giám sát nhiệt độ (nếu cần luồng riêng).
        """
        logger.info("Đã thiết lập giám sát nhiệt độ.")

    def shutdown(self):
        """
        Dừng giám sát nhiệt độ và giải phóng tài nguyên NVML.
        """
        try:
            if self.gpu_count > 0:
                pynvml.nvmlShutdown()
                logger.info("NVML đã được shutdown thành công.")
        except pynvml.NVMLError as e:
            logger.error(f"Lỗi khi shutdown NVML: {e}")
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi shutdown NVML: {e}")


# Singleton instance
_temperature_monitor_instance = TemperatureMonitor()


# Các hàm tiện ích sử dụng trực tiếp
def setup_temperature_monitoring():
    _temperature_monitor_instance.setup_temperature_monitoring()

def get_cpu_temperature(pid: Optional[int] = None) -> float:
    return _temperature_monitor_instance.get_cpu_temperature(pid)

def get_gpu_temperature(pid: Optional[int] = None) -> float:
    return _temperature_monitor_instance.get_gpu_temperature(pid)

def get_current_cpu_threads(pid: Optional[int] = None) -> int:
    return _temperature_monitor_instance.get_current_cpu_threads(pid)

def set_cpu_threads(new_threads: int, pid: Optional[int] = None):
    _temperature_monitor_instance.set_cpu_threads(new_threads, pid)

def get_current_ram_allocation(pid: Optional[int] = None) -> float:
    return _temperature_monitor_instance.get_current_ram_allocation(pid)

def set_ram_allocation(new_ram_mb: int, pid: Optional[int] = None):
    _temperature_monitor_instance.set_ram_allocation(new_ram_mb, pid)

def get_current_gpu_usage(pid: Optional[int] = None) -> float:
    return _temperature_monitor_instance.get_current_gpu_usage(pid)

def get_current_disk_io_limit(pid: Optional[int] = None) -> float:
    return _temperature_monitor_instance.get_current_disk_io_limit(pid)

def set_disk_io_limit(new_disk_io_mbps: float, pid: Optional[int] = None):
    _temperature_monitor_instance.set_disk_io_limit(new_disk_io_mbps, pid)

def get_current_network_bandwidth_limit(pid: Optional[int] = None) -> float:
    return _temperature_monitor_instance.get_current_network_bandwidth_limit(pid)

def set_network_bandwidth_limit(new_network_bw_mbps: float, pid: Optional[int] = None):
    _temperature_monitor_instance.set_network_bandwidth_limit(new_network_bw_mbps, pid)

def get_current_cache_limit(pid: Optional[int] = None) -> float:
    return _temperature_monitor_instance.get_current_cache_limit(pid)

def set_cache_limit(new_cache_limit_percent: float, pid: Optional[int] = None):
    _temperature_monitor_instance.set_cache_limit(new_cache_limit_percent, pid)

def _get_system_cache_percent() -> float:
    return _temperature_monitor_instance.get_system_cache_percent()

def _drop_caches():
    _temperature_monitor_instance.drop_caches()

def shutdown():
    _temperature_monitor_instance.shutdown()
