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
    TemperatureMonitor là một singleton class chịu trách nhiệm giám sát nhiệt độ của CPU và GPU,
    cũng như quản lý các tài nguyên liên quan đến tiến trình khai thác.
    """
    _instance = None
    _lock = asyncio.Lock()

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        # Quản lý pynvml
        self._nvml_initialized = False
        self.gpu_count = 0  # Số lượng GPU, sẽ được cập nhật sau khi khởi tạo pynvml

        # Cache limit percentage (có thể được cập nhật qua set_cache_limit)
        self.cache_limit_percent = 70.0

    @classmethod
    async def create(cls) -> 'TemperatureMonitor':
        """
        Async factory method để tạo và khởi tạo instance của TemperatureMonitor.
        Đảm bảo rằng chỉ có một instance được tạo ra (singleton).
        """
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                await cls._instance.initialize()
            return cls._instance

    async def initialize(self):
        """
        Phương thức async để khởi tạo các thuộc tính bất đồng bộ nếu cần thiết.
        Trong trường hợp này, khởi tạo pynvml.
        """
        if not self._nvml_initialized:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, pynvml.nvmlInit)
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self._nvml_initialized = True
                logger.info(f"TemperatureMonitor: Đã khởi tạo pynvml thành công với {self.gpu_count} GPU.")
            except pynvml.NVMLError as e:
                logger.error(f"TemperatureMonitor: Lỗi khi khởi tạo pynvml: {e}")
                self.gpu_count = 0
        else:
            logger.debug("TemperatureMonitor: pynvml đã được khởi tạo trước đó.")

    def _ensure_nvml_initialized(self):
        """
        Đảm bảo rằng pynvml đã được khởi tạo trước khi sử dụng.
        Nếu chưa, thực hiện khởi tạo.
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

    async def get_cpu_temperature(self, pid: Optional[int] = None) -> float:
        """
        Lấy nhiệt độ hiện tại của CPU. 
        Trả về 0.0 nếu không thể lấy.

        Args:
            pid (Optional[int]): PID của tiến trình (không sử dụng).

        Returns:
            float: Nhiệt độ CPU trung bình (°C).
        """
        try:
            loop = asyncio.get_event_loop()
            temps = await loop.run_in_executor(None, psutil.sensors_temperatures)
            if not temps:
                logger.warning("TemperatureMonitor: Không tìm thấy cảm biến nhiệt độ CPU.")
                return 0.0

            # Tìm kiếm cảm biến CPU
            for name, entries in temps.items():
                if 'coretemp' in name.lower() or 'cpu' in name.lower():
                    cpu_temps = [entry.current for entry in entries if 'core' in entry.label.lower()]
                    if cpu_temps:
                        avg_temp = sum(cpu_temps) / len(cpu_temps)
                        logger.debug(f"TemperatureMonitor: Nhiệt độ CPU trung bình: {avg_temp}°C")
                        return float(avg_temp)
            logger.warning("TemperatureMonitor: Không tìm thấy nhãn nhiệt độ CPU phù hợp.")
            return 0.0
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy nhiệt độ CPU: {e}")
            return 0.0

    async def get_gpu_temperature(self, pid: Optional[int] = None) -> float:
        """
        Lấy nhiệt độ hiện tại của từng GPU và trả về nhiệt độ trung bình.
        Trả về 0.0 nếu không có GPU hoặc gặp lỗi.

        Args:
            pid (Optional[int]): PID của tiến trình (không sử dụng).

        Returns:
            float: Nhiệt độ GPU trung bình (°C).
        """
        if self.gpu_count == 0:
            logger.warning("TemperatureMonitor: Không có GPU nào được phát hiện để giám sát nhiệt độ.")
            return 0.0

        gpu_temps = []
        try:
            self._ensure_nvml_initialized()
            if not self._nvml_initialized:
                return 0.0

            loop = asyncio.get_event_loop()
            for i in range(self.gpu_count):
                temp = await loop.run_in_executor(None, self._get_single_gpu_temperature, i)
                if temp is not None:
                    gpu_temps.append(temp)
                    logger.debug(f"TemperatureMonitor: GPU {i} Temperature: {temp}°C")
                else:
                    gpu_temps.append(0.0)
            if gpu_temps:
                avg_temp = sum(gpu_temps) / len(gpu_temps)
                logger.debug(f"TemperatureMonitor: Nhiệt độ GPU trung bình: {avg_temp}°C")
                return float(avg_temp)
            else:
                logger.warning("TemperatureMonitor: Không có dữ liệu nhiệt độ GPU để tính trung bình.")
                return 0.0
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy nhiệt độ GPU: {e}")
            return 0.0

    def _get_single_gpu_temperature(self, gpu_index: int) -> Optional[float]:
        """
        Lấy nhiệt độ của một GPU cụ thể.

        Args:
            gpu_index (int): Chỉ số GPU.

        Returns:
            Optional[float]: Nhiệt độ GPU (°C) hoặc None nếu lỗi.
        """
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            return float(temp)
        except pynvml.NVMLError as e:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy nhiệt độ GPU {gpu_index}: {e}")
            return None

    async def get_current_cpu_threads(self, pid: Optional[int] = None) -> int:
        """
        Lấy số lượng CPU threads hiện tại được gán cho tiến trình (nếu pid=None, tìm mining process).

        Args:
            pid (Optional[int]): PID của tiến trình (không sử dụng).

        Returns:
            int: Số lượng CPU threads.
        """
        try:
            loop = asyncio.get_event_loop()
            proc = await loop.run_in_executor(None, self._find_mining_process)
            if proc:
                affinity = await loop.run_in_executor(None, proc.cpu_affinity)
                num_threads = len(affinity)
                logger.debug(
                    f"TemperatureMonitor: Số lượng CPU threads cho tiến trình '{proc.name()}' (PID {proc.pid}): {num_threads}"
                )
                return num_threads
            else:
                logger.warning("TemperatureMonitor: Không tìm thấy tiến trình khai thác.")
                return 0
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy số lượng CPU threads: {e}")
            return 0

    async def set_cpu_threads(self, new_threads: int, pid: Optional[int] = None):
        """
        Gán số lượng CPU threads cho tiến trình (nếu pid=None, tìm mining process).

        Args:
            new_threads (int): Số lượng threads mới.
            pid (Optional[int]): PID của tiến trình (không sử dụng).
        """
        try:
            loop = asyncio.get_event_loop()
            proc = await loop.run_in_executor(None, self._find_mining_process)
            if proc:
                total_cores = psutil.cpu_count(logical=True)
                if new_threads > total_cores:
                    logger.warning(
                        f"TemperatureMonitor: Số threads mới ({new_threads}) vượt quá số lõi CPU ({total_cores}). "
                        f"Đặt lại thành {total_cores}."
                    )
                    new_threads = total_cores

                new_affinity = list(range(new_threads))
                await loop.run_in_executor(None, proc.cpu_affinity, new_affinity)
                logger.info(
                    f"TemperatureMonitor: Đã gán tiến trình '{proc.name()}' (PID {proc.pid}) vào CPU cores: {new_affinity}"
                )
            else:
                logger.warning("TemperatureMonitor: Không tìm thấy tiến trình khai thác để gán CPU threads.")
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi gán CPU threads: {e}")

    async def get_current_ram_allocation(self, pid: Optional[int] = None) -> float:
        """
        Lấy lượng RAM hiện tại (MB) cho tiến trình.
        Trả về 0.0 nếu không thể lấy.

        Args:
            pid (Optional[int]): PID của tiến trình (không sử dụng).

        Returns:
            float: Lượng RAM hiện tại (MB).
        """
        try:
            loop = asyncio.get_event_loop()
            proc = await loop.run_in_executor(None, self._find_mining_process)
            if proc:
                mem_info = await loop.run_in_executor(None, proc.memory_info)
                ram_mb = mem_info.rss / (1024 * 1024)
                logger.debug(f"TemperatureMonitor: Lượng RAM hiện tại cho tiến trình '{proc.name()}': {ram_mb} MB")
                return float(ram_mb)
            else:
                logger.warning("TemperatureMonitor: Không tìm thấy tiến trình khai thác để lấy RAM allocation.")
                return 0.0
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy RAM allocation: {e}")
            return 0.0

    async def set_ram_allocation(self, new_ram_mb: int, pid: Optional[int] = None):
        """
        Thiết lập giới hạn RAM (MB) cho tiến trình, giả lập bằng cgroups (nếu có quyền).

        Args:
            new_ram_mb (int): Lượng RAM mới (MB).
            pid (Optional[int]): PID của tiến trình (không sử dụng).
        """
        try:
            loop = asyncio.get_event_loop()
            proc = await loop.run_in_executor(None, self._find_mining_process)
            if proc:
                cgroup_dir = Path(f"/sys/fs/cgroup/memory/temperature_monitor/{proc.pid}")
                cgroup_dir.mkdir(parents=True, exist_ok=True)
                cgroup_path = cgroup_dir / 'memory.limit_in_bytes'
                new_limit_bytes = new_ram_mb * 1024 * 1024
                await loop.run_in_executor(None, self._write_to_file, cgroup_path, str(new_limit_bytes))
                logger.info(
                    f"TemperatureMonitor: Đã thiết lập giới hạn RAM cho tiến trình '{proc.name()}' thành {new_ram_mb} MB."
                )
            else:
                logger.warning("TemperatureMonitor: Không tìm thấy tiến trình khai thác để thiết lập RAM allocation.")
        except PermissionError:
            logger.error("TemperatureMonitor: Không có quyền để thiết lập RAM allocation qua cgroup.")
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi thiết lập RAM allocation: {e}")

    async def get_current_gpu_usage(self, pid: Optional[int] = None) -> float:
        """
        Lấy mức sử dụng GPU hiện tại (%) cho tiến trình pid hoặc tổng thể.
        Trả về 0.0 nếu không có GPU hoặc gặp lỗi.

        Args:
            pid (Optional[int]): PID của tiến trình (không sử dụng).

        Returns:
            float: Mức sử dụng GPU trung bình (%).
        """
        # Giả định rằng việc lấy mức sử dụng GPU là nhiệt độ GPU
        return await self.get_gpu_temperature(pid)

    async def get_current_disk_io_limit(self, pid: Optional[int] = None) -> float:
        """
        Lấy giới hạn Disk I/O hiện tại (Mbps) giả lập thông qua cgroup.
        Trả về 0.0 nếu không thể lấy.

        Args:
            pid (Optional[int]): PID của tiến trình (không sử dụng).

        Returns:
            float: Giới hạn Disk I/O (Mbps).
        """
        return await self._get_current_disk_io_limit(pid)

    async def _get_current_disk_io_limit(self, pid: Optional[int] = None) -> float:
        """
        Hàm nội bộ để lấy giới hạn Disk I/O, đảm bảo luôn trả về float.

        Args:
            pid (Optional[int]): PID của tiến trình (không sử dụng).

        Returns:
            float: Giới hạn Disk I/O (Mbps).
        """
        try:
            loop = asyncio.get_event_loop()
            if pid:
                cgroup_path_read = f"/sys/fs/cgroup/blkio/temperature_monitor/{pid}/blkio.throttle.read_bps_device"
                cgroup_path_write = f"/sys/fs/cgroup/blkio/temperature_monitor/{pid}/blkio.throttle.write_bps_device"
            else:
                cgroup_path_read = None
                cgroup_path_write = None

            read_limit = 0.0
            write_limit = 0.0

            if cgroup_path_read and Path(cgroup_path_read).exists():
                content = await loop.run_in_executor(None, Path(cgroup_path_read).read_text)
                parts = content.strip().split()
                # parts = ["8:0", "1048576"]
                if len(parts) == 2 and parts[1].isdigit():
                    read_limit = int(parts[1]) * 8 / (1024 * 1024)  # B/s -> Mbps
                    logger.debug(f"TemperatureMonitor: Disk I/O limit Read: {read_limit} Mbps")

            if cgroup_path_write and Path(cgroup_path_write).exists():
                content = await loop.run_in_executor(None, Path(cgroup_path_write).read_text)
                parts = content.strip().split()
                if len(parts) == 2 and parts[1].isdigit():
                    write_limit = int(parts[1]) * 8 / (1024 * 1024)  # B/s -> Mbps
                    logger.debug(f"TemperatureMonitor: Disk I/O limit Write: {write_limit} Mbps")

            if read_limit > 0.0 and write_limit > 0.0:
                disk_io_limit = max(read_limit, write_limit)
                logger.debug(f"TemperatureMonitor: Disk I/O limit (cgroup): Read={read_limit}Mbps, Write={write_limit}Mbps. Sử dụng max: {disk_io_limit}Mbps")
                return disk_io_limit
            elif read_limit > 0.0:
                logger.debug(f"TemperatureMonitor: Disk I/O limit (cgroup) chỉ giới hạn Read={read_limit}Mbps")
                return read_limit
            elif write_limit > 0.0:
                logger.debug(f"TemperatureMonitor: Disk I/O limit (cgroup) chỉ giới hạn Write={write_limit}Mbps")
                return write_limit
            else:
                logger.warning("TemperatureMonitor: Không thể lấy giới hạn Disk I/O thông qua cgroup blkio. Gán giá trị mặc định 0.0 Mbps.")
                return 0.0
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy giới hạn Disk I/O: {e}")
            return 0.0

    async def set_disk_io_limit(self, new_disk_io_mbps: float, pid: Optional[int] = None):
        """
        Thiết lập giới hạn Disk I/O (Mbps) cho tiến trình qua cgroup blkio (giả lập).

        Args:
            new_disk_io_mbps (float): Giới hạn Disk I/O mới (Mbps).
            pid (Optional[int]): PID của tiến trình (không sử dụng).
        """
        try:
            loop = asyncio.get_event_loop()
            proc = await loop.run_in_executor(None, self._find_mining_process)
            if proc:
                cgroup_dir = Path(f"/sys/fs/cgroup/blkio/temperature_monitor/{proc.pid}")
                cgroup_dir.mkdir(parents=True, exist_ok=True)

                device_major = 8
                device_minor = 0
                bps_limit = int(new_disk_io_mbps * 1024 * 1024 / 8)

                cgroup_path_read = cgroup_dir / 'blkio.throttle.read_bps_device'
                cgroup_path_write = cgroup_dir / 'blkio.throttle.write_bps_device'

                await loop.run_in_executor(None, self._write_to_file, cgroup_path_read, f"{device_major}:{device_minor} {bps_limit}\n")
                await loop.run_in_executor(None, self._write_to_file, cgroup_path_write, f"{device_major}:{device_minor} {bps_limit}\n")

                logger.info(
                    f"TemperatureMonitor: Đã thiết lập giới hạn Disk I/O cho tiến trình '{proc.name()}' thành {new_disk_io_mbps} Mbps."
                )
            else:
                logger.warning("TemperatureMonitor: Không tìm thấy tiến trình khai thác để thiết lập Disk I/O limit.")
        except PermissionError:
            logger.error("TemperatureMonitor: Không có quyền cgroup để thiết lập Disk I/O limit.")
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi thiết lập Disk I/O limit: {e}")

    async def get_current_network_bandwidth_limit(self, pid: Optional[int] = None) -> float:
        """
        Lấy giới hạn băng thông mạng (Mbps) giả lập qua tc.
        Trả về 0.0 nếu không thể lấy.

        Args:
            pid (Optional[int]): PID của tiến trình (không sử dụng).

        Returns:
            float: Giới hạn băng thông mạng (Mbps).
        """
        try:
            loop = asyncio.get_event_loop()
            network_interface = 'eth0'
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.check_output(
                    ['tc', 'class', 'show', 'dev', network_interface],
                    stderr=subprocess.STDOUT
                )
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
                        logger.debug(f"TemperatureMonitor: Network bandwidth limit current: {bw} Mbps")
                        return bw
            logger.warning("TemperatureMonitor: Không tìm thấy giới hạn băng thông mạng. Gán giá trị mặc định 0.0 Mbps.")
            return 0.0
        except subprocess.CalledProcessError as e:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy giới hạn băng thông mạng: {e.output.decode().strip()}")
            return 0.0
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy giới hạn băng thông mạng: {e}")
            return 0.0

    async def set_network_bandwidth_limit(self, new_network_bw_mbps: float, pid: Optional[int] = None):
        """
        Thiết lập giới hạn băng thông mạng (Mbps) qua tc.

        Args:
            new_network_bw_mbps (float): Giới hạn băng thông mạng mới (Mbps).
            pid (Optional[int]): PID của tiến trình (không sử dụng).
        """
        try:
            loop = asyncio.get_event_loop()
            network_interface = 'eth0'
            class_id = '1:1'
            # Xóa qdisc hiện tại nếu có
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(['tc', 'qdisc', 'del', 'dev', network_interface, 'root'], stderr=subprocess.DEVNULL)
            )
            # Thêm qdisc mới
            await loop.run_in_executor(None, lambda: subprocess.run(['tc', 'qdisc', 'add', 'dev', network_interface, 'root', 'handle', '1:', 'htb'], check=True))
            # Thêm class mới với rate
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [
                        'tc', 'class', 'add', 'dev', network_interface, 'parent', '1:', 'classid', class_id, 'htb',
                        'rate', f'{new_network_bw_mbps}mbit'
                    ],
                    check=True
                )
            )
            logger.info(
                f"TemperatureMonitor: Đã thiết lập giới hạn băng thông mạng thành {new_network_bw_mbps} Mbps trên {network_interface}."
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"TemperatureMonitor: Lỗi khi thiết lập giới hạn băng thông mạng: {e}")
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi thiết lập giới hạn băng thông mạng: {e}")

    async def get_current_cache_limit(self, pid: Optional[int] = None) -> float:
        """
        Lấy giới hạn Cache hiện tại (%).
        Trả về 0.0 nếu không thể lấy.

        Args:
            pid (Optional[int]): PID của tiến trình (không sử dụng).

        Returns:
            float: Giới hạn Cache (%).
        """
        try:
            logger.debug(f"TemperatureMonitor: Giới hạn Cache hiện tại: {self.cache_limit_percent}%")
            return float(self.cache_limit_percent)
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy giới hạn Cache: {e}")
            return 0.0

    async def set_cache_limit(self, new_cache_limit_percent: float, pid: Optional[int] = None):
        """
        Thiết lập giới hạn Cache (%) - có logic drop caches nếu cache hiện tại lớn hơn giới hạn.

        Args:
            new_cache_limit_percent (float): Giới hạn Cache mới (%).
            pid (Optional[int]): PID của tiến trình (không sử dụng).
        """
        try:
            if not (0 < new_cache_limit_percent <= 100):
                logger.error("TemperatureMonitor: Giới hạn Cache phải trong khoảng (0, 100].")
                return

            self.cache_limit_percent = new_cache_limit_percent
            current_cache = await self.get_system_cache_percent()
            if current_cache > self.cache_limit_percent:
                await self.drop_caches()
                logger.info(f"TemperatureMonitor: Đã drop caches để duy trì giới hạn Cache ở mức {self.cache_limit_percent}%.")
            else:
                logger.info(f"TemperatureMonitor: Giới hạn Cache đã được thiết lập thành công: {self.cache_limit_percent}%.")
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi thiết lập Cache limit: {e}")

    async def get_system_cache_percent(self) -> float:
        """
        Lấy phần trăm Cache hiện tại của hệ thống.
        Trả về 0.0 nếu không thể lấy.

        Returns:
            float: Phần trăm Cache hiện tại.
        """
        try:
            loop = asyncio.get_event_loop()
            mem = await loop.run_in_executor(None, psutil.virtual_memory)
            cache_percent = mem.cached / mem.total * 100
            logger.debug(f"TemperatureMonitor: Cache hiện tại: {cache_percent:.2f}%")
            return float(cache_percent)
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi lấy phần trăm Cache: {e}")
            return 0.0

    async def drop_caches(self):
        """
        Drop caches hệ thống (cần quyền root).
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._write_to_file, '/proc/sys/vm/drop_caches', '3\n')
            logger.info("TemperatureMonitor: Đã drop caches thành công.")
        except PermissionError:
            logger.error("TemperatureMonitor: Không có quyền để drop caches.")
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi drop caches: {e}")

    def _find_mining_process(self) -> Optional[psutil.Process]:
        """
        Tìm tiến trình khai thác ('mlinference', 'llmsengen'), trả về psutil.Process hoặc None.

        Returns:
            Optional[psutil.Process]: Tiến trình khai thác hoặc None.
        """
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] in ['ml-inference', 'inference-cuda']:
                    logger.debug(f"TemperatureMonitor: Tiến trình '{proc.info['name']}' được tìm thấy: PID {proc.pid}")
                    return proc
            logger.debug("TemperatureMonitor: Không tìm thấy tiến trình 'mlinference' hoặc 'llmsengen'.")
            return None
        except Exception as e:
            logger.error(f"TemperatureMonitor: Lỗi khi tìm tiến trình khai thác: {e}")
            return None

    async def setup_temperature_monitoring(self):
        """
        Thiết lập giám sát nhiệt độ (nếu cần luồng riêng).
        """
        logger.info("TemperatureMonitor: Đã thiết lập giám sát nhiệt độ.")

    async def shutdown(self):
        """
        Dừng giám sát nhiệt độ và giải phóng tài nguyên GPU.
        """
        try:
            if self._nvml_initialized:
                pynvml.nvmlShutdown()
                self._nvml_initialized = False
                logger.info("TemperatureMonitor: Đã shutdown thành công pynvml.")
        except pynvml.NVMLError as e:
            logger.error(f"TemperatureMonitor: Lỗi khi shutdown pynvml: {e}")

    @staticmethod
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
