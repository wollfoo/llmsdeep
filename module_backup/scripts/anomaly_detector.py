# anomaly_detector.py

import psutil
import logging
import traceback
import asyncio
from asyncio import Event
from typing import List, Dict, Any
from threading import Lock

import pynvml

from .base_manager import BaseManager
from .utils import MiningProcess
from .auxiliary_modules.power_management import get_cpu_power, get_gpu_power
from .auxiliary_modules.temperature_monitor import get_cpu_temperature, get_gpu_temperature
from .interfaces import IResourceManager

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

class SafeRestoreEvaluator:
    """
    Lớp đánh giá điều kiện an toàn để khôi phục tài nguyên cho tiến trình.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, resource_manager: IResourceManager):
        self.config = config
        self.logger = logger
        self.resource_manager = resource_manager

        # Ngưỡng baseline
        baseline_thresholds = self.config.get('baseline_thresholds', {})
        self.baseline_cpu_usage_percent = baseline_thresholds.get('cpu_usage_percent', 80)
        self.baseline_gpu_usage_percent = baseline_thresholds.get('gpu_usage_percent', 80)
        self.baseline_ram_usage_percent = baseline_thresholds.get('ram_usage_percent', 80)
        self.baseline_disk_io_usage_mbps = baseline_thresholds.get('disk_io_usage_mbps', 80)
        self.baseline_network_usage_mbps = baseline_thresholds.get('network_usage_mbps', 80)

        # Giới hạn nhiệt độ
        temperature_limits = self.config.get("temperature_limits", {})
        self.cpu_max_temp = temperature_limits.get("cpu_max_celsius", 75)
        self.gpu_max_temp = temperature_limits.get("gpu_max_celsius", 75)

        # Giới hạn công suất
        power_limits = self.config.get("power_limits", {})
        per_device_power = power_limits.get("per_device_power_watts", {})
        self.cpu_max_power = per_device_power.get("cpu", 100)
        self.gpu_max_power = per_device_power.get("gpu", 200)

    async def is_safe_to_restore(self, process: MiningProcess) -> bool:
        """
        Kiểm tra các điều kiện an toàn để khôi phục tài nguyên cho tiến trình.
        """
        # 1) Kiểm tra PID tồn tại
        if not psutil.pid_exists(process.pid):
            self.logger.warning(f"Tiến trình PID {process.pid} không tồn tại.")
            return False

        # 2) Kiểm tra nhiệt độ CPU
        try:
            cpu_temp = await asyncio.get_event_loop().run_in_executor(None, get_cpu_temperature, process.pid)
            if cpu_temp is not None and cpu_temp >= self.cpu_max_temp:
                self.logger.info(f"Nhiệt độ CPU {cpu_temp}°C vẫn cao (PID={process.pid}).")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi kiểm tra nhiệt độ CPU PID={process.pid}: {e}\n{traceback.format_exc()}")
            return False

        # 3) Kiểm tra nhiệt độ GPU
        try:
            if not is_nvml_initialized():
                initialize_nvml()

            gpu_temps = await asyncio.get_event_loop().run_in_executor(None, get_gpu_temperature, process.pid)
            if gpu_temps and any(temp >= self.gpu_max_temp for temp in gpu_temps):
                self.logger.info(f"Nhiệt độ GPU {gpu_temps}°C vẫn cao (PID={process.pid}).")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi kiểm tra nhiệt độ GPU PID={process.pid}: {e}\n{traceback.format_exc()}")
            return False

        # 4) Kiểm tra công suất CPU
        try:
            cpu_power = await asyncio.get_event_loop().run_in_executor(None, get_cpu_power, process.pid)
            if cpu_power is not None and cpu_power >= self.cpu_max_power:
                self.logger.info(f"Công suất CPU {cpu_power}W vẫn cao (PID={process.pid}).")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi kiểm tra công suất CPU PID={process.pid}: {e}\n{traceback.format_exc()}")
            return False

        # 5) Kiểm tra công suất GPU
        try:
            if not is_nvml_initialized():
                initialize_nvml()

            gpu_power = await asyncio.get_event_loop().run_in_executor(None, get_gpu_power, process.pid)
            if isinstance(gpu_power, list):
                if sum(gpu_power) >= self.gpu_max_power:
                    self.logger.info(f"Công suất GPU {gpu_power}W vẫn cao (PID={process.pid}).")
                    return False
            else:
                if gpu_power >= self.gpu_max_power:
                    self.logger.info(f"Công suất GPU {gpu_power}W vẫn cao (PID={process.pid}).")
                    return False
        except Exception as e:
            self.logger.error(f"Lỗi kiểm tra công suất GPU PID={process.pid}: {e}\n{traceback.format_exc()}")
            return False

        # 6) Kiểm tra CPU usage tổng thể
        try:
            total_cpu_usage = psutil.cpu_percent(interval=1)
            if total_cpu_usage >= self.baseline_cpu_usage_percent:
                self.logger.info(f"Sử dụng CPU tổng thể {total_cpu_usage}% vẫn cao.")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi kiểm tra CPU tổng thể: {e}\n{traceback.format_exc()}")
            return False

        # 7) Kiểm tra RAM
        try:
            ram = psutil.virtual_memory()
            if ram.percent >= self.baseline_ram_usage_percent:
                self.logger.info(f"Sử dụng RAM tổng thể {ram.percent}% vẫn cao.")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi kiểm tra RAM tổng thể: {e}\n{traceback.format_exc()}")
            return False

        # 8) Kiểm tra Disk I/O
        try:
            disk_io_counters = psutil.disk_io_counters()
            total_disk_io_usage_mbps = (disk_io_counters.read_bytes + disk_io_counters.write_bytes) / (1024 * 1024)
            if total_disk_io_usage_mbps >= self.baseline_disk_io_usage_mbps:
                self.logger.info(f"Sử dụng Disk I/O {total_disk_io_usage_mbps:.2f} MBps vẫn cao.")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi kiểm tra Disk I/O: {e}\n{traceback.format_exc()}")
            return False

        # 9) Kiểm tra mạng
        try:
            net_io_counters = psutil.net_io_counters()
            total_network_usage_mbps = (net_io_counters.bytes_sent + net_io_counters.bytes_recv) / (1024 * 1024)
            if total_network_usage_mbps >= self.baseline_network_usage_mbps:
                self.logger.info(f"Sử dụng mạng {total_network_usage_mbps:.2f} MBps vẫn cao.")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi kiểm tra mạng: {e}\n{traceback.format_exc()}")
            return False

        # 10) Kiểm tra bất thường qua Azure AnomalyDetector
        try:
            current_state = await self.resource_manager.collect_metrics(process)
            anomalies_detected = self.resource_manager.azure_anomaly_detector_client.detect_anomalies(current_state)
            if anomalies_detected:
                self.logger.info(f"Azure Anomaly Detector phát hiện bất thường (PID={process.pid}).")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi qua Azure Anomaly Detector PID={process.pid}: {e}\n{traceback.format_exc()}")
            return False

        # Tất cả điều kiện ok => True
        self.logger.info(f"Đủ điều kiện an toàn để khôi phục cho PID={process.pid}.")
        return True

###############################################################################
#                           LỚP CHÍNH: AnomalyDetector                        #
###############################################################################

class AnomalyDetector(BaseManager):
    """
    Lớp phát hiện bất thường cho tiến trình khai thác.
    Sử dụng event-driven thay vì polling liên tục.
    """
    _instance = None
    _instance_lock = Lock()

    def __new__(cls, config: Dict[str, Any], logger: logging.Logger, resource_manager: IResourceManager):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super(AnomalyDetector, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, resource_manager: IResourceManager):
        BaseManager.__init__(self, config, logger)

        if getattr(self, '_initialized', False):
            return
        self._initialized = True

        self.logger = logger
        self.config = config
        self.resource_manager = resource_manager

        self.stop_event = Event()
        self.mining_processes: List[MiningProcess] = []
        self.mining_processes_lock = asyncio.Lock()

        # Lớp hỗ trợ đánh giá khi nào an toàn để khôi phục
        self.safe_restore_evaluator = SafeRestoreEvaluator(config, logger, resource_manager)

        # Danh sách các task chạy ngầm
        self.task_futures = []
        loop = asyncio.get_event_loop()
        self.task_futures.append(loop.create_task(self.anomaly_detection()))
        self.task_futures.append(loop.create_task(self.monitor_restoration()))

        self.logger.info("AnomalyDetector đã được khởi tạo với ResourceManager.")

    async def start(self):
        """
        Khởi động AnomalyDetector. Gọi discover_mining_processes_async() ban đầu.
        """
        self.logger.info("Đang khởi động AnomalyDetector...")
        await self.discover_mining_processes_async()
        self.logger.info("AnomalyDetector khởi động thành công.")

    async def discover_mining_processes_async(self):
        """
        Tìm các tiến trình khai thác dựa trên config (CPU/GPU).
        """
        cpu_name = self.config['processes'].get('CPU', '').lower()
        gpu_name = self.config['processes'].get('GPU', '').lower()

        async with self.mining_processes_lock:
            self.mining_processes.clear()
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_name = proc.info['name'].lower()
                    if cpu_name in proc_name or gpu_name in proc_name:
                        priority = self.get_process_priority(proc.info['name'])
                        net_if = self.config.get('network_interface', 'eth0')
                        mp = MiningProcess(proc.info['pid'], proc.info['name'], priority, net_if, self.logger)
                        mp.is_cloaked = False
                        self.mining_processes.append(mp)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            self.logger.info(f"Đã phát hiện {len(self.mining_processes)} tiến trình khai thác.")

    def get_process_priority(self, process_name: str) -> int:
        """Lấy độ ưu tiên của tiến trình dựa trên config."""
        priority_map = self.config.get('process_priority_map', {})
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
        Coroutine chính để phát hiện bất thường cho các tiến trình.
        """
        detection_interval = self.config.get("monitoring_parameters", {}).get("detection_interval_seconds", 3600)
        cloak_delay = self.config.get("monitoring_parameters", {}).get("cloak_activation_delay_seconds", 5)

        while not self.stop_event.is_set():
            try:
                await self.discover_mining_processes_async()

                if not self.resource_manager:
                    self.logger.error("ResourceManager chưa được thiết lập.")
                else:
                    async with self.mining_processes_lock:
                        processes = list(self.mining_processes)

                    tasks = [
                        self.evaluate_process_anomaly(proc, cloak_delay)
                        for proc in processes
                    ]
                    await asyncio.gather(*tasks)

            except Exception as e:
                self.logger.error(f"Lỗi anomaly_detection: {e}\n{traceback.format_exc()}")

            # Nghỉ cho đến lần quét tiếp theo
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
            anomalies_detected = self.resource_manager.azure_anomaly_detector_client.detect_anomalies(current_state)
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
        """Dừng AnomalyDetector, hủy các task ngầm."""
        self.logger.info("Đang dừng AnomalyDetector...")
        self.stop_event.set()

        for t in self.task_futures:
            t.cancel()
        await asyncio.gather(*self.task_futures, return_exceptions=True)

        self.logger.info("AnomalyDetector đã dừng thành công.")
