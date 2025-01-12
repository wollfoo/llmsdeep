# anomaly_detector.py

import psutil
import logging
import traceback
import pynvml
import asyncio
from asyncio import Lock
from typing import List, Dict, Any, Optional
from time import time

from .base_manager import BaseManager
from .utils import MiningProcess, GPUManager
from .auxiliary_modules.power_management import get_cpu_power, get_gpu_power
from .auxiliary_modules.temperature_monitor import get_cpu_temperature, get_gpu_temperature

# Import Interface
from .interfaces import IResourceManager
from .resource_manager import acquire_lock_with_timeout

from threading import Lock as ThreadLock, Event as ThreadEvent

class SafeRestoreEvaluator:
    """
    Lớp đánh giá điều kiện an toàn để khôi phục tài nguyên cho các tiến trình.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, resource_manager: IResourceManager):
        self.config = config
        self.logger = logger
        self.resource_manager = resource_manager

        baseline_thresholds = self.config.get('baseline_thresholds', {})
        self.baseline_cpu_usage_percent = baseline_thresholds.get('cpu_usage_percent', 80)
        self.baseline_gpu_usage_percent = baseline_thresholds.get('gpu_usage_percent', 80)
        self.baseline_ram_usage_percent = baseline_thresholds.get('ram_usage_percent', 80)
        self.baseline_disk_io_usage_mbps = baseline_thresholds.get('disk_io_usage_mbps', 80)
        self.baseline_network_usage_mbps = baseline_thresholds.get('network_usage_mbps', 80)

        temperature_limits = self.config.get("temperature_limits", {})
        self.cpu_max_temp = temperature_limits.get("cpu_max_celsius", 75)
        self.gpu_max_temp = temperature_limits.get("gpu_max_celsius", 75)

        power_limits = self.config.get("power_limits", {})
        per_device_power = power_limits.get("per_device_power_watts", {})
        self.cpu_max_power = per_device_power.get("cpu", 100)
        self.gpu_max_power = per_device_power.get("gpu", 200)

    async def is_safe_to_restore(self, process: MiningProcess) -> bool:
        """
        Kiểm tra xem điều kiện có đủ an toàn để khôi phục tài nguyên cho tiến trình hay không.
        """
        if not psutil.pid_exists(process.pid):
            self.logger.warning(f"Tiến trình PID {process.pid} không tồn tại.")
            return False

        # 1) Kiểm tra nhiệt độ CPU
        try:
            cpu_temp = await asyncio.get_event_loop().run_in_executor(None, get_cpu_temperature, process.pid)
            if cpu_temp is not None and cpu_temp >= self.cpu_max_temp:
                self.logger.info(
                    f"Nhiệt độ CPU {cpu_temp}°C vẫn cao cho tiến trình {process.name} (PID={process.pid})."
                )
                return False
        except Exception as e:
            self.logger.error(
                f"Lỗi khi kiểm tra nhiệt độ CPU cho PID={process.pid}: {e}\n{traceback.format_exc()}"
            )
            return False

        # 2) Kiểm tra nhiệt độ GPU
        try:
            if self.resource_manager.is_gpu_initialized():
                gpu_temps = await asyncio.get_event_loop().run_in_executor(None, get_gpu_temperature, process.pid)
                if gpu_temps and any(temp >= self.gpu_max_temp for temp in gpu_temps):
                    self.logger.info(
                        f"Nhiệt độ GPU {gpu_temps}°C vẫn cao cho tiến trình {process.name} (PID={process.pid})."
                    )
                    return False
        except Exception as e:
            self.logger.error(
                f"Lỗi khi kiểm tra nhiệt độ GPU cho PID={process.pid}: {e}\n{traceback.format_exc()}"
            )
            return False

        # 3) Kiểm tra công suất CPU
        try:
            cpu_power = await asyncio.get_event_loop().run_in_executor(None, get_cpu_power, process.pid)
            if cpu_power is not None and cpu_power >= self.cpu_max_power:
                self.logger.info(
                    f"Công suất CPU {cpu_power}W vẫn cao cho tiến trình {process.name} (PID={process.pid})."
                )
                return False
        except Exception as e:
            self.logger.error(
                f"Lỗi khi kiểm tra công suất CPU cho PID={process.pid}: {e}\n{traceback.format_exc()}"
            )
            return False

        # 4) Kiểm tra công suất GPU
        try:
            if self.resource_manager.is_gpu_initialized():
                gpu_power = await asyncio.get_event_loop().run_in_executor(None, get_gpu_power, process.pid)
                # gpu_power có thể là list => lấy tổng
                if isinstance(gpu_power, list):
                    gpu_power = sum(gpu_power)
                if gpu_power >= self.gpu_max_power:
                    self.logger.info(
                        f"Công suất GPU {gpu_power}W vẫn cao cho tiến trình {process.name} (PID={process.pid})."
                    )
                    return False
        except Exception as e:
            self.logger.error(
                f"Lỗi khi kiểm tra công suất GPU cho PID={process.pid}: {e}\n{traceback.format_exc()}"
            )
            return False

        # 5) Kiểm tra sử dụng CPU tổng thể
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

        # 6) Kiểm tra sử dụng RAM tổng thể
        try:
            ram = psutil.virtual_memory()
            if ram.percent >= self.baseline_ram_usage_percent:
                self.logger.info(f"Sử dụng RAM tổng thể {ram.percent}% vẫn cao.")
                return False
        except Exception as e:
            self.logger.error(
                f"Lỗi khi kiểm tra sử dụng RAM tổng thể: {e}\n{traceback.format_exc()}"
            )
            return False

        # 7) Kiểm tra Disk I/O
        try:
            disk_io_counters = psutil.disk_io_counters()
            total_disk_io_usage_mbps = (
                (disk_io_counters.read_bytes + disk_io_counters.write_bytes) / (1024 * 1024)
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

        # 8) Kiểm tra băng thông mạng
        try:
            net_io_counters = psutil.net_io_counters()
            total_network_usage_mbps = (
                (net_io_counters.bytes_sent + net_io_counters.bytes_recv) / (1024 * 1024)
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

        # 9) Kiểm tra anomaly qua Azure Anomaly Detector (nếu logic config sẵn)
        try:
            current_state = await self.resource_manager.collect_metrics(process)
            anomalies_detected = self.resource_manager.azure_anomaly_detector_client.detect_anomalies(current_state)
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

        self.logger.info(
            f"Điều kiện an toàn để khôi phục tài nguyên cho tiến trình {process.name} (PID: {process.pid})."
        )
        return True


class AnomalyDetector(BaseManager):
    """
    Lớp phát hiện bất thường cho các tiến trình khai thác, sử dụng Event-Driven
    thay vì loop-based polling liên tục.
    """
    _instance = None
    _instance_lock = ThreadLock()

    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super(AnomalyDetector, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        BaseManager.__init__(self, config, logger)
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        self.logger = logger
        self.config = config

        self.stop_event = ThreadEvent()

        self.mining_processes: List[MiningProcess] = []
        self.mining_processes_lock = Lock()

        self.gpu_manager = GPUManager()
        self.gpu_initialized = self.gpu_manager.gpu_initialized
        if self.gpu_initialized:
            self.logger.info("GPUManager đã được khởi tạo thành công.")
        else:
            self.logger.warning("GPUManager không được khởi tạo. Các chức năng liên quan đến GPU sẽ bị vô hiệu hóa.")

        self.resource_manager: Optional[IResourceManager] = None
        self.safe_restore_evaluator: Optional[SafeRestoreEvaluator] = None

        # Khởi tạo tasks (vòng lặp phát hiện anomaly và phục hồi)
        loop = asyncio.get_event_loop()
        self.task_anomaly_check = loop.create_task(self.anomaly_check_loop())
        self.task_restoration_check = loop.create_task(self.restoration_check_loop())

    def set_resource_manager(self, resource_manager: IResourceManager):
        """
        Thiết lập ResourceManager cho AnomalyDetector.
        """
        self.resource_manager = resource_manager
        self.logger.info("ResourceManager đã được thiết lập cho AnomalyDetector.")

        self.safe_restore_evaluator = SafeRestoreEvaluator(
            self.config, self.logger, self.resource_manager
        )

    async def discover_mining_processes_async(self):
        """
        Khám phá các tiến trình khai thác.
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
        """
        priority_map = self.config.get('process_priority_map', {})
        priority = priority_map.get(process_name.lower(), 1)
        if not isinstance(priority, int):
            self.logger.warning(
                f"Độ ưu tiên cho tiến trình '{process_name}' không phải là int. Sử dụng mặc định = 1. priority={priority}"
            )
            priority = 1
        return priority

    # ------------------- MỚI: Tách riêng check_temperature_and_power() ------------------- #
    async def check_temperature_and_power(self, process: MiningProcess, cloak_activation_delay: int) -> bool:
        """
        Kiểm tra nhiệt độ & công suất CPU/GPU. Nếu vượt ngưỡng => phát event cloaking.
        Trả về True nếu đã cloaking, ngược lại False.
        """
        if process.is_cloaked:
            return False  # Đã cloak rồi thì bỏ qua

        temperature_limits = self.config.get("temperature_limits", {})
        cpu_max_temp = temperature_limits.get("cpu_max_celsius", 75)
        gpu_max_temp = temperature_limits.get("gpu_max_celsius", 75)

        power_limits = self.config.get("power_limits", {})
        per_device_power = power_limits.get("per_device_power_watts", {})
        cpu_max_power = per_device_power.get("cpu", 100)
        gpu_max_power = per_device_power.get("gpu", 200)

        try:
            # 1) Nhiệt độ CPU
            cpu_temp = await asyncio.get_event_loop().run_in_executor(None, get_cpu_temperature, process.pid)
            if cpu_temp is not None and cpu_temp > cpu_max_temp:
                self.logger.warning(
                    f"Nhiệt độ CPU {cpu_temp}°C > {cpu_max_temp}°C => Cloaking (PID={process.pid})."
                )
                await asyncio.sleep(cloak_activation_delay)
                await self.resource_manager.emit_cloaking_event(process)
                return True

            # 2) Nhiệt độ GPU
            if self.resource_manager and self.resource_manager.is_gpu_initialized():
                gpu_temps = await asyncio.get_event_loop().run_in_executor(None, get_gpu_temperature, process.pid)
                if gpu_temps and any(t > gpu_max_temp for t in gpu_temps):
                    self.logger.warning(
                        f"Nhiệt độ GPU {gpu_temps}°C > {gpu_max_temp}°C => Cloaking (PID={process.pid})."
                    )
                    await asyncio.sleep(cloak_activation_delay)
                    await self.resource_manager.emit_cloaking_event(process)
                    return True

            # 3) Công suất CPU
            cpu_power = await asyncio.get_event_loop().run_in_executor(None, get_cpu_power, process.pid)
            if cpu_power and cpu_power > cpu_max_power:
                self.logger.warning(
                    f"CPU Power {cpu_power}W > {cpu_max_power}W => Cloaking (PID={process.pid})."
                )
                await asyncio.sleep(cloak_activation_delay)
                await self.resource_manager.emit_cloaking_event(process)
                return True

            # 4) Công suất GPU
            if self.resource_manager and self.resource_manager.is_gpu_initialized():
                gpu_power_val = await asyncio.get_event_loop().run_in_executor(None, get_gpu_power, process.pid)
                if isinstance(gpu_power_val, list):
                    gpu_power_val = sum(gpu_power_val)
                if gpu_power_val and gpu_power_val > gpu_max_power:
                    self.logger.warning(
                        f"GPU Power {gpu_power_val}W > {gpu_max_power}W => Cloaking (PID={process.pid})."
                    )
                    await asyncio.sleep(cloak_activation_delay)
                    await self.resource_manager.emit_cloaking_event(process)
                    return True

        except psutil.NoSuchProcess:
            self.logger.warning(f"Tiến trình PID={process.pid} không còn tồn tại.")
        except Exception as e:
            self.logger.error(f"check_temperature_and_power() bị lỗi: {e}\n{traceback.format_exc()}")

        return False  # Chưa cloak

    # ------------------- Event-Driven: Phát hiện anomaly ------------------- #
    async def anomaly_check_loop(self):
        """
        Định kỳ quét, gọi check_temperature_and_power() trước,
        sau đó nếu chưa cloak => gọi evaluate_process_anomaly().
        """
        detection_interval = self.config.get("monitoring_parameters", {}).get("detection_interval_seconds", 360)
        cloak_activation_delay = self.config.get("monitoring_parameters", {}).get("cloak_activation_delay_seconds", 5)
        last_detection_time = 0

        while not self.stop_event.is_set():
            current_time = time()
            if current_time - last_detection_time >= detection_interval:
                try:
                    # Quét danh sách process
                    await self.discover_mining_processes_async()

                    if self.resource_manager is None:
                        self.logger.error("ResourceManager chưa được thiết lập. Bỏ qua anomaly_check_loop.")
                    else:
                        # Lấy snapshot
                        async with acquire_lock_with_timeout(self.mining_processes_lock, 'read', timeout=5) as read_lock:
                            if read_lock is None:
                                self.logger.error("Không thể acquire mining_processes_lock trong anomaly_check_loop.")
                                await asyncio.sleep(1)
                                continue
                            processes = list(self.mining_processes)

                        # Kiểm tra temperature/power, rồi anomaly
                        for process in processes:
                            if process.is_cloaked:
                                continue  # Skip nếu đã cloak
                            cloaked = await self.check_temperature_and_power(process, cloak_activation_delay)
                            if cloaked:
                                process.is_cloaked = True
                                continue
                            # Nếu chưa cloak => kiểm tra anomaly
                            await self.evaluate_process_anomaly(process, cloak_activation_delay)

                except Exception as e:
                    self.logger.error(f"Lỗi trong anomaly_check_loop: {e}\n{traceback.format_exc()}")

                last_detection_time = current_time

            await asyncio.sleep(2)  # Tạm nghỉ

    async def evaluate_process_anomaly(self, process: MiningProcess, cloak_activation_delay: int):
        """
        Kiểm tra anomaly qua Azure Anomaly Detector. Nếu có => emit event cloaking.
        """
        if not psutil.pid_exists(process.pid):
            return

        if self.resource_manager is None:
            return

        try:
            current_state = await self.resource_manager.collect_metrics(process)
            anomalies_detected = self.resource_manager.azure_anomaly_detector_client.detect_anomalies(current_state)
            if anomalies_detected:
                self.logger.warning(
                    f"Phát hiện bất thường (Azure Anomaly) ở tiến trình {process.name} (PID={process.pid}) => CLOAKING sau {cloak_activation_delay}s."
                )
                await asyncio.sleep(cloak_activation_delay)
                await self.resource_manager.emit_cloaking_event(process)
                process.is_cloaked = True
        except Exception as e:
            self.logger.error(f"Lỗi evaluate_process_anomaly cho PID={process.pid}: {e}\n{traceback.format_exc()}")

    # ------------------- Event-Driven: Kiểm tra phục hồi tài nguyên ------------------- #
    async def restoration_check_loop(self):
        """
        Định kỳ kiểm tra xem có tiến trình nào đã bị cloak trước đó
        mà hiện tại đáp ứng điều kiện an toàn để khôi phục chưa.
        """
        check_interval = 60  # Kiểm tra mỗi 60s

        while not self.stop_event.is_set():
            try:
                async with acquire_lock_with_timeout(self.mining_processes_lock, 'read', timeout=5) as read_lock:
                    if read_lock is None:
                        self.logger.error("Không thể acquire lock trong restoration_check_loop.")
                        await asyncio.sleep(check_interval)
                        continue
                    cloaked_procs = [p for p in self.mining_processes if getattr(p, "is_cloaked", False) is True]

                if not self.safe_restore_evaluator:
                    await asyncio.sleep(check_interval)
                    continue

                tasks = []
                for proc in cloaked_procs:
                    tasks.append(self.safe_restore_evaluator.is_safe_to_restore(proc))

                results = await asyncio.gather(*tasks)
                for proc, is_safe in zip(cloaked_procs, results):
                    if is_safe:
                        self.logger.info(
                            f"Điều kiện đã đạt để khôi phục tài nguyên cho PID={proc.pid}."
                        )
                        await self.resource_manager.emit_restoration_event(proc)
                        proc.is_cloaked = False
                    else:
                        self.logger.debug(
                            f"Điều kiện chưa đạt để khôi phục tài nguyên cho PID={proc.pid}."
                        )

            except Exception as e:
                self.logger.error(f"Lỗi trong restoration_check_loop: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(check_interval)

    async def stop(self):
        """
        Dừng AnomalyDetector, hủy các task.
        """
        self.logger.info("Đang dừng AnomalyDetector...")
        self.stop_event.set()
        self.task_anomaly_check.cancel()
        self.task_restoration_check.cancel()
        await asyncio.sleep(0.5)
        self.logger.info("AnomalyDetector đã dừng thành công.")
