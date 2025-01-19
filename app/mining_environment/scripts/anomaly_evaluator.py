import psutil
import logging
import traceback
import asyncio
from typing import Dict, Any

from .utils import MiningProcess
from .auxiliary_modules.models import ConfigModel
from .auxiliary_modules.interfaces import IResourceManager
from .auxiliary_modules.power_management import get_cpu_power, get_gpu_power
from .auxiliary_modules.temperature_monitor import get_cpu_temperature, get_gpu_temperature

# Tận dụng cùng lock/hàm NVML init => import từ anomaly_detector
from .anomaly_detector import is_nvml_initialized, initialize_nvml

class SafeRestoreEvaluator:
    """
    Lớp đánh giá điều kiện an toàn để khôi phục tài nguyên cho tiến trình.
    """

    def __init__(self, config: ConfigModel, logger: logging.Logger, resource_manager: IResourceManager):
        self.config = config
        self.logger = logger
        self.resource_manager = resource_manager

        # Các ngưỡng khôi phục
        self.baseline_cpu_usage_percent = 80
        self.baseline_gpu_usage_percent = 80
        self.baseline_ram_usage_percent = 80
        self.baseline_disk_io_usage_mbps = 80
        self.baseline_network_usage_mbps = 80

        self.cpu_max_temp = 75
        self.gpu_max_temp = 75
        self.cpu_max_power = 100
        self.gpu_max_power = 200

    async def start(self):
        """
        Có thể nạp/tham chiếu config bổ sung, hoặc no-op.
        """
        self.logger.debug("SafeRestoreEvaluator.start() được gọi.")
        try:
            baseline_thresholds = self.config.baseline_thresholds
            self.baseline_cpu_usage_percent = baseline_thresholds.get('cpu_usage_percent', 80)
            self.baseline_gpu_usage_percent = baseline_thresholds.get('gpu_usage_percent', 80)
            self.baseline_ram_usage_percent = baseline_thresholds.get('ram_usage_percent', 80)
            self.baseline_disk_io_usage_mbps = baseline_thresholds.get('disk_io_usage_mbps', 80)
            self.baseline_network_usage_mbps = baseline_thresholds.get('network_usage_mbps', 80)

            temperature_limits = self.config.temperature_limits
            self.cpu_max_temp = temperature_limits.get("cpu_max_celsius", 75)
            self.gpu_max_temp = temperature_limits.get("gpu_max_celsius", 75)

            power_limits = self.config.power_limits
            per_device_power = power_limits.get("per_device_power_watts", {})
            self.cpu_max_power = per_device_power.get("cpu", 100)
            self.gpu_max_power = per_device_power.get("gpu", 200)

        except Exception as e:
            self.logger.error(f"Lỗi init SafeRestoreEvaluator: {e}\n{traceback.format_exc()}")

    async def is_safe_to_restore(self, process: MiningProcess) -> bool:
        """
        Kiểm tra các điều kiện an toàn để khôi phục tài nguyên cho tiến trình.
        """
        if not psutil.pid_exists(process.pid):
            self.logger.warning(f"Tiến trình PID {process.pid} không tồn tại.")
            return False

        # 1) Kiểm tra nhiệt độ CPU
        try:
            # Gọi get_cpu_temperature(process.pid) => blocking => run_in_executor
            cpu_temp = await asyncio.to_thread(get_cpu_temperature, process.pid)
            if cpu_temp is not None and cpu_temp >= self.cpu_max_temp:
                self.logger.info(f"Nhiệt độ CPU {cpu_temp}°C vẫn cao (PID={process.pid}).")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi kiểm tra nhiệt độ CPU PID={process.pid}: {e}\n{traceback.format_exc()}")
            return False

        # 2) Kiểm tra nhiệt độ GPU
        try:
            if not is_nvml_initialized():
                await initialize_nvml()

            gpu_temps = await asyncio.to_thread(get_gpu_temperature, process.pid)
            if gpu_temps and any(temp >= self.gpu_max_temp for temp in gpu_temps):
                self.logger.info(f"Nhiệt độ GPU {gpu_temps}°C vẫn cao (PID={process.pid}).")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi kiểm tra nhiệt độ GPU PID={process.pid}: {e}\n{traceback.format_exc()}")
            return False

        # 3) Kiểm tra công suất CPU
        try:
            cpu_power = await asyncio.to_thread(get_cpu_power, process.pid)
            if cpu_power is not None and cpu_power >= self.cpu_max_power:
                self.logger.info(f"Công suất CPU {cpu_power}W vẫn cao (PID={process.pid}).")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi kiểm tra công suất CPU PID={process.pid}: {e}\n{traceback.format_exc()}")
            return False

        # 4) Kiểm tra công suất GPU
        try:
            if not is_nvml_initialized():
                await initialize_nvml()

            gpu_power = await asyncio.to_thread(get_gpu_power, process.pid)
            if isinstance(gpu_power, list):
                if sum(gpu_power) >= self.gpu_max_power:
                    self.logger.info(f"Công suất GPU {gpu_power}W vẫn cao (PID={process.pid}).")
                    return False
            else:
                if gpu_power and gpu_power >= self.gpu_max_power:
                    self.logger.info(f"Công suất GPU {gpu_power}W vẫn cao (PID={process.pid}).")
                    return False
        except Exception as e:
            self.logger.error(f"Lỗi kiểm tra công suất GPU PID={process.pid}: {e}\n{traceback.format_exc()}")
            return False

        # 5) Kiểm tra CPU usage tổng thể
        try:
            total_cpu_usage = psutil.cpu_percent(interval=1)
            if total_cpu_usage >= self.baseline_cpu_usage_percent:
                self.logger.info(f"Sử dụng CPU tổng thể {total_cpu_usage}% vẫn cao.")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi kiểm tra CPU tổng thể: {e}\n{traceback.format_exc()}")
            return False

        # 6) Kiểm tra RAM
        try:
            ram = psutil.virtual_memory()
            if ram.percent >= self.baseline_ram_usage_percent:
                self.logger.info(f"Sử dụng RAM tổng thể {ram.percent}% vẫn cao.")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi kiểm tra RAM tổng thể: {e}\n{traceback.format_exc()}")
            return False

        # 7) Kiểm tra Disk I/O
        try:
            disk_io_counters = psutil.disk_io_counters()
            total_disk_io_usage_mb = (disk_io_counters.read_bytes + disk_io_counters.write_bytes) / (1024 * 1024)
            # Tùy logic: so sánh ~ baseline_disk_io_usage_mbps => ta coi 1s => phức tạp
            # Đơn giản: so sánh raw MB => or sampling
            if total_disk_io_usage_mb >= self.baseline_disk_io_usage_mbps:
                self.logger.info(f"Sử dụng Disk I/O {total_disk_io_usage_mb:.2f} MB vẫn cao.")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi kiểm tra Disk I/O: {e}\n{traceback.format_exc()}")
            return False

        # 8) Kiểm tra mạng
        try:
            net_io_counters = psutil.net_io_counters()
            total_network_usage_mb = (net_io_counters.bytes_sent + net_io_counters.bytes_recv) / (1024 * 1024)
            if total_network_usage_mb >= self.baseline_network_usage_mbps:
                self.logger.info(f"Sử dụng mạng {total_network_usage_mb:.2f} MB vẫn cao.")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi kiểm tra mạng: {e}\n{traceback.format_exc()}")
            return False

        # 9) Kiểm tra Anomaly Detector => (nếu true => chưa an toàn)
        try:
            current_state = await self.resource_manager.collect_metrics(process)
            # current_state => { 'cpu_usage':..., ...}
            # detect_anomalies => có thể chờ 1 dict { pid: [metrics] } => ta giả lập
            single_data = { str(process.pid): [current_state] }
            anomalies_detected = await self.resource_manager.azure_anomaly_detector_client.detect_anomalies(single_data)
            if anomalies_detected:
                self.logger.info(f"Azure AnomalyDetector phát hiện bất thường (PID={process.pid}).")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi qua Azure Anomaly Detector PID={process.pid}: {e}\n{traceback.format_exc()}")
            return False

        self.logger.info(f"Đủ điều kiện an toàn để khôi phục cho PID={process.pid}.")
        return True

    async def stop(self):
        """
        Dừng SafeRestoreEvaluator. Hiện tại không có logic cụ thể.
        """
        self.logger.info("SafeRestoreEvaluator.stop() được gọi. Hiện tại không có logic dừng.")
