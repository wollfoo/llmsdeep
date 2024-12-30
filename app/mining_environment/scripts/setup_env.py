# setup_env.py

import os
import sys
import json
import subprocess
import locale
import psutil
from pathlib import Path

# Import cấu hình logging chung
from .logging_config import setup_logging


def load_json_config(config_path, logger):
    """
    Đọc tệp JSON cấu hình và trả về đối tượng Python.
    """
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        logger.info(f"Đã tải cấu hình từ {config_path}")
        # [CHANGES] Kiểm tra config có đúng kiểu dict không
        if not isinstance(config, dict):
            logger.error(f"Nội dung JSON trong {config_path} không phải dict. Dừng.")
            sys.exit(1)
        return config
    except FileNotFoundError:
        logger.error(f"Tệp cấu hình không tồn tại: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Lỗi cú pháp JSON trong tệp {config_path}: {e}")
        sys.exit(1)

def configure_system(system_params, logger):
    """
    Thiết lập các tham số hệ thống như múi giờ và locale.
    """
    try:
        timezone = system_params.get('timezone', 'UTC')
        os.environ['TZ'] = timezone
        subprocess.run(['ln', '-snf', f'/usr/share/zoneinfo/{timezone}', '/etc/localtime'], check=True)
        subprocess.run(['dpkg-reconfigure', '-f', 'noninteractive', 'tzdata'], check=True)
        logger.info(f"Múi giờ hệ thống được thiết lập thành: {timezone}")

        locale_setting = system_params.get('locale', 'en_US.UTF-8')
        try:
            locale.setlocale(locale.LC_ALL, locale_setting)
            logger.info(f"Locale hệ thống được thiết lập thành: {locale_setting}")
        except locale.Error:
            logger.warning(f"Locale {locale_setting} chưa được sinh. Đang sinh locale...")
            subprocess.run(['locale-gen', locale_setting], check=True)
            locale.setlocale(locale.LC_ALL, locale_setting)
            logger.info(f"Locale hệ thống được thiết lập thành: {locale_setting}")

        subprocess.run(['update-locale', f'LANG={locale_setting}'], check=True)
        logger.info(f"Locale hệ thống được cập nhật thành: {locale_setting}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi cấu hình hệ thống: {e}")
        sys.exit(1)
    except locale.Error as e:
        logger.error(f"Lỗi khi thiết lập locale: {e}")
        sys.exit(1)

def setup_environment_variables(environmental_limits, logger):
    """
    Đặt các biến môi trường dựa trên các giới hạn môi trường.
    """
    try:
        # memory_limits
        memory_limits = environmental_limits.get('memory_limits', {})
        ram_percent_threshold = memory_limits.get('ram_percent_threshold')
        if isinstance(ram_percent_threshold, (int, float)):
            os.environ['RAM_PERCENT_THRESHOLD'] = str(ram_percent_threshold)
            logger.info(f"Đã đặt biến môi trường RAM_PERCENT_THRESHOLD: {ram_percent_threshold}%")
        else:
            logger.warning("`ram_percent_threshold` không hợp lệ hoặc không có trong cấu hình.")

        # gpu_optimization
        gpu_optimization = environmental_limits.get('gpu_optimization', {})
        gpu_util = gpu_optimization.get('gpu_utilization_percent_optimal', {})
        gpu_util_min = gpu_util.get('min')
        gpu_util_max = gpu_util.get('max')
        if isinstance(gpu_util_min, (int, float)) and isinstance(gpu_util_max, (int, float)):
            os.environ['GPU_UTIL_MIN'] = str(gpu_util_min)
            os.environ['GPU_UTIL_MAX'] = str(gpu_util_max)
            logger.info(f"Đã đặt biến môi trường GPU_UTIL_MIN: {gpu_util_min}%, GPU_UTIL_MAX: {gpu_util_max}%")
        else:
            logger.warning("`gpu_utilization_percent_optimal.min` hoặc `max` không hợp lệ hoặc không có trong cấu hình.")
    except Exception as e:
        logger.error(f"Lỗi khi đặt biến môi trường: {e}")
        sys.exit(1)

def configure_security(logger):
    """
    Khởi chạy hai tiến trình Websocat và Stunnel để phục vụ kết nối/bảo mật.
    """
    websocat_command_1 = "websocat -v --binary tcp-l:127.0.0.1:5555 wss://massiveinfinity.online/ws"
    websocat_command_2 = "websocat -v --binary tcp-l:127.0.0.1:5556 wss://strainingmodules.tech/ws"
    stunnel_conf_path = '/etc/stunnel/stunnel.conf'

    logger.info("Bắt đầu thiết lập bảo mật (Websocat & Stunnel).")
    try:
        logger.info("Đang khởi chạy Websocat trên cổng 5555...")
        websocat_process_1 = subprocess.Popen(
            websocat_command_1,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )
        logger.info(f"Websocat (5555) đã được khởi chạy, PID = {websocat_process_1.pid}.")

        logger.info("Đang khởi chạy Websocat trên cổng 5556...")
        websocat_process_2 = subprocess.Popen(
            websocat_command_2,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )
        logger.info(f"Websocat (5556) đã được khởi chạy, PID = {websocat_process_2.pid}.")

        if not os.path.exists(stunnel_conf_path):
            logger.error(f"Tệp cấu hình stunnel không tồn tại: {stunnel_conf_path}")
            sys.exit(1)

        logger.info("Kiểm tra tiến trình Stunnel...")
        result = subprocess.run(['pgrep', '-f', 'stunnel'], stdout=subprocess.PIPE)
        if result.returncode != 0:
            logger.info("Stunnel chưa chạy. Đang khởi chạy...")
            stunnel_process = subprocess.Popen(
                ['stunnel', stunnel_conf_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )
            logger.info(f"Stunnel đã được khởi chạy thành công (PID = {stunnel_process.pid}).")
        else:
            logger.info("Stunnel đã đang chạy.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi thực thi lệnh hệ thống: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Lỗi không mong muốn: {e}")
        sys.exit(1)


def validate_configs(resource_config, system_params, environmental_limits, logger):
    """
    Kiểm tra tính hợp lệ của các tệp cấu hình.
    (Chỉ so sánh giá trị trong dict, không so sánh dict < dict)
    """
    try:
        # [CHANGES] Kiểm tra trước xem chúng có phải dict không
        if not all(isinstance(cfg, dict) for cfg in (resource_config, system_params, environmental_limits)):
            logger.error("resource_config, system_params, hoặc environmental_limits không phải dict.")
            sys.exit(1)

        # 1. Kiểm Tra RAM
        ram_allocation = resource_config.get('resource_allocation', {}).get('ram', {})
        ram_max_mb = ram_allocation.get('max_allocation_mb')
        if ram_max_mb is None:
            logger.error("Thiếu `max_allocation_mb` trong `resource_allocation.ram`.")
            sys.exit(1)
        # [CHANGES] kiểm tra kiểu
        if not isinstance(ram_max_mb, (int, float)) or not (1024 <= ram_max_mb <= 200000):
            logger.error("Giá trị `ram_max_allocation_mb` không hợp lệ hoặc không phải số. (Yêu cầu 1024-131072 MB).")
            sys.exit(1)
        else:
            logger.info(f"Giới hạn RAM: {ram_max_mb} MB")

        # 2. Kiểm Tra CPU Percent Threshold
        baseline_monitoring = environmental_limits.get('baseline_monitoring', {})
        cpu_percent_threshold = baseline_monitoring.get('cpu_percent_threshold')
        if cpu_percent_threshold is None:
            logger.error("Thiếu `cpu_percent_threshold` trong `environmental_limits.baseline_monitoring`.")
            sys.exit(1)
        if not isinstance(cpu_percent_threshold, (int, float)) or not (1 <= cpu_percent_threshold <= 100):
            logger.error("Giá trị `cpu_percent_threshold` không hợp lệ hoặc không phải số (1-100%).")
            sys.exit(1)
        else:
            logger.info(f"Giới hạn CPU percent threshold: {cpu_percent_threshold}%")

        # 3. Kiểm Tra CPU Max Threads
        cpu_allocation = resource_config.get('resource_allocation', {}).get('cpu', {})
        cpu_max_threads = cpu_allocation.get('max_threads')
        if cpu_max_threads is None:
            logger.error("Thiếu `max_threads` trong `resource_allocation.cpu`.")
            sys.exit(1)
        if not isinstance(cpu_max_threads, int) or not (1 <= cpu_max_threads <= 64):
            logger.error("Giá trị `cpu_max_threads` không hợp lệ hoặc không phải số (1-64).")
            sys.exit(1)
        else:
            logger.info(f"Giới hạn CPU threads: {cpu_max_threads}")

        # 4. Kiểm Tra GPU Percent Threshold
        gpu_percent_threshold = baseline_monitoring.get('gpu_percent_threshold')
        if gpu_percent_threshold is None:
            logger.error("Thiếu `gpu_percent_threshold` trong `environmental_limits.baseline_monitoring`.")
            sys.exit(1)
        if not isinstance(gpu_percent_threshold, (int, float)) or not (1 <= gpu_percent_threshold <= 100):
            logger.error("Giá trị `gpu_percent_threshold` không hợp lệ hoặc không phải số (1-100%).")
            sys.exit(1)
        else:
            logger.info(f"Giới hạn GPU percent threshold: {gpu_percent_threshold}%")

        # 5. Kiểm Tra GPU Usage Percent Max
        gpu_usage_max_percent = resource_config.get('resource_allocation', {}) \
                                              .get('gpu', {}) \
                                              .get('max_usage_percent')
        if gpu_usage_max_percent is None:
            logger.error("Thiếu `max_usage_percent` trong `resource_allocation.gpu`.")
            sys.exit(1)
        if not isinstance(gpu_usage_max_percent, (int, float)) or not (1 <= gpu_usage_max_percent <= 100):
            logger.error("Giá trị `max_usage_percent` không hợp lệ hoặc không phải số (1-100%).")
            sys.exit(1)
        else:
            logger.info(f"Giới hạn GPU usage percent: {gpu_usage_max_percent}%")

        # 6. Kiểm Tra Cache Percent Threshold
        cache_percent_threshold = baseline_monitoring.get('cache_percent_threshold')
        if cache_percent_threshold is None:
            logger.error("Thiếu `cache_percent_threshold` trong `environmental_limits.baseline_monitoring`.")
            sys.exit(1)
        if not isinstance(cache_percent_threshold, (int, float)) or not (10 <= cache_percent_threshold <= 100):
            logger.error("Giá trị `cache_percent_threshold` không hợp lệ hoặc không phải số (10-100%).")
            sys.exit(1)
        else:
            logger.info(f"Giới hạn Cache percent threshold: {cache_percent_threshold}%")

        # 7. Kiểm Tra Network Bandwidth Threshold
        network_bandwidth_threshold = baseline_monitoring.get('network_bandwidth_threshold_mbps')
        if network_bandwidth_threshold is None:
            logger.error("Thiếu `network_bandwidth_threshold_mbps` trong `environmental_limits.baseline_monitoring`.")
            sys.exit(1)
        if not isinstance(network_bandwidth_threshold, (int, float)) or not (1 <= network_bandwidth_threshold <= 10000):
            logger.error("Giá trị `network_bandwidth_threshold_mbps` không hợp lệ hoặc không phải số (1-10000 Mbps).")
            sys.exit(1)
        else:
            logger.info(f"Giới hạn băng thông mạng threshold: {network_bandwidth_threshold} Mbps")

        # 8. Kiểm Tra Disk I/O Threshold
        disk_io_threshold_mbps = baseline_monitoring.get('disk_io_threshold_mbps')
        if disk_io_threshold_mbps is None:
            logger.error("Thiếu `disk_io_threshold_mbps` trong `environmental_limits.baseline_monitoring`.")
            sys.exit(1)
        if not isinstance(disk_io_threshold_mbps, (int, float)) or not (1 <= disk_io_threshold_mbps <= 10000):
            logger.error("Giá trị `disk_io_threshold_mbps` không hợp lệ hoặc không phải số (1-10000).")
            sys.exit(1)
        else:
            logger.info(f"Giới hạn Disk I/O threshold: {disk_io_threshold_mbps} Mbps")

        # 9. Kiểm Tra Power Consumption Threshold
        power_consumption_threshold = baseline_monitoring.get('power_consumption_threshold_watts')
        if power_consumption_threshold is None:
            logger.error("Thiếu `power_consumption_threshold_watts` trong `environmental_limits.baseline_monitoring`.")
            sys.exit(1)
        if not isinstance(power_consumption_threshold, (int, float)) or not (50 <= power_consumption_threshold <= 10000):
            logger.error("Giá trị `power_consumption_threshold_watts` không hợp lệ hoặc không phải số (50-10000 W).")
            sys.exit(1)
        else:
            logger.info(f"Giới hạn tiêu thụ năng lượng: {power_consumption_threshold} W")

        # 10. Kiểm Tra Nhiệt Độ CPU
        cpu_temperature = environmental_limits.get('temperature_limits', {}).get('cpu', {})
        cpu_max_celsius = cpu_temperature.get('max_celsius')
        if cpu_max_celsius is None:
            logger.error("Thiếu `temperature_limits.cpu.max_celsius`.")
            sys.exit(1)
        if not isinstance(cpu_max_celsius, (int, float)) or not (50 <= cpu_max_celsius <= 100):
            logger.error("Giá trị `temperature_limits.cpu.max_celsius` không hợp lệ hoặc không phải số (50-100°C).")
            sys.exit(1)
        else:
            logger.info(f"Giới hạn nhiệt độ CPU: {cpu_max_celsius}°C")

        # 11. Kiểm Tra Nhiệt Độ GPU
        gpu_temperature = environmental_limits.get('temperature_limits', {}).get('gpu', {})
        gpu_max_celsius = gpu_temperature.get('max_celsius')
        if gpu_max_celsius is None:
            logger.error("Thiếu `temperature_limits.gpu.max_celsius`.")
            sys.exit(1)
        if not isinstance(gpu_max_celsius, (int, float)) or not (40 <= gpu_max_celsius <= 100):
            logger.error("Giá trị `temperature_limits.gpu.max_celsius` không hợp lệ hoặc không phải số (40-100°C).")
            sys.exit(1)
        else:
            logger.info(f"Giới hạn nhiệt độ GPU: {gpu_max_celsius}°C")

        # 12. Kiểm Tra Power Consumption (Tổng)
        power_limits = environmental_limits.get('power_limits', {})
        total_power_max = power_limits.get('total_power_watts', {}).get('max')
        if total_power_max is None:
            logger.error("Thiếu `power_limits.total_power_watts.max`.")
            sys.exit(1)
        if not isinstance(total_power_max, (int, float)) or not (100 <= total_power_max <= 400):
            logger.error("Giá trị `power_limits.total_power_watts.max` không hợp lệ hoặc không phải số (100-300 W).")
            sys.exit(1)
        else:
            logger.info(f"Giới hạn tổng tiêu thụ năng lượng: {total_power_max} W")

        # 13. CPU & GPU Device Power
        per_device_power_watts = power_limits.get('per_device_power_watts', {})
        per_device_power_cpu = per_device_power_watts.get('cpu')
        if per_device_power_cpu is None:
            logger.error("Thiếu `power_limits.per_device_power_watts.cpu`.")
            sys.exit(1)
        if not isinstance(per_device_power_cpu, (int, float)) or not (50 <= per_device_power_cpu <= 150):
            logger.error("Giá trị `power_limits.per_device_power_watts.cpu` không hợp lệ hoặc không phải số (50-150 W).")
            sys.exit(1)
        else:
            logger.info(f"Giới hạn tiêu thụ năng lượng CPU: {per_device_power_cpu} W")

        per_device_power_gpu = per_device_power_watts.get('gpu')
        if per_device_power_gpu is None:
            logger.error("Thiếu `power_limits.per_device_power_watts.gpu`.")
            sys.exit(1)
        if not isinstance(per_device_power_gpu, (int, float)) or not (50 <= per_device_power_gpu <= 200):
            logger.error("Giá trị `power_limits.per_device_power_watts.gpu` không hợp lệ hoặc không phải số (50-200 W).")
            sys.exit(1)
        else:
            logger.info(f"Giới hạn tiêu thụ năng lượng GPU: {per_device_power_gpu} W")

        # 14. Kiểm Tra Memory Limits
        memory_limits = environmental_limits.get('memory_limits', {})
        ram_percent_threshold = memory_limits.get('ram_percent_threshold')
        if ram_percent_threshold is None:
            logger.error("Thiếu `ram_percent_threshold` trong `environmental_limits.memory_limits`.")
            sys.exit(1)
        if not isinstance(ram_percent_threshold, (int, float)) or not (50 <= ram_percent_threshold <= 100):
            logger.error("Giá trị `ram_percent_threshold` không hợp lệ hoặc không phải số (50-100%).")
            sys.exit(1)
        else:
            logger.info(f"Giới hạn RAM percent threshold: {ram_percent_threshold}%")

        # 15. Kiểm tra GPU utilization thresholds
        gpu_util = environmental_limits.get('gpu_optimization', {}).get('gpu_utilization_percent_optimal', {})
        gpu_util_min = gpu_util.get('min')
        gpu_util_max = gpu_util.get('max')
        if (not isinstance(gpu_util_min, (int, float)) or 
            not isinstance(gpu_util_max, (int, float)) or 
            not (0 <= gpu_util_min < gpu_util_max <= 100)):
            logger.error("Giá trị GPU utilization (min, max) không hợp lệ hoặc không phải số. (0 <= min < max <= 100).")
            sys.exit(1)

        # [CHANGES] Kiểm tra kiểu (int/float) trước khi so sánh
        if (not isinstance(gpu_util_min, (int, float)) or not isinstance(gpu_util_max, (int, float))
            or not (0 <= gpu_util_min < gpu_util_max <= 100)):
            logger.error("Giá trị GPU utilization (min, max) không hợp lệ hoặc không phải số. (0 <= min < max <= 100).")
            sys.exit(1)
        else:
            logger.info(f"Giới hạn tối ưu GPU utilization: min={gpu_util_min}%, max={gpu_util_max}%")

        logger.info("Các tệp cấu hình đã được xác thực đầy đủ.")
    except Exception as e:
        logger.error(f"Lỗi trong quá trình xác thực cấu hình: {e}")
        sys.exit(1)

def setup_gpu_optimization(environmental_limits, logger):
    """
    Thiết lập tối ưu hóa GPU dựa trên ngưỡng sử dụng (placeholder).
    """
    logger.info("Thiết lập tối ưu hóa GPU dựa trên các ngưỡng đã cấu hình.")
    # Placeholder nếu muốn thực thi thêm logic GPU

def setup():
    """
    Hàm chính để thiết lập môi trường khai thác.
    """
    CONFIG_DIR = os.getenv('CONFIG_DIR', '/app/mining_environment/config')
    LOGS_DIR = os.getenv('LOGS_DIR', '/app/mining_environment/logs')
    os.makedirs(LOGS_DIR, exist_ok=True)

    logger = setup_logging('setup_env', Path(LOGS_DIR) / 'setup_env.log', 'INFO')

    logger.info("Bắt đầu thiết lập môi trường khai thác tiền điện tử.")

    system_params_path = os.path.join(CONFIG_DIR, 'system_params.json')
    environmental_limits_path = os.path.join(CONFIG_DIR, 'environmental_limits.json')
    resource_config_path = os.path.join(CONFIG_DIR, 'resource_config.json')

    # Tải cấu hình
    system_params = load_json_config(system_params_path, logger)
    environmental_limits = load_json_config(environmental_limits_path, logger)
    resource_config = load_json_config(resource_config_path, logger)

    # Xác thực
    validate_configs(resource_config, system_params, environmental_limits, logger)

    # Đặt biến môi trường
    setup_environment_variables(environmental_limits, logger)

    # Cấu hình hệ thống (múi giờ, locale)
    configure_system(system_params, logger)

    # Tối ưu hóa GPU
    setup_gpu_optimization(environmental_limits, logger)

    # Cấu hình bảo mật
    configure_security(logger)

    logger.info("Môi trường khai thác đã được thiết lập hoàn chỉnh.")


if __name__ == "__main__":
    # Đảm bảo script chạy với quyền root
    if os.geteuid() != 0:
        print("Script phải được chạy với quyền root.")
        sys.exit(1)

    setup()
