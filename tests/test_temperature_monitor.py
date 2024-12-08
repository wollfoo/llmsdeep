import os
import sys
import pytest
import psutil
import subprocess
from unittest.mock import patch, MagicMock, mock_open, call
from pathlib import Path

# Thêm SCRIPTS_DIR vào sys.path nếu chưa có
APP_DIR = Path("/home/llmss/llmsdeep/app")
SCRIPTS_DIR = APP_DIR / "mining_environment" / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from unittest.mock import patch, MagicMock, mock_open, call

from auxiliary_modules.temperature_monitor import TemperatureMonitor

from auxiliary_modules.temperature_monitor import (
    setup_temperature_monitoring,
    get_cpu_temperature,
    get_gpu_temperature,
    get_current_cpu_threads,
    set_cpu_threads,
    get_current_ram_allocation,
    set_ram_allocation,
    get_current_gpu_usage,
    get_current_disk_io_limit,
    set_disk_io_limit,
    get_current_network_bandwidth_limit,
    set_network_bandwidth_limit,
    get_current_cache_limit,
    set_cache_limit,
    _drop_caches,
    shutdown,
)

# Mocking NVML initialization to prevent real hardware dependencies
@pytest.fixture(autouse=True)
def mock_nvml_init():
    with patch("auxiliary_modules.temperature_monitor.pynvml.nvmlInit") as nvml_init_mock:
        with patch("auxiliary_modules.temperature_monitor.pynvml.nvmlDeviceGetCount", return_value=2):
            nvml_init_mock.return_value = None
            yield nvml_init_mock


# Test setup_temperature_monitoring
def test_setup_temperature_monitoring():
    with patch("auxiliary_modules.temperature_monitor.logger.info") as mock_logger:
        setup_temperature_monitoring()
        mock_logger.assert_called_once_with("Đã thiết lập giám sát nhiệt độ.")

# Test get_cpu_temperature
def test_get_cpu_temperature():
    # Mock cấu trúc trả về của psutil.sensors_temperatures
    mock_entry_1 = MagicMock(label="Core 0", current=50)
    mock_entry_2 = MagicMock(label="Core 1", current=55)
    mock_sensors = {"coretemp": [mock_entry_1, mock_entry_2]}

    with patch("auxiliary_modules.temperature_monitor.psutil.sensors_temperatures", return_value=mock_sensors):
        result = get_cpu_temperature()
        assert result == 52.5  # Average of 50 and 55

    # Trường hợp không có cảm biến
    with patch("auxiliary_modules.temperature_monitor.psutil.sensors_temperatures", return_value={}):
        assert get_cpu_temperature() is None

# Test get_gpu_temperature
def test_get_gpu_temperature():
    # Khởi tạo đối tượng TemperatureMonitor
    with patch("auxiliary_modules.temperature_monitor.pynvml.nvmlInit"):
        monitor = TemperatureMonitor()

    # Mô phỏng nvmlDeviceGetCount để đảm bảo gpu_count > 0
    with patch("auxiliary_modules.temperature_monitor.pynvml.nvmlDeviceGetCount", return_value=2):
        monitor.gpu_count = 2  # Đảm bảo gpu_count được thiết lập

        mock_handle = MagicMock()
        with patch("auxiliary_modules.temperature_monitor.pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle):
            with patch("auxiliary_modules.temperature_monitor.pynvml.nvmlDeviceGetTemperature", side_effect=[60, 70]):
                result = monitor.get_gpu_temperature()
                assert result == [60, 70]

# Test get_current_cpu_threads
def test_get_current_cpu_threads():
    mock_proc = MagicMock()
    mock_proc.cpu_affinity.return_value = [0, 1, 2, 3]
    with patch("auxiliary_modules.temperature_monitor.psutil.Process", return_value=mock_proc):
        assert get_current_cpu_threads(1234) == 4

# Test set_cpu_threads
def test_set_cpu_threads():
    mock_proc = MagicMock()
    with patch("auxiliary_modules.temperature_monitor.psutil.Process", return_value=mock_proc):
        set_cpu_threads(2, 1234)
        mock_proc.cpu_affinity.assert_called_once_with([0, 1])

# Test get_current_ram_allocation
def test_get_current_ram_allocation():
    mock_proc = MagicMock()
    mock_proc.memory_info.return_value = MagicMock(rss=1024 * 1024 * 512)  # 512 MB
    with patch("auxiliary_modules.temperature_monitor.psutil.Process", return_value=mock_proc):
        assert get_current_ram_allocation(1234) == 512

# Test set_ram_allocation
def test_set_ram_allocation():
    mock_proc = MagicMock()
    mock_proc.pid = 1234
    mock_proc.name.return_value = "mock_process"
    
    with patch("auxiliary_modules.temperature_monitor.TemperatureMonitor._find_mining_process", return_value=mock_proc):
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            with patch("builtins.open", mock_open()) as mock_file:
                set_ram_allocation(512, None)  # Không truyền pid, sẽ sử dụng _find_mining_process
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
                mock_file().write.assert_called_once_with(str(512 * 1024 * 1024))

# Test get_current_gpu_usage
def test_get_current_gpu_usage():
    # Khởi tạo đối tượng TemperatureMonitor
    with patch("auxiliary_modules.temperature_monitor.pynvml.nvmlInit"):
        monitor = TemperatureMonitor()

    # Mô phỏng nvmlDeviceGetCount để đảm bảo gpu_count > 0
    with patch("auxiliary_modules.temperature_monitor.pynvml.nvmlDeviceGetCount", return_value=2):
        monitor.gpu_count = 2  # Thiết lập số lượng GPU

        mock_handle = MagicMock()
        with patch("auxiliary_modules.temperature_monitor.pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle):
            with patch("auxiliary_modules.temperature_monitor.pynvml.nvmlDeviceGetUtilizationRates", return_value=MagicMock(gpu=50)):
                result = monitor.get_current_gpu_usage()
                assert result == [50, 50]  # Kết quả mong đợi là danh sách [50, 50]

# Test _drop_caches
def test_drop_caches():
    with patch("builtins.open", mock_open()) as mock_file:
        _drop_caches()
        mock_file.assert_called_once_with("/proc/sys/vm/drop_caches", "w")
        mock_file().write.assert_called_once_with("3\n")


def test_shutdown():
    # Khởi tạo đối tượng TemperatureMonitor
    with patch("auxiliary_modules.temperature_monitor.pynvml.nvmlInit"):
        monitor = TemperatureMonitor()

    # Giả lập gpu_count để đảm bảo điều kiện if self.gpu_count > 0 được thỏa mãn
    with patch("auxiliary_modules.temperature_monitor.pynvml.nvmlDeviceGetCount", return_value=2):
        monitor.gpu_count = 2  # Thiết lập gpu_count lớn hơn 0

        # Giả lập nvmlShutdown và kiểm tra
        with patch("auxiliary_modules.temperature_monitor.pynvml.nvmlShutdown") as mock_nvml_shutdown:
            monitor.shutdown()  # Gọi phương thức shutdown
            mock_nvml_shutdown.assert_called_once()  # Kiểm tra xem nvmlShutdown có được gọi không
