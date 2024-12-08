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


from auxiliary_modules import power_management



@pytest.fixture
def mock_logger():
    with patch('auxiliary_modules.power_management.logger') as mock_logger:
        yield mock_logger

@pytest.fixture
def mock_psutil():
    with patch('auxiliary_modules.power_management.psutil') as mock_psutil:
        yield mock_psutil

@pytest.fixture
def mock_pynvml():
    with patch('auxiliary_modules.power_management.pynvml', autospec=True) as mock_pynvml:
        yield mock_pynvml


@pytest.fixture
def power_manager_instance(mock_logger):
    return power_management.PowerManager()



def test_power_manager_initialization(mock_pynvml, mock_logger):
    # Reset singleton
    power_management.PowerManager._instance = None

    # Cấu hình giá trị trả về khi nvmlDeviceGetCount được gọi
    mock_pynvml.nvmlDeviceGetCount.return_value = 2

    # Khởi tạo PowerManager
    power_manager = power_management.PowerManager()

    # Xác minh nvmlInit được gọi một lần
    mock_pynvml.nvmlInit.assert_called_once()

    # Xác minh thông báo logger
    mock_logger.info.assert_called_with("NVML initialized successfully. Số lượng GPU: 2")

    # Kiểm tra giá trị gpu_count
    assert power_manager.gpu_count == 2


# Test CPU Power Retrieval
def test_get_cpu_power(mock_psutil, mock_logger, power_manager_instance):
    mock_psutil.cpu_percent.return_value = 50.0
    result = power_manager_instance.get_cpu_power()

    assert result == 80.0  # base power + 50% of (max - base)
    mock_logger.debug.assert_called_with("CPU Load: 50.0%, Estimated CPU Power: 80.00W")

def test_get_cpu_power_error(mock_psutil, mock_logger, power_manager_instance):
    mock_psutil.cpu_percent.side_effect = Exception("Test Exception")
    result = power_manager_instance.get_cpu_power()

    assert result == 0.0
    mock_logger.error.assert_called_with("Lỗi khi ước tính công suất CPU: Test Exception")

# Test GPU Power Retrieval
def test_get_gpu_power(mock_pynvml, mock_logger, power_manager_instance):
    mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = [50000, 75000]
    power_manager_instance.gpu_count = 2
    result = power_manager_instance.get_gpu_power()

    assert result == [50.0, 75.0]
    mock_logger.debug.assert_has_calls([
        call("GPU 0: 50.0W"),
        call("GPU 1: 75.0W")
    ])

def test_get_gpu_power_no_gpus(mock_logger, power_manager_instance):
    power_manager_instance.gpu_count = 0
    result = power_manager_instance.get_gpu_power()

    assert result == []
    mock_logger.warning.assert_called_with("Không có GPU nào được phát hiện để giám sát công suất.")

# Test CPU Power Reduction
def test_reduce_cpu_power(mock_psutil, mock_logger, power_manager_instance):
    # Giả lập tần số CPU hiện tại
    mock_psutil.cpu_freq.return_value = MagicMock(current=2400)
    mock_psutil.cpu_count.return_value = 4  # Giả lập hệ thống có 4 lõi CPU

    with patch('auxiliary_modules.power_management.subprocess.run') as mock_run:
        power_manager_instance.reduce_cpu_power(reduction_percentage=50.0)

        # Do new_freq bị giới hạn bởi min_freq = 1800 MHz
        mock_run.assert_has_calls([
            call(['cpufreq-set', '-c', str(cpu), '-f', '1800MHz'], check=True)
            for cpu in range(4)
        ])
        mock_logger.info.assert_called_with("Đã giảm tần số CPU xuống 1800MHz (50.0% giảm).")


def test_reduce_cpu_power_invalid_percentage(mock_logger, power_manager_instance):
    power_manager_instance.reduce_cpu_power(reduction_percentage=150.0)

    mock_logger.error.assert_called_with("Reduction percentage phải nằm trong khoảng (0, 100).")

# Test GPU Power Reduction
def test_reduce_gpu_power(mock_pynvml, mock_logger, power_manager_instance):
    power_manager_instance.gpu_count = 1
    handle = MagicMock()
    mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = handle
    mock_pynvml.nvmlDeviceGetPowerManagementLimit.return_value = 100000
    mock_pynvml.nvmlDeviceGetPowerManagementLimitConstraints.return_value = MagicMock(
        minPowerLimit=50000, maxPowerLimit=100000)

    power_manager_instance.reduce_gpu_power(reduction_percentage=50.0)

    mock_pynvml.nvmlDeviceSetPowerManagementLimit.assert_called_with(handle, 50000)
    mock_logger.info.assert_called_with("Đã giảm giới hạn công suất GPU 0 xuống 50000W (50.0% giảm).")

# Test GPU Usage Setting
def test_set_gpu_usage(mock_pynvml, mock_logger, power_manager_instance):
    power_manager_instance.gpu_count = 2
    mock_pynvml.nvmlDeviceGetPowerManagementLimit.side_effect = [100000, 200000]
    constraints = MagicMock(minPowerLimit=50000, maxPowerLimit=200000)
    mock_pynvml.nvmlDeviceGetPowerManagementLimitConstraints.return_value = constraints

    power_manager_instance.set_gpu_usage([50.0, 75.0])

    mock_pynvml.nvmlDeviceSetPowerManagementLimit.assert_has_calls([
        call(mock_pynvml.nvmlDeviceGetHandleByIndex(0), 50000),
        call(mock_pynvml.nvmlDeviceGetHandleByIndex(1), 150000)
    ])
    mock_logger.info.assert_has_calls([
        call("Đã thiết lập giới hạn công suất GPU 0 thành 50000W (50.0%)."),
        call("Đã thiết lập giới hạn công suất GPU 1 thành 150000W (75.0%).")
    ])

# Test Shutdown
def test_shutdown(mock_pynvml, mock_logger, power_manager_instance):
    power_manager_instance.gpu_count = 1

    power_manager_instance.shutdown()

    mock_pynvml.nvmlShutdown.assert_called_once()
    mock_logger.info.assert_called_with("NVML đã được shutdown thành công.")

def test_shutdown_no_gpus(mock_logger, power_manager_instance):
    power_manager_instance.gpu_count = 0

    power_manager_instance.shutdown()

    mock_logger.info.assert_not_called()
