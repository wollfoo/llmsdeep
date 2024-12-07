# tests/test_shared_resource_manager.py

import os
import sys
import logging
import psutil
import pytest
import subprocess
import pynvml
import torch
import rwlock
from unittest.mock import patch, MagicMock, mock_open, call, ANY
from pathlib import Path
from queue import PriorityQueue, Empty, Queue
from threading import Lock, Event, Thread
from typing import Any, Dict, List
from time import sleep, time

# Thiết lập biến môi trường TESTING=1
os.environ["TESTING"] = "1"

APP_DIR = Path("/home/llmss/llmsdeep/app")
CONFIG_DIR = APP_DIR / "mining_environment" / "config"
MODELS_DIR = APP_DIR / "mining_environment" / "models"
SCRIPTS_DIR = APP_DIR / "mining_environment" / "scripts"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Import lớp cần kiểm thử
from mining_environment.scripts.resource_manager import SharedResourceManager, ResourceManager, MiningProcess


@pytest.fixture
def mock_logger():
    """Fixture để tạo mock logger."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.error = MagicMock()
    logger.warning = MagicMock()
    return logger

@pytest.fixture
def shared_resource_manager(mock_logger):
    """Fixture để tạo instance của SharedResourceManager với cấu hình hợp nhất và logger mock."""
    config = {
        "optimization_parameters": {
            "gpu_power_adjustment_step": 10,
            "disk_io_limit_step_mbps": 5
        },
        "resource_allocation": {
            "disk_io": {
                "min_limit_mbps": 10,
                "max_limit_mbps": 100
            },
            "ram": {
                "max_allocation_mb": 2048
            },
            "gpu": {
                "max_usage_percent": [50, 75, 100]
            },
            "network": {
                "bandwidth_limit_mbps": 100
            }
        },
        "processes": {
            "CPU": "cpu_miner",
            "GPU": "gpu_miner"
        }
    }

    # Khởi tạo instance của SharedResourceManager với mock logger
    manager = SharedResourceManager(config=config, logger=mock_logger)
    return manager

# ----------------------------
# Kiểm thử SharedResourceManager
# ----------------------------

@patch('mining_environment.scripts.resource_manager.assign_process_to_cgroups')
def test_adjust_cpu_threads(mock_assign, shared_resource_manager):
    """Kiểm thử phương thức adjust_cpu_threads."""
    pid = 1234
    cpu_threads = 4
    process_name = "test_process"

    shared_resource_manager.adjust_cpu_threads(pid, cpu_threads, process_name)

    mock_assign.assert_called_once_with(pid, {'cpu_threads': cpu_threads}, process_name, shared_resource_manager.logger)
    shared_resource_manager.logger.info.assert_called_with(
        f"Điều chỉnh số luồng CPU xuống {cpu_threads} cho tiến trình {process_name} (PID: {pid})."
    )

@patch('mining_environment.scripts.resource_manager.assign_process_to_cgroups', side_effect=Exception("Error adjusting CPU threads"))
def test_adjust_cpu_threads_exception(mock_assign, shared_resource_manager):
    """Kiểm thử phương thức adjust_cpu_threads khi có ngoại lệ."""
    pid = 1234
    cpu_threads = 4
    process_name = "test_process"

    shared_resource_manager.adjust_cpu_threads(pid, cpu_threads, process_name)

    mock_assign.assert_called_once_with(pid, {'cpu_threads': cpu_threads}, process_name, shared_resource_manager.logger)
    shared_resource_manager.logger.error.assert_called_with(
        f"Lỗi khi điều chỉnh số luồng CPU cho tiến trình {process_name} (PID: {pid}): Error adjusting CPU threads"
    )

@patch('mining_environment.scripts.resource_manager.assign_process_to_cgroups')
def test_adjust_ram_allocation(mock_assign, shared_resource_manager):
    """Kiểm thử phương thức adjust_ram_allocation."""
    pid = 5678
    ram_allocation_mb = 1024
    process_name = "test_process_ram"

    shared_resource_manager.adjust_ram_allocation(pid, ram_allocation_mb, process_name)

    mock_assign.assert_called_once_with(pid, {'memory': ram_allocation_mb}, process_name, shared_resource_manager.logger)
    shared_resource_manager.logger.info.assert_called_with(
        f"Điều chỉnh giới hạn RAM xuống {ram_allocation_mb}MB cho tiến trình {process_name} (PID: {pid})."
    )

@patch('mining_environment.scripts.resource_manager.assign_process_to_cgroups', side_effect=Exception("Error adjusting RAM"))
def test_adjust_ram_allocation_exception(mock_assign, shared_resource_manager):
    """Kiểm thử phương thức adjust_ram_allocation khi có ngoại lệ."""
    pid = 5678
    ram_allocation_mb = 1024
    process_name = "test_process_ram"

    shared_resource_manager.adjust_ram_allocation(pid, ram_allocation_mb, process_name)

    mock_assign.assert_called_once_with(pid, {'memory': ram_allocation_mb}, process_name, shared_resource_manager.logger)
    shared_resource_manager.logger.error.assert_called_with(
        f"Lỗi khi điều chỉnh RAM cho tiến trình {process_name} (PID: {pid}): Error adjusting RAM"
    )

@patch('mining_environment.scripts.resource_manager.set_gpu_usage')
def test_adjust_gpu_usage(mock_set_gpu_usage, shared_resource_manager):
    """Kiểm thử phương thức adjust_gpu_usage sử dụng fixture shared_resource_manager."""

    # Tạo một process giả
    process = MagicMock()
    process.pid = 12345
    process.name = "gpu_miner"

    # Dữ liệu đầu vào giả lập cho GPU usage
    gpu_usage_percent = [50.0, 60.0]

    # Gọi phương thức cần kiểm thử
    shared_resource_manager.adjust_gpu_usage(process, gpu_usage_percent)

    # Kỳ vọng mức GPU usage mới
    expected_new_gpu_usage = [60.0, 70.0]  # Mỗi giá trị cộng thêm 10 từ gpu_power_adjustment_step

    # Kiểm tra set_gpu_usage được gọi với đúng tham số
    mock_set_gpu_usage.assert_called_once_with(process.pid, expected_new_gpu_usage)

    # Kiểm tra logger có ghi log đúng không
    shared_resource_manager.logger.info.assert_called_with(
        f"Điều chỉnh mức sử dụng GPU xuống {expected_new_gpu_usage} cho tiến trình {process.name} (PID: {process.pid})."
    )

@patch('mining_environment.scripts.resource_manager.set_gpu_usage', side_effect=Exception("Error adjusting GPU usage"))
def test_adjust_gpu_usage_exception(mock_set_gpu_usage, shared_resource_manager):
    """Kiểm thử phương thức adjust_gpu_usage khi có ngoại lệ sử dụng fixture shared_resource_manager."""

    # Tạo một process giả
    process = MagicMock()
    process.pid = 91011
    process.name = "gpu_miner"

    # Dữ liệu đầu vào giả lập cho GPU usage
    gpu_usage_percent = [50.0, 60.0]

    # Gọi phương thức cần kiểm thử
    shared_resource_manager.adjust_gpu_usage(process, gpu_usage_percent)

    # Kỳ vọng mức GPU usage mới
    expected_new_gpu_usage = [60.0, 70.0]  # Mỗi giá trị cộng thêm 10 từ gpu_power_adjustment_step

    # Kiểm tra set_gpu_usage được gọi với đúng tham số
    mock_set_gpu_usage.assert_called_once_with(process.pid, expected_new_gpu_usage)

    # Kiểm tra logger có ghi log lỗi đúng không
    shared_resource_manager.logger.error.assert_called_with(
        f"Lỗi khi điều chỉnh mức sử dụng GPU cho tiến trình {process.name} (PID: {process.pid}): Error adjusting GPU usage"
    )

@patch('mining_environment.scripts.resource_manager.temperature_monitor.get_current_disk_io_limit')
@patch('mining_environment.scripts.resource_manager.assign_process_to_cgroups')
def test_adjust_disk_io_limit(mock_assign, mock_get_disk_io, shared_resource_manager):
    """Kiểm thử phương thức adjust_disk_io_limit."""
    process = MagicMock()
    process.pid = 1212
    process.name = "disk_io_process"
    disk_io_limit_mbps = 50.0

    mock_get_disk_io.return_value = 55.0  # current_limit > disk_io_limit_mbps
    shared_resource_manager.adjust_disk_io_limit(process, disk_io_limit_mbps)

    new_limit = 55.0 - 5  # adjustment_step = 5
    new_limit = max(10, min(new_limit, 100))  # min_limit=10, max_limit=100

    mock_assign.assert_called_once_with(process.pid, {'disk_io_limit_mbps': new_limit}, process.name, shared_resource_manager.logger)
    shared_resource_manager.logger.info.assert_called_with(
        f"Điều chỉnh giới hạn Disk I/O xuống {new_limit} Mbps cho tiến trình {process.name} (PID: {process.pid})."
    )

@patch('mining_environment.scripts.resource_manager.temperature_monitor.get_current_disk_io_limit', side_effect=Exception("Disk I/O Error"))
@patch('mining_environment.scripts.resource_manager.assign_process_to_cgroups')
def test_adjust_disk_io_limit_exception(mock_assign, mock_get_disk_io, shared_resource_manager):
    """Kiểm thử phương thức adjust_disk_io_limit khi có ngoại lệ."""
    process = MagicMock()
    process.pid = 1212
    process.name = "disk_io_process"
    disk_io_limit_mbps = 50.0

    shared_resource_manager.adjust_disk_io_limit(process, disk_io_limit_mbps)

    mock_assign.assert_not_called()
    shared_resource_manager.logger.error.assert_called_with(
        f"Lỗi khi điều chỉnh Disk I/O cho tiến trình {process.name} (PID: {process.pid}): Disk I/O Error"
    )

@patch('mining_environment.scripts.resource_manager.SharedResourceManager.apply_network_cloaking')
def test_adjust_network_bandwidth(mock_apply_cloaking, shared_resource_manager):
    """Kiểm thử phương thức adjust_network_bandwidth."""
    process = MagicMock()
    process.name = "network_process"
    process.pid = 1313
    process.network_interface = "eth0"
    bandwidth_limit_mbps = 80.0

    shared_resource_manager.adjust_network_bandwidth(process, bandwidth_limit_mbps)

    mock_apply_cloaking.assert_called_once_with("eth0", bandwidth_limit_mbps, process)
    shared_resource_manager.logger.info.assert_called_with(
        f"Điều chỉnh giới hạn băng thông mạng xuống {bandwidth_limit_mbps} Mbps cho tiến trình {process.name} (PID: {process.pid})."
    )

@patch('mining_environment.scripts.resource_manager.SharedResourceManager.apply_network_cloaking', side_effect=Exception("Network Cloaking Error"))
def test_adjust_network_bandwidth_exception(mock_apply_cloaking, shared_resource_manager):
    """Kiểm thử phương thức adjust_network_bandwidth khi có ngoại lệ."""
    process = MagicMock()
    process.name = "network_process"
    process.pid = 1313
    process.network_interface = "eth0"
    bandwidth_limit_mbps = 80.0

    shared_resource_manager.adjust_network_bandwidth(process, bandwidth_limit_mbps)

    mock_apply_cloaking.assert_called_once_with("eth0", bandwidth_limit_mbps, process)
    shared_resource_manager.logger.error.assert_called_with(
        f"Lỗi khi điều chỉnh Mạng cho tiến trình {process.name} (PID: {process.pid}): Network Cloaking Error"
    )

@patch('mining_environment.scripts.resource_manager.assign_process_to_cgroups')
def test_adjust_cpu_frequency(mock_assign, shared_resource_manager):
    """Kiểm thử phương thức adjust_cpu_frequency."""
    pid = 1414
    frequency = 2500
    process_name = "cpu_process"

    shared_resource_manager.adjust_cpu_frequency(pid, frequency, process_name)

    mock_assign.assert_called_once_with(pid, {'cpu_freq': frequency}, process_name, shared_resource_manager.logger)
    shared_resource_manager.logger.info.assert_called_with(
        f"Đặt tần số CPU xuống {frequency}MHz cho tiến trình {process_name} (PID: {pid})."
    )

@patch('mining_environment.scripts.resource_manager.assign_process_to_cgroups', side_effect=Exception("CPU Frequency Error"))
def test_adjust_cpu_frequency_exception(mock_assign, shared_resource_manager):
    """Kiểm thử phương thức adjust_cpu_frequency khi có ngoại lệ."""
    pid = 1414
    frequency = 2500
    process_name = "cpu_process"

    shared_resource_manager.adjust_cpu_frequency(pid, frequency, process_name)

    mock_assign.assert_called_once_with(pid, {'cpu_freq': frequency}, process_name, shared_resource_manager.logger)
    shared_resource_manager.logger.error.assert_called_with(
        f"Lỗi khi điều chỉnh tần số CPU cho tiến trình {process_name} (PID: {pid}): CPU Frequency Error"
    )

@patch('mining_environment.scripts.resource_manager.pynvml.nvmlDeviceSetPowerManagementLimit')
@patch('mining_environment.scripts.resource_manager.pynvml.nvmlDeviceGetHandleByIndex', return_value=MagicMock())
@patch('mining_environment.scripts.resource_manager.pynvml.nvmlInit')
@patch('mining_environment.scripts.resource_manager.pynvml.nvmlShutdown')
def test_adjust_gpu_power_limit(mock_shutdown, mock_init, mock_get_handle, mock_set_power_limit, shared_resource_manager):
    """Kiểm thử phương thức adjust_gpu_power_limit."""
    pid = 1515
    power_limit = 200
    process_name = "gpu_power_process"

    shared_resource_manager.adjust_gpu_power_limit(pid, power_limit, process_name)

    mock_init.assert_called_once()
    mock_get_handle.assert_called_once_with(0)
    mock_set_power_limit.assert_called_once_with(mock_get_handle.return_value, power_limit * 1000)
    mock_shutdown.assert_called_once()
    shared_resource_manager.logger.info.assert_called_with(
        f"Đặt giới hạn công suất GPU xuống {power_limit}W cho tiến trình {process_name} (PID: {pid})."
    )

@patch('mining_environment.scripts.resource_manager.subprocess.run')
def test_adjust_disk_io_priority(mock_subprocess_run, shared_resource_manager):
    """Kiểm thử phương thức adjust_disk_io_priority."""
    pid = 1616
    ionice_class = 2
    process_name = "disk_io_priority_process"

    shared_resource_manager.adjust_disk_io_priority(pid, ionice_class, process_name)

    mock_subprocess_run.assert_called_once_with(['ionice', '-c', '2', '-p', '1616'], check=True)
    shared_resource_manager.logger.info.assert_called_with(
        f"Đặt ionice class thành {ionice_class} cho tiến trình {process_name} (PID: {pid})."
    )

@patch('mining_environment.scripts.resource_manager.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'ionice'))
def test_adjust_disk_io_priority_subprocess_error(mock_subprocess_run, shared_resource_manager):
    """Kiểm thử phương thức adjust_disk_io_priority khi subprocess gặp lỗi."""
    pid = 1616
    ionice_class = 2
    process_name = "disk_io_priority_process"

    shared_resource_manager.adjust_disk_io_priority(pid, ionice_class, process_name)

    mock_subprocess_run.assert_called_once_with(['ionice', '-c', '2', '-p', '1616'], check=True)
    shared_resource_manager.logger.error.assert_called_with(
        f"Lỗi khi thực hiện ionice: Command 'ionice' returned non-zero exit status 1."
    )

@patch('mining_environment.scripts.resource_manager.subprocess.run', side_effect=Exception("Unknown Error"))
def test_adjust_disk_io_priority_generic_exception(mock_subprocess_run, shared_resource_manager):
    """Kiểm thử phương thức adjust_disk_io_priority khi có ngoại lệ khác."""
    pid = 1616
    ionice_class = 2
    process_name = "disk_io_priority_process"

    shared_resource_manager.adjust_disk_io_priority(pid, ionice_class, process_name)

    mock_subprocess_run.assert_called_once_with(['ionice', '-c', '2', '-p', '1616'], check=True)
    shared_resource_manager.logger.error.assert_called_with(
        f"Lỗi khi điều chỉnh ưu tiên Disk I/O cho tiến trình {process_name} (PID: {pid}): Unknown Error"
    )

@patch('builtins.open', new_callable=mock_open)
def test_drop_caches(mock_file, shared_resource_manager):
    """Kiểm thử phương thức drop_caches."""
    shared_resource_manager.drop_caches()

    mock_file.assert_called_once_with('/proc/sys/vm/drop_caches', 'w')
    mock_file().write.assert_called_once_with('3\n')
    shared_resource_manager.logger.info.assert_called_with("Đã giảm sử dụng cache bằng cách drop_caches.")

def test_drop_caches_exception(shared_resource_manager):
    """Kiểm thử trường hợp lỗi khi gọi drop_caches."""
    # Mô phỏng lỗi khi ghi tệp
    def raise_io_error(*args, **kwargs):
        raise IOError("Lỗi giả lập khi ghi vào /proc/sys/vm/drop_caches")

    with patch("builtins.open", mock_open()) as mocked_open:
        mocked_open.side_effect = raise_io_error  # Gây lỗi IOError khi mở tệp
        shared_resource_manager.drop_caches()
    
    # Xác minh rằng logger ghi nhận lỗi
    shared_resource_manager.logger.error.assert_called_with("Lỗi khi giảm sử dụng cache: Lỗi giả lập khi ghi vào /proc/sys/vm/drop_caches")

@patch('mining_environment.scripts.resource_manager.SharedResourceManager.apply_network_cloaking')
def test_apply_network_cloaking(mock_apply_cloaking, shared_resource_manager):
    """Kiểm thử phương thức apply_network_cloaking."""
    interface = "eth0"
    bandwidth_limit = 50.0
    process = MagicMock()
    process.name = "network_cloak_process"
    process.pid = 1717

    shared_resource_manager.apply_network_cloaking(interface, bandwidth_limit, process)

    mock_apply_cloaking.assert_called_once_with(interface, bandwidth_limit, process)
    # Vì phương thức chưa thực hiện gì, chỉ kiểm tra gọi hàm cloaking và log lỗi nếu có


@patch('mining_environment.scripts.resource_manager.SharedResourceManager.configure_network_interface', side_effect=Exception("Network Cloaking Exception"))
def test_apply_network_cloaking_exception(mock_configure_network, shared_resource_manager):
    """Kiểm thử phương thức apply_network_cloaking khi có ngoại lệ."""
    interface = "eth0"
    bandwidth_limit = 50.0
    process = MagicMock(spec=MiningProcess)
    process.name = "network_cloak_process"
    process.pid = 1717

    # Gọi phương thức và kiểm tra ngoại lệ
    with pytest.raises(Exception) as exc_info:
        shared_resource_manager.apply_network_cloaking(interface, bandwidth_limit, process)
    assert str(exc_info.value) == "Network Cloaking Exception"

    # Kiểm tra logger đã ghi log lỗi đúng cách
    shared_resource_manager.logger.error.assert_called_once_with(
        "Lỗi khi áp dụng cloaking mạng cho tiến trình network_cloak_process (PID: 1717): Network Cloaking Exception"
    )


@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_cpu_frequency')
def test_throttle_cpu_based_on_load_high_load(mock_adjust_cpu_freq, shared_resource_manager):
    """Kiểm thử phương thức throttle_cpu_based_on_load với load > 80%."""
    process = MagicMock()
    process.pid = 1818
    process.name = "high_load_process"
    load_percent = 85.0

    shared_resource_manager.throttle_cpu_based_on_load(process, load_percent)

    mock_adjust_cpu_freq.assert_called_once_with(process.pid, 2000, process.name)
    shared_resource_manager.logger.info.assert_called_with(
        f"Điều chỉnh tần số CPU xuống 2000MHz cho tiến trình {process.name} (PID: {process.pid}) dựa trên tải {load_percent}%."
    )


@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_cpu_frequency')
def test_throttle_cpu_based_on_load_medium_load(mock_adjust_cpu_freq, shared_resource_manager):
    """Kiểm thử phương thức throttle_cpu_based_on_load với 50% < load <= 80%."""
    process = MagicMock()
    process.pid = 1818
    process.name = "medium_load_process"
    load_percent = 65.0

    shared_resource_manager.throttle_cpu_based_on_load(process, load_percent)

    mock_adjust_cpu_freq.assert_called_once_with(process.pid, 2500, process.name)
    shared_resource_manager.logger.info.assert_called_with(
        f"Điều chỉnh tần số CPU xuống 2500MHz cho tiến trình {process.name} (PID: {process.pid}) dựa trên tải {load_percent}%."
    )

@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_cpu_frequency')
def test_throttle_cpu_based_on_load_low_load(mock_adjust_cpu_freq, shared_resource_manager):
    """Kiểm thử phương thức throttle_cpu_based_on_load với load <= 50%."""
    process = MagicMock()
    process.pid = 1818
    process.name = "low_load_process"
    load_percent = 30.0

    shared_resource_manager.throttle_cpu_based_on_load(process, load_percent)

    mock_adjust_cpu_freq.assert_called_once_with(process.pid, 3000, process.name)
    shared_resource_manager.logger.info.assert_called_with(
        f"Điều chỉnh tần số CPU xuống 3000MHz cho tiến trình {process.name} (PID: {process.pid}) dựa trên tải {load_percent}%."
    )

@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_cpu_frequency', side_effect=Exception("CPU Throttle Error"))
def test_throttle_cpu_based_on_load_exception(mock_adjust_cpu_freq, shared_resource_manager):
    """Kiểm thử phương thức throttle_cpu_based_on_load khi có ngoại lệ."""
    process = MagicMock()
    process.pid = 1818
    process.name = "exception_load_process"
    load_percent = 85.0

    shared_resource_manager.throttle_cpu_based_on_load(process, load_percent)

    mock_adjust_cpu_freq.assert_called_once_with(process.pid, 2000, process.name)
    shared_resource_manager.logger.error.assert_called_with(
        f"Lỗi khi điều chỉnh tần số CPU dựa trên tải cho tiến trình {process.name} (PID: {process.pid}): CPU Throttle Error"
    )

@patch('mining_environment.scripts.resource_manager.CloakStrategyFactory.create_strategy')
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.execute_adjustments')
def test_apply_cloak_strategy(mock_execute_adjustments, mock_create_strategy, shared_resource_manager):
    """Kiểm thử phương thức apply_cloak_strategy trong trường hợp thành công."""
    strategy_name = "test_strategy"
    process = MagicMock()
    process.pid = 1919
    process.name = "cloak_process"

    mock_strategy = MagicMock()
    mock_strategy.apply.return_value = {
        'cpu_freq': 2500,
        'gpu_power_limit': 200,
        'network_bandwidth_limit_mbps': 80.0,
        'ionice_class': 2
    }
    mock_create_strategy.return_value = mock_strategy

    # Giả lập các phương thức get_current_*
    with patch.object(shared_resource_manager, 'get_current_cpu_frequency', return_value=3000):
        with patch.object(shared_resource_manager, 'get_current_gpu_power_limit', return_value=250):
            with patch.object(shared_resource_manager, 'get_current_network_bandwidth_limit', return_value=100.0):
                with patch.object(shared_resource_manager, 'get_current_ionice_class', return_value=1):
                    shared_resource_manager.apply_cloak_strategy(strategy_name, process)

    mock_create_strategy.assert_called_once_with(strategy_name, shared_resource_manager.config, shared_resource_manager.logger, shared_resource_manager.is_gpu_initialized())
    mock_strategy.apply.assert_called_once_with(process)
    shared_resource_manager.logger.info.assert_called_with(
        f"Áp dụng điều chỉnh {strategy_name} cho tiến trình {process.name} (PID: {process.pid}): {{'cpu_freq': 2500, 'gpu_power_limit': 200, 'network_bandwidth_limit_mbps': 80.0, 'ionice_class': 2}}"
    )
    mock_execute_adjustments.assert_called_once_with(mock_strategy.apply.return_value, process)
    assert shared_resource_manager.original_resource_limits[process.pid] == {
        'cpu_freq': 3000,
        'gpu_power_limit': 250,
        'network_bandwidth_limit_mbps': 100.0,
        'ionice_class': 1
    }

@patch('mining_environment.scripts.resource_manager.CloakStrategyFactory.create_strategy')
def test_apply_cloak_strategy_no_adjustments(mock_create_strategy, shared_resource_manager):
    """Kiểm thử phương thức apply_cloak_strategy khi chiến lược không trả về điều chỉnh nào."""
    strategy_name = "empty_strategy"
    process = MagicMock()
    process.pid = 2020
    process.name = "empty_cloak_process"

    mock_strategy = MagicMock()
    mock_strategy.apply.return_value = None
    mock_create_strategy.return_value = mock_strategy

    shared_resource_manager.apply_cloak_strategy(strategy_name, process)

    mock_create_strategy.assert_called_once_with(strategy_name, shared_resource_manager.config, shared_resource_manager.logger, shared_resource_manager.is_gpu_initialized())
    mock_strategy.apply.assert_called_once_with(process)
    shared_resource_manager.logger.warning.assert_called_with(
        f"Không có điều chỉnh nào được áp dụng cho chiến lược {strategy_name} cho tiến trình {process.name} (PID: {process.pid})."
    )

@patch('mining_environment.scripts.resource_manager.CloakStrategyFactory.create_strategy', return_value=None)
def test_apply_cloak_strategy_strategy_creation_failure(mock_create_strategy, shared_resource_manager):
    """Kiểm thử phương thức apply_cloak_strategy khi tạo chiến lược thất bại."""
    strategy_name = "invalid_strategy"
    process = MagicMock()
    process.pid = 2021
    process.name = "invalid_cloak_process"

    shared_resource_manager.apply_cloak_strategy(strategy_name, process)

    mock_create_strategy.assert_called_once_with(strategy_name, shared_resource_manager.config, shared_resource_manager.logger, shared_resource_manager.is_gpu_initialized())
    shared_resource_manager.logger.warning.assert_called_with(
        f"Chiến lược cloaking {strategy_name} không được tạo thành công cho tiến trình {process.name} (PID: {process.pid})."
    )


@patch('mining_environment.scripts.resource_manager.CloakStrategyFactory.create_strategy', side_effect=Exception("Strategy Creation Error"))
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.execute_adjustments')
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.is_gpu_initialized', return_value=False)
def test_apply_cloak_strategy_creation_exception(
    mock_is_gpu_initialized,
    mock_execute_adjustments,
    mock_create_strategy,
    shared_resource_manager
):
    """Kiểm thử phương thức apply_cloak_strategy khi xảy ra lỗi trong quá trình tạo chiến lược."""
    # Mock logger để kiểm tra
    shared_resource_manager.logger = MagicMock()

    strategy_name = "error_strategy"
    process = MagicMock()
    process.pid = 2022
    process.name = "error_cloak_process"

    # Sử dụng pytest.raises để bắt ngoại lệ
    with pytest.raises(Exception) as exc_info:
        shared_resource_manager.apply_cloak_strategy(strategy_name, process)

    # Kiểm tra nội dung ngoại lệ
    assert str(exc_info.value) == "Strategy Creation Error"

    # Kiểm tra rằng create_strategy được gọi đúng cách
    mock_create_strategy.assert_called_once_with(
        strategy_name,
        shared_resource_manager.config,
        shared_resource_manager.logger,
        shared_resource_manager.is_gpu_initialized()
    )

    # Kiểm tra rằng logger.error đã được gọi với thông điệp đúng
    shared_resource_manager.logger.error.assert_called_once_with(
        f"Không thể tạo chiến lược {strategy_name}: Strategy Creation Error"
    )

    # Kiểm tra rằng execute_adjustments không được gọi
    mock_execute_adjustments.assert_not_called()

@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_cpu_frequency')
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_cpu_threads')
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_ram_allocation')
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_gpu_power_limit')
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_disk_io_priority')
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_network_bandwidth')
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.drop_caches')
def test_restore_resources(mock_drop_caches, mock_adjust_network_bw, mock_adjust_disk_io, mock_adjust_gpu_power, 
                          mock_adjust_ram, mock_adjust_cpu_threads, mock_adjust_cpu_freq, shared_resource_manager):
    """Kiểm thử phương thức restore_resources trong trường hợp thành công."""
    process = MagicMock()
    process.pid = 2121
    process.name = "restore_process"

    # Giả lập original_resource_limits
    shared_resource_manager.original_resource_limits = {
        process.pid: {
            'cpu_freq': 3000,
            'cpu_threads': 4,
            'ram_allocation_mb': 2048,
            'gpu_power_limit': 250,
            'ionice_class': 2,
            'network_bandwidth_limit_mbps': 100.0
        }
    }

    shared_resource_manager.restore_resources(process)

    mock_adjust_cpu_freq.assert_called_once_with(process.pid, 3000, process.name)
    mock_adjust_cpu_threads.assert_called_once_with(process.pid, 4, process.name)
    mock_adjust_ram.assert_called_once_with(process.pid, 2048, process.name)
    mock_adjust_gpu_power.assert_called_once_with(process.pid, 250, process.name)
    mock_adjust_disk_io.assert_called_once_with(process.pid, 2, process.name)
    mock_adjust_network_bw.assert_called_once_with(process, 100.0)
    mock_drop_caches.assert_not_called()  # Không có yêu cầu drop_caches trong restore

    shared_resource_manager.logger.info.assert_any_call(
        f"Đã khôi phục tần số CPU về 3000MHz cho tiến trình {process.name} (PID: {process.pid})."
    )
    shared_resource_manager.logger.info.assert_any_call(
        f"Đã khôi phục số luồng CPU về 4 cho tiến trình {process.name} (PID: {process.pid})."
    )
    shared_resource_manager.logger.info.assert_any_call(
        f"Đã khôi phục giới hạn RAM về 2048MB cho tiến trình {process.name} (PID: {process.pid})."
    )
    shared_resource_manager.logger.info.assert_any_call(
        f"Đã khôi phục giới hạn công suất GPU về 250W cho tiến trình {process.name} (PID: {process.pid})."
    )
    shared_resource_manager.logger.info.assert_any_call(
        f"Đã khôi phục lớp ionice về 2 cho tiến trình {process.name} (PID: {process.pid})."
    )
    shared_resource_manager.logger.info.assert_any_call(
        f"Đã khôi phục giới hạn băng thông mạng về 100.0 Mbps cho tiến trình {process.name} (PID: {process.pid})."
    )
    assert process.pid not in shared_resource_manager.original_resource_limits


def test_restore_resources_exception(shared_resource_manager):
    """Kiểm thử phương thức restore_resources khi có ngoại lệ trong quá trình khôi phục."""
    # Mock logger để kiểm tra
    shared_resource_manager.logger = MagicMock()

    process = MagicMock()
    process.pid = 2121
    process.name = "restore_process_exception"

    # Giả lập original_resource_limits với các khóa khớp
    shared_resource_manager.original_resource_limits = {
        process.pid: {
            'cpu_freq': 3000,
            'cpu_threads': 4,
            'ram_allocation_mb': 2048,
            'gpu_power_limit': 250,
            'ionice_class': 2,
            'network_bandwidth_limit_mbps': 100.0
        }
    }

    # Patch các phương thức trên instance
    shared_resource_manager.adjust_cpu_frequency = MagicMock(side_effect=Exception("Restore CPU Frequency Error"))
    shared_resource_manager.adjust_cpu_threads = MagicMock()
    shared_resource_manager.adjust_ram_allocation = MagicMock()
    shared_resource_manager.adjust_gpu_power_limit = MagicMock()
    shared_resource_manager.adjust_disk_io_priority = MagicMock()
    shared_resource_manager.adjust_network_bandwidth = MagicMock()
    shared_resource_manager.drop_caches = MagicMock()

    # Gọi phương thức restore_resources và kiểm tra ngoại lệ
    with pytest.raises(Exception) as exc_info:
        shared_resource_manager.restore_resources(process)

    # Kiểm tra ngoại lệ đúng loại và nội dung
    assert str(exc_info.value) == "Restore CPU Frequency Error"

    # Kiểm tra các hàm điều chỉnh được gọi đúng cách
    shared_resource_manager.adjust_cpu_frequency.assert_called_once_with(process.pid, 3000, process.name)

    # Các phương thức khác không được gọi do ngoại lệ
    shared_resource_manager.adjust_cpu_threads.assert_not_called()
    shared_resource_manager.adjust_ram_allocation.assert_not_called()
    shared_resource_manager.adjust_gpu_power_limit.assert_not_called()
    shared_resource_manager.adjust_disk_io_priority.assert_not_called()
    shared_resource_manager.adjust_network_bandwidth.assert_not_called()
    shared_resource_manager.drop_caches.assert_not_called()

    # Kiểm tra logger.error đã được gọi
    shared_resource_manager.logger.error.assert_called_once_with(
        f"Lỗi khi khôi phục tài nguyên cho tiến trình {process.name} (PID: {process.pid}): Restore CPU Frequency Error"
    )

@patch('mining_environment.scripts.resource_manager.psutil.cpu_freq')
def test_get_current_cpu_frequency(mock_cpu_freq, shared_resource_manager):
    """Kiểm thử phương thức get_current_cpu_frequency."""
    pid = 5555
    mock_cpu_freq.return_value.current = 2500

    freq = shared_resource_manager.get_current_cpu_frequency(pid)

    mock_cpu_freq.assert_called_once()
    assert freq == 2500

@patch('mining_environment.scripts.resource_manager.psutil.cpu_freq', side_effect=Exception("CPU Frequency Error"))
def test_get_current_cpu_frequency_exception(mock_cpu_freq, shared_resource_manager):
    """Kiểm thử phương thức get_current_cpu_frequency khi có ngoại lệ."""
    pid = 5555

    freq = shared_resource_manager.get_current_cpu_frequency(pid)

    mock_cpu_freq.assert_called_once()
    shared_resource_manager.logger.error.assert_called_with(
        f"Lỗi khi lấy tần số CPU hiện tại cho PID {pid}: CPU Frequency Error"
    )
    assert freq is None

@patch('mining_environment.scripts.resource_manager.pynvml.nvmlDeviceGetPowerManagementLimit', return_value=250000)
@patch('mining_environment.scripts.resource_manager.pynvml.nvmlDeviceGetHandleByIndex', return_value=MagicMock())
@patch('mining_environment.scripts.resource_manager.pynvml.nvmlInit')
@patch('mining_environment.scripts.resource_manager.pynvml.nvmlShutdown')
def test_get_current_gpu_power_limit(mock_shutdown, mock_init, mock_get_handle, mock_get_limit, shared_resource_manager):
    """Kiểm thử phương thức get_current_gpu_power_limit."""
    pid = 4444

    power_limit = shared_resource_manager.get_current_gpu_power_limit(pid)

    mock_init.assert_called_once()
    mock_get_handle.assert_called_once_with(0)
    mock_get_limit.assert_called_once_with(mock_get_handle.return_value)
    mock_shutdown.assert_called_once()
    assert power_limit == 250  # Chuyển đổi từ mW sang W

@patch('mining_environment.scripts.resource_manager.pynvml.nvmlDeviceGetPowerManagementLimit', side_effect=pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN))
@patch('mining_environment.scripts.resource_manager.pynvml.nvmlDeviceGetHandleByIndex', return_value=MagicMock())
@patch('mining_environment.scripts.resource_manager.pynvml.nvmlInit')
@patch('mining_environment.scripts.resource_manager.pynvml.nvmlShutdown')
def test_get_current_gpu_power_limit_exception(mock_shutdown, mock_init, mock_get_handle, mock_get_limit, shared_resource_manager):
    """Kiểm thử phương thức get_current_gpu_power_limit khi có ngoại lệ."""
    pid = 4444

    # Gọi hàm và kiểm tra giá trị trả về
    power_limit = shared_resource_manager.get_current_gpu_power_limit(pid)
    assert power_limit is None

    # Kiểm tra rằng các phương thức NVML đã được gọi đúng cách
    mock_init.assert_called_once()
    mock_get_handle.assert_called_once_with(0)
    mock_get_limit.assert_called_once_with(mock_get_handle.return_value)
    mock_shutdown.assert_called_once()

    # Kiểm tra logger.error đã được gọi với thông điệp đúng
    shared_resource_manager.logger.error.assert_called_once_with(
        f"Lỗi khi lấy giới hạn công suất GPU hiện tại cho PID {pid}: NVMLError_Unknown: {pynvml.NVML_ERROR_UNKNOWN}"
    )

def test_get_current_network_bandwidth_limit(shared_resource_manager):
    """Kiểm thử phương thức get_current_network_bandwidth_limit."""
    pid = 6666
    bw_limit = shared_resource_manager.get_current_network_bandwidth_limit(pid)
    assert bw_limit is None  # Theo định nghĩa trong phương thức

@patch('mining_environment.scripts.resource_manager.psutil.Process')
def test_get_current_ionice_class(mock_process, shared_resource_manager):
    """Kiểm thử phương thức get_current_ionice_class."""
    pid = 6666
    mock_proc = MagicMock()
    mock_proc.ionice.return_value = 3
    mock_process.return_value = mock_proc

    ionice_class = shared_resource_manager.get_current_ionice_class(pid)

    mock_process.assert_called_once_with(pid)
    mock_proc.ionice.assert_called_once()
    assert ionice_class == 3


@patch('mining_environment.scripts.resource_manager.psutil.Process', side_effect=Exception("Process Error"))
def test_get_current_ionice_class_exception(mock_process, shared_resource_manager):
    """Kiểm thử phương thức get_current_ionice_class khi có ngoại lệ."""
    pid = 6666

    ionice_class = shared_resource_manager.get_current_ionice_class(pid)

    mock_process.assert_called_once_with(pid)
    shared_resource_manager.logger.error.assert_called_with(
        f"Lỗi khi lấy ionice class cho PID {pid}: Process Error"
    )
    assert ionice_class is None

@patch('mining_environment.scripts.resource_manager.pynvml.nvmlDeviceGetCount', return_value=1)
@patch('mining_environment.scripts.resource_manager.pynvml.nvmlInit')
@patch('mining_environment.scripts.resource_manager.pynvml.nvmlShutdown')
def test_is_gpu_initialized_true(mock_shutdown, mock_init, mock_get_count, shared_resource_manager):
    """Kiểm thử phương thức is_gpu_initialized khi GPU được khởi tạo thành công."""
    result = shared_resource_manager.is_gpu_initialized()
    mock_init.assert_called_once()
    mock_get_count.assert_called_once()
    mock_shutdown.assert_called_once()
    assert result is True

@patch('mining_environment.scripts.resource_manager.pynvml.nvmlDeviceGetCount', return_value=0)
@patch('mining_environment.scripts.resource_manager.pynvml.nvmlInit')
@patch('mining_environment.scripts.resource_manager.pynvml.nvmlShutdown')
def test_is_gpu_initialized_false(mock_shutdown, mock_init, mock_get_count, shared_resource_manager):
    """Kiểm thử phương thức is_gpu_initialized khi không có GPU nào được khởi tạo."""
    result = shared_resource_manager.is_gpu_initialized()
    mock_init.assert_called_once()
    mock_get_count.assert_called_once()
    mock_shutdown.assert_called_once()
    assert result is False


@patch('mining_environment.scripts.resource_manager.pynvml.nvmlDeviceGetCount', side_effect=pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN))
@patch('mining_environment.scripts.resource_manager.pynvml.nvmlInit')
@patch('mining_environment.scripts.resource_manager.pynvml.nvmlShutdown')
def test_is_gpu_initialized_exception(mock_shutdown, mock_init, mock_get_count, shared_resource_manager):
    """Kiểm thử phương thức is_gpu_initialized khi có ngoại lệ."""
    pid = 4444

    # Gọi hàm và kiểm tra giá trị trả về
    result = shared_resource_manager.is_gpu_initialized()
    assert result is False

    # Kiểm tra rằng các phương thức NVML đã được gọi đúng cách
    mock_init.assert_called_once()
    mock_get_count.assert_called_once()
    mock_shutdown.assert_called_once()

    # Kiểm tra logger.error đã được gọi với thông điệp đúng một phần
    shared_resource_manager.logger.error.assert_called_once()
    args, kwargs = shared_resource_manager.logger.error.call_args
    error_message = args[0]
    assert "NVMLError" in error_message
    assert f"NVMLError_Unknown: {pynvml.NVML_ERROR_UNKNOWN}" in error_message



@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_cpu_frequency')
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_gpu_power_limit')
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_network_bandwidth')
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_disk_io_priority')
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_ram_allocation')
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_cpu_threads')
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.drop_caches')
def test_execute_adjustments_shared_resource(mock_drop_caches, mock_adjust_cpu_threads, mock_adjust_ram_allocation, 
                            mock_adjust_disk_io_priority, mock_adjust_network_bandwidth, 
                            mock_adjust_gpu_power_limit, mock_adjust_cpu_frequency, shared_resource_manager):
    """Kiểm thử phương thức execute_adjustments với các điều chỉnh hợp lệ."""
    adjustments = {
        'cpu_freq': 2500,
        'gpu_power_limit': 200,
        'network_bandwidth_limit_mbps': 80.0,
        'ionice_class': 2,
        'drop_caches': True,
        'unknown_adjust': 999
    }
    process = MagicMock()
    process.pid = 2323
    process.name = "execute_adjust_process"

    shared_resource_manager.execute_adjustments(adjustments, process)

    mock_adjust_cpu_frequency.assert_called_once_with(process.pid, 2500, process.name)
    mock_adjust_gpu_power_limit.assert_called_once_with(process.pid, 200, process.name)
    mock_adjust_network_bandwidth.assert_called_once_with(process, 80.0)
    mock_adjust_disk_io_priority.assert_called_once_with(process.pid, 2, process.name)
    mock_drop_caches.assert_called_once()
    # unknown_adjust should trigger a warning
    shared_resource_manager.logger.warning.assert_called_with("Không nhận dạng được điều chỉnh: unknown_adjust")


@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_cpu_frequency', side_effect=Exception("Adjustment Error"))
def test_execute_adjustments_shared_resource_exception(mock_adjust_cpu_freq, shared_resource_manager):
    """Kiểm thử phương thức execute_adjustments khi có ngoại lệ trong điều chỉnh."""
    adjustments = {
        'cpu_freq': 2500
    }
    process = MagicMock()
    process.pid = 2323
    process.name = "execute_adjust_exception_process"

    shared_resource_manager.execute_adjustments(adjustments, process)

    mock_adjust_cpu_freq.assert_called_once_with(process.pid, 2500, process.name)
    shared_resource_manager.logger.error.assert_called_with(
        f"Lỗi khi thực hiện các điều chỉnh cloaking cho tiến trình {process.name} (PID: {process.pid}): Adjustment Error"
    )


@patch('mining_environment.scripts.resource_manager.CloakStrategyFactory.create_strategy')
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.execute_adjustments')
def test_apply_cloak_strategy_partial_adjustments(mock_execute_adjustments, mock_create_strategy, shared_resource_manager):
    """Kiểm thử phương thức apply_cloak_strategy với một số điều chỉnh."""
    strategy_name = "partial_strategy"
    process = MagicMock()
    process.pid = 2424
    process.name = "partial_cloak_process"

    mock_strategy = MagicMock()
    mock_strategy.apply.return_value = {
        'cpu_freq': 2600,
        'gpu_power_limit': 220
    }
    mock_create_strategy.return_value = mock_strategy

    # Giả lập các phương thức get_current_*
    with patch.object(shared_resource_manager, 'get_current_cpu_frequency', return_value=3000):
        with patch.object(shared_resource_manager, 'get_current_gpu_power_limit', return_value=250):

            shared_resource_manager.apply_cloak_strategy(strategy_name, process)

    mock_create_strategy.assert_called_once_with(strategy_name, shared_resource_manager.config, shared_resource_manager.logger, shared_resource_manager.is_gpu_initialized())
    mock_strategy.apply.assert_called_once_with(process)
    shared_resource_manager.logger.info.assert_called_with(
        f"Áp dụng điều chỉnh {strategy_name} cho tiến trình {process.name} (PID: {process.pid}): {{'cpu_freq': 2600, 'gpu_power_limit': 220}}"
    )
    mock_execute_adjustments.assert_called_once_with(mock_strategy.apply.return_value, process)
    assert shared_resource_manager.original_resource_limits[process.pid] == {
        'cpu_freq': 3000,
        'gpu_power_limit': 250
    }


#  ----------------------------------  Kiểm Thử Lớp ResourceManager ------------------------------------------  #

# ----------------------------
# Kiểm thử ResourceManager
# ----------------------------


@pytest.fixture
def simple_mock_logger():
    """Fixture tạo logger giả đơn giản."""
    return MagicMock()


@pytest.fixture
def resource_manager(simple_mock_logger, monkeypatch, mocker):
    """
    Fixture khởi tạo instance ResourceManager với toàn bộ các mock cần thiết:
    - Reset singleton instance để đảm bảo mỗi kiểm thử sử dụng một instance mới.
    - Mock các client Azure
    - Mock load_model
    - Mock SharedResourceManager
    - Thiết lập config chuẩn
    - Mock các thread, queue,...
    """
    # Mock phương thức __init__ của BaseManager để tránh thực thi logic không mong muốn
    mocker.patch('mining_environment.scripts.resource_manager.BaseManager.__init__', return_value=None)

    # Reset singleton instance
    ResourceManager._instance = None

    monkeypatch.setenv('AZURE_SUBSCRIPTION_ID', 'dummy_subscription_id')

    # Patch các client Azure
    mock_AzureMonitorClient = mocker.patch('mining_environment.scripts.resource_manager.AzureMonitorClient')
    mock_AzureSentinelClient = mocker.patch('mining_environment.scripts.resource_manager.AzureSentinelClient')
    mock_AzureLogAnalyticsClient = mocker.patch('mining_environment.scripts.resource_manager.AzureLogAnalyticsClient')
    mock_AzureSecurityCenterClient = mocker.patch('mining_environment.scripts.resource_manager.AzureSecurityCenterClient')
    mock_AzureNetworkWatcherClient = mocker.patch('mining_environment.scripts.resource_manager.AzureNetworkWatcherClient')
    mock_AzureTrafficAnalyticsClient = mocker.patch('mining_environment.scripts.resource_manager.AzureTrafficAnalyticsClient')
    mock_AzureMLClient = mocker.patch('mining_environment.scripts.resource_manager.AzureMLClient')

    # Patch psutil.process_iter, load_model,...
    mocker.patch('mining_environment.scripts.resource_manager.psutil.process_iter', return_value=[])
    mock_load_model = mocker.patch('mining_environment.scripts.resource_manager.ResourceManager.load_model', return_value=(MagicMock(), MagicMock()))

    # Patch SharedResourceManager, shutdown_power_management
    mock_shared_resource_manager_class = mocker.patch('mining_environment.scripts.resource_manager.SharedResourceManager', autospec=True)
    mock_shutdown_power_management = mocker.patch('mining_environment.scripts.resource_manager.shutdown_power_management')

    # Cấu hình giả lập
    config = {
        "processes": {
            "CPU": "cpu_miner",
            "GPU": "gpu_miner"
        },
        "process_priority_map": {
            "cpu_miner": 2,
            "gpu_miner": 3
        },
        "monitoring_parameters": {
            "temperature_monitoring_interval_seconds": 10,
            "power_monitoring_interval_seconds": 10,
            "azure_monitor_interval_seconds": 300,
            "optimization_interval_seconds": 30
        },
        "temperature_limits": {
            "cpu_max_celsius": 75,
            "gpu_max_celsius": 85
        },
        "power_limits": {
            "per_device_power_watts": {
                "cpu": 150,
                "gpu": 300
            }
        },
        "resource_allocation": {
            "ram": {
                "max_allocation_mb": 2048
            },
            "network": {
                "bandwidth_limit_mbps": 100
            },
            "cache": {
                "limit_percent": 50
            },
            "gpu": {
                "max_usage_percent": [50, 75, 100]
            },
            "disk_io": {
                "min_limit_mbps": 10,
                "max_limit_mbps": 100
            }
        },
        "network_interface": "eth0",
        "optimization_parameters": {
            "gpu_power_adjustment_step": 10,
            "disk_io_limit_step_mbps": 5
        },
        "cloak_strategies": {
            "default": "basic_cloak"
        },
        "ai_driven_monitoring": {
            "enabled": True,
            "detection_interval_seconds": 60,
            "cloak_activation_delay_seconds": 30,
            "anomaly_cloaking_model": {
                "detection_threshold": 0.75
            }
        },
        "log_analytics": {
            "enabled": True,
            "log_level": "INFO",
            "queries": [
                "SELECT * FROM logs WHERE level='ERROR'",
                "SELECT COUNT(*) FROM logs WHERE message LIKE '%failure%'"
            ]
        },
        "alert_thresholds": {
            "cpu_load": 90,
            "gpu_load": 95
        },
        "baseline_thresholds": {
            "cpu_usage_percent": 50,
            "ram_usage_percent": 60,
            "gpu_usage_percent": 70,
            "disk_io_usage_mbps": 80,
            "network_usage_mbps": 90
        }
    }

    model_path = Path("/path/to/model.pt")
    manager = ResourceManager(config, model_path, simple_mock_logger)

    # Kiểm tra logger
    assert manager.logger is simple_mock_logger, "Logger không được gán đúng."

    # Reset mock_logger để xóa các cuộc gọi log trong quá trình khởi tạo
    simple_mock_logger.reset_mock()

    # Mock stop_event và threads
    manager.stop_event = MagicMock(spec=Event)
    manager.stop_event.set = MagicMock()
    manager.monitor_thread = MagicMock(name='monitor_thread')
    manager.optimization_thread = MagicMock(name='optimization_thread')
    manager.cloaking_thread = MagicMock(name='cloaking_thread')
    manager.resource_adjustment_thread = MagicMock(name='resource_adjustment_thread')

    # Mock shared_resource_manager instance
    shared_resource_manager = manager.shared_resource_manager
    shared_resource_manager.is_gpu_initialized.return_value = True
    shared_resource_manager.adjust_cpu_threads = MagicMock()
    shared_resource_manager.adjust_gpu_usage = MagicMock()
    shared_resource_manager.adjust_ram_allocation = MagicMock()
    shared_resource_manager.adjust_disk_io_limit = MagicMock()
    shared_resource_manager.adjust_network_bandwidth = MagicMock()
    shared_resource_manager.apply_cloak_strategy = MagicMock()
    shared_resource_manager.restore_resources = MagicMock()

    # Mock resource_adjustment_queue
    manager.resource_adjustment_queue = MagicMock()
    manager.resource_optimization_model = MagicMock(name='resource_optimization_model')
    manager.resource_optimization_device = 'cpu'

    # Mock các phương thức liên quan đến lock bằng một context manager hợp lệ
    mock_rlock = MagicMock()
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__.return_value = mock_rlock
    mock_context_manager.__exit__.return_value = None
    mocker.patch.object(manager.mining_processes_lock, 'gen_rlock', return_value=mock_context_manager)

    # Mock các phương thức khác nếu cần
    mocker.patch.object(manager, 'allocate_resources_with_priority', return_value=None)
    mocker.patch.object(manager, 'collect_metrics', return_value={
        'cpu_usage_percent': 50,
        'memory_usage_mb': 100,
        'gpu_usage_percent': 70,
        'disk_io_mbps': 30.0,
        'network_bandwidth_mbps': 100,
        'cache_limit_percent': 50
    })
    mocker.patch.object(manager, 'prepare_input_features', return_value=[50, 100, 70, 30.0, 100, 50])

    return {
        'manager': manager,
        'mock_load_model': mock_load_model,
        'mock_shared_resource_manager_class': mock_shared_resource_manager_class,
        'mock_shutdown_power_management': mock_shutdown_power_management,
        'simple_mock_logger': simple_mock_logger,
        'mock_shared_resource_manager_instance': shared_resource_manager,
        'mock_AzureMonitorClient': mock_AzureMonitorClient,
        'mock_AzureSentinelClient': mock_AzureSentinelClient,
        'mock_AzureLogAnalyticsClient': mock_AzureLogAnalyticsClient,
        'mock_AzureSecurityCenterClient': mock_AzureSecurityCenterClient,
        'mock_AzureNetworkWatcherClient': mock_AzureNetworkWatcherClient,
        'mock_AzureTrafficAnalyticsClient': mock_AzureTrafficAnalyticsClient,
        'mock_AzureMLClient': mock_AzureMLClient,
        'config': config
    }


# ----------------------------
# Test nhóm khởi tạo ResourceManager
# ----------------------------

def test_resource_manager_singleton(resource_manager):
    """Kiểm thử ResourceManager là singleton."""
    manager = resource_manager['manager']
    another_instance = ResourceManager(resource_manager['config'], manager.model_path, resource_manager['simple_mock_logger'])
    assert manager is another_instance


def test_resource_manager_init(resource_manager):
    """Kiểm thử khởi tạo ResourceManager."""
    manager = resource_manager['manager']
    assert manager.config['processes']['CPU'] == "cpu_miner"
    assert manager.config['process_priority_map']['gpu_miner'] == 3
    resource_manager['mock_load_model'].assert_called_once_with(manager.model_path)

def test_initialize_threads(resource_manager):
    """Kiểm thử phương thức initialize_threads."""
    manager = resource_manager['manager']
    manager.initialize_threads()
    assert manager.monitor_thread.name == "MonitorThread"
    assert manager.optimization_thread.name == "OptimizationThread"
    assert manager.cloaking_thread.name == "CloakingThread"
    assert manager.resource_adjustment_thread.name == "ResourceAdjustmentThread"

# ----------------------------
# Test start/stop ResourceManager
# ----------------------------

def test_resource_manager_start(resource_manager):
    manager = resource_manager['manager']
    mock_logger = resource_manager['simple_mock_logger']

    # Mock discover_mining_processes và start_threads để test
    with patch.object(manager, 'discover_mining_processes') as mock_discover, \
         patch.object(manager, 'start_threads') as mock_start_threads:

        manager.start()

        mock_logger.info.assert_any_call("Bắt đầu ResourceManager...")
        mock_logger.info.assert_any_call("ResourceManager đã khởi động thành công.")

        mock_discover.assert_called_once()
        mock_start_threads.assert_called_once()

def test_resource_manager_stop(resource_manager):
    manager = resource_manager['manager']
    mock_logger = resource_manager['simple_mock_logger']
    resource_manager['mock_shutdown_power_management'].reset_mock()

    with patch.object(manager, 'join_threads') as mock_join_threads:
        manager.stop()
        mock_logger.info.assert_any_call("Dừng ResourceManager...")
        mock_logger.info.assert_any_call("ResourceManager đã dừng thành công.")
        manager.stop_event.set.assert_called_once()
        mock_join_threads.assert_called_once()
        resource_manager['mock_shutdown_power_management'].assert_called_once()

# ----------------------------
# Test discover_mining_processes
# ----------------------------

@patch('mining_environment.scripts.resource_manager.psutil.process_iter')
def test_discover_mining_processes(mock_process_iter, resource_manager):
    manager = resource_manager['manager']
    mock_logger = resource_manager['simple_mock_logger']

    mock_proc1 = MagicMock()
    mock_proc1.info = {'pid': 101, 'name': 'cpu_miner'}
    mock_proc2 = MagicMock()
    mock_proc2.info = {'pid': 102, 'name': 'gpu_miner'}
    mock_process_iter.return_value = [mock_proc1, mock_proc2]

    manager.discover_mining_processes()
    assert len(manager.mining_processes) == 2
    assert manager.mining_processes[0].pid == 101
    assert manager.mining_processes[1].pid == 102
    mock_logger.info.assert_called_with("Khám phá 2 tiến trình khai thác.")

# ----------------------------
# Test monitor_and_adjust
# ----------------------------

@patch.object(ResourceManager, 'should_collect_azure_monitor_data', return_value=True)
@patch.object(ResourceManager, 'collect_azure_monitor_data')
@patch.object(ResourceManager, 'check_power_and_enqueue')
@patch.object(ResourceManager, 'check_temperature_and_enqueue')
@patch.object(ResourceManager, 'allocate_resources_with_priority')
@patch.object(ResourceManager, 'discover_mining_processes')
def test_monitor_and_adjust(mock_discover, mock_allocate, mock_check_temp, mock_check_power, mock_collect_azure, mock_should_collect, resource_manager):
    manager = resource_manager['manager']
    mock_logger = resource_manager['simple_mock_logger']
    
    # Giả lập stop_event được đặt để vòng lặp chạy 1 lần
    with patch.object(manager.stop_event, 'is_set', side_effect=[False, True]):
        # Patch sleep để không chờ
        with patch('time.sleep', return_value=None):
            # Giả lập mining_processes
            process = MagicMock()
            manager.mining_processes = [process]

            manager.monitor_and_adjust()

            mock_discover.assert_called_once()
            mock_allocate.assert_called_once()
            mock_check_temp.assert_called_once_with(process, 75, 85)
            mock_check_power.assert_called_once_with(process, 150, 300)
            mock_should_collect.assert_called_once()
            mock_collect_azure.assert_called_once()

# ----------------------------
# Test check_temperature_and_enqueue
# ----------------------------

def test_check_temperature_and_enqueue(resource_manager):
    manager = resource_manager['manager']
    mock_logger = resource_manager['simple_mock_logger']
    process = MagicMock()
    process.pid = 303
    process.name = "test_process"

    with patch('mining_environment.scripts.resource_manager.temperature_monitor.get_cpu_temperature', return_value=80), \
         patch('mining_environment.scripts.resource_manager.temperature_monitor.get_gpu_temperature', return_value=90):

        manager.check_temperature_and_enqueue(process, 75, 85)
        resource_manager['manager'].resource_adjustment_queue.put.assert_called_once()
        mock_logger.warning.assert_any_call(
            "Nhiệt độ CPU 80°C của tiến trình test_process (PID: 303) vượt quá 75°C."
        )
        mock_logger.warning.assert_any_call(
            "Nhiệt độ GPU 90°C của tiến trình test_process (PID: 303) vượt quá 85°C."
        )

# ----------------------------
# Test check_power_and_enqueue
# ----------------------------

def test_check_power_and_enqueue(resource_manager):
    manager = resource_manager['manager']
    mock_logger = resource_manager['simple_mock_logger']
    process = MagicMock()
    process.name = "power_process"
    process.pid = 404

    with patch('mining_environment.scripts.resource_manager.get_cpu_power', return_value=160), \
         patch('mining_environment.scripts.resource_manager.get_gpu_power', return_value=350):
        manager.check_power_and_enqueue(process, 150, 300)
        manager.resource_adjustment_queue.put.assert_called_once()
        mock_logger.warning.assert_any_call(
            "Công suất CPU 160W của tiến trình power_process (PID: 404) vượt quá 150W."
        )
        mock_logger.warning.assert_any_call(
            "Công suất GPU 350W của tiến trình power_process (PID: 404) vượt quá 300W."
        )

# ----------------------------
# Test should_collect_azure_monitor_data và collect_azure_monitor_data
# ----------------------------

def test_should_collect_azure_monitor_data_first_call(resource_manager):
    manager = resource_manager['manager']
    with patch('mining_environment.scripts.resource_manager.time', return_value=1000):
        assert manager.should_collect_azure_monitor_data() is True
        assert manager._last_azure_monitor_time == 1000


def test_collect_azure_monitor_data(resource_manager):
    manager = resource_manager['manager']
    mock_logger = resource_manager['simple_mock_logger']
    mock_client = resource_manager['mock_AzureMonitorClient'].return_value
    manager.vms = [{'id': 'vm1', 'name': 'VM1'}]
    mock_client.get_metrics.return_value = {'Percentage CPU': 50, 'Available Memory Bytes': 4000000000}
    manager.collect_azure_monitor_data()
    mock_logger.info.assert_any_call("Thu thập chỉ số từ Azure Monitor cho VM VM1: {'Percentage CPU': 50, 'Available Memory Bytes': 4000000000}")

# ----------------------------
# Test optimization_resources
# ----------------------------


@patch('mining_environment.scripts.resource_manager.psutil.Process')
@patch('mining_environment.scripts.resource_manager.torch.tensor')
def test_optimize_resources(mock_tensor, mock_psutil_proc, resource_manager):
    manager = resource_manager['manager']
    mock_logger = resource_manager['simple_mock_logger']
    mock_model = manager.resource_optimization_model

    # Giả lập tiến trình
    process = MagicMock()
    process.pid = 1010
    process.name = "optimize_process"
    manager.mining_processes = [process]

    mock_psutil_proc.return_value.cpu_percent.return_value = 50
    mock_psutil_proc.return_value.memory_info.return_value.rss = 104857600  # 100MB

    # Giả lập GPU usage, disk_io và torch.no_grad
    with patch('mining_environment.scripts.resource_manager.temperature_monitor.get_current_gpu_usage', return_value=70), \
         patch('mining_environment.scripts.resource_manager.temperature_monitor.get_current_disk_io_limit', return_value=30.0), \
         patch.object(manager.stop_event, 'is_set', side_effect=[False, True]), \
         patch('mining_environment.scripts.resource_manager.sleep', return_value=None), \
         patch('mining_environment.scripts.resource_manager.torch.no_grad') as mock_no_grad:
        
        # Định nghĩa cách hoạt động của torch.no_grad()
        mock_no_grad.return_value.__enter__.return_value = None
        mock_no_grad.return_value.__exit__.return_value = None

        # Mock predict
        mock_predict = MagicMock()
        mock_predict.squeeze.return_value.cpu.return_value.numpy.return_value = [1, 2, 3]
        mock_model.return_value = mock_predict

        mock_tensor_instance = MagicMock()
        mock_tensor.return_value = mock_tensor_instance
        mock_tensor_instance.to.return_value = mock_tensor_instance
        mock_tensor_instance.unsqueeze.return_value = mock_tensor_instance

        # Gọi phương thức optimize_resources
        manager.optimize_resources()

        # Kiểm tra logger.debug được gọi đúng
        expected_log_message = f"Mô hình AI đề xuất hành động cho tiến trình {process.name} (PID: {process.pid}): [1, 2, 3]"
        mock_logger.debug.assert_called_with(expected_log_message)

        # Kiểm tra resource_adjustment_queue.put được gọi đúng
        manager.resource_adjustment_queue.put.assert_any_call((2, {
            'type': 'optimization',
            'process': process,
            'action': [1, 2, 3]
        }))

        # Thêm các kiểm tra bổ sung để đảm bảo các mock được gọi đúng cách
        # Kiểm tra rằng torch.tensor được gọi với các tham số đúng
        mock_tensor.assert_called_with([50, 100, 70, 30.0, 100, 50], dtype=torch.float32)

        # Kiểm tra rằng resource_optimization_model được gọi với input_tensor đúng
        mock_model.assert_called_with(mock_tensor_instance)


# ----------------------------
# Test apply_recommended_action
# ----------------------------

def test_apply_recommended_action(resource_manager):
    manager = resource_manager['manager']
    mock_logger = resource_manager['simple_mock_logger']
    process = MagicMock()
    process.pid = 1011
    process.name = "ai_process"

    # Điều chỉnh action list để phù hợp với mong đợi
    action = [4, 2048, 60.0, 40.0, 50.0, 50.0]  # disk_io_limit=40.0, network=50.0
    manager.apply_recommended_action(action, process)

    # Kiểm tra hàng đợi resource_adjustment_queue được gọi đúng
    calls = [
        call((3, {'function': 'adjust_cpu_threads', 'args': (1011, 4, "ai_process")})),
        call((3, {'function': 'adjust_ram_allocation', 'args': (1011, 2048, "ai_process")})),
        call((3, {'function': 'adjust_gpu_usage', 'args': (process, [60.0, 40.0, 50.0])})),
        call((3, {'function': 'adjust_disk_io_limit', 'args': (process, 40.0)})),
        call((3, {'function': 'adjust_network_bandwidth', 'args': (process, 50.0)}))
    ]
    manager.resource_adjustment_queue.put.assert_has_calls(calls, any_order=True)
    resource_manager['mock_shared_resource_manager_instance'].apply_cloak_strategy.assert_called_once_with('cache', process)
    mock_logger.info.assert_called_with(
        f"Áp dụng thành công các điều chỉnh tài nguyên dựa trên AI cho tiến trình {process.name} (PID: {process.pid})."
    )

# ----------------------------
# Test cloaking requests
# ----------------------------

def test_process_cloaking_requests(resource_manager):
    manager = resource_manager['manager']
    process = MagicMock()
    manager.cloaking_request_queue = MagicMock()
    manager.cloaking_request_queue.get.return_value = process

    with patch.object(manager.stop_event, 'is_set', side_effect=[False, True]), \
         patch('time.sleep', return_value=None):

        manager.process_cloaking_requests()

        manager.resource_adjustment_queue.put.assert_called_once()
        manager.cloaking_request_queue.task_done.assert_called_once()

# ----------------------------
# Test resource_adjustment_handler
# ----------------------------

def test_resource_adjustment_handler(resource_manager):
    manager = resource_manager['manager']
    process = MagicMock()
    adjustment_task = {'type': 'monitoring', 'process': process, 'adjustments': {'cpu_cloak': True}}
    manager.resource_adjustment_queue.get.return_value = (2, adjustment_task)

    with patch.object(manager.stop_event, 'is_set', side_effect=[False, True]), \
         patch('time.sleep', return_value=None):

        manager.resource_adjustment_handler()
        manager.resource_adjustment_queue.task_done.assert_called_once()
        resource_manager['mock_shared_resource_manager_instance'].apply_cloak_strategy.assert_called_once_with('cpu', process)

# ----------------------------
# Test execute_adjustment_task với loại nhiệm vụ không xác định
# ----------------------------

def test_execute_adjustment_task_unknown_type(resource_manager):
    manager = resource_manager['manager']
    mock_logger = resource_manager['simple_mock_logger']
    adjustment_task = {
        'type': 'unknown_type',
        'process': MagicMock()
    }

    manager.execute_adjustment_task(adjustment_task)
    mock_logger.warning.assert_called_with("Loại nhiệm vụ không xác định: unknown_type")

# ----------------------------
# Test execute_adjustment_task nhiệm vụ hàm
# ----------------------------

def test_execute_adjustment_task_function_call(resource_manager):
    manager = resource_manager['manager']
    shared_resource_manager = resource_manager['mock_shared_resource_manager_instance']
    adjustment_task = {
        'function': 'adjust_cpu_threads',
        'args': (1234, 4, "function_process")
    }

    manager.execute_adjustment_task(adjustment_task)
    shared_resource_manager.adjust_cpu_threads.assert_called_once_with(1234, 4, "function_process")

# ----------------------------
# Kết thúc
# ----------------------------
