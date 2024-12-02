# test_resource_manager.py

import os
import sys
import pytest
import json
from unittest.mock import patch, MagicMock, mock_open, call, Mock
from pathlib import Path

# Thiết lập biến môi trường TESTING=1 trước khi import bất kỳ module nào
os.environ["TESTING"] = "1"

# Định nghĩa các thư mục cần thiết dựa trên cấu trúc dự án
APP_DIR = Path("/home/llmss/llmsdeep/app")
CONFIG_DIR = APP_DIR / "mining_environment" / "config"
MODELS_DIR = APP_DIR / "mining_environment" / "models"
SCRIPTS_DIR = APP_DIR / "mining_environment" / "scripts"

# Thêm thư mục APP_DIR vào sys.path để Python có thể tìm thấy các module
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Import các module cần thiết
with patch.dict('sys.modules', {
    'psutil': MagicMock(),
    'torch': MagicMock(),
    'pynvml': MagicMock(),
    'readerwriterlock': MagicMock(),
    'resource_manager.azure_clients': MagicMock(),
    'resource_manager.auxiliary_modules': MagicMock(),
    'resource_manager.auxiliary_modules.cgroup_manager': MagicMock(),
    'resource_manager.auxiliary_modules.temperature_monitor': MagicMock(),
    'resource_manager.auxiliary_modules.power_management': MagicMock(),
    'resource_manager.cloak_strategies': MagicMock(),
}):
    from resource_manager import SharedResourceManager, ResourceManager
    from resource_manager.utils import MiningProcess

@pytest.fixture
def shared_resource_manager():
    config = {
        "optimization_parameters": {
            "gpu_power_adjustment_step": 10,
            "disk_io_limit_step_mbps": 1
        },
        "resource_allocation": {
            "disk_io": {
                "min_limit_mbps": 10,
                "max_limit_mbps": 100
            },
            "ram": {
                "max_allocation_mb": 1024
            }
        }
    }
    logger = MagicMock()
    return SharedResourceManager(config, logger)

@pytest.fixture
def resource_manager():
    config = {
        "processes": {
            "CPU": "cpu_miner",
            "GPU": "gpu_miner"
        },
        "process_priority_map": {
            "cpu_miner": 1,
            "gpu_miner": 2
        },
        "resource_allocation": {
            "ram": {
                "max_allocation_mb": 1024
            },
            "gpu": {
                "max_usage_percent": [100]
            },
            "disk_io": {
                "min_limit_mbps": 10,
                "max_limit_mbps": 100
            },
            "network": {
                "bandwidth_limit_mbps": 100
            },
            "cache": {
                "limit_percent": 50
            }
        },
        "monitoring_parameters": {
            "temperature_monitoring_interval_seconds": 10,
            "power_monitoring_interval_seconds": 10,
            "optimization_interval_seconds": 30,
            "azure_monitor_interval_seconds": 300
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
        }
    }
    model_path = Path("/path/to/model")
    logger = MagicMock()
    with patch('resource_manager.ResourceManager.load_model', return_value=(MagicMock(), MagicMock())):
        rm = ResourceManager(config, model_path, logger)
    return rm

def test_adjust_cpu_threads(shared_resource_manager):
    pid = 1234
    cpu_threads = 2
    process_name = "test_process"

    with patch('resource_manager.assign_process_to_cgroups') as mock_assign:
        shared_resource_manager.adjust_cpu_threads(pid, cpu_threads, process_name)
        mock_assign.assert_called_with(pid, {'cpu_threads': cpu_threads}, process_name, shared_resource_manager.logger)
        shared_resource_manager.logger.info.assert_called_with(
            f"Điều chỉnh số luồng CPU xuống {cpu_threads} cho tiến trình {process_name} (PID: {pid})."
        )

def test_adjust_cpu_threads_exception(shared_resource_manager):
    pid = 1234
    cpu_threads = 2
    process_name = "test_process"

    with patch('resource_manager.assign_process_to_cgroups', side_effect=Exception("Test Exception")):
        shared_resource_manager.adjust_cpu_threads(pid, cpu_threads, process_name)
        shared_resource_manager.logger.error.assert_called_with(
            f"Lỗi khi điều chỉnh số luồng CPU cho tiến trình {process_name} (PID: {pid}): Test Exception"
        )

def test_adjust_ram_allocation(shared_resource_manager):
    pid = 1234
    ram_allocation_mb = 1024
    process_name = "test_process"

    with patch('resource_manager.assign_process_to_cgroups') as mock_assign:
        shared_resource_manager.adjust_ram_allocation(pid, ram_allocation_mb, process_name)
        mock_assign.assert_called_with(pid, {'memory': ram_allocation_mb}, process_name, shared_resource_manager.logger)
        shared_resource_manager.logger.info.assert_called_with(
            f"Điều chỉnh giới hạn RAM xuống {ram_allocation_mb}MB cho tiến trình {process_name} (PID: {pid})."
        )

def test_adjust_ram_allocation_exception(shared_resource_manager):
    pid = 1234
    ram_allocation_mb = 1024
    process_name = "test_process"

    with patch('resource_manager.assign_process_to_cgroups', side_effect=Exception("Test Exception")):
        shared_resource_manager.adjust_ram_allocation(pid, ram_allocation_mb, process_name)
        shared_resource_manager.logger.error.assert_called_with(
            f"Lỗi khi điều chỉnh RAM cho tiến trình {process_name} (PID: {pid}): Test Exception"
        )

def test_adjust_gpu_usage(shared_resource_manager):
    process = MagicMock()
    process.pid = 1234
    process.name = "test_process"
    gpu_usage_percent = [50.0, 60.0]

    with patch('resource_manager.set_gpu_usage') as mock_set_gpu_usage:
        shared_resource_manager.adjust_gpu_usage(process, gpu_usage_percent)
        mock_set_gpu_usage.assert_called_with(process.pid, [50.0, 60.0])
        shared_resource_manager.logger.info.assert_called_with(
            f"Điều chỉnh mức sử dụng GPU xuống [50.0, 60.0] cho tiến trình {process.name} (PID: {process.pid})."
        )

def test_adjust_gpu_usage_exception(shared_resource_manager):
    process = MagicMock()
    process.pid = 1234
    process.name = "test_process"
    gpu_usage_percent = [50.0, 60.0]

    with patch('resource_manager.set_gpu_usage', side_effect=Exception("Test Exception")):
        shared_resource_manager.adjust_gpu_usage(process, gpu_usage_percent)
        shared_resource_manager.logger.error.assert_called_with(
            f"Lỗi khi điều chỉnh mức sử dụng GPU cho tiến trình {process.name} (PID: {process.pid}): Test Exception"
        )

def test_adjust_disk_io_limit(shared_resource_manager):
    process = MagicMock()
    process.pid = 1234
    process.name = "test_process"
    disk_io_limit_mbps = 50.0

    with patch('resource_manager.temperature_monitor.get_current_disk_io_limit', return_value=60.0) as mock_get_limit, \
         patch('resource_manager.assign_process_to_cgroups') as mock_assign:

        shared_resource_manager.adjust_disk_io_limit(process, disk_io_limit_mbps)
        mock_get_limit.assert_called_with(process.pid)
        mock_assign.assert_called_with(process.pid, {'disk_io_limit_mbps': 59.0}, process.name, shared_resource_manager.logger)
        shared_resource_manager.logger.info.assert_called_with(
            f"Điều chỉnh giới hạn Disk I/O xuống 59.0 Mbps cho tiến trình {process.name} (PID: {process.pid})."
        )

def test_adjust_disk_io_limit_exception(shared_resource_manager):
    process = MagicMock()
    process.pid = 1234
    process.name = "test_process"
    disk_io_limit_mbps = 50.0

    with patch('resource_manager.temperature_monitor.get_current_disk_io_limit', side_effect=Exception("Test Exception")):
        shared_resource_manager.adjust_disk_io_limit(process, disk_io_limit_mbps)
        shared_resource_manager.logger.error.assert_called_with(
            f"Lỗi khi điều chỉnh Disk I/O cho tiến trình {process.name} (PID: {process.pid}): Test Exception"
        )

def test_adjust_network_bandwidth(shared_resource_manager):
    process = MagicMock()
    process.network_interface = "eth0"
    process.name = "test_process"
    process.pid = 1234
    bandwidth_limit_mbps = 100.0

    with patch.object(shared_resource_manager, 'apply_network_cloaking') as mock_apply_cloaking:
        shared_resource_manager.adjust_network_bandwidth(process, bandwidth_limit_mbps)
        mock_apply_cloaking.assert_called_with("eth0", 100.0, process)
        shared_resource_manager.logger.info.assert_called_with(
            f"Điều chỉnh giới hạn băng thông mạng xuống {bandwidth_limit_mbps} Mbps cho tiến trình {process.name} (PID: {process.pid})."
        )

def test_adjust_network_bandwidth_exception(shared_resource_manager):
    process = MagicMock()
    process.network_interface = "eth0"
    process.name = "test_process"
    process.pid = 1234
    bandwidth_limit_mbps = 100.0

    with patch.object(shared_resource_manager, 'apply_network_cloaking', side_effect=Exception("Test Exception")):
        shared_resource_manager.adjust_network_bandwidth(process, bandwidth_limit_mbps)
        shared_resource_manager.logger.error.assert_called_with(
            f"Lỗi khi điều chỉnh Mạng cho tiến trình {process.name} (PID: {process.pid}): Test Exception"
        )

def test_adjust_cpu_frequency(shared_resource_manager):
    pid = 1234
    frequency = 2000
    process_name = "test_process"

    with patch('resource_manager.assign_process_to_cgroups') as mock_assign:
        shared_resource_manager.adjust_cpu_frequency(pid, frequency, process_name)
        mock_assign.assert_called_with(pid, {'cpu_freq': frequency}, process_name, shared_resource_manager.logger)
        shared_resource_manager.logger.info.assert_called_with(
            f"Đặt tần số CPU xuống {frequency}MHz cho tiến trình {process_name} (PID: {pid})."
        )

def test_adjust_cpu_frequency_exception(shared_resource_manager):
    pid = 1234
    frequency = 2000
    process_name = "test_process"

    with patch('resource_manager.assign_process_to_cgroups', side_effect=Exception("Test Exception")):
        shared_resource_manager.adjust_cpu_frequency(pid, frequency, process_name)
        shared_resource_manager.logger.error.assert_called_with(
            f"Lỗi khi điều chỉnh tần số CPU cho tiến trình {process_name} (PID: {pid}): Test Exception"
        )

def test_adjust_gpu_power_limit(shared_resource_manager):
    pid = 1234
    power_limit = 150
    process_name = "test_process"

    with patch('resource_manager.pynvml.nvmlInit') as mock_nvmlInit, \
         patch('resource_manager.pynvml.nvmlDeviceGetHandleByIndex', return_value='handle') as mock_get_handle, \
         patch('resource_manager.pynvml.nvmlDeviceSetPowerManagementLimit') as mock_set_limit, \
         patch('resource_manager.pynvml.nvmlShutdown') as mock_nvmlShutdown:

        shared_resource_manager.adjust_gpu_power_limit(pid, power_limit, process_name)
        mock_nvmlInit.assert_called_once()
        mock_get_handle.assert_called_with(0)
        mock_set_limit.assert_called_with('handle', power_limit * 1000)
        mock_nvmlShutdown.assert_called_once()
        shared_resource_manager.logger.info.assert_called_with(
            f"Đặt giới hạn công suất GPU xuống {power_limit}W cho tiến trình {process_name} (PID: {pid})."
        )

def test_adjust_gpu_power_limit_exception(shared_resource_manager):
    pid = 1234
    power_limit = 150
    process_name = "test_process"

    with patch('resource_manager.pynvml.nvmlInit', side_effect=Exception("Test Exception")):
        shared_resource_manager.adjust_gpu_power_limit(pid, power_limit, process_name)
        shared_resource_manager.logger.error.assert_called_with(
            f"Lỗi khi điều chỉnh công suất GPU cho tiến trình {process_name} (PID: {pid}): Test Exception"
        )

def test_adjust_disk_io_priority(shared_resource_manager):
    pid = 1234
    ionice_class = 3
    process_name = "test_process"

    with patch('resource_manager.subprocess.run') as mock_run:
        shared_resource_manager.adjust_disk_io_priority(pid, ionice_class, process_name)
        mock_run.assert_called_with(['ionice', '-c', str(ionice_class), '-p', str(pid)], check=True)
        shared_resource_manager.logger.info.assert_called_with(
            f"Đặt ionice class thành {ionice_class} cho tiến trình {process_name} (PID: {pid})."
        )

def test_adjust_disk_io_priority_subprocess_error(shared_resource_manager):
    pid = 1234
    ionice_class = 3
    process_name = "test_process"

    with patch('resource_manager.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'ionice')):
        shared_resource_manager.adjust_disk_io_priority(pid, ionice_class, process_name)
        shared_resource_manager.logger.error.assert_called_with(
            f"Lỗi khi thực hiện ionice: Command 'ionice' returned non-zero exit status 1."
        )

def test_adjust_disk_io_priority_exception(shared_resource_manager):
    pid = 1234
    ionice_class = 3
    process_name = "test_process"

    with patch('resource_manager.subprocess.run', side_effect=Exception("Test Exception")):
        shared_resource_manager.adjust_disk_io_priority(pid, ionice_class, process_name)
        shared_resource_manager.logger.error.assert_called_with(
            f"Lỗi khi điều chỉnh ưu tiên Disk I/O cho tiến trình {process_name} (PID: {pid}): Test Exception"
        )

def test_drop_caches(shared_resource_manager):
    with patch('builtins.open', mock_open()) as mock_file:
        shared_resource_manager.drop_caches()
        mock_file.assert_called_with('/proc/sys/vm/drop_caches', 'w')
        mock_file().write.assert_called_with('3\n')
        shared_resource_manager.logger.info.assert_called_with(
            "Đã giảm sử dụng cache bằng cách drop_caches."
        )

def test_drop_caches_exception(shared_resource_manager):
    with patch('builtins.open', side_effect=Exception("Test Exception")):
        shared_resource_manager.drop_caches()
        shared_resource_manager.logger.error.assert_called_with(
            f"Lỗi khi giảm sử dụng cache: Test Exception"
        )

def test_throttle_cpu_based_on_load(shared_resource_manager):
    process = MagicMock()
    process.pid = 1234
    process.name = "test_process"
    load_percent = 85

    with patch.object(shared_resource_manager, 'adjust_cpu_frequency') as mock_adjust_cpu_frequency:
        shared_resource_manager.throttle_cpu_based_on_load(process, load_percent)
        mock_adjust_cpu_frequency.assert_called_with(1234, 2000, "test_process")
        shared_resource_manager.logger.info.assert_called_with(
            f"Điều chỉnh tần số CPU xuống 2000MHz cho tiến trình {process.name} (PID: {process.pid}) dựa trên tải {load_percent}%."
        )

def test_throttle_cpu_based_on_load_exception(shared_resource_manager):
    process = MagicMock()
    process.pid = 1234
    process.name = "test_process"
    load_percent = 85

    with patch.object(shared_resource_manager, 'adjust_cpu_frequency', side_effect=Exception("Test Exception")):
        shared_resource_manager.throttle_cpu_based_on_load(process, load_percent)
        shared_resource_manager.logger.error.assert_called_with(
            f"Lỗi khi điều chỉnh tần số CPU dựa trên tải cho tiến trình {process.name} (PID: {process.pid}): Test Exception"
        )

def test_apply_cloak_strategy(shared_resource_manager):
    strategy_name = 'cpu'
    process = MagicMock()
    process.pid = 1234
    process.name = "test_process"

    mock_strategy = MagicMock()
    mock_strategy.apply.return_value = {'cpu_freq': 2000}

    with patch('resource_manager.CloakStrategyFactory.create_strategy', return_value=mock_strategy):
        with patch.object(shared_resource_manager, 'get_current_cpu_frequency', return_value=2500):
            with patch.object(shared_resource_manager, 'execute_adjustments') as mock_execute_adjustments:
                shared_resource_manager.apply_cloak_strategy(strategy_name, process)
                shared_resource_manager.logger.info.assert_called_with(
                    f"Áp dụng điều chỉnh {strategy_name} cho tiến trình {process.name} (PID: {process.pid}): {{'cpu_freq': 2000}}"
                )
                assert process.pid in shared_resource_manager.original_resource_limits
                assert shared_resource_manager.original_resource_limits[process.pid]['cpu_freq'] == 2500
                mock_execute_adjustments.assert_called_with({'cpu_freq': 2000}, process)

def test_apply_cloak_strategy_no_strategy(shared_resource_manager):
    strategy_name = 'unknown'
    process = MagicMock()
    process.pid = 1234
    process.name = "test_process"

    with patch('resource_manager.CloakStrategyFactory.create_strategy', return_value=None):
        shared_resource_manager.apply_cloak_strategy(strategy_name, process)
        shared_resource_manager.logger.warning.assert_called_with(
            f"Chiến lược cloaking {strategy_name} không được tạo thành công cho tiến trình {process.name} (PID: {process.pid})."
        )

def test_restore_resources(shared_resource_manager):
    process = MagicMock()
    process.pid = 1234
    process.name = "test_process"
    shared_resource_manager.original_resource_limits[1234] = {
        'cpu_freq': 2500,
        'cpu_threads': 4,
        'ram_allocation_mb': 1024,
        'gpu_power_limit': 150,
        'ionice_class': 2,
        'network_bandwidth_limit_mbps': 100
    }

    with patch.object(shared_resource_manager, 'adjust_cpu_frequency') as mock_adjust_cpu_frequency, \
         patch.object(shared_resource_manager, 'adjust_cpu_threads') as mock_adjust_cpu_threads, \
         patch.object(shared_resource_manager, 'adjust_ram_allocation') as mock_adjust_ram_allocation, \
         patch.object(shared_resource_manager, 'adjust_gpu_power_limit') as mock_adjust_gpu_power_limit, \
         patch.object(shared_resource_manager, 'adjust_disk_io_priority') as mock_adjust_disk_io_priority, \
         patch.object(shared_resource_manager, 'adjust_network_bandwidth') as mock_adjust_network_bandwidth:

        shared_resource_manager.restore_resources(process)

        mock_adjust_cpu_frequency.assert_called_with(1234, 2500, "test_process")
        mock_adjust_cpu_threads.assert_called_with(1234, 4, "test_process")
        mock_adjust_ram_allocation.assert_called_with(1234, 1024, "test_process")
        mock_adjust_gpu_power_limit.assert_called_with(1234, 150, "test_process")
        mock_adjust_disk_io_priority.assert_called_with(1234, 2, "test_process")
        mock_adjust_network_bandwidth.assert_called_with(process, 100)

        shared_resource_manager.logger.info.assert_any_call(
            f"Đã khôi phục tần số CPU về 2500MHz cho tiến trình {process.name} (PID: {process.pid})."
        )
        shared_resource_manager.logger.info.assert_any_call(
            f"Đã khôi phục số luồng CPU về 4 cho tiến trình {process.name} (PID: {process.pid})."
        )
        shared_resource_manager.logger.info.assert_any_call(
            f"Đã khôi phục giới hạn RAM về 1024MB cho tiến trình {process.name} (PID: {process.pid})."
        )
        shared_resource_manager.logger.info.assert_any_call(
            f"Đã khôi phục giới hạn công suất GPU về 150W cho tiến trình {process.name} (PID: {process.pid})."
        )
        shared_resource_manager.logger.info.assert_any_call(
            f"Đã khôi phục lớp ionice về 2 cho tiến trình {process.name} (PID: {process.pid})."
        )
        shared_resource_manager.logger.info.assert_any_call(
            f"Đã khôi phục giới hạn băng thông mạng về 100 Mbps cho tiến trình {process.name} (PID: {process.pid})."
        )
        shared_resource_manager.logger.info.assert_called_with(
            f"Đã khôi phục tất cả tài nguyên cho tiến trình {process.name} (PID: {process.pid})."
        )
        assert 1234 not in shared_resource_manager.original_resource_limits

def test_restore_resources_no_original_limits(shared_resource_manager):
    process = MagicMock()
    process.pid = 1234
    process.name = "test_process"

    shared_resource_manager.restore_resources(process)
    shared_resource_manager.logger.warning.assert_called_with(
        f"Không tìm thấy giới hạn tài nguyên ban đầu cho tiến trình {process.name} (PID: {process.pid})."
    )


def test_is_gpu_initialized(shared_resource_manager):
    with patch('resource_manager.pynvml.nvmlInit') as mock_nvmlInit, \
         patch('resource_manager.pynvml.nvmlDeviceGetCount', return_value=1) as mock_get_count, \
         patch('resource_manager.pynvml.nvmlShutdown') as mock_nvmlShutdown:
        result = shared_resource_manager.is_gpu_initialized()
        assert result is True
        mock_nvmlInit.assert_called_once()
        mock_get_count.assert_called_once()
        mock_nvmlShutdown.assert_called_once()

def test_is_gpu_initialized_no_gpu(shared_resource_manager):
    with patch('resource_manager.pynvml.nvmlInit') as mock_nvmlInit, \
         patch('resource_manager.pynvml.nvmlDeviceGetCount', return_value=0) as mock_get_count, \
         patch('resource_manager.pynvml.nvmlShutdown') as mock_nvmlShutdown:
        result = shared_resource_manager.is_gpu_initialized()
        assert result is False
        mock_nvmlInit.assert_called_once()
        mock_get_count.assert_called_once()
        mock_nvmlShutdown.assert_called_once()

def test_is_gpu_initialized_exception(shared_resource_manager):
    with patch('resource_manager.pynvml.nvmlInit', side_effect=Exception("Test Exception")):
        result = shared_resource_manager.is_gpu_initialized()
        assert result is False
        shared_resource_manager.logger.error.assert_called_with(
            f"Lỗi khi kiểm tra GPU: Test Exception"
        )

def test_execute_adjustments(shared_resource_manager):
    adjustments = {
        'cpu_freq': 2000,
        'gpu_power_limit': 150,
        'unknown_key': 123
    }
    process = MagicMock()
    process.pid = 1234
    process.name = "test_process"

    with patch.object(shared_resource_manager, 'adjust_cpu_frequency') as mock_adjust_cpu_freq, \
         patch.object(shared_resource_manager, 'adjust_gpu_power_limit') as mock_adjust_gpu_power_limit:
        shared_resource_manager.execute_adjustments(adjustments, process)
        mock_adjust_cpu_freq.assert_called_with(1234, 2000, "test_process")
        mock_adjust_gpu_power_limit.assert_called_with(1234, 150, "test_process")
        shared_resource_manager.logger.warning.assert_called_with(
            f"Không nhận dạng được điều chỉnh: unknown_key"
        )

def test_execute_adjustments_exception(shared_resource_manager):
    adjustments = {
        'cpu_freq': 2000
    }
    process = MagicMock()
    process.pid = 1234
    process.name = "test_process"

    with patch.object(shared_resource_manager, 'adjust_cpu_frequency', side_effect=Exception("Test Exception")):
        shared_resource_manager.execute_adjustments(adjustments, process)
        shared_resource_manager.logger.error.assert_called_with(
            f"Lỗi khi thực hiện các điều chỉnh cloaking cho tiến trình {process.name} (PID: {process.pid}): Test Exception"
        )

def test_resource_manager_singleton():
    config = {}
    model_path = Path("/path/to/model")
    logger = MagicMock()

    with patch('resource_manager.ResourceManager.load_model', return_value=(MagicMock(), MagicMock())):
        rm1 = ResourceManager(config, model_path, logger)
        rm2 = ResourceManager(config, model_path, logger)
        assert rm1 is rm2

def test_resource_manager_start_stop(resource_manager):
    with patch.object(resource_manager, 'discover_mining_processes') as mock_discover, \
         patch.object(resource_manager, 'start_threads') as mock_start_threads, \
         patch.object(resource_manager, 'join_threads') as mock_join_threads, \
         patch.object(resource_manager, 'shutdown_power_management') as mock_shutdown_power_management:

        resource_manager.start()
        mock_discover.assert_called_once()
        mock_start_threads.assert_called_once()
        resource_manager.logger.info.assert_any_call("Bắt đầu ResourceManager...")

        resource_manager.stop()
        resource_manager.stop_event.set.assert_called_once()
        mock_join_threads.assert_called_once()
        mock_shutdown_power_management.assert_called_once()
        resource_manager.logger.info.assert_any_call("Dừng ResourceManager...")

def test_initialize_threads(resource_manager):
    resource_manager.initialize_threads()
    assert resource_manager.monitor_thread.name == "MonitorThread"
    assert resource_manager.optimization_thread.name == "OptimizationThread"
    assert resource_manager.cloaking_thread.name == "CloakingThread"
    assert resource_manager.resource_adjustment_thread.name == "ResourceAdjustmentThread"

def test_start_threads(resource_manager):
    resource_manager.monitor_thread = MagicMock()
    resource_manager.optimization_thread = MagicMock()
    resource_manager.cloaking_thread = MagicMock()
    resource_manager.resource_adjustment_thread = MagicMock()

    resource_manager.start_threads()
    resource_manager.monitor_thread.start.assert_called_once()
    resource_manager.optimization_thread.start.assert_called_once()
    resource_manager.cloaking_thread.start.assert_called_once()
    resource_manager.resource_adjustment_thread.start.assert_called_once()

def test_join_threads(resource_manager):
    resource_manager.monitor_thread = MagicMock()
    resource_manager.optimization_thread = MagicMock()
    resource_manager.cloaking_thread = MagicMock()
    resource_manager.resource_adjustment_thread = MagicMock()

    resource_manager.join_threads()
    resource_manager.monitor_thread.join.assert_called_once()
    resource_manager.optimization_thread.join.assert_called_once()
    resource_manager.cloaking_thread.join.assert_called_once()
    resource_manager.resource_adjustment_thread.join.assert_called_once()

def test_load_model(resource_manager):
    model_path = Path("/path/to/model")
    with patch('resource_manager.torch.load', return_value=MagicMock()) as mock_torch_load, \
         patch('resource_manager.torch.device', return_value='cpu') as mock_device:

        model, device = resource_manager.load_model(model_path)
        mock_torch_load.assert_called_with(model_path)
        resource_manager.logger.info.assert_called_with(
            f"Tải mô hình tối ưu hóa tài nguyên từ {model_path} vào cpu."
        )

def test_load_model_exception(resource_manager):
    model_path = Path("/path/to/model")
    with patch('resource_manager.torch.load', side_effect=Exception("Test Exception")):
        with pytest.raises(Exception):
            resource_manager.load_model(model_path)
            resource_manager.logger.error.assert_called_with(
                f"Không thể tải mô hình AI từ {model_path}: Test Exception"
            )

def test_discover_mining_processes(resource_manager):
    resource_manager.config['processes'] = {'CPU': 'cpu_miner', 'GPU': 'gpu_miner'}
    resource_manager.config['process_priority_map'] = {'cpu_miner': 1, 'gpu_miner': 2}
    process_iter = [
        MagicMock(info={'pid': 1, 'name': 'cpu_miner'}),
        MagicMock(info={'pid': 2, 'name': 'gpu_miner'}),
        MagicMock(info={'pid': 3, 'name': 'other_process'})
    ]

    with patch('resource_manager.psutil.process_iter', return_value=process_iter):
        resource_manager.discover_mining_processes()
        assert len(resource_manager.mining_processes) == 2
        resource_manager.logger.info.assert_called_with(f"Khám phá 2 tiến trình khai thác.")

def test_get_process_priority(resource_manager):
    priority = resource_manager.get_process_priority('cpu_miner')
    assert priority == 1
    priority = resource_manager.get_process_priority('gpu_miner')
    assert priority == 2
    priority = resource_manager.get_process_priority('unknown_process')
    assert priority == 1  # Default priority

def test_monitor_and_adjust(resource_manager):
    resource_manager.stop_event = MagicMock()
    resource_manager.stop_event.is_set.side_effect = [False, True]
    resource_manager.config['monitoring_parameters'] = {
        'temperature_monitoring_interval_seconds': 0,
        'power_monitoring_interval_seconds': 0
    }

    with patch.object(resource_manager, 'discover_mining_processes') as mock_discover, \
         patch.object(resource_manager, 'allocate_resources_with_priority') as mock_allocate, \
         patch.object(resource_manager, 'check_temperature_and_enqueue') as mock_check_temp, \
         patch.object(resource_manager, 'check_power_and_enqueue') as mock_check_power, \
         patch.object(resource_manager, 'should_collect_azure_monitor_data', return_value=False):

        resource_manager.mining_processes = [MagicMock()]
        resource_manager.monitor_and_adjust()
        mock_discover.assert_called_once()
        mock_allocate.assert_called_once()
        mock_check_temp.assert_called_once()
        mock_check_power.assert_called_once()

def test_should_collect_azure_monitor_data(resource_manager):
    resource_manager._last_azure_monitor_time = 0
    resource_manager.config['monitoring_parameters'] = {'azure_monitor_interval_seconds': 300}
    with patch('resource_manager.time', return_value=301):
        result = resource_manager.should_collect_azure_monitor_data()
        assert result is True

def test_collect_metrics(resource_manager):
    process = MagicMock()
    process.pid = 1234
    with patch('resource_manager.psutil.Process') as mock_psutil_process, \
         patch('resource_manager.temperature_monitor.get_current_gpu_usage', return_value=50), \
         patch('resource_manager.temperature_monitor.get_current_disk_io_limit', return_value=100):
        mock_psutil_process().cpu_percent.return_value = 10.0
        mock_psutil_process().memory_info().rss = 1024 * 1024 * 512  # 512 MB

        metrics = resource_manager.collect_metrics(process)
        assert metrics['cpu_usage_percent'] == 10.0
        assert metrics['memory_usage_mb'] == 512.0
        assert metrics['gpu_usage_percent'] == 50
        assert metrics['disk_io_mbps'] == 100
        assert metrics['network_bandwidth_mbps'] == 100
        assert metrics['cache_limit_percent'] == 50

def test_prepare_input_features(resource_manager):
    metrics = {
        'cpu_usage_percent': 10.0,
        'memory_usage_mb': 512.0,
        'gpu_usage_percent': 50,
        'disk_io_mbps': 100,
        'network_bandwidth_mbps': 100,
        'cache_limit_percent': 50
    }
    features = resource_manager.prepare_input_features(metrics)
    assert features == [10.0, 512.0, 50, 100, 100, 50]

def test_process_cloaking_requests(resource_manager):
    resource_manager.stop_event = MagicMock()
    resource_manager.stop_event.is_set.side_effect = [False, True]
    resource_manager.cloaking_request_queue = MagicMock()
    process = MagicMock()
    resource_manager.cloaking_request_queue.get.return_value = process

    with patch.object(resource_manager.resource_adjustment_queue, 'put') as mock_put:
        resource_manager.process_cloaking_requests()
        mock_put.assert_called_with((1, {
            'type': 'cloaking',
            'process': process,
            'strategies': ['cpu', 'gpu', 'network', 'disk_io', 'cache']
        }))

def test_resource_adjustment_handler(resource_manager):
    resource_manager.stop_event = MagicMock()
    resource_manager.stop_event.is_set.side_effect = [False, True]
    resource_manager.resource_adjustment_queue = MagicMock()
    resource_manager.resource_adjustment_queue.get.return_value = (1, {'function': 'adjust_cpu_threads', 'args': (1234, 2, 'test_process')})

    with patch.object(resource_manager, 'execute_adjustment_task') as mock_execute:
        resource_manager.resource_adjustment_handler()
        mock_execute.assert_called_with({'function': 'adjust_cpu_threads', 'args': (1234, 2, 'test_process')})

def test_execute_adjustment_task_function(resource_manager):
    adjustment_task = {'function': 'adjust_cpu_threads', 'args': (1234, 2, 'test_process')}
    with patch.object(resource_manager.shared_resource_manager, 'adjust_cpu_threads') as mock_adjust_cpu_threads:
        resource_manager.execute_adjustment_task(adjustment_task)
        mock_adjust_cpu_threads.assert_called_with(1234, 2, 'test_process')

def test_execute_adjustment_task_cloaking(resource_manager):
    process = MagicMock()
    adjustment_task = {
        'type': 'cloaking',
        'process': process,
        'strategies': ['cpu', 'gpu']
    }
    with patch.object(resource_manager.shared_resource_manager, 'apply_cloak_strategy') as mock_apply_cloak_strategy:
        resource_manager.execute_adjustment_task(adjustment_task)
        calls = [call('cpu', process), call('gpu', process)]
        mock_apply_cloak_strategy.assert_has_calls(calls)
        resource_manager.logger.info.assert_called_with(
            f"Hoàn thành cloaking cho tiến trình {process.name} (PID: {process.pid})."
        )

def test_execute_adjustment_task_restore(resource_manager):
    process = MagicMock()
    adjustment_task = {
        'type': 'restore',
        'process': process
    }
    with patch.object(resource_manager.shared_resource_manager, 'restore_resources') as mock_restore_resources:
        resource_manager.execute_adjustment_task(adjustment_task)
        mock_restore_resources.assert_called_with(process)
        resource_manager.logger.info.assert_called_with(
            f"Đã khôi phục tài nguyên cho tiến trình {process.name} (PID: {process.pid})."
        )

def test_apply_monitoring_adjustments(resource_manager):
    process = MagicMock()
    process.pid = 1234
    process.name = "test_process"
    adjustments = {'cpu_cloak': True, 'gpu_cloak': True}

    with patch.object(resource_manager.shared_resource_manager, 'apply_cloak_strategy') as mock_apply_cloak_strategy, \
         patch('resource_manager.psutil.cpu_percent', return_value=80), \
         patch.object(resource_manager.shared_resource_manager, 'throttle_cpu_based_on_load') as mock_throttle_cpu:
        resource_manager.apply_monitoring_adjustments(adjustments, process)
        mock_apply_cloak_strategy.assert_any_call('cpu', process)
        mock_apply_cloak_strategy.assert_any_call('gpu', process)
        resource_manager.logger.info.assert_called_with(
            f"Áp dụng điều chỉnh từ MonitorThread cho tiến trình {process.name} (PID: {process.pid})."
        )

def test_apply_monitoring_adjustments_exception(resource_manager):
    process = MagicMock()
    process.pid = 1234
    process.name = "test_process"
    adjustments = {'cpu_cloak': True}

    with patch.object(resource_manager.shared_resource_manager, 'apply_cloak_strategy', side_effect=Exception("Test Exception")):
        resource_manager.apply_monitoring_adjustments(adjustments, process)
        resource_manager.logger.error.assert_called_with(
            f"Lỗi khi áp dụng điều chỉnh từ MonitorThread cho tiến trình {process.name} (PID: {process.pid}): Test Exception"
        )

def test_apply_recommended_action(resource_manager):
    action = [2, 1024, 50.0, 60.0, 50.0, 100.0, 50.0]
    process = MagicMock()
    process.pid = 1234
    process.name = "test_process"

    with patch.object(resource_manager.resource_adjustment_queue, 'put') as mock_put, \
         patch.object(resource_manager.shared_resource_manager, 'apply_cloak_strategy') as mock_apply_cloak_strategy:

        resource_manager.apply_recommended_action(action, process)
        assert mock_put.call_count == 4  # adjust_cpu_threads, adjust_ram_allocation, adjust_gpu_usage, adjust_disk_io_limit

        resource_manager.logger.info.assert_called_with(
            f"Áp dụng thành công các điều chỉnh tài nguyên dựa trên AI cho tiến trình {process.name} (PID: {process.pid})."
        )

def test_apply_recommended_action_exception(resource_manager):
    action = [2, 1024, 50.0]
    process = MagicMock()
    process.pid = 1234
    process.name = "test_process"

    with patch.object(resource_manager.resource_adjustment_queue, 'put', side_effect=Exception("Test Exception")):
        resource_manager.apply_recommended_action(action, process)
        resource_manager.logger.error.assert_called_with(
            f"Lỗi khi áp dụng các điều chỉnh tài nguyên dựa trên AI cho tiến trình {process.name} (PID: {process.pid}): Test Exception"
        )

def test_shutdown_power_management(resource_manager):
    with patch('resource_manager.shutdown_power_management') as mock_shutdown:
        resource_manager.shutdown_power_management()
        mock_shutdown.assert_called_once()
        resource_manager.logger.info.assert_called_with("Đóng các dịch vụ quản lý công suất thành công.")

def test_shutdown_power_management_exception(resource_manager):
    with patch('resource_manager.shutdown_power_management', side_effect=Exception("Test Exception")):
        resource_manager.shutdown_power_management()
        resource_manager.logger.error.assert_called_with(
            f"Lỗi khi đóng các dịch vụ quản lý công suất: Test Exception"
        )

def test_discover_azure_resources(resource_manager):
    resource_manager.azure_monitor_client = MagicMock()
    resource_manager.azure_network_watcher_client = MagicMock()
    resource_manager.azure_traffic_analytics_client = MagicMock()
    resource_manager.azure_ml_client = MagicMock()

    resource_manager.azure_monitor_client.discover_resources.return_value = [{'name': 'vm1'}]
    resource_manager.azure_network_watcher_client.discover_resources.return_value = [{'name': 'nw1'}]
    resource_manager.azure_traffic_analytics_client.get_traffic_workspace_ids.return_value = ['workspace1']
    resource_manager.azure_ml_client.discover_ml_clusters.return_value = ['cluster1']

    resource_manager.discover_azure_resources()
    resource_manager.logger.info.assert_any_call("Khám phá 1 Máy ảo.")
    resource_manager.logger.info.assert_any_call("Khám phá 1 Network Watchers.")
    resource_manager.logger.info.assert_any_call("Khám phá 1 Traffic Analytics Workspaces.")
    resource_manager.logger.info.assert_any_call("Khám phá 1 Azure ML Clusters.")

def test_discover_azure_resources_exception(resource_manager):
    resource_manager.azure_monitor_client = MagicMock()
    resource_manager.azure_monitor_client.discover_resources.side_effect = Exception("Test Exception")

    resource_manager.discover_azure_resources()
    resource_manager.logger.error.assert_called_with(
        f"Lỗi khi khám phá tài nguyên Azure: Test Exception"
    )

