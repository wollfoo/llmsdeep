import os
import sys
import json
import pytest
import locale
import subprocess  # Thêm dòng này
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Giả sử rằng setup_env.py nằm trong sys.path hoặc ta đã thêm trước
from pathlib import Path

# Thêm đường dẫn của app để import được module
APP_DIR = Path("/home/llmss/llmsdeep/app")
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from mining_environment.scripts.setup_env import (
    load_json_config,
    configure_system,
    setup_environment_variables,
    validate_configs,
    configure_security,
    setup_gpu_optimization,
    setup
)


@pytest.fixture
def logger_mock():
    class LoggerMock:
        def info(self, msg):
            pass
        def warning(self, msg):
            pass
        def error(self, msg):
            pass
    return LoggerMock()


@pytest.fixture
def mock_configs():
    # Mock dữ liệu config JSON hợp lệ
    system_params = {
        "timezone": "UTC",
        "locale": "en_US.UTF-8"
    }

    environmental_limits = {
        "baseline_monitoring": {
            "cpu_percent_threshold": 50,
            "gpu_percent_threshold": 50,
            "cache_percent_threshold": 50,
            "network_bandwidth_threshold_mbps": 100,
            "disk_io_threshold_mbps": 200,
            "power_consumption_threshold_watts": 1000
        },
        "temperature_limits": {
            "cpu": {"max_celsius": 60},
            "gpu": {"max_celsius": 70}
        },
        "power_limits": {
            "total_power_watts": {"max": 200},
            "per_device_power_watts": {
                "cpu": {"max": 100},
                "gpu": {"max": 100}
            }
        },
        "memory_limits": {
            "ram_percent_threshold": 80
        },
        "gpu_optimization": {
            "gpu_utilization_percent_optimal": {
                "min": 10,
                "max": 90
            }
        }
    }

    resource_config = {
        "resource_allocation": {
            "ram": {"max_allocation_mb": 4096},
            "cpu": {"max_threads": 4},
            "gpu": {"usage_percent_range": {"max": 90}}
        }
    }

    return system_params, environmental_limits, resource_config


### Bài Kiểm Thử 1: Tải Cấu Hình JSON ###

def test_load_json_config_success(logger_mock):
    sample_json = '{"key": "value"}'
    with patch("builtins.open", mock_open(read_data=sample_json)):
        config = load_json_config("dummy_path.json", logger_mock)
        assert config["key"] == "value"


def test_load_json_config_not_found(logger_mock):
    with patch("builtins.open", side_effect=FileNotFoundError):
        with pytest.raises(SystemExit):
            load_json_config("non_existent.json", logger_mock)


def test_load_json_config_invalid_json(logger_mock):
    invalid_json = "{key: 'value'}"  # thiếu dấu ngoặc kép
    with patch("builtins.open", mock_open(read_data=invalid_json)):
        with pytest.raises(SystemExit):
            load_json_config("invalid.json", logger_mock)


### Bài Kiểm Thử 2: Cấu Hình Hệ Thống ###

@patch("subprocess.run")
def test_configure_system_success(mock_run, logger_mock, mock_configs):
    system_params, _, _ = mock_configs
    with patch("locale.setlocale") as mock_setlocale:
        configure_system(system_params, logger_mock)
        mock_run.assert_any_call(['ln', '-snf', '/usr/share/zoneinfo/UTC', '/etc/localtime'], check=True)
        mock_run.assert_any_call(['dpkg-reconfigure', '-f', 'noninteractive', 'tzdata'], check=True)
        mock_setlocale.assert_called_once_with(locale.LC_ALL, 'en_US.UTF-8')


@patch("mining_environment.scripts.setup_env.subprocess.run")
def test_configure_system_locale_error(mock_run, logger_mock):
    system_params = {"timezone": "UTC", "locale": "invalid_LOCALE"}
    with patch("mining_environment.scripts.setup_env.locale.setlocale", side_effect=locale.Error):
        # Giả lập subprocess.run thành công cho các lệnh khác
        mock_run.return_value = MagicMock(returncode=0)
        with pytest.raises(SystemExit):
            configure_system(system_params, logger_mock)

@patch("mining_environment.scripts.setup_env.subprocess.run", side_effect=subprocess.CalledProcessError(returncode=1, cmd=['ln', '-snf', '/usr/share/zoneinfo/UTC', '/etc/localtime']))
def test_configure_system_subprocess_error(mock_run, logger_mock):
    system_params = {"timezone": "UTC", "locale": "en_US.UTF-8"}
    with pytest.raises(SystemExit):
        configure_system(system_params, logger_mock)

### Bài Kiểm Thử 3: Đặt Biến Môi Trường ###

def test_setup_environment_variables(logger_mock, mock_configs):
    _, environmental_limits, _ = mock_configs
    setup_environment_variables(environmental_limits, logger_mock)
    assert os.environ["RAM_PERCENT_THRESHOLD"] == "80"
    assert os.environ["GPU_UTIL_MIN"] == "10"
    assert os.environ["GPU_UTIL_MAX"] == "90"


def test_setup_environment_variables_missing_ram_threshold(monkeypatch, logger_mock):
    # Đảm bảo biến môi trường RAM_PERCENT_THRESHOLD không tồn tại trước khi test
    monkeypatch.delenv('RAM_PERCENT_THRESHOLD', raising=False)

    environmental_limits = {
        "gpu_optimization": {
            "gpu_utilization_percent_optimal": {"min": 10, "max": 90}
        }
    }
    setup_environment_variables(environmental_limits, logger_mock)
    
    assert "RAM_PERCENT_THRESHOLD" not in os.environ


### Bài Kiểm Thử 4: Xác Thực Cấu Hình ###

def test_validate_configs_success(logger_mock, mock_configs):
    system_params, environmental_limits, resource_config = mock_configs
    # Không có exception nghĩa là pass
    validate_configs(resource_config, system_params, environmental_limits, logger_mock)


def test_validate_configs_missing_ram_max_allocation(logger_mock, mock_configs):
    system_params, environmental_limits, resource_config = mock_configs
    del resource_config["resource_allocation"]["ram"]["max_allocation_mb"]
    with pytest.raises(SystemExit):
        validate_configs(resource_config, system_params, environmental_limits, logger_mock)


def test_validate_configs_invalid_cpu_threshold(logger_mock, mock_configs):
    system_params, environmental_limits, resource_config = mock_configs
    environmental_limits["baseline_monitoring"]["cpu_percent_threshold"] = 200
    with pytest.raises(SystemExit):
        validate_configs(resource_config, system_params, environmental_limits, logger_mock)


### Bài Kiểm Thử 5: Thiết Lập Bảo Mật ###

@patch("os.path.exists", return_value=True)
@patch("subprocess.run")
@patch("subprocess.Popen")
def test_configure_security_success(mock_popen, mock_run, mock_exists, logger_mock):
    # Giả lập chưa có stunnel chạy (pgrep return code != 0)
    mock_run.return_value.returncode = 1
    configure_security(logger_mock)
    mock_popen.assert_called_once()


@patch("os.path.exists", return_value=False)
def test_configure_security_no_conf(mock_exists, logger_mock):
    with pytest.raises(SystemExit):
        configure_security(logger_mock)


### Bài Kiểm Thử 6: Tối Ưu GPU ###

def test_setup_gpu_optimization(logger_mock, mock_configs):
    _, environmental_limits, _ = mock_configs
    # Chỉ cần chạy không lỗi là được
    setup_gpu_optimization(environmental_limits, logger_mock)


### Bài Kiểm Thử 7: Kiểm Thử tích hợp cgroup_manager ###

# Cần giả lập cgroup_manager.setup_cgroups và assign_process_to_cgroups
# để tránh lỗi khi chạy test mà không có module thực tế.
@patch("mining_environment.scripts.setup_env.setup_cgroups")
def test_setup_calls_setup_cgroups(mock_setup_cgroups, logger_mock):
    # Giả lập môi trường test:
    os.environ["CONFIG_DIR"] = "/some/config"
    os.environ["LOGS_DIR"] = "/tmp/logs"
    # Giả lập file JSON
    system_params = json.dumps({"timezone": "UTC", "locale": "en_US.UTF-8"})
    environmental_limits = json.dumps({
        "baseline_monitoring": {
            "cpu_percent_threshold": 50,
            "gpu_percent_threshold": 50,
            "cache_percent_threshold": 50,
            "network_bandwidth_threshold_mbps": 100,
            "disk_io_threshold_mbps": 200,
            "power_consumption_threshold_watts": 1000
        },
        "temperature_limits": {
            "cpu": {"max_celsius": 60},
            "gpu": {"max_celsius": 70}
        },
        "power_limits": {
            "total_power_watts": {"max": 200},
            "per_device_power_watts": {
                "cpu": {"max": 100},
                "gpu": {"max": 100}
            }
        },
        "memory_limits": {
            "ram_percent_threshold": 80
        },
        "gpu_optimization": {
            "gpu_utilization_percent_optimal": {
                "min": 10,
                "max": 90
            }
        }
    })
    resource_config = json.dumps({
        "resource_allocation": {
            "ram": {"max_allocation_mb": 4096},
            "cpu": {"max_threads": 4},
            "gpu": {"usage_percent_range": {"max": 90}}
        }
    })

    # Mô phỏng open để đọc file json
    m = mock_open()
    m.side_effect = [
        mock_open(read_data=system_params).return_value,
        mock_open(read_data=environmental_limits).return_value,
        mock_open(read_data=resource_config).return_value
    ]

    with patch("builtins.open", m), \
         patch("os.geteuid", return_value=0), \
         patch("mining_environment.scripts.setup_env.configure_system"), \
         patch("mining_environment.scripts.setup_env.configure_security"), \
         patch("mining_environment.scripts.setup_env.setup_gpu_optimization"), \
         patch("mining_environment.scripts.setup_env.setup_logging"):

        setup()
        mock_setup_cgroups.assert_called_once()


### Kiểm Thử Thiếu Quyền Root ###

def test_setup_no_root(logger_mock):
    # Giả lập không có root
    with patch("os.geteuid", return_value=1000):
        with pytest.raises(SystemExit):
            setup()
