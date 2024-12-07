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

    # Thiết lập original_resource_limits để phương thức restore_resources hoạt động đúng
    manager.original_resource_limits = {
        1111: {  # PID của process trong kiểm thử
            'cpu_freq': 2400,
            'cpu_threads': 4,
            'ram_allocation_mb': 2048,
            'gpu_power_limit': 300,
            'ionice_class': 2,
            'network_bandwidth_limit_mbps': 100
        }
    }

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
