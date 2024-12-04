@pytest.fixture
def resource_manager(simple_mock_logger, monkeypatch):
    """Fixture để tạo instance của ResourceManager với các tham số mock và patch các phương thức phụ thuộc."""
    # 1. Thiết lập biến môi trường AZURE_SUBSCRIPTION_ID
    monkeypatch.setenv('AZURE_SUBSCRIPTION_ID', 'dummy_subscription_id')

    # 2. Patch các phương thức phụ thuộc trước khi khởi tạo ResourceManager
    with patch('mining_environment.scripts.resource_manager.psutil.process_iter', return_value=[]), \
         patch('mining_environment.scripts.resource_manager.ResourceManager.load_model', return_value=(MagicMock(), MagicMock())) as mock_load_model, \
         patch('mining_environment.scripts.resource_manager.ResourceManager.initialize_azure_clients') as mock_init_azure_clients, \
         patch('mining_environment.scripts.resource_manager.ResourceManager.discover_azure_resources') as mock_discover_azure_resources, \
         patch('mining_environment.scripts.resource_manager.ResourceManager.initialize_threads') as mock_initialize_threads, \
         patch('mining_environment.scripts.resource_manager.SharedResourceManager', autospec=True) as mock_shared_resource_manager_class, \
         patch('mining_environment.scripts.resource_manager.ResourceManager.shutdown_power_management') as mock_shutdown_power_management, \
         patch('mining_environment.scripts.resource_manager.ResourceManager.join_threads') as mock_join_threads:

        # 3. Định nghĩa cấu hình đầy đủ với tất cả các khóa bắt buộc
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

        # 4. Khởi tạo ResourceManager với cấu hình đã được patch
        manager = ResourceManager(config, model_path, simple_mock_logger)

        # 5. Thiết lập các thuộc tính thread bằng MagicMock
        manager.monitor_thread = MagicMock()
        manager.optimization_thread = MagicMock()
        manager.cloaking_thread = MagicMock()
        manager.resource_adjustment_thread = MagicMock()

        # 6. Trực tiếp mock các phương thức phụ thuộc trên instance
        manager.shutdown_power_management = mock_shutdown_power_management
        manager.join_threads = mock_join_threads

        # 7. Trực tiếp mock phương thức 'set' của 'stop_event'
        manager.stop_event.set = MagicMock()

        # 8. Truy cập SharedResourceManager instance
        shared_resource_manager = manager.shared_resource_manager

        # 9. Truy cập và thiết lập các phương thức của SharedResourceManager
        shared_resource_manager.is_gpu_initialized.return_value = True
        shared_resource_manager.adjust_cpu_threads = MagicMock()
        shared_resource_manager.adjust_gpu_usage = MagicMock()
        shared_resource_manager.adjust_ram_allocation = MagicMock()

        # 10. Trực tiếp mock resource_adjustment_queue để tránh lỗi TypeError
        manager.resource_adjustment_queue = MagicMock()

        # 11. Trả về manager và các mock để kiểm tra trong hàm kiểm thử
        return {
            'manager': manager,
            'mock_load_model': mock_load_model,
            'mock_init_azure_clients': mock_init_azure_clients,
            'mock_discover_azure_resources': mock_discover_azure_resources,
            'mock_initialize_threads': mock_initialize_threads,
            'mock_shared_resource_manager_class': mock_shared_resource_manager_class,
            'mock_shutdown_power_management': mock_shutdown_power_management,
            'mock_join_threads': mock_join_threads,
            'mock_event_set': manager.stop_event.set,
            'simple_mock_logger': simple_mock_logger,
            'mock_shared_resource_manager_instance': shared_resource_manager,
            # Bổ sung mining_processes để truy cập trực tiếp trong kiểm thử
            'mining_processes': manager.mining_processes
        }
