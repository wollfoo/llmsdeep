import os
import sys
import pytest
import torch
import pynvml
import psutil
from unittest.mock import patch, MagicMock, mock_open, call
from pathlib import Path

# Add the root directory to sys.path for absolute imports
ROOT_DIR = Path(__file__).parent.parent  # /home/llmss/llmsdeep
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.mining_environment.scripts.anomaly_detector import AnomalyDetector, SafeRestoreEvaluator

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def mock_resource_manager():
    rm = MagicMock()
    rm.shared_resource_manager.is_gpu_initialized.return_value = True
    rm.azure_sentinel_client.get_recent_alerts.return_value = []
    rm.azure_log_analytics_client.query_logs.return_value = []
    rm.azure_security_center_client.get_security_recommendations.return_value = []
    rm.azure_traffic_analytics_client.get_traffic_data.return_value = []
    rm.vms = []
    rm.nsgs = []
    rm.network_watchers = []
    rm.cloaking_request_queue = MagicMock()
    rm.resource_adjustment_queue = MagicMock()
    rm.collect_metrics.return_value = {}
    rm.prepare_input_features.return_value = []
    return rm

@pytest.fixture
def mock_model():
    mock_m = MagicMock()
    mock_m.eval.return_value = None
    mock_m.to.return_value = None
    mock_m.return_value.item.return_value = 0.5  # Example anomaly score
    return mock_m

@pytest.fixture
def config():
    return {
        'baseline_thresholds': {
            'cpu_usage_percent': 80,
            'gpu_usage_percent': 80,
            'ram_usage_percent': 80,
            'disk_io_usage_mbps': 80,
            'network_usage_mbps': 80
        },
        'temperature_limits': {
            'cpu_max_celsius': 75,
            'gpu_max_celsius': 85
        },
        'power_limits': {
            'per_device_power_watts': {
                'cpu': 150,
                'gpu': 300
            }
        },
        'processes': {
            'CPU': 'cpu_miner',
            'GPU': 'gpu_miner'
        },
        'process_priority_map': {
            'cpu_miner': 2,
            'gpu_miner': 3
        },
        'network_interface': 'eth0',
        'ai_driven_monitoring': {
            'detection_interval_seconds': 60,
            'cloak_activation_delay_seconds': 5,
            'anomaly_cloaking_model': {
                'detection_threshold': 0.7
            }
        },
        'resource_allocation': {
            'ram': {
                'max_allocation_mb': 16000,  # 16 GB
                'min_allocation_mb': 8000    # 8 GB
            },
            'gpu': {
                'max_usage_percent': 90
            },
            'disk_io': {
                'min_limit_mbps': 100,
                'max_limit_mbps': 1000
            },
            'network': {
                'bandwidth_limit_mbps': 100
            },
            'cache': {
                'limit_percent': 50
            }
        },
        'monitoring_parameters': {
            'monitoring_level': 'detailed',
            'temperature_monitoring_interval_seconds': 60,
            'power_monitoring_interval_seconds': 60,
            'azure_monitor_interval_seconds': 60,
            'optimization_interval_seconds': 60
        },
        'optimization_parameters': {
            'optimization_level': 'high',
            'gpu_power_adjustment_step': 10,
            'disk_io_limit_step_mbps': 100
        },
        'cloak_strategies': {
            'strategy_type': 'basic'
        },
        'log_analytics': {
            'log_level': 'INFO',
            'log_file_path': '/var/log/anomaly_detector.log'
        },
        'alert_thresholds': {
            'cpu_alert_threshold': 90,
            'gpu_alert_threshold': 90
        }
    }

@pytest.fixture
def model_path(tmp_path):
    # Tạo một tệp mô hình giả để sử dụng trong kiểm thử
    model_file = tmp_path / "model.pth"
    model_file.write_text("dummy model content")
    return model_file


@patch('app.mining_environment.scripts.anomaly_detector.pynvml.nvmlDeviceGetMemoryInfo')
@patch('app.mining_environment.scripts.anomaly_detector.pynvml.nvmlDeviceGetHandleByIndex')
@patch('app.mining_environment.scripts.anomaly_detector.pynvml.nvmlDeviceGetCount', return_value=2)
@patch('app.mining_environment.scripts.anomaly_detector.pynvml.nvmlInit')  # Đúng đường dẫn
@patch('app.mining_environment.scripts.anomaly_detector.torch.load')  # Đúng đường dẫn
@patch('app.mining_environment.scripts.anomaly_detector.torch.no_grad')  # Đúng đường dẫn
@patch('app.mining_environment.scripts.anomaly_detector.torch.tensor')  # Đúng đường dẫn
@patch('app.mining_environment.scripts.anomaly_detector.AnomalyDetector.prepare_input_features', return_value=[0.5])
@patch('app.mining_environment.scripts.anomaly_detector.AnomalyDetector.collect_metrics', return_value={})
@patch('app.mining_environment.scripts.anomaly_detector.AnomalyDetector.discover_mining_processes')
def test_anomaly_detection(
    mock_discover_mining_processes,
    mock_collect_metrics,
    mock_prepare_input_features,
    mock_tensor,
    mock_no_grad,
    mock_torch_load,
    mock_nvml_init,
    mock_nvml_device_get_count,
    mock_nvml_device_get_handle_by_index,
    mock_nvml_device_get_memory_info,
    config, 
    model_path, 
    mock_logger, 
    mock_resource_manager, 
    mock_model
):
    # Thiết lập mock torch.load để trả về mock_model
    mock_torch_load.return_value = mock_model
    
    # Thiết lập các giá trị trả về cho các mock đã patch
    mock_virtual_mem = MagicMock(percent=50)
    mock_disk_io = MagicMock(read_bytes=100 * 1024 * 1024, write_bytes=100 * 1024 * 1024)
    mock_net_io = MagicMock(bytes_sent=50 * 1024 * 1024, bytes_recv=50 * 1024 * 1024)
    
    # Khởi tạo AnomalyDetector
    detector = AnomalyDetector(config=config, model_path=model_path, logger=mock_logger)
    detector.set_resource_manager(mock_resource_manager)
    
    # Giả lập một tiến trình khai thác
    mock_process = MagicMock()
    mock_process.cpu_usage = 30
    mock_process.memory_usage = 50
    mock_process.gpu_usage = 60
    mock_process.disk_io = 100
    mock_process.network_io = 100
    mock_process.name = 'gpu_miner'
    mock_process.pid = 1234
    mock_process.is_cloaked = False
    
    detector.mining_processes = [mock_process]
    
    # Giả lập is_safe_to_restore trả về True
    detector.safe_restore_evaluator = MagicMock()
    detector.safe_restore_evaluator.is_safe_to_restore.return_value = True
    
    # Giả lập một vòng lặp ngắn cho anomaly_detection
    with patch('app.mining_environment.scripts.anomaly_detector.AnomalyDetector.discover_mining_processes') as mock_discover:
        with patch('app.mining_environment.scripts.anomaly_detector.AnomalyDetector.collect_metrics', return_value={}):
            with patch('app.mining_environment.scripts.anomaly_detector.AnomalyDetector.prepare_input_features', return_value=[0.5]):
                with patch('app.mining_environment.scripts.anomaly_detector.torch.tensor') as mock_tensor_inner:
                    mock_tensor_inner.return_value.to.return_value.unsqueeze.return_value = mock_model
                    with patch('app.mining_environment.scripts.anomaly_detector.torch.no_grad'):
                        # Chạy anomaly_detection một lần
                        detector.anomaly_detection()
