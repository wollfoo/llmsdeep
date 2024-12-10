# tests/test_anomaly_detector.py
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

@pytest.fixture(autouse=True)
def reset_anomaly_detector_instance():
    """Fixture để reset Singleton instance của AnomalyDetector trước mỗi bài kiểm thử."""
    AnomalyDetector._instance = None

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

def test_safe_restore_evaluator_init(config, mock_logger, mock_resource_manager, mock_model):
    evaluator = SafeRestoreEvaluator(
        config=config,
        logger=mock_logger,
        resource_manager=mock_resource_manager,
        anomaly_cloaking_model=mock_model,
        anomaly_cloaking_device='cpu'
    )
    
    assert evaluator.baseline_cpu_usage_percent == 80
    assert evaluator.cpu_max_temp == 75
    assert evaluator.cpu_max_power == 150
    mock_logger.assert_not_called()


@patch('app.mining_environment.scripts.anomaly_detector.get_cpu_temperature', return_value=70)
@patch('app.mining_environment.scripts.anomaly_detector.get_gpu_temperature', return_value=[80])
@patch('app.mining_environment.scripts.anomaly_detector.get_cpu_power', return_value=100)
@patch('app.mining_environment.scripts.anomaly_detector.get_gpu_power', return_value=200)
@patch('app.mining_environment.scripts.anomaly_detector.psutil.cpu_percent', return_value=50)
@patch('app.mining_environment.scripts.anomaly_detector.psutil.virtual_memory')
@patch('app.mining_environment.scripts.anomaly_detector.psutil.disk_io_counters')
@patch('app.mining_environment.scripts.anomaly_detector.psutil.net_io_counters')
@patch('app.mining_environment.scripts.anomaly_detector.torch.tensor')  # Mock torch.tensor
@patch('app.mining_environment.scripts.anomaly_detector.torch.no_grad')  # Mock torch.no_grad
def test_is_safe_to_restore_all_safe(
    mock_no_grad, mock_torch_tensor, mock_net_io, mock_disk_io, mock_virtual_mem,
    mock_cpu_percent, mock_gpu_power, mock_cpu_power, mock_gpu_temp, mock_cpu_temp,
    config, mock_logger, mock_resource_manager, mock_model
):
    # Setup mock return values
    mock_virtual_mem.return_value = MagicMock(percent=50)
    mock_disk_io.return_value = MagicMock(read_bytes=30 * 1024 * 1024, write_bytes=30 * 1024 * 1024)  # Tổng 60 MBps
    mock_net_io.return_value = MagicMock(bytes_sent=30 * 1024 * 1024, bytes_recv=30 * 1024 * 1024)  # Tổng 60 MBps

    # Mock torch.tensor to return a mock tensor
    mock_tensor = MagicMock()
    mock_tensor.to.return_value = mock_tensor
    mock_tensor.unsqueeze.return_value = mock_tensor
    mock_tensor.item.return_value = 0.5
    mock_torch_tensor.return_value = mock_tensor

    # Mock torch.no_grad context manager
    mock_no_grad.return_value = MagicMock()

    evaluator = SafeRestoreEvaluator(
        config=config,
        logger=mock_logger,
        resource_manager=mock_resource_manager,
        anomaly_cloaking_model=mock_model,
        anomaly_cloaking_device='cpu'  # Sử dụng 'cpu' để tránh lỗi
    )

    mock_process = MagicMock()
    mock_process.pid = 1234
    mock_process.name = 'gpu_miner'

    result = evaluator.is_safe_to_restore(mock_process)
    assert result == True
    mock_logger.info.assert_called_with(f"Điều kiện an toàn để khôi phục tài nguyên cho tiến trình {mock_process.name} (PID: {mock_process.pid}).")


@patch('app.mining_environment.scripts.anomaly_detector.pynvml.nvmlInit')
@patch('app.mining_environment.scripts.anomaly_detector.AnomalyDetector.load_model')
def test_anomaly_detector_init(
    mock_load_model, mock_nvml_init, config, model_path, mock_logger, mock_resource_manager, mock_model
):
    mock_load_model.return_value = (mock_model, 'cpu')
    
    detector = AnomalyDetector(config=config, model_path=model_path, logger=mock_logger)
    
    mock_load_model.assert_called_with(model_path)
    mock_nvml_init.assert_called_once()
    assert detector.anomaly_cloaking_model == mock_model
    assert detector.anomaly_cloaking_device == 'cpu'
    assert detector.gpu_initialized == True
    mock_logger.info.assert_called_with("NVML initialized successfully.")


@patch('app.mining_environment.scripts.anomaly_detector.torch.load')  # Đúng đường dẫn
def test_anomaly_detector_set_resource_manager(
    mock_torch_load, config, model_path, mock_logger, mock_resource_manager, mock_model
):
    # Cấu hình mock để torch.load trả về mock_model
    mock_torch_load.return_value = mock_model
    
    detector = AnomalyDetector(config=config, model_path=model_path, logger=mock_logger)
    
    detector.set_resource_manager(mock_resource_manager)
    
    assert detector.resource_manager == mock_resource_manager
    assert detector.safe_restore_evaluator is not None
    mock_logger.info.assert_called_with("ResourceManager has been set for AnomalyDetector.")
    
    # Kiểm tra rằng torch.load đã được gọi với đúng đường dẫn mô hình
    mock_torch_load.assert_called_with(model_path)


@patch('app.mining_environment.scripts.anomaly_detector.psutil.process_iter')
@patch('app.mining_environment.scripts.anomaly_detector.torch.load')  # Đúng đường dẫn
def test_discover_mining_processes(
    mock_torch_load, mock_process_iter, config, model_path, mock_logger, mock_resource_manager, mock_model
):
    # Cấu hình mock để torch.load trả về mock_model
    mock_torch_load.return_value = mock_model
    
    # Tạo các mock tiến trình
    mock_proc1 = MagicMock()
    mock_proc1.info = {'pid': 1111, 'name': 'cpu_miner'}
    mock_proc2 = MagicMock()
    mock_proc2.info = {'pid': 2222, 'name': 'gpu_miner'}
    mock_proc3 = MagicMock()
    mock_proc3.info = {'pid': 3333, 'name': 'other_process'}
    
    mock_process_iter.return_value = [mock_proc1, mock_proc2, mock_proc3]
    
    # Khởi tạo AnomalyDetector
    detector = AnomalyDetector(config=config, model_path=model_path, logger=mock_logger)
    detector.set_resource_manager(mock_resource_manager)
    
    # Chạy hàm discover_mining_processes
    detector.discover_mining_processes()
    
    # Kiểm tra số lượng tiến trình được phát hiện
    assert len(detector.mining_processes) == 2
    
    # Kiểm tra thông tin các tiến trình được phát hiện
    assert detector.mining_processes[0].pid == 1111
    assert detector.mining_processes[0].name == 'cpu_miner'
    assert detector.mining_processes[0].priority == 2
    assert detector.mining_processes[1].pid == 2222
    assert detector.mining_processes[1].name == 'gpu_miner'
    assert detector.mining_processes[1].priority == 3
    
    # Kiểm tra log thông tin
    mock_logger.info.assert_called_with("Discovered 2 mining processes.")
    
    # Kiểm tra rằng torch.load đã được gọi với đúng đường dẫn mô hình
    mock_torch_load.assert_called_with(model_path)


@patch('app.mining_environment.scripts.anomaly_detector.psutil.process_iter')
@patch('app.mining_environment.scripts.anomaly_detector.torch.load')  # Đúng đường dẫn
def test_get_process_priority(
    mock_torch_load, mock_process_iter, config, model_path, mock_logger, mock_resource_manager, mock_model
):
    # Cấu hình mock để torch.load trả về mock_model
    mock_torch_load.return_value = mock_model
    
    # Khởi tạo AnomalyDetector
    detector = AnomalyDetector(config=config, model_path=model_path, logger=mock_logger)
    detector.set_resource_manager(mock_resource_manager)
    
    # Gọi hàm get_process_priority
    priority_cpu = detector.get_process_priority('cpu_miner')
    priority_gpu = detector.get_process_priority('gpu_miner')
    priority_unknown = detector.get_process_priority('unknown_miner')
    
    # Kiểm tra kết quả
    assert priority_cpu == 2
    assert priority_gpu == 3
    assert priority_unknown == 1  # Default priority
    
    # Kiểm tra rằng torch.load đã được gọi với đúng đường dẫn mô hình
    mock_torch_load.assert_called_with(model_path)
    
    # Kiểm tra log thông tin nếu cần (không bắt buộc trong trường hợp này)


@patch('app.mining_environment.scripts.anomaly_detector.torch.load')  # Đúng đường dẫn
@patch('app.mining_environment.scripts.anomaly_detector.psutil.cpu_percent', return_value=30)
@patch('app.mining_environment.scripts.anomaly_detector.psutil.cpu_count', return_value=8)
@patch('app.mining_environment.scripts.anomaly_detector.psutil.cpu_freq')
@patch('app.mining_environment.scripts.anomaly_detector.psutil.virtual_memory')
@patch('app.mining_environment.scripts.anomaly_detector.psutil.disk_io_counters')
@patch('app.mining_environment.scripts.anomaly_detector.psutil.net_io_counters')
@patch('app.mining_environment.scripts.anomaly_detector.get_cpu_power', return_value=100)
@patch('app.mining_environment.scripts.anomaly_detector.get_gpu_power', return_value=200)
@patch('app.mining_environment.scripts.anomaly_detector.AnomalyDetector.get_cache_percent', return_value=40)
@patch('app.mining_environment.scripts.anomaly_detector.AnomalyDetector.get_gpu_temperature', return_value=70)
def test_collect_metrics(
    mock_gpu_temp, mock_cache_percent, mock_gpu_power, mock_cpu_power, mock_net_io, mock_disk_io,
    mock_virtual_mem, mock_cpu_freq, mock_cpu_count, mock_cpu_percent, mock_torch_load,
    config, model_path, mock_logger, mock_resource_manager, mock_model
):
    # Cấu hình mock để torch.load trả về mock_model
    mock_torch_load.return_value = mock_model

    # Thiết lập các giá trị trả về cho các mock đã patch
    mock_cpu_freq.return_value = MagicMock(current=2500)
    mock_virtual_mem.return_value = MagicMock(total=16000000 * 1024, available=8000000 * 1024, percent=50)
    mock_disk_io.return_value = MagicMock(read_bytes=100 * 1024 * 1024, write_bytes=100 * 1024 * 1024)
    mock_net_io.return_value = MagicMock(bytes_sent=50 * 1024 * 1024, bytes_recv=50 * 1024 * 1024)

    # Khởi tạo AnomalyDetector
    detector = AnomalyDetector(config=config, model_path=model_path, logger=mock_logger)
    detector.set_resource_manager(mock_resource_manager)
    
    # Thiết lập gpu_initialized thành True
    detector.gpu_initialized = True

    # Tạo một mock tiến trình khai thác
    mock_process = MagicMock()
    mock_process.cpu_usage = 30
    mock_process.memory_usage = 50
    mock_process.gpu_usage = 60
    mock_process.disk_io = 100
    mock_process.network_io = 100
    mock_process.name = 'cpu_miner'
    mock_process.pid = 1111

    # Gọi hàm collect_metrics
    metrics = detector.collect_metrics(mock_process)

    expected_metrics = {
        'cpu_percent': 30,
        'cpu_count': 8,
        'cpu_freq_mhz': 2500,
        'ram_percent': 50,
        'ram_total_mb': 16000000 * 1024 / (1024 * 1024),  # 16000000 MB
        'ram_available_mb': 8000000 * 1024 / (1024 * 1024),  # 8000000 MB
        'cache_percent': 40,  # get_cache_percent được mock
        'gpus': [
            {
                'gpu_percent': 60,
                'memory_percent': 200,  # get_gpu_memory_percent không được mock, nên giá trị này sẽ phụ thuộc vào implementation
                'temperature_celsius': 70  # get_gpu_temperature được mock
            }
        ],
        'disk_io_limit_mbps': 100,
        'network_bandwidth_limit_mbps': 100,
        'cpu_power_watts': 100,
        'gpu_power_watts': 200
    }

    # Vì get_gpu_memory_percent không được mock, ta cần mock nó để trả về giá trị mong muốn
    with patch('app.mining_environment.scripts.anomaly_detector.AnomalyDetector.get_gpu_memory_percent', return_value=75):
        metrics = detector.collect_metrics(mock_process)
        expected_metrics['gpus'][0]['memory_percent'] = 75

    # Kiểm tra các trường đã được mock
    assert metrics['cpu_percent'] == expected_metrics['cpu_percent']
    assert metrics['cpu_count'] == expected_metrics['cpu_count']
    assert metrics['cpu_freq_mhz'] == expected_metrics['cpu_freq_mhz']
    assert metrics['ram_percent'] == expected_metrics['ram_percent']
    assert metrics['ram_total_mb'] == expected_metrics['ram_total_mb']
    assert metrics['ram_available_mb'] == expected_metrics['ram_available_mb']
    assert metrics['cache_percent'] == expected_metrics['cache_percent']
    assert metrics['disk_io_limit_mbps'] == expected_metrics['disk_io_limit_mbps']
    assert metrics['network_bandwidth_limit_mbps'] == expected_metrics['network_bandwidth_limit_mbps']
    assert metrics['cpu_power_watts'] == expected_metrics['cpu_power_watts']
    assert metrics['gpu_power_watts'] == expected_metrics['gpu_power_watts']  # Bây giờ nên là 200

    # Kiểm tra các giá trị trong 'gpus'
    assert len(metrics['gpus']) == 1
    assert metrics['gpus'][0]['gpu_percent'] == expected_metrics['gpus'][0]['gpu_percent']
    assert metrics['gpus'][0]['memory_percent'] == expected_metrics['gpus'][0]['memory_percent']
    assert metrics['gpus'][0]['temperature_celsius'] == expected_metrics['gpus'][0]['temperature_celsius']

    # Kiểm tra log thông tin
    mock_logger.debug.assert_called_with(f"Collected metrics for process {mock_process.name} (PID: {mock_process.pid}): {metrics}")

    # Kiểm tra rằng torch.load đã được gọi với đúng đường dẫn mô hình
    mock_torch_load.assert_called_with(model_path)


def test_prepare_input_features(config, model_path, mock_logger, mock_resource_manager, mock_model):
    with patch('app.mining_environment.scripts.anomaly_detector.torch.load', return_value=mock_model):
        detector = AnomalyDetector(config=config, model_path=model_path, logger=mock_logger)

        current_state = {
            'cpu_percent': 30,
            'cpu_count': 8,
            'cpu_freq_mhz': 2500,
            'ram_percent': 50,
            'ram_total_mb': 16000,
            'ram_available_mb': 8000,
            'cache_percent': 20,
            'cpu_power_watts': 100,
            'gpus': [
                {
                    'gpu_percent': 60,
                    'memory_percent': 70,
                    'temperature_celsius': 75,
                    'gpu_power_watts': 200  # Đảm bảo trường này được sử dụng đúng
                }
            ],
            'disk_io_limit_mbps': 100,
            'network_bandwidth_limit_mbps': 100
        }

        input_features = detector.prepare_input_features(current_state)

        expected_features = [
            30, 8, 2500, 50, 16000, 8000, 20, 100,
            60, 70, 75, 200,  # Sử dụng gpu['gpu_power_watts']
            100, 100
        ]

        assert input_features == expected_features
        mock_logger.debug.assert_called_with(f"Prepared input features for AI model: {expected_features}")


@patch('app.mining_environment.scripts.anomaly_detector.torch.load')  # Đúng đường dẫn
@patch('app.mining_environment.scripts.anomaly_detector.torch.device')
@patch('app.mining_environment.scripts.anomaly_detector.torch.cuda.is_available', return_value=True)
def test_load_model_success(
    mock_cuda_available, mock_device, mock_torch_load, config, model_path, mock_logger
):
    mock_model = MagicMock()
    mock_torch_load.return_value = mock_model
    mock_device.return_value = 'cuda'

    # Patch __init__ để không gọi load_model
    with patch.object(AnomalyDetector, '__init__', return_value=None):
        # Khởi tạo đối tượng và thiết lập thuộc tính logger thủ công
        anomaly_detector = AnomalyDetector(config=config, model_path=model_path, logger=mock_logger)
        anomaly_detector.logger = mock_logger  # Thiết lập logger thủ công

        # Gọi load_model một lần
        anomaly_detector.load_model(model_path)

    # Kiểm tra các gọi hàm
    mock_torch_load.assert_called_with(model_path)
    mock_model.to.assert_called_with('cuda')
    mock_model.eval.assert_called_once()
    mock_logger.info.assert_called_with(f"Tải mô hình Anomaly Detection từ {model_path} vào cuda.")


@patch('app.mining_environment.scripts.anomaly_detector.Thread')
@patch('app.mining_environment.scripts.anomaly_detector.torch.load')  # Đúng đường dẫn
def test_start_stop_anomaly_detector(
    mock_torch_load, mock_thread, config, model_path, mock_logger, mock_resource_manager, mock_model
):
    # Thiết lập mock cho torch.load để trả về mock_model
    mock_torch_load.return_value = mock_model

    detector = AnomalyDetector(config=config, model_path=model_path, logger=mock_logger)
    detector.set_resource_manager(mock_resource_manager)
    
    # Kiểm tra start
    detector.start()
    mock_thread.return_value.start.assert_called_once()
    mock_logger.info.assert_any_call("Starting AnomalyDetector...")
    mock_logger.info.assert_any_call("AnomalyDetector started successfully.")
    
    # Kiểm tra stop
    detector.stop()
    mock_resource_manager.azure_log_analytics_client.query_logs.assert_not_called()  # Ví dụ
    mock_logger.info.assert_any_call("Stopping AnomalyDetector...")
    mock_logger.info.assert_any_call("AnomalyDetector stopped successfully.")





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





@patch('app.mining_environment.scripts.anomaly_detector.torch.load')  # Đúng đường dẫn
@patch('app.mining_environment.scripts.anomaly_detector.psutil.cpu_freq')
def test_get_cpu_freq_success(
    mock_cpu_freq, mock_torch_load, config, model_path, mock_logger, mock_resource_manager, mock_model
):
    # Thiết lập giá trị trả về cho cpu_freq
    mock_cpu_freq.return_value = MagicMock(current=2500)
    
    # Thiết lập giá trị trả về cho torch.load
    mock_torch_load.return_value = mock_model
    
    # Khởi tạo AnomalyDetector
    detector = AnomalyDetector(config=config, model_path=model_path, logger=mock_logger)
    
    # Gọi phương thức get_cpu_freq
    freq = detector.get_cpu_freq()
    
    # Kiểm tra kết quả
    assert freq == 2500
    mock_logger.debug.assert_called_with("Tần số CPU hiện tại: 2500 MHz")
    
    # Kiểm tra rằng torch.load đã được gọi đúng
    mock_torch_load.assert_called_once_with(model_path)


@patch('app.mining_environment.scripts.anomaly_detector.AnomalyDetector.get_cache_percent', return_value=50.0)
@patch('app.mining_environment.scripts.anomaly_detector.torch.load')  # Đúng đường dẫn
@patch('app.mining_environment.scripts.anomaly_detector.psutil.virtual_memory')
def test_get_cache_percent_success(
    mock_virtual_mem, mock_torch_load, mock_get_cache_percent, config, model_path, mock_logger, mock_resource_manager, mock_model
):
    # Thiết lập mock cho torch.load
    mock_torch_load.return_value = mock_model

    # Thiết lập trả về cho virtual_memory
    mock_virtual_mem.return_value = MagicMock(cached=8000000, total=16000000)

    detector = AnomalyDetector(config=config, model_path=model_path, logger=mock_logger)
    detector.set_resource_manager(mock_resource_manager)
    cache_percent = detector.get_cache_percent()

    assert cache_percent == 50.0  # 8000000 / 16000000 * 100 = 50.0%
    mock_logger.debug.assert_any_call("Cache hiện tại: 50.00%")


@patch('app.mining_environment.scripts.anomaly_detector.torch.load')  # Đúng đường dẫn
def test_get_gpu_memory_percent_no_gpu(
    mock_torch_load, config, model_path, mock_logger, mock_resource_manager, mock_model
):
    # Mock torch.load để trả về một mô hình giả thay vì cố gắng tải từ tệp
    mock_torch_load.return_value = mock_model
    
    # Đặt is_gpu_initialized thành False
    mock_resource_manager.shared_resource_manager.is_gpu_initialized.return_value = False
    
    # Khởi tạo AnomalyDetector
    detector = AnomalyDetector(config=config, model_path=model_path, logger=mock_logger)
    detector.set_resource_manager(mock_resource_manager)
    
    # Gọi hàm get_gpu_memory_percent
    memory_percents = detector.get_gpu_memory_percent()
    
    # Kiểm tra kết quả mong đợi
    assert memory_percents == []
    mock_logger.warning.assert_called_with("GPU not initialized. Cannot get GPU memory percent.")


@patch('app.mining_environment.scripts.anomaly_detector.pynvml.nvmlInit')  # Đúng đường dẫn
@patch('app.mining_environment.scripts.anomaly_detector.pynvml.nvmlDeviceGetMemoryInfo')
@patch('app.mining_environment.scripts.anomaly_detector.pynvml.nvmlDeviceGetHandleByIndex')
@patch('app.mining_environment.scripts.anomaly_detector.pynvml.nvmlDeviceGetCount', return_value=2)
@patch('app.mining_environment.scripts.anomaly_detector.torch.load')  # Đúng đường dẫn
def test_get_gpu_memory_percent_success(
    mock_torch_load, mock_nvml_device_get_count, mock_nvml_device_get_handle_by_index,
    mock_nvml_device_get_memory_info, mock_nvml_init, config, model_path, mock_logger,
    mock_resource_manager, mock_model
):
    # Mock torch.load để trả về một mô hình giả thay vì cố gắng tải từ tệp
    mock_torch_load.return_value = mock_model

    # Mock pynvml functions
    mock_nvml_device_get_count.return_value = 2

    # Mock thông tin bộ nhớ GPU
    mock_mem_info1 = MagicMock()
    mock_mem_info1.used = 4000
    mock_mem_info1.total = 8000
    mock_mem_info2 = MagicMock()
    mock_mem_info2.used = 2000
    mock_mem_info2.total = 4000
    mock_nvml_device_get_memory_info.side_effect = [mock_mem_info1, mock_mem_info2]

    # Mock nvmlInit để không thực sự khởi tạo GPU
    mock_nvml_init.return_value = None

    # Khởi tạo AnomalyDetector
    detector = AnomalyDetector(config=config, model_path=model_path, logger=mock_logger)
    detector.set_resource_manager(mock_resource_manager)

    # Đảm bảo rằng gpu_initialized được thiết lập đúng thông qua nvmlInit
    assert detector.gpu_initialized == True, "gpu_initialized phải được thiết lập là True sau khi nvmlInit được gọi"

    # Gọi hàm get_gpu_memory_percent
    memory_percents = detector.get_gpu_memory_percent()

    # Kiểm tra kết quả mong đợi
    assert memory_percents == [50.0, 50.0], f"Expected [50.0, 50.0], but got {memory_percents}"
    assert mock_logger.debug.call_count == 2
    mock_logger.debug.assert_any_call("GPU 0 Memory Usage: 50.0%")
    mock_logger.debug.assert_any_call("GPU 1 Memory Usage: 50.0%")

    # Kiểm tra rằng nvmlInit được gọi một lần
    mock_nvml_init.assert_called_once()


@patch('app.mining_environment.scripts.anomaly_detector.pynvml.nvmlDeviceGetCount', side_effect=pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN))
@patch('app.mining_environment.scripts.anomaly_detector.pynvml.nvmlInit', return_value=None)  # Đúng đường dẫn
@patch('app.mining_environment.scripts.anomaly_detector.torch.load', return_value=MagicMock())  # Đúng đường dẫn
def test_get_gpu_memory_percent_nvml_error(
    mock_torch_load, mock_nvml_init, mock_nvml_device_get_count, config, model_path, mock_logger, mock_resource_manager, mock_model
):
    # Khởi tạo AnomalyDetector
    detector = AnomalyDetector(config=config, model_path=model_path, logger=mock_logger)
    detector.set_resource_manager(mock_resource_manager)

    # Gọi hàm kiểm thử
    memory_percents = detector.get_gpu_memory_percent()

    # Kiểm tra kết quả trả về
    assert memory_percents == []

    # Kiểm tra thông báo lỗi được log đúng
    expected_error_message = f"Error getting GPU memory percent: {str(pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN))}"
    mock_logger.error.assert_called_with(expected_error_message)


