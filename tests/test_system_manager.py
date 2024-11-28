# tests/test_system_manager.py

import os
import sys
import pytest
import json
from unittest import mock
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Thiết lập biến môi trường TESTING=1 trước khi import bất kỳ module nào
os.environ["TESTING"] = "1"

# Định nghĩa các thư mục cần thiết dựa trên cấu trúc dự án
APP_DIR = Path(__file__).resolve().parent.parent / "app"
CONFIG_DIR = APP_DIR / "mining_environment" / "config"
MODELS_DIR = APP_DIR / "mining_environment" / "models"
SCRIPTS_DIR = APP_DIR / "mining_environment" / "scripts"

# Thêm thư mục APP_DIR vào sys.path để Python có thể tìm thấy các module
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Import các module cần thiết sau khi đã thiết lập đường dẫn
from mining_environment.scripts.system_manager import load_config, SystemManager, start, stop

@pytest.fixture
def mock_logging():
    with patch('mining_environment.scripts.system_manager.setup_logging') as mock_setup_logging:
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        yield mock_logger

@pytest.fixture
def mock_resource_manager():
    with patch('mining_environment.scripts.system_manager.ResourceManager') as mock_rm:
        yield mock_rm

@pytest.fixture
def mock_anomaly_detector():
    with patch('mining_environment.scripts.system_manager.AnomalyDetector') as mock_ad:
        yield mock_ad

@pytest.fixture
def mock_sys_exit():
    with patch('mining_environment.scripts.system_manager.sys.exit') as mock_exit:
        yield mock_exit

@pytest.fixture
def mock_sleep():
    with patch('mining_environment.scripts.system_manager.sleep') as mock_sleep:
        yield mock_sleep

@pytest.fixture
def mock_system_manager_start_stop():
    with patch.object(SystemManager, 'start') as mock_start, \
         patch.object(SystemManager, 'stop') as mock_stop:
        yield mock_start, mock_stop

# 1. Kiểm thử cho hàm load_config
def test_load_config_success(mock_logging):
    config_content = {"key": "value"}
    m = mock_open(read_data=json.dumps(config_content))
    with patch('builtins.open', m):
        config_path = Path("/fake/config.json")
        result = load_config(config_path)
        m.assert_called_once_with(config_path, 'r')
        assert result == config_content
        mock_logging.info.assert_called_with(f"Đã tải cấu hình từ {config_path}")

def test_load_config_file_not_found(mock_logging):
    with patch('builtins.open', side_effect=FileNotFoundError):
        config_path = Path("/fake/missing_config.json")
        with pytest.raises(SystemExit) as exc_info:
            load_config(config_path)
        assert exc_info.value.code == 1
        mock_logging.error.assert_called_with(f"Tệp cấu hình không tìm thấy: {config_path}")

def test_load_config_json_error(mock_logging):
    m = mock_open(read_data="Invalid JSON")
    with patch('builtins.open', m):
        config_path = Path("/fake/bad_config.json")
        with pytest.raises(SystemExit) as exc_info:
            load_config(config_path)
        assert exc_info.value.code == 1
        mock_logging.error.assert_called()
        args, _ = mock_logging.error.call_args
        assert "Lỗi cú pháp JSON trong tệp cấu hình" in args[0]

# 2. Kiểm thử cho lớp SystemManager
def test_system_manager_init(mock_logging, mock_resource_manager, mock_anomaly_detector):
    config = {"config_key": "config_value"}
    system_manager = SystemManager(config)
    
    # Kiểm tra lưu cấu hình
    assert system_manager.config == config
    
    # Kiểm tra logger được gán đúng
    assert system_manager.system_logger == mock_logging
    assert system_manager.resource_logger == mock_logging
    assert system_manager.anomaly_logger == mock_logging
    
    # Kiểm tra ResourceManager và AnomalyDetector được khởi tạo với đúng tham số
    mock_resource_manager.assert_called_once_with(
        config,
        MODELS_DIR / "resource_optimization_model.pt",
        mock_logging
    )
    mock_anomaly_detector.assert_called_once_with(
        config,
        MODELS_DIR / "anomaly_cloaking_model.pt",
        mock_logging
    )
    
    # Kiểm tra set_resource_manager được gọi
    system_manager.anomaly_detector.set_resource_manager.assert_called_once_with(system_manager.resource_manager)
    
    # Kiểm tra log thông báo khởi tạo thành công
    mock_logging.info.assert_called_with("SystemManager đã được khởi tạo thành công.")

def test_system_manager_start_success(mock_logging, mock_resource_manager, mock_anomaly_detector):
    config = {"config_key": "config_value"}
    system_manager = SystemManager(config)
    
    system_manager.start()
    
    # Kiểm tra rằng start được gọi trên các thành phần
    system_manager.resource_manager.start.assert_called_once()
    system_manager.anomaly_detector.start.assert_called_once()
    
    # Kiểm tra log thông báo
    mock_logging.info.assert_any_call("Đang khởi động SystemManager...")
    mock_logging.info.assert_any_call("SystemManager đã khởi động thành công.")

def test_system_manager_start_failure(mock_logging, mock_resource_manager, mock_anomaly_detector, mock_sys_exit):
    config = {"config_key": "config_value"}
    system_manager = SystemManager(config)
    
    # Giả lập lỗi khi khởi động ResourceManager
    system_manager.resource_manager.start.side_effect = Exception("Start error")
    
    with pytest.raises(Exception) as exc_info:
        system_manager.start()
    
    assert str(exc_info.value) == "Start error"
    
    # Kiểm tra rằng stop được gọi trong trường hợp lỗi
    system_manager.stop.assert_called_once()
    
    # Kiểm tra log lỗi
    mock_logging.error.assert_called_with("Lỗi khi khởi động SystemManager: Start error")

def test_system_manager_stop_success(mock_logging, mock_resource_manager, mock_anomaly_detector):
    config = {"config_key": "config_value"}
    system_manager = SystemManager(config)
    
    system_manager.stop()
    
    # Kiểm tra rằng stop được gọi trên các thành phần
    system_manager.resource_manager.stop.assert_called_once()
    system_manager.anomaly_detector.stop.assert_called_once()
    
    # Kiểm tra log thông báo dừng thành công
    mock_logging.info.assert_any_call("Đang dừng SystemManager...")
    mock_logging.info.assert_any_call("SystemManager đã dừng thành công.")

def test_system_manager_stop_failure(mock_logging, mock_resource_manager, mock_anomaly_detector):
    config = {"config_key": "config_value"}
    system_manager = SystemManager(config)
    
    # Giả lập lỗi khi dừng AnomalyDetector
    system_manager.anomaly_detector.stop.side_effect = Exception("Stop error")
    
    with pytest.raises(Exception) as exc_info:
        system_manager.stop()
    
    assert str(exc_info.value) == "Stop error"
    
    # Kiểm tra log lỗi
    mock_logging.error.assert_called_with("Lỗi khi dừng SystemManager: Stop error")

# 3. Kiểm thử cho hàm start và stop toàn hệ thống
def test_start_success(mock_logging, mock_resource_manager, mock_anomaly_detector, mock_sys_exit):
    # Giả lập tồn tại các mô hình AI
    with patch.object(Path, 'exists', return_value=True):
        # Giả lập load_config thành công
        config_data = {"key": "value"}
        m = mock_open(read_data=json.dumps(config_data))
        with patch('builtins.open', m):
            # Giả lập khởi tạo SystemManager và start thành công
            with patch('mining_environment.scripts.system_manager.SystemManager') as mock_sm:
                instance = mock_sm.return_value
                instance.start.return_value = None
                
                # Giả lập vòng lặp không thực sự chạy để tránh vòng lặp vô hạn
                with patch('mining_environment.scripts.system_manager.sleep', side_effect=KeyboardInterrupt):
                    with pytest.raises(KeyboardInterrupt):
                        start()
                
                # Kiểm tra load_config được gọi cho từng tệp cấu hình
                m.assert_any_call(CONFIG_DIR / "resource_config.json", 'r')
                m.assert_any_call(CONFIG_DIR / "process_config.json", 'r')
                
                # Kiểm tra SystemManager được khởi tạo với cấu hình hợp nhất
                expected_config = {**config_data, **config_data}
                mock_sm.assert_called_once_with(expected_config)
                
                # Kiểm tra start được gọi trên SystemManager
                instance.start.assert_called_once()
                
                # Kiểm tra log thông báo hệ thống đang chạy
                mock_logging.info.assert_any_call("SystemManager đang chạy. Nhấn Ctrl+C để dừng.")

def test_start_missing_ai_model(mock_logging, mock_resource_manager, mock_anomaly_detector, mock_sys_exit):
    # Giả lập một trong các mô hình AI không tồn tại
    def exists_side_effect(path):
        if path == MODELS_DIR / "resource_optimization_model.pt":
            return False
        return True

    with patch.object(Path, 'exists', side_effect=exists_side_effect):
        # Giả lập load_config thành công
        config_data = {"key": "value"}
        m = mock_open(read_data=json.dumps(config_data))
        with patch('builtins.open', m):
            with patch('mining_environment.scripts.system_manager.SystemManager') as mock_sm:
                with pytest.raises(SystemExit) as exc_info:
                    start()
                assert exc_info.value.code == 1
                # Kiểm tra log lỗi về mô hình AI
                mock_logging.error.assert_called_with(f"Mô hình AI không tìm thấy tại: {MODELS_DIR / 'resource_optimization_model.pt'}")
                # Kiểm tra rằng SystemManager không được khởi tạo
                mock_sm.assert_not_called()

def test_start_load_config_failure(mock_logging, mock_resource_manager, mock_anomaly_detector, mock_sys_exit):
    # Giả lập load_config thất bại
    with patch('builtins.open', side_effect=FileNotFoundError):
        with pytest.raises(SystemExit) as exc_info:
            start()
        assert exc_info.value.code == 1
        # Kiểm tra log lỗi được ghi bởi load_config
        mock_logging.error.assert_called()
        # Kiểm tra rằng SystemManager không được khởi tạo
        with patch('mining_environment.scripts.system_manager.SystemManager') as mock_sm:
            mock_sm.assert_not_called()

def test_stop_when_system_manager_exists(mock_logging, mock_system_manager_start_stop):
    mock_start, mock_stop = mock_system_manager_start_stop
    # Giả lập rằng _system_manager_instance đã được khởi tạo
    with patch('mining_environment.scripts.system_manager._system_manager_instance') as mock_instance:
        # Đặt mock_instance trả về một instance mock
        mock_instance = MagicMock()
        # Gán _system_manager_instance để trả về mock_instance
        with patch('mining_environment.scripts.system_manager._system_manager_instance', mock_instance):
            stop()
            # Kiểm tra rằng stop được gọi trên SystemManager
            mock_instance.stop.assert_called_once()
            # Kiểm tra log thông báo dừng
            mock_logging.info.assert_any_call("Đang dừng SystemManager...")
            mock_logging.info.assert_any_call("SystemManager đã dừng thành công.")

def test_stop_when_system_manager_not_exists(mock_logging):
    with patch('mining_environment.scripts.system_manager._system_manager_instance', None):
        stop()
        # Kiểm tra log warning khi SystemManager chưa được khởi tạo
        mock_logging.warning.assert_called_with("SystemManager instance chưa được khởi tạo.")

# 4. Kiểm thử toàn bộ luồng start và stop với xử lý lỗi
def test_full_flow_with_keyboard_interrupt(mock_logging, mock_resource_manager, mock_anomaly_detector, mock_sleep):
    with patch('builtins.open', mock_open(read_data=json.dumps({"key": "value"}))), \
         patch.object(Path, 'exists', return_value=True), \
         patch('mining_environment.scripts.system_manager.SystemManager') as mock_sm, \
         patch('mining_environment.scripts.system_manager.sleep', side_effect=KeyboardInterrupt):
        
        instance = mock_sm.return_value
        instance.start.return_value = None
        instance.stop.return_value = None
        
        with pytest.raises(KeyboardInterrupt):
            start()
        
        # Kiểm tra start được gọi
        instance.start.assert_called_once()
        
        # Kiểm tra stop được gọi khi nhận KeyboardInterrupt
        instance.stop.assert_called_once()
        
        # Kiểm tra log thông báo dừng
        mock_logging.info.assert_any_call("Nhận tín hiệu dừng từ người dùng. Đang dừng SystemManager...")

def test_full_flow_with_unexpected_error(mock_logging, mock_resource_manager, mock_anomaly_detector, mock_sleep, mock_sys_exit):
    with patch('builtins.open', mock_open(read_data=json.dumps({"key": "value"}))), \
         patch.object(Path, 'exists', return_value=True), \
         patch('mining_environment.scripts.system_manager.SystemManager') as mock_sm, \
         patch('mining_environment.scripts.system_manager.sleep', side_effect=Exception("Unexpected error")), \
         patch('mining_environment.scripts.system_manager.sys.exit') as mock_exit:
        
        instance = mock_sm.return_value
        instance.start.side_effect = None
        instance.stop.side_effect = None
        
        with pytest.raises(Exception):
            start()
        
        # Kiểm tra stop được gọi khi có lỗi không mong muốn
        instance.stop.assert_called_once()
        
        # Kiểm tra log lỗi
        mock_logging.error.assert_called_with("Lỗi không mong muốn trong SystemManager: Unexpected error")
        
        # Kiểm tra sys.exit được gọi với code 1
        mock_exit.assert_called_once_with(1)

# 5. Tạo báo cáo kiểm thử
# Để tạo báo cáo kiểm thử, bạn có thể sử dụng các plugin của pytest như pytest-html hoặc pytest-xml.
# Ví dụ, để tạo báo cáo HTML, bạn có thể chạy pytest với tham số --html:
# pytest tests/test_system_manager.py --verbose --capture=tee-sys -vv --html=report.html --self-contained-html
