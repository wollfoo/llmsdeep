# tests/test_system_manager.py

import os
import sys
import pytest
import json
from unittest.mock import patch, MagicMock, mock_open
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


# 1. Kiểm thử cho hàm load_config
def test_load_config_success(caplog):
    with patch('builtins.open', mock_open(read_data=json.dumps({"key": "value"}))):
        from mining_environment.scripts.system_manager import load_config

        config_path = Path("/fake/config.json")
        result = load_config(config_path)

        # Kiểm tra rằng hàm open đã được gọi đúng cách
        assert result == {"key": "value"}
        assert "Đã tải cấu hình từ /fake/config.json" in caplog.text


def test_load_config_file_not_found(caplog):
    with patch('builtins.open', side_effect=FileNotFoundError):
        from mining_environment.scripts.system_manager import load_config

        config_path = Path("/fake/missing_config.json")
        with pytest.raises(SystemExit) as exc_info:
            load_config(config_path)
        assert exc_info.value.code == 1
        assert f"Tệp cấu hình không tìm thấy: {config_path}" in caplog.text


def test_load_config_json_error(caplog):
    with patch('builtins.open', mock_open(read_data="Invalid JSON")):
        from mining_environment.scripts.system_manager import load_config

        config_path = Path("/fake/bad_config.json")
        with pytest.raises(SystemExit) as exc_info:
            load_config(config_path)
        assert exc_info.value.code == 1
        assert f"Lỗi cú pháp JSON trong tệp cấu hình {config_path}: Expecting value: line 1 column 1 (char 0)" in caplog.text


# 2. Kiểm thử cho lớp SystemManager
def test_system_manager_init(caplog):
    with patch('mining_environment.scripts.system_manager.ResourceManager') as mock_resource_manager, \
         patch('mining_environment.scripts.system_manager.AnomalyDetector') as mock_anomaly_detector:
        
        from mining_environment.scripts.system_manager import SystemManager, MODELS_DIR

        config = {"config_key": "config_value"}
        system_manager = SystemManager(config)

        # Kiểm tra lưu cấu hình
        assert system_manager.config == config

        # Kiểm tra ResourceManager và AnomalyDetector được khởi tạo với đúng tham số
        mock_resource_manager.assert_called_once_with(
            config,
            MODELS_DIR / "resource_optimization_model.pt",
            system_manager.resource_logger
        )
        mock_anomaly_detector.assert_called_once_with(
            config,
            MODELS_DIR / "anomaly_cloaking_model.pt",
            system_manager.anomaly_logger
        )

        # Kiểm tra set_resource_manager được gọi
        system_manager.anomaly_detector.set_resource_manager.assert_called_once_with(system_manager.resource_manager)

        # Kiểm tra log thông báo khởi tạo thành công
        assert "SystemManager đã được khởi tạo thành công." in caplog.text


def test_system_manager_start_success(caplog):
    with patch('mining_environment.scripts.system_manager.ResourceManager') as mock_resource_manager, \
         patch('mining_environment.scripts.system_manager.AnomalyDetector') as mock_anomaly_detector:
        
        from mining_environment.scripts.system_manager import SystemManager

        config = {"config_key": "config_value"}
        system_manager = SystemManager(config)

        with patch.object(system_manager.resource_manager, 'start') as mock_rm_start, \
             patch.object(system_manager.anomaly_detector, 'start') as mock_ad_start:
            system_manager.start()
            mock_rm_start.assert_called_once()
            mock_ad_start.assert_called_once()
            assert "Đang khởi động SystemManager..." in caplog.text
            assert "SystemManager đã khởi động thành công." in caplog.text


def test_system_manager_start_failure(caplog):
    with patch('mining_environment.scripts.system_manager.ResourceManager') as mock_resource_manager, \
         patch('mining_environment.scripts.system_manager.AnomalyDetector') as mock_anomaly_detector:
        
        from mining_environment.scripts.system_manager import SystemManager

        config = {"config_key": "config_value"}
        system_manager = SystemManager(config)

        # Giả lập lỗi khi khởi động ResourceManager
        with patch.object(system_manager.resource_manager, 'start', side_effect=Exception("Start error")), \
             patch.object(system_manager, 'stop') as mock_stop:
            
            with pytest.raises(Exception) as exc_info:
                system_manager.start()

            assert str(exc_info.value) == "Start error"
            mock_stop.assert_called_once()
            assert "Lỗi khi khởi động SystemManager: Start error" in caplog.text


def test_system_manager_stop_success(caplog):
    with patch('mining_environment.scripts.system_manager.ResourceManager') as mock_resource_manager, \
         patch('mining_environment.scripts.system_manager.AnomalyDetector') as mock_anomaly_detector:
        
        from mining_environment.scripts.system_manager import SystemManager

        config = {"config_key": "config_value"}
        system_manager = SystemManager(config)

        with patch.object(system_manager.resource_manager, 'stop') as mock_rm_stop, \
             patch.object(system_manager.anomaly_detector, 'stop') as mock_ad_stop:
            system_manager.stop()
            mock_rm_stop.assert_called_once()
            mock_ad_stop.assert_called_once()
            assert "Đang dừng SystemManager..." in caplog.text
            assert "SystemManager đã dừng thành công." in caplog.text


def test_system_manager_stop_failure(caplog):
    with patch('mining_environment.scripts.system_manager.ResourceManager') as mock_resource_manager, \
         patch('mining_environment.scripts.system_manager.AnomalyDetector') as mock_anomaly_detector:
        
        from mining_environment.scripts.system_manager import SystemManager

        config = {"config_key": "config_value"}
        system_manager = SystemManager(config)

        # Giả lập lỗi khi dừng AnomalyDetector
        with patch.object(system_manager.anomaly_detector, 'stop', side_effect=Exception("Stop error")):
            with pytest.raises(Exception) as exc_info:
                system_manager.stop()

            assert str(exc_info.value) == "Stop error"
            assert "Lỗi khi dừng SystemManager: Stop error" in caplog.text


# 3. Kiểm thử cho hàm start và stop toàn hệ thống
def test_start_success(caplog):
    with patch('builtins.open', mock_open(read_data=json.dumps({"key": "value"}))), \
         patch.object(Path, 'exists', return_value=True), \
         patch('mining_environment.scripts.system_manager.SystemManager') as mock_sm, \
         patch('mining_environment.scripts.system_manager.sleep', side_effect=KeyboardInterrupt):
        
        from mining_environment.scripts.system_manager import start, SystemManager, CONFIG_DIR, MODELS_DIR

        instance = mock_sm.return_value
        instance.start.return_value = None
        instance.stop.return_value = None

        # Không sử dụng pytest.raises vì KeyboardInterrupt được xử lý trong code
        start()

        # Kiểm tra load_config được gọi cho từng tệp cấu hình
        mock_sm.assert_called_once_with({"key": "value", "key": "value"})

        # Kiểm tra start được gọi trên SystemManager
        instance.start.assert_called_once()

        # Kiểm tra stop được gọi khi nhận KeyboardInterrupt
        instance.stop.assert_called_once()

        # Kiểm tra log thông báo hệ thống đang chạy và dừng
        assert "SystemManager đang chạy. Nhấn Ctrl+C để dừng." in caplog.text
        assert "Nhận tín hiệu dừng từ người dùng. Đang dừng SystemManager..." in caplog.text



def test_start_missing_ai_model(caplog):
    from mining_environment.scripts.system_manager import start, RESOURCE_OPTIMIZATION_MODEL_PATH

    # Tạo một mock Path object với spec là Path để đảm bảo tính tương thích
    mock_path = MagicMock(spec=Path)
    # Đặt phương thức exists() trả về False
    mock_path.exists.return_value = False
    # Đặt phương thức __str__ của mock_path trả về chuỗi đường dẫn gốc để log đúng thông tin
    mock_path.__str__.return_value = str(RESOURCE_OPTIMIZATION_MODEL_PATH)

    with patch('builtins.open', mock_open(read_data=json.dumps({"key": "value"}))):
        # Patch đối tượng RESOURCE_OPTIMIZATION_MODEL_PATH bằng mock_path
        with patch('mining_environment.scripts.system_manager.RESOURCE_OPTIMIZATION_MODEL_PATH', mock_path):
            # Patch SystemManager để kiểm tra nó không được gọi khi mô hình AI không tồn tại
            with patch('mining_environment.scripts.system_manager.SystemManager') as mock_sm:
                # Kiểm tra rằng start() sẽ gọi SystemExit khi mô hình AI không tồn tại
                with pytest.raises(SystemExit) as exc_info:
                    start()

                # Kiểm tra rằng mã thoát là 1
                assert exc_info.value.code == 1

                # Kiểm tra rằng log lỗi chứa thông tin về mô hình AI không tìm thấy
                assert f"Mô hình AI không tìm thấy tại: {RESOURCE_OPTIMIZATION_MODEL_PATH}" in caplog.text

                # Kiểm tra rằng SystemManager không được khởi tạo
                mock_sm.assert_not_called()

def test_start_load_config_failure(caplog):
    with patch('builtins.open', side_effect=FileNotFoundError):
        from mining_environment.scripts.system_manager import start, CONFIG_DIR

        with pytest.raises(SystemExit) as exc_info:
            start()
        assert exc_info.value.code == 1
        # Kiểm tra log lỗi được ghi bởi load_config
        assert f"Tệp cấu hình không tìm thấy: {CONFIG_DIR / 'resource_config.json'}" in caplog.text
        # Không cần kiểm tra tệp thứ hai vì chương trình dừng sau lỗi đầu tiên
        assert f"Tệp cấu hình không tìm thấy: {CONFIG_DIR / 'process_config.json'}" not in caplog.text
        # Kiểm tra rằng SystemManager không được khởi tạo
        with patch('mining_environment.scripts.system_manager.SystemManager') as mock_sm:
            mock_sm.assert_not_called()


def test_stop_when_system_manager_exists(caplog):
    with patch('mining_environment.scripts.system_manager._system_manager_instance') as mock_instance:
        mock_instance = MagicMock()
        with patch('mining_environment.scripts.system_manager._system_manager_instance', mock_instance):
            from mining_environment.scripts.system_manager import stop
            stop()
            # Kiểm tra rằng stop được gọi trên SystemManager
            mock_instance.stop.assert_called_once()
            # Kiểm tra log thông báo dừng
            assert "Đang dừng SystemManager..." in caplog.text
            assert "SystemManager đã dừng thành công." in caplog.text


def test_stop_when_system_manager_not_exists(caplog):
    with patch('mining_environment.scripts.system_manager._system_manager_instance', None):
        from mining_environment.scripts.system_manager import stop
        stop()
        # Kiểm tra log warning khi SystemManager chưa được khởi tạo
        assert "SystemManager instance chưa được khởi tạo." in caplog.text


# 4. Kiểm thử toàn bộ luồng start và stop với xử lý lỗi
def test_full_flow_with_keyboard_interrupt(caplog):
    with patch('builtins.open', mock_open(read_data=json.dumps({"key": "value"}))), \
         patch.object(Path, 'exists', return_value=True), \
         patch('mining_environment.scripts.system_manager.SystemManager') as mock_sm, \
         patch('mining_environment.scripts.system_manager.sleep', side_effect=KeyboardInterrupt):
        
        from mining_environment.scripts.system_manager import start, SystemManager, CONFIG_DIR, MODELS_DIR

        instance = mock_sm.return_value
        instance.start.return_value = None
        instance.stop.return_value = None

        # Không sử dụng pytest.raises vì KeyboardInterrupt được xử lý trong code
        start()

        # Kiểm tra load_config được gọi cho từng tệp cấu hình
        mock_sm.assert_called_once_with({"key": "value", "key": "value"})

        # Kiểm tra start được gọi trên SystemManager
        instance.start.assert_called_once()

        # Kiểm tra stop được gọi khi nhận KeyboardInterrupt
        instance.stop.assert_called_once()

        # Kiểm tra log thông báo hệ thống đang chạy và dừng
        assert "SystemManager đang chạy. Nhấn Ctrl+C để dừng." in caplog.text
        assert "Nhận tín hiệu dừng từ người dùng. Đang dừng SystemManager..." in caplog.text

def test_full_flow_with_unexpected_error(caplog):
    with patch('builtins.open', mock_open(read_data=json.dumps({"key": "value"}))), \
         patch.object(Path, 'exists', return_value=True), \
         patch('mining_environment.scripts.system_manager.SystemManager') as mock_sm, \
         patch('mining_environment.scripts.system_manager.sleep', side_effect=Exception("Unexpected error")), \
         patch('mining_environment.scripts.system_manager.sys.exit', side_effect=SystemExit(1)) as mock_exit:
        
        from mining_environment.scripts.system_manager import start, SystemManager, CONFIG_DIR

        instance = mock_sm.return_value
        instance.start.side_effect = None
        instance.stop.side_effect = None

        with pytest.raises(SystemExit) as exc_info:
            start()

        # Kiểm tra rằng sys.exit được gọi với mã 1
        mock_exit.assert_called_once_with(1)

        # Kiểm tra stop được gọi khi có lỗi không mong muốn
        instance.stop.assert_called_once()

        # Kiểm tra log lỗi
        assert "Lỗi không mong muốn trong SystemManager: Unexpected error" in caplog.text
