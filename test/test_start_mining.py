# test/test_start_mining.py

import sys
import os
import pytest
from unittest.mock import patch, MagicMock
import signal
import logging
from pathlib import Path

# Thiết lập biến môi trường TESTING=1 trước khi import bất kỳ module nào
os.environ["TESTING"] = "1"

# Thêm thư mục `app` và `app/mining_environment/scripts` vào sys.path
APP_DIR = Path(__file__).resolve().parent.parent / "app"
SCRIPTS_DIR = APP_DIR / "mining_environment" / "scripts"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

print("Current sys.path:", sys.path)

# Kiểm thử initialize_environment
@patch("setup_env.setup")
def test_initialize_environment(mock_setup, caplog):
    # Đảm bảo caplog ghi nhận các log từ logger "start_mining"
    caplog.set_level(logging.INFO, logger="start_mining")

    # Import start_mining sau khi caplog đã được thiết lập
    import start_mining

    start_mining.initialize_environment()
    mock_setup.assert_called_once()
    assert "Bắt đầu thiết lập môi trường khai thác." in caplog.text
    assert "Thiết lập môi trường thành công." in caplog.text

@patch("setup_env.setup", side_effect=Exception("Setup failed"))
def test_initialize_environment_failure(mock_setup, caplog):
    caplog.set_level(logging.ERROR, logger="start_mining")

    import start_mining

    with pytest.raises(SystemExit):
        start_mining.initialize_environment()
    assert "Lỗi khi thiết lập môi trường: Setup failed" in caplog.text

# Kiểm thử start_system_manager
@patch("start_mining.system_manager.start", autospec=True)
@patch("start_mining.system_manager.stop", autospec=True)
def test_start_system_manager(mock_stop, mock_start, caplog):
    caplog.set_level(logging.INFO, logger="start_mining")

    import start_mining

    start_mining.start_system_manager()
    mock_start.assert_called_once()
    assert "Khởi động Resource Manager." in caplog.text
    assert "Resource Manager đã được khởi động." in caplog.text

@patch("start_mining.system_manager.start", side_effect=Exception("Manager failed"))
@patch("start_mining.system_manager.stop", autospec=True)
def test_start_system_manager_failure(mock_stop, mock_start, caplog):
    caplog.set_level(logging.ERROR, logger="start_mining")

    import start_mining

    start_mining.start_system_manager()
    assert "Lỗi khi khởi động Resource Manager: Manager failed" in caplog.text
    assert start_mining.stop_event.is_set()  # Đảm bảo hệ thống dừng nếu lỗi

# Kiểm thử start_mining_process
@patch("subprocess.Popen")
@patch("os.getenv", side_effect=lambda key, default=None: default if key != "PSUTIL_DEBUG" else None)
def test_start_mining_process(mock_getenv, mock_popen, caplog):
    caplog.set_level(logging.INFO, logger="start_mining")

    import start_mining

    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_popen.return_value = mock_process

    process = start_mining.start_mining_process(retries=1, delay=1)
    assert process is not None
    mock_popen.assert_called_once()
    assert "Quá trình khai thác đang chạy." in caplog.text

@patch("subprocess.Popen", side_effect=Exception("Start failed"))
@patch("os.getenv", side_effect=lambda key, default=None: default if key != "PSUTIL_DEBUG" else None)
def test_start_mining_process_failure(mock_getenv, mock_popen, caplog):
    caplog.set_level(logging.ERROR, logger="start_mining")

    import start_mining

    process = start_mining.start_mining_process(retries=1, delay=1)
    assert process is None
    assert "Lỗi khi khởi động quá trình khai thác: Start failed" in caplog.text
    assert start_mining.stop_event.is_set()

# Kiểm thử is_mining_process_running
def test_is_mining_process_running():
    import start_mining

    mock_process = MagicMock()
    mock_process.poll.return_value = None
    assert start_mining.is_mining_process_running(mock_process)

    mock_process.poll.return_value = 1
    assert not start_mining.is_mining_process_running(mock_process)

# Kiểm thử signal_handler
def test_signal_handler(caplog):
    caplog.set_level(logging.INFO, logger="start_mining")

    import start_mining

    start_mining.signal_handler(signal.SIGINT, None)
    assert "Nhận tín hiệu dừng (2). Đang dừng hệ thống khai thác..." in caplog.text
    assert start_mining.stop_event.is_set()

# Kiểm thử hàm main khi quá trình khai thác đang chạy
@patch("start_mining.system_manager.stop", autospec=True)
@patch("start_mining.system_manager.start", autospec=True)
def test_main_mining_running(mock_start, mock_stop, caplog):
    caplog.set_level(logging.INFO, logger="start_mining")

    with patch("start_mining.start_mining_process", return_value=MagicMock()) as mock_start_mining:
        with patch("start_mining.initialize_environment") as mock_init_env:
            # Thay đổi side_effect để cho phép vòng lặp chạy nhiều lần hơn
            with patch("start_mining.stop_event.is_set", side_effect=[False, False, True]) as mock_stop_event:
                import start_mining

                # Thiết lập mock_start_mining để poll trả về None (quá trình khai thác đang chạy)
                mining_process_mock = mock_start_mining.return_value
                mining_process_mock.poll.return_value = None

                # Sử dụng side_effect để gọi run của thread
                with patch("threading.Thread.start", side_effect=lambda thread: thread.run()):
                    start_mining.main()

                    # After main() completes, assert the expected calls
                    mock_init_env.assert_called_once()
                    mock_start.assert_called_once()
                    mock_start_mining.assert_called_once()
                    mock_stop.assert_called_once()

                    assert "===== Bắt đầu hoạt động khai thác tiền điện tử =====" in caplog.text
                    assert "===== Hoạt động khai thác tiền điện tử đã dừng thành công =====" in caplog.text

# Kiểm thử hàm main khi quá trình khai thác không thành công
@patch("start_mining.system_manager.stop", autospec=True)
@patch("start_mining.system_manager.start", autospec=True)
def test_main_mining_failure(mock_start, mock_stop, caplog):
    caplog.set_level(logging.INFO, logger="start_mining")

    with patch("start_mining.start_mining_process", return_value=MagicMock()) as mock_start_mining:
        with patch("start_mining.initialize_environment") as mock_init_env:
            with patch("start_mining.stop_event.is_set", side_effect=[False, True]) as mock_stop_event:
                import start_mining

                # Thiết lập mock_start_mining để poll trả về 1 (quá trình khai thác đã kết thúc không thành công)
                mining_process_mock = mock_start_mining.return_value
                mining_process_mock.poll.return_value = 1

                # Sử dụng side_effect để gọi run của thread
                with patch("threading.Thread.start", side_effect=lambda self: self.run()):
                    with pytest.raises(SystemExit) as exc_info:
                        start_mining.main()

                    assert exc_info.value.code == 1  # Kiểm tra mã thoát là 1
                    mock_init_env.assert_called_once()
                    mock_start.assert_not_called()
                    mock_start_mining.assert_called_once()
                    mock_stop.assert_called_once_with()

                    assert "===== Bắt đầu hoạt động khai thác tiền điện tử =====" in caplog.text
                    assert "Quá trình khai thác không khởi động thành công sau nhiều cố gắng. Dừng hệ thống khai thác." in caplog.text

# Kiểm thử đơn giản để xác minh caplog hoạt động
def test_caplog_capture(caplog):
    caplog.set_level(logging.INFO, logger="start_mining")
    import start_mining
    logger = logging.getLogger("start_mining")
    logger.info("Testing caplog capture.")
    assert "Testing caplog capture." in caplog.text
