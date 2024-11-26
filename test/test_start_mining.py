import pytest
from unittest.mock import patch, MagicMock
import threading
import signal
from start_mining import (
    initialize_environment,
    start_system_manager,
    start_mining_process,
    is_mining_process_running,
    signal_handler,
    main,
    stop_event,
)

# Kiểm thử initialize_environment
@patch("setup_env.setup")
def test_initialize_environment(mock_setup, caplog):
    initialize_environment()
    mock_setup.assert_called_once()
    assert "Bắt đầu thiết lập môi trường khai thác." in caplog.text
    assert "Thiết lập môi trường thành công." in caplog.text

@patch("setup_env.setup", side_effect=Exception("Setup failed"))
def test_initialize_environment_failure(mock_setup, caplog):
    with pytest.raises(SystemExit):
        initialize_environment()
    assert "Lỗi khi thiết lập môi trường: Setup failed" in caplog.text

# Kiểm thử start_system_manager
@patch("system_manager.start")
def test_start_system_manager(mock_start, caplog):
    start_system_manager()
    mock_start.assert_called_once()
    assert "Khởi động Resource Manager." in caplog.text
    assert "Resource Manager đã được khởi động." in caplog.text

@patch("system_manager.start", side_effect=Exception("Manager failed"))
def test_start_system_manager_failure(mock_start, caplog):
    start_system_manager()
    assert "Lỗi khi khởi động Resource Manager: Manager failed" in caplog.text
    assert stop_event.is_set()  # Đảm bảo hệ thống dừng nếu lỗi

# Kiểm thử start_mining_process
@patch("subprocess.Popen")
@patch("os.getenv", side_effect=lambda x, default: default)
def test_start_mining_process(mock_getenv, mock_popen, caplog):
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_popen.return_value = mock_process

    process = start_mining_process(retries=1, delay=1)
    assert process is not None
    mock_popen.assert_called_once()
    assert "Quá trình khai thác đang chạy." in caplog.text

@patch("subprocess.Popen", side_effect=Exception("Start failed"))
@patch("os.getenv", side_effect=lambda x, default: default)
def test_start_mining_process_failure(mock_getenv, mock_popen, caplog):
    process = start_mining_process(retries=1, delay=1)
    assert process is None
    assert "Lỗi khi khởi động quá trình khai thác: Start failed" in caplog.text
    assert stop_event.is_set()

# Kiểm thử is_mining_process_running
def test_is_mining_process_running():
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    assert is_mining_process_running(mock_process)

    mock_process.poll.return_value = 1
    assert not is_mining_process_running(mock_process)

# Kiểm thử signal_handler
def test_signal_handler(caplog):
    signal_handler(signal.SIGINT, None)
    assert "Nhận tín hiệu dừng (2). Đang dừng hệ thống khai thác..." in caplog.text
    assert stop_event.is_set()

# Kiểm thử hàm main
@patch("start_mining.start_mining_process", return_value=MagicMock())
@patch("start_mining.initialize_environment")
@patch("start_mining.start_system_manager")
@patch("start_mining.stop_event.is_set", side_effect=[False, True])
def test_main(mock_stop_event, mock_start_manager, mock_init_env, mock_start_mining, caplog):
    with patch("threading.Thread.start"):
        main()
    mock_init_env.assert_called_once()
    mock_start_mining.assert_called_once()
    mock_start_manager.assert_called_once()
    assert "===== Bắt đầu hoạt động khai thác tiền điện tử =====" in caplog.text
    assert "===== Hoạt động khai thác tiền điện tử đã dừng thành công =====" in caplog.text
