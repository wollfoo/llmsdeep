# test_logging_config.py
import os
import sys
import pytest
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
from cryptography.fernet import Fernet

# Thêm SCRIPTS_DIR vào sys.path nếu chưa có
APP_DIR = Path("/home/llmss/llmsdeep/app")
SCRIPTS_DIR = APP_DIR / "mining_environment" / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from logging_config import setup_logging, ObfuscatedEncryptedFileHandler

@pytest.fixture
def temp_log_file(tmp_path):
    """Fixture tạo một tệp log tạm thời."""
    return tmp_path / "test_log.log"

@pytest.fixture
def mock_fernet():
    """Fixture tạo đối tượng Fernet mock."""
    mock_fernet_instance = MagicMock()

    # Khi encrypt được gọi với dữ liệu bytes, trả về dữ liệu bytes đã được mã hóa giả định
    def encrypt_side_effect(data):
        return b'encrypted_' + data

    # Khi decrypt được gọi với dữ liệu bytes, trả về dữ liệu bytes đã được giải mã giả định
    def decrypt_side_effect(data):
        if data.startswith(b'encrypted_'):
            return data[len(b'encrypted_'):]
        else:
            raise ValueError("Invalid encrypted data")

    mock_fernet_instance.encrypt.side_effect = encrypt_side_effect
    mock_fernet_instance.decrypt.side_effect = decrypt_side_effect

    return mock_fernet_instance

@pytest.fixture(autouse=True)
def reset_loggers():
    """Fixture tự động reset các logger sau mỗi bài kiểm thử."""
    yield
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    for logger_name in list(logging.Logger.manager.loggerDict):
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.propagate = True

@patch('random.choices', return_value=list('ABCDEFGH'))
def test_obfuscated_encrypted_file_handler_emit(mock_random_choices, temp_log_file, mock_fernet):
    """Kiểm tra phương thức emit của ObfuscatedEncryptedFileHandler."""
    handler = ObfuscatedEncryptedFileHandler(temp_log_file, mock_fernet)
    handler.setFormatter(logging.Formatter('%(message)s'))
    log_record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Test log",
        args=(),
        exc_info=None
    )

    handler.emit(log_record)
    handler.close()

    with open(temp_log_file, 'rb') as log_file:
        encrypted_log = log_file.read().strip()
        decrypted_log = mock_fernet.decrypt(encrypted_log).decode('utf-8')
        assert "Test log" in decrypted_log
        assert len(decrypted_log.split()) == 3  # "Test log" + random_suffix
        assert decrypted_log == "Test log ABCDEFGH"

def test_obfuscated_encrypted_file_handler_error_handling(temp_log_file, mock_fernet):
    """Kiểm tra xử lý lỗi khi emit log."""
    handler = ObfuscatedEncryptedFileHandler(temp_log_file, mock_fernet)
    handler.format = MagicMock(side_effect=Exception("Formatting error"))
    log_record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Test log",
        args=(),
        exc_info=None
    )

    # Kiểm tra không ném lỗi khi có lỗi định dạng
    handler.emit(log_record)
    handler.close()

    with open(temp_log_file, 'rb') as log_file:
        assert log_file.read() == b''  # Không ghi log nếu xảy ra lỗi

@patch("logging_config.Fernet")
def test_setup_logging_creates_encrypted_handler(mock_fernet_class, temp_log_file):
    """Kiểm tra setup_logging tạo handler mã hóa."""
    mock_key = Fernet.generate_key()
    mock_fernet_instance = MagicMock()
    mock_fernet_class.return_value = mock_fernet_instance

    # Thiết lập Fernet.generate_key() trả về mock_key
    mock_fernet_class.generate_key.return_value = mock_key

    with patch.dict(os.environ, {}, clear=True):  # Loại bỏ tất cả biến môi trường
        setup_logging("test_module", str(temp_log_file))
        assert any(isinstance(h, ObfuscatedEncryptedFileHandler) for h in logging.getLogger("test_module").handlers)

@patch("logging_config.Fernet")
def test_setup_logging_generates_new_key(mock_fernet_class, temp_log_file):
    """Kiểm tra setup_logging tạo khóa mới nếu không có khóa mã hóa."""
    real_key = Fernet.generate_key()
    mock_fernet_class.generate_key.return_value = real_key
    mock_fernet_instance = MagicMock()
    mock_fernet_class.return_value = mock_fernet_instance

    with patch.dict(os.environ, {}, clear=True):  # Loại bỏ tất cả biến môi trường
        setup_logging("test_module", str(temp_log_file))
        assert os.environ["LOG_ENCRYPTION_KEY"] == real_key.decode()

@patch("logging_config.Fernet")
def test_setup_logging_uses_existing_key(mock_fernet_class, temp_log_file):
    """Kiểm tra setup_logging sử dụng khóa mã hóa hiện có."""
    real_key = Fernet.generate_key().decode()
    with patch.dict(os.environ, {"LOG_ENCRYPTION_KEY": real_key}, clear=True):  # Loại bỏ biến TESTING
        setup_logging("test_module", str(temp_log_file))
        mock_fernet_class.assert_called_once_with(real_key.encode())

@patch("logging_config.Fernet")
def test_setup_logging_handles_invalid_key(mock_fernet_class, temp_log_file):
    """Kiểm tra xử lý lỗi khi khóa mã hóa không hợp lệ."""
    mock_fernet_class.side_effect = Exception("Invalid key")
    with patch.dict(os.environ, {"LOG_ENCRYPTION_KEY": "invalid_key"}, clear=True):
        logger = setup_logging("test_module", str(temp_log_file))
        assert logger is None

@patch.object(Path, 'mkdir')
def test_setup_logging_creates_log_directory(mock_mkdir, temp_log_file):
    """Kiểm tra setup_logging tạo thư mục log nếu chưa tồn tại."""
    with patch.dict(os.environ, {}, clear=True):  # Loại bỏ biến TESTING
        setup_logging("test_module", str(temp_log_file))
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

def test_setup_logging_in_test_env(temp_log_file):
    """Kiểm tra setup_logging không thêm handler trong môi trường kiểm thử."""
    with patch.dict(os.environ, {"TESTING": "1"}):  # Đặt biến TESTING
        logger = setup_logging("test_module", str(temp_log_file))
        assert not logger.handlers
