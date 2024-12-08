# test_logging_config.py
import os
import sys
import pytest
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
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
    mock_key = Fernet.generate_key()
    return Fernet(mock_key)


@patch("logging_config.Fernet")
@patch("sys.exit")  # Mô phỏng sys.exit
def test_setup_logging_handles_invalid_key(mock_exit, mock_fernet, temp_log_file):
    """Kiểm tra xử lý lỗi khi khóa mã hóa không hợp lệ."""
    mock_fernet.side_effect = Exception("Invalid key")
    with patch.dict(os.environ, {"LOG_ENCRYPTION_KEY": "invalid_key"}, clear=True):
        setup_logging("test_module", str(temp_log_file))
        mock_exit.assert_called_once_with(1)  # Kiểm tra sys.exit được gọi với mã 1
