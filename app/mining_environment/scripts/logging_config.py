# logging_config.py

import logging
import os
import sys
import random
import string
from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path
from cryptography.fernet import Fernet

class ObfuscatedEncryptedFileHandler(logging.Handler):
    """
    Custom logging handler để mã hóa và làm rối các log trước khi ghi vào tệp.
    """
    def __init__(self, filename, fernet, level=logging.NOTSET):
        super().__init__(level)
        self.filename = filename
        self.fernet = fernet
        self.file = open(filename, 'ab')  # Ghi ở chế độ binary

    def emit(self, record):
        try:
            msg = self.format(record)
            # Thêm chuỗi ngẫu nhiên để làm rối
            random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            obfuscated_msg = f"{msg} {random_suffix}"
            # Mã hóa thông điệp
            encrypted_msg = self.fernet.encrypt(obfuscated_msg.encode('utf-8'))
            # Ghi vào tệp
            self.file.write(encrypted_msg + b'\n')
            self.file.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        self.file.close()
        super().close()

def setup_logging(module_name: str, log_file: str, log_level: str = 'INFO') -> Logger:
    """
    Thiết lập logging cho một module cụ thể với mã hóa và làm rối.

    Args:
        module_name (str): Tên của module để lấy logger.
        log_file (str): Đường dẫn tới tệp log của module.
        log_level (str): Mức độ logging (default: 'INFO').

    Returns:
        Logger: Đối tượng logger đã được cấu hình.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False  # Ngăn chặn việc logger truyền log lên các logger cha

    if not logger.handlers:
        # Tạo thư mục cho log nếu chưa tồn tại
        log_path = Path(log_file).parent
        log_path.mkdir(parents=True, exist_ok=True)

        # Lấy khóa mã hóa từ biến môi trường hoặc tạo mới
        encryption_key = os.getenv('LOG_ENCRYPTION_KEY')
        if not encryption_key:
            # Tạo khóa mới và lưu vào biến môi trường
            encryption_key = Fernet.generate_key().decode()
            os.environ['LOG_ENCRYPTION_KEY'] = encryption_key
            print(f"Đã tạo khóa mã hóa mới: {encryption_key} (Lưu lại để sử dụng tiếp)")

        try:
            fernet = Fernet(encryption_key.encode())
        except Exception as e:
            print(f"Lỗi khi tạo đối tượng Fernet với khóa mã hóa: {e}", file=sys.stderr)
            sys.exit(1)

        # Tạo ObfuscatedEncryptedFileHandler
        encrypted_handler = ObfuscatedEncryptedFileHandler(log_file, fernet)
        encrypted_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Tạo Formatter cho log
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        encrypted_handler.setFormatter(formatter)

        # Thêm Handler vào logger
        logger.addHandler(encrypted_handler)

        # Tùy chọn: Thêm StreamHandler để ghi log ra console (không mã hóa)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
