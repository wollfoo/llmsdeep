# logging_config.py

import logging
import os
import sys
from logging import Logger
from pathlib import Path
from cryptography.fernet import Fernet
import random
import string

class ObfuscatedEncryptedFileHandler(logging.Handler):
    """
    Custom logging handler để mã hóa và làm rối các log trước khi ghi vào tệp.
    """
    def __init__(self, filename: str, fernet: Fernet, level=logging.NOTSET):
        super().__init__(level)
        self.filename = filename
        self.fernet = fernet
        # [CHANGES] Kiểm tra đường dẫn cha, tạo nếu chưa có
        file_parent = Path(filename).parent
        file_parent.mkdir(parents=True, exist_ok=True)
        # Mở file ở chế độ 'ab' (append-binary)
        self.file = open(filename, 'ab')

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            # Thêm chuỗi ngẫu nhiên để làm rối
            random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            obfuscated_msg = f"{msg} {random_suffix}"
            # Mã hóa thông điệp
            encrypted_msg = self.fernet.encrypt(obfuscated_msg.encode('utf-8'))
            # Ghi vào tệp (dạng nhị phân, thêm newline)
            self.file.write(encrypted_msg + b'\n')
            self.file.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        if not self.file.closed:
            self.file.close()
        super().close()


def setup_logging(module_name: str, log_file: str, log_level: str = 'INFO') -> Logger:
    """
    Thiết lập logger cho module, hỗ trợ mã hóa log bằng ObfuscatedEncryptedFileHandler.
    
    Args:
        module_name (str): Tên module (logger name).
        log_file (str): Đường dẫn file log.
        log_level (str): Mức log (DEBUG, INFO, WARN, ERROR...).
    
    Returns:
        Logger: Đối tượng logger đã được thiết lập.
    """
    logger = logging.getLogger(module_name)
    # [CHANGES] Lấy log_level bằng getattr an toàn
    safe_log_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(safe_log_level)
    
    # Kiểm tra xem có đang trong môi trường kiểm thử hay không
    in_test = "TESTING" in os.environ
    if in_test:
        logger.propagate = True
        print("Logger propagate set to True for testing mode.")
    else:
        # Không propagate nếu không phải test
        logger.propagate = False

    # Nếu logger chưa có handler nào, ta thêm
    if not logger.handlers:
        if in_test:
            print("Skip adding StreamHandler due to testing mode.")
            return logger

        # [CHANGES] Đảm bảo thư mục log tồn tại
        log_path = Path(log_file).parent
        log_path.mkdir(parents=True, exist_ok=True)

        # Lấy khóa mã hóa từ biến môi trường hoặc tự tạo
        encryption_key = os.getenv('LOG_ENCRYPTION_KEY')
        if not encryption_key:
            encryption_key = Fernet.generate_key().decode()
            os.environ['LOG_ENCRYPTION_KEY'] = encryption_key
            print(f"Đã tạo khóa mã hóa mới: {encryption_key} (hãy lưu lại để sử dụng tiếp).")

        try:
            fernet = Fernet(encryption_key.encode())
        except Exception as e:
            print(f"Lỗi khi tạo Fernet với khóa mã hóa: {e}", file=sys.stderr)
            return logger

        # Tạo handler mã hóa
        encrypted_handler = ObfuscatedEncryptedFileHandler(log_file, fernet)
        encrypted_handler.setLevel(safe_log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        encrypted_handler.setFormatter(formatter)
        logger.addHandler(encrypted_handler)

        # Thêm StreamHandler (log ra console) nếu không phải testing
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(safe_log_level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
