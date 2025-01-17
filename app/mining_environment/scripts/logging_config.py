# logging_config.py


import os
import sys
import logging
from logging import Logger
from pathlib import Path
from cryptography.fernet import Fernet
import random
import string
from contextvars import ContextVar
from typing import Optional



###############################################################################
#                           ĐỊNH NGHĨA CORRELATION ID                        #
###############################################################################

# Định nghĩa một ContextVar để lưu trữ Correlation ID cho mỗi ngữ cảnh.
correlation_id: ContextVar[str] = ContextVar('correlation_id', default='unknown')

###############################################################################
#                           CLASS: CorrelationIdFilter                       #
###############################################################################

class CorrelationIdFilter(logging.Filter):
    """
    Bộ lọc logging để thêm Correlation ID vào mỗi bản ghi log.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Thêm Correlation ID vào bản ghi log.
        
        Args:
            record (logging.LogRecord): Bản ghi log hiện tại.
        
        Returns:
            bool: Luôn trả về True để cho phép bản ghi log được xử lý.
        """
        record.correlation_id = correlation_id.get()
        return True

###############################################################################
#                           CLASS: ObfuscatedEncryptedFileHandler          #
###############################################################################

class ObfuscatedEncryptedFileHandler(logging.Handler):
    """
    Custom logging handler để mã hóa và làm rối các log trước khi ghi vào tệp.
    """
    def __init__(self, filename: str, fernet: Fernet, level=logging.NOTSET):
        """
        Khởi tạo ObfuscatedEncryptedFileHandler.
        
        Args:
            filename (str): Đường dẫn đến tệp log.
            fernet (Fernet): Đối tượng Fernet để mã hóa log.
            level (int, optional): Mức độ log để xử lý. Mặc định là NOTSET.
        """
        super().__init__(level)
        self.filename = filename
        self.fernet = fernet
        # Kiểm tra và tạo thư mục cha nếu chưa tồn tại
        file_parent = Path(filename).parent
        file_parent.mkdir(parents=True, exist_ok=True)
        # Mở file ở chế độ 'ab' (append-binary)
        self.file = open(filename, 'ab')

    def emit(self, record: logging.LogRecord):
        """
        Xử lý và ghi bản ghi log vào tệp sau khi mã hóa và làm rối.
        
        Args:
            record (logging.LogRecord): Bản ghi log cần xử lý.
        """
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
        """
        Đóng tệp log khi handler được đóng.
        """
        if not self.file.closed:
            self.file.close()
        super().close()

###############################################################################
#                           FUNCTION: setup_logging                         #
###############################################################################

# def setup_logging(module_name: str, log_file: str, log_level: str = 'INFO', **kwargs) -> Logger:
#     """
#     Thiết lập logger cho module, hỗ trợ mã hóa log bằng ObfuscatedEncryptedFileHandler
#     và thêm Correlation ID vào mỗi bản ghi log.
    
#     Args:
#         module_name (str): Tên module (tên logger).
#         log_file (str): Đường dẫn đến tệp log.
#         log_level (str, optional): Mức log (DEBUG, INFO, WARN, ERROR...). Mặc định là 'INFO'.
    
#     Returns:
#         Logger: Đối tượng logger đã được thiết lập.
#     """
#     logger = logging.getLogger(module_name)
#     # Lấy log_level an toàn bằng getattr
#     safe_log_level = getattr(logging, log_level.upper(), logging.INFO)
#     logger.setLevel(safe_log_level)
    
#     # Kiểm tra xem có đang trong môi trường kiểm thử hay không
#     in_test = "TESTING" in os.environ
#     if in_test:
#         logger.propagate = True
#         print("Logger propagate set to True for testing mode.")
#     else:
#         # Không propagate nếu không phải test
#         logger.propagate = False

#     # Nếu logger chưa có handler nào, ta thêm
#     if not logger.handlers:
#         if in_test:
#             print("Skip adding StreamHandler due to testing mode.")
#             return logger

#         # Đảm bảo thư mục log tồn tại
#         log_path = Path(log_file).parent
#         log_path.mkdir(parents=True, exist_ok=True)

#         # Lấy khóa mã hóa từ biến môi trường hoặc tự tạo
#         encryption_key = os.getenv('LOG_ENCRYPTION_KEY')
#         if not encryption_key:
#             encryption_key = Fernet.generate_key().decode()
#             os.environ['LOG_ENCRYPTION_KEY'] = encryption_key
#             print(f"Đã tạo khóa mã hóa mới: {encryption_key} (hãy lưu lại để sử dụng tiếp).")

#         try:
#             fernet = Fernet(encryption_key.encode())
#         except Exception as e:
#             print(f"Lỗi khi tạo Fernet với khóa mã hóa: {e}", file=sys.stderr)
#             return logger

#         # Tạo handler mã hóa
#         encrypted_handler = ObfuscatedEncryptedFileHandler(log_file, fernet)
#         encrypted_handler.setLevel(safe_log_level)
#         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(message)s')
#         encrypted_handler.setFormatter(formatter)
#         # Thêm CorrelationIdFilter vào handler
#         encrypted_handler.addFilter(CorrelationIdFilter())
#         logger.addHandler(encrypted_handler)

#         # Thêm StreamHandler (log ra console) nếu không phải testing
#         stream_handler = logging.StreamHandler(sys.stdout)
#         stream_handler.setLevel(safe_log_level)
#         stream_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(message)s')
#         stream_handler.setFormatter(stream_formatter)
#         # Thêm CorrelationIdFilter vào handler
#         stream_handler.addFilter(CorrelationIdFilter())
#         logger.addHandler(stream_handler)

#     return logger


def setup_logging(module_name: str, log_file: str, log_level: str = 'ERROR', **kwargs) -> Logger:
    """
    Thiết lập logger với mức log linh hoạt và chi tiết hơn.

    Args:
        module_name (str): Tên module (tên logger).
        log_file (str): Đường dẫn đến tệp log.
        log_level (str, optional): Mức log (DEBUG, INFO, WARN, ERROR...). Mặc định là 'ERROR'.
        **kwargs: Các tham số bổ sung (tùy chọn cho các handlers khác, chẳng hạn như log_format, log_rotation...).

    Returns:
        Logger: Đối tượng logger đã được thiết lập.
    """
    # Mức độ log từ chuỗi (log_level) chuyển sang giá trị của logging
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }

    # Nếu mức log không hợp lệ, mặc định là ERROR
    level = log_levels.get(log_level.upper(), logging.ERROR)

    # Thiết lập logger
    logger = logging.getLogger(module_name)
    logger.setLevel(level)

    # Kiểm tra nếu logger đã có handler để tránh duplicate
    if not logger.handlers:
        # Đảm bảo thư mục log tồn tại
        log_path = Path(log_file).parent
        log_path.mkdir(parents=True, exist_ok=True)

        # Lấy các tham số bổ sung từ kwargs (nếu có), ví dụ: log_format, log_rotation...
        log_format = kwargs.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        datefmt = kwargs.get('datefmt', '%Y-%m-%d %H:%M:%S')

        # Formatter chi tiết
        formatter = logging.Formatter(log_format, datefmt=datefmt)

        # File handler: Ghi log vào file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)  # Ghi log theo mức độ được chỉ định
        file_handler.setFormatter(formatter)
        file_handler.addFilter(CorrelationIdFilter())  # Nếu cần sử dụng CorrelationIdFilter
        logger.addHandler(file_handler)

        # Stream handler: Ghi log ra console
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.DEBUG)  # Console có thể ghi cả DEBUG (chi tiết hơn)
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(CorrelationIdFilter())
        logger.addHandler(stream_handler)

        # Có thể thêm các handler khác tại đây (ví dụ, ghi vào một service log ngoài như ELK, Sentry...)
        # Ví dụ về handler gửi log qua HTTP, Kafka, hoặc các hệ thống phân tán khác.

    return logger
