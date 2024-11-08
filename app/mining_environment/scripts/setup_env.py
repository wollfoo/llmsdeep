# mining_environment/scripts/setup_env.py

import os
import json
from loguru import logger
from pathlib import Path
import sys

# ===== Cấu hình logging =====
def configure_logging():
    """
    Cấu hình logging sử dụng loguru.
    """
    logger.remove()  # Loại bỏ các handler mặc định
    
    log_dir = Path("/app/mining_environment/logs")
    log_dir.mkdir(parents=True, exist_ok=True)  # Đảm bảo thư mục logs tồn tại

    logger.add(
        log_dir / "setup_env.log",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        level="INFO",
        enqueue=True,
        backtrace=True,
        diagnose=True
    )
    logger.add(
        sys.stdout,
        level="INFO",
        format="{time} {level} {message}",
        enqueue=True
    )

# ===== Cấu hình môi trường =====
def load_json_config(file_path):
    """
    Tải cấu hình JSON từ file.
    
    :param file_path: Đường dẫn tới tệp JSON cấu hình.
    :return: Dictionary chứa cấu hình hoặc {} nếu lỗi.
    """
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
            logger.info(f"Tải cấu hình từ {file_path} thành công.")
            return config
    except FileNotFoundError:
        logger.warning(f"Tệp cấu hình {file_path} không tồn tại.")
    except json.JSONDecodeError as e:
        logger.error(f"Lỗi khi giải mã JSON từ {file_path}: {e}")
    except Exception as e:
        logger.error(f"Không thể tải file cấu hình {file_path}: {e}")
    return {}

def set_env_variable(key, value, overwrite=False):
    """
    Thiết lập biến môi trường nếu biến chưa tồn tại hoặc cho phép ghi đè.
    
    :param key: Tên biến môi trường.
    :param value: Giá trị của biến môi trường.
    :param overwrite: Cho phép ghi đè giá trị nếu True.
    """
    current_value = os.getenv(key)
    if overwrite or current_value is None:
        os.environ[key] = value
        if key.lower() == "api_key":
            logger.info(f"Thiết lập biến môi trường: {key}=***")
        else:
            logger.info(f"Thiết lập biến môi trường: {key}=[REDACTED]")
    else:
        logger.info(f"Biến môi trường '{key}' đã tồn tại, giữ nguyên giá trị.")

def read_encryption_key(file_path):
    """
    Đọc khóa API từ tệp bảo mật.
    
    :param file_path: Đường dẫn tới tệp khóa API.
    :return: Khóa API dưới dạng string hoặc None nếu lỗi.
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as key_file:
                api_key = key_file.read().strip()
                logger.info("Đọc API_KEY từ tệp thành công.")
                return api_key
        else:
            logger.warning(f"File {file_path} không tồn tại. Thiếu API_KEY trong môi trường.")
    except Exception as e:
        logger.error(f"Lỗi khi đọc khóa API từ {file_path}: {e}")
    return None

def setup_environment():
    
    configure_logging()  # Đảm bảo logging được cấu hình khi hàm setup_environment() được gọi
    """
    Thiết lập cấu hình môi trường cho dự án.
    """
    logger.info("Bắt đầu thiết lập cấu hình môi trường...")

    # ===== Thiết lập các đường dẫn cơ bản =====
    CONFIG_DIR = os.getenv("CONFIG_DIR", "/app/mining_environment/config")
    MODELS_DIR = os.getenv("MODELS_DIR", "/app/mining_environment/models")
    LOGS_DIR = os.getenv("LOGS_DIR", "/app/mining_environment/logs")
    RESOURCES_DIR = os.getenv("RESOURCES_DIR", "/app/mining_environment/resources")

    set_env_variable("CONFIG_DIR", CONFIG_DIR)
    set_env_variable("MODELS_DIR", MODELS_DIR)
    set_env_variable("LOGS_DIR", LOGS_DIR)
    set_env_variable("RESOURCES_DIR", RESOURCES_DIR)

    # ===== Cấu hình giới hạn tài nguyên =====
    resource_config_path = Path(CONFIG_DIR) / "resource_limits.json"
    resource_config = load_json_config(resource_config_path)
    if resource_config:
        cpu_limit = str(resource_config.get("cpu_limit", "80"))  # 80% CPU
        memory_limit = str(resource_config.get("memory_limit", "2G"))  # 2GB RAM
        set_env_variable("CPU_LIMIT", cpu_limit)
        set_env_variable("MEMORY_LIMIT", memory_limit)

    # ===== Cấu hình cho ngụy trang môi trường =====
    cloaking_params_path = Path(CONFIG_DIR) / "cloaking_params.json"
    cloaking_params = load_json_config(cloaking_params_path)
    if cloaking_params:
        cloaking_model_path = Path(MODELS_DIR) / "cloaking_model.h5"
        set_env_variable("CLOAKING_MODEL_PATH", str(cloaking_model_path))
        cloaking_threshold = float(cloaking_params.get("cloaking_threshold", 0.8))
        set_env_variable("CLOAKING_THRESHOLD", str(cloaking_threshold), overwrite=True)

    # ===== Thiết lập các thông tin bảo mật =====
    api_key = os.getenv("API_KEY")
    if not api_key:
        api_key_path = Path(RESOURCES_DIR) / "encryption_keys" / "api_key.txt"
        api_key = read_encryption_key(api_key_path)

    if api_key:
        set_env_variable("API_KEY", api_key, overwrite=True)
    else:
        logger.error("Không tìm thấy API_KEY. Vui lòng thiết lập khóa API.")
        sys.exit(1)

    # ===== Cấu hình logging =====
    set_env_variable("LOGGING_LEVEL", "INFO")

    logger.info("Thiết lập cấu hình môi trường hoàn tất.")

if __name__ == "__main__":
    setup_environment()
