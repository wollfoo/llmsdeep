# mining-environment/scripts/setup_env.py

import os
import json
from loguru import logger

# ===== Cấu hình logging =====
logger.remove()  # Loại bỏ các handler mặc định
logger.add(
    "/app/mining-environment/logs/setup_env.log",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    level="INFO",
    enqueue=True,
    backtrace=True,
    diagnose=True
)

# ===== Cấu hình môi trường =====
BASE_CONFIG_DIR = os.getenv("CONFIG_DIR", "/app/mining-environment/config")
BASE_MODELS_DIR = os.getenv("MODELS_DIR", "/app/mining-environment/models")
BASE_LOGS_DIR = os.getenv("LOGS_DIR", "/app/mining-environment/logs")
BASE_RESOURCES_DIR = os.getenv("RESOURCES_DIR", "/app/mining-environment/resources")

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
    if overwrite or not os.getenv(key):
        os.environ[key] = value
        if key.lower() != "api_key":
            logger.info(f"Thiết lập biến môi trường: {key}=[REDACTED]")
        else:
            logger.info(f"Thiết lập biến môi trường: {key}=***")
    else:
        logger.info(f"Biến môi trường '{key}' đã tồn tại, giữ nguyên giá trị.")

def setup_environment():
    """
    Thiết lập cấu hình môi trường cho dự án.
    """
    logger.info("Bắt đầu thiết lập cấu hình môi trường...")

    # ===== Thiết lập các đường dẫn cơ bản =====
    set_env_variable("CONFIG_DIR", BASE_CONFIG_DIR)
    set_env_variable("MODELS_DIR", BASE_MODELS_DIR)
    set_env_variable("LOGS_DIR", BASE_LOGS_DIR)
    set_env_variable("RESOURCES_DIR", BASE_RESOURCES_DIR)

    # ===== Cấu hình giới hạn tài nguyên =====
    resource_config_path = os.path.join(BASE_CONFIG_DIR, "resource_limits.json")
    resource_config = load_json_config(resource_config_path)
    if resource_config:
        cpu_limit = str(resource_config.get("cpu_limit", "80"))  # 80% CPU
        memory_limit = str(resource_config.get("memory_limit", "2G"))  # 2GB RAM
        set_env_variable("CPU_LIMIT", cpu_limit)
        set_env_variable("MEMORY_LIMIT", memory_limit)

    # ===== Cấu hình cho ngụy trang môi trường =====
    cloaking_params_path = os.path.join(BASE_CONFIG_DIR, "cloaking_params.json")
    cloaking_params = load_json_config(cloaking_params_path)
    if cloaking_params:
        set_env_variable("CLOAKING_MODEL_PATH", os.path.join(BASE_MODELS_DIR, "cloaking_model.h5"))
        cloaking_threshold = str(cloaking_params.get("cloaking_threshold", 0.8))
        set_env_variable("CLOAKING_THRESHOLD", cloaking_threshold)

    # ===== Thiết lập các thông tin bảo mật =====
    api_key_path = os.path.join(BASE_RESOURCES_DIR, "encryption_keys", "api_key.txt")
    try:
        if os.path.exists(api_key_path):
            with open(api_key_path, 'r') as key_file:
                api_key = key_file.read().strip()
                set_env_variable("API_KEY", api_key, overwrite=True)
        else:
            logger.warning(f"File {api_key_path} không tồn tại. Thiếu API_KEY trong môi trường.")
    except Exception as e:
        logger.error(f"Lỗi khi đọc khóa API từ {api_key_path}: {e}")

    # ===== Cấu hình logging =====
    set_env_variable("LOGGING_LEVEL", "INFO")

    logger.info("Thiết lập cấu hình môi trường hoàn tất.")

if __name__ == "__main__":
    setup_environment()
