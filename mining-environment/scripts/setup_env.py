import os
import json
from loguru import logger

# Đường dẫn thư mục chính của cấu hình và tài nguyên
BASE_CONFIG_DIR = "/app/mining-environment/config"
BASE_MODELS_DIR = "/app/mining-environment/models"
BASE_LOGS_DIR = "/app/mining-environment/logs"
BASE_RESOURCES_DIR = "/app/mining-environment/resources"

# Cấu hình logging
logger.add(f"{BASE_LOGS_DIR}/setup_env.log", rotation="10 MB")

def load_json_config(file_path):
    """
    Tải cấu hình JSON từ file.
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        logger.error(f"Không thể tải file cấu hình {file_path}: {e}")
        return {}

def set_env_variable(key, value, overwrite=False):
    """
    Thiết lập biến môi trường nếu biến chưa tồn tại hoặc cho phép ghi đè.
    """
    if overwrite or not os.getenv(key):
        os.environ[key] = value
        logger.info(f"Thiết lập biến môi trường: {key}={value}")
    else:
        logger.info(f"Biến môi trường '{key}' đã tồn tại, giữ nguyên giá trị.")

def setup_environment():
    """
    Thiết lập cấu hình môi trường cho dự án, bao gồm:
    - Đường dẫn tệp, cấu hình giới hạn tài nguyên.
    - Khóa API và các biến môi trường bảo mật.
    """

    logger.info("Bắt đầu thiết lập cấu hình môi trường...")

    # ===== Thiết lập các đường dẫn cơ bản =====
    set_env_variable("CONFIG_DIR", BASE_CONFIG_DIR)
    set_env_variable("MODELS_DIR", BASE_MODELS_DIR)
    set_env_variable("LOGS_DIR", BASE_LOGS_DIR)
    set_env_variable("RESOURCES_DIR", BASE_RESOURCES_DIR)

    # ===== Cấu hình giới hạn tài nguyên =====
    resource_config = load_json_config(f"{BASE_CONFIG_DIR}/resource_limits.json")
    if resource_config:
        set_env_variable("CPU_LIMIT", str(resource_config.get("cpu_limit", "80")))  # 80% CPU
        set_env_variable("MEMORY_LIMIT", str(resource_config.get("memory_limit", "2G")))  # 2GB RAM

    # ===== Cấu hình cho ngụy trang môi trường =====
    cloaking_params = load_json_config(f"{BASE_CONFIG_DIR}/cloaking_params.json")
    if cloaking_params:
        set_env_variable("CLOAKING_MODEL_PATH", f"{BASE_MODELS_DIR}/cloaking_model.h5")
        set_env_variable("ANOMALY_DETECTION_MODEL_PATH", f"{BASE_MODELS_DIR}/anomaly_detection_model.h5")
        set_env_variable("CLOAKING_THRESHOLD", str(cloaking_params.get("cloaking_threshold", 0.8)))

    # ===== Thiết lập các thông tin bảo mật =====
    try:
        with open(f"{BASE_RESOURCES_DIR}/encryption_keys/api_key.txt", 'r') as key_file:
            api_key = key_file.read().strip()
            set_env_variable("API_KEY", api_key, overwrite=True)
    except FileNotFoundError:
        logger.warning("File api_key.txt không tồn tại. Thiếu API_KEY trong môi trường.")
    except Exception as e:
        logger.error(f"Lỗi khi đọc khóa API: {e}")

    # ===== Cấu hình logging =====
    set_env_variable("LOGGING_LEVEL", "INFO")
    
    logger.info("Thiết lập cấu hình môi trường hoàn tất.")

if __name__ == "__main__":
    setup_environment()
