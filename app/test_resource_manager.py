import os
import threading
from logging import getLogger, StreamHandler, INFO
from mining_environment.scripts.resource_manager import ResourceManager
from mining_environment.scripts.utils import MiningProcess  # Đảm bảo rằng MiningProcess được định nghĩa và import đúng cách
import json

# Thiết lập biến môi trường CONFIG_DIR nếu chưa được thiết lập
if 'CONFIG_DIR' not in os.environ:
    os.environ['CONFIG_DIR'] = '/app/mining_environment/config'

# Khởi tạo logger cho việc kiểm tra
test_logger = getLogger("test_logger")
test_logger.setLevel(INFO)
test_logger.addHandler(StreamHandler())

# Đọc cấu hình từ file resource_config.json
CONFIG_DIR = Path(os.getenv('CONFIG_DIR', '/app/mining_environment/config'))
resource_config_path = CONFIG_DIR / "resource_config.json"



try:
    with open(resource_config_path, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Không tìm thấy file cấu hình: {resource_config_path}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Lỗi khi parse JSON: {e}")
    exit(1)

# Khởi tạo ResourceManager
try:
    rm = ResourceManager(config, test_logger)
    print("ResourceManager initialized successfully!")
except Exception as e:
    print(f"Error initializing ResourceManager: {e}")
    exit(1)  # Thoát chương trình nếu không khởi tạo được

# Nếu khởi tạo thành công, tiếp tục kiểm tra các hàm

# 1. Kiểm tra resource_adjustment_handler trong thread
def test_resource_adjustment_handler():
    try:
        print("Testing resource_adjustment_handler...")
        # Thêm một task mẫu vào queue để xử lý
        sample_task = {'function': 'adjust_cpu_threads', 'args': (1234, 2, 'ml-inference')}
        rm.resource_adjustment_queue.put((3, sample_task))
        rm.resource_adjustment_handler()  # Thực thi một lần, đảm bảo hàm có thể xử lý task
    except Exception as e:
        print(f"Error in resource_adjustment_handler: {e}")

handler_thread = threading.Thread(target=test_resource_adjustment_handler, daemon=True)
handler_thread.start()
handler_thread.join()  # Chờ thread hoàn thành

# 2. Kiểm tra optimize_resources trong thread
def test_optimize_resources():
    try:
        print("Testing optimize_resources...")
        # Thêm một tiến trình mẫu vào danh sách mining_processes
        sample_process = MiningProcess(pid=1234, name="ml-inference", priority=1, network_interface="eth0", logger=test_logger)
        rm.mining_processes.append(sample_process)
        rm.allocate_resources_with_priority()  # Chạy phân bổ tài nguyên
        rm.optimize_resources()  # Thực thi một lần
    except Exception as e:
        print(f"Error in optimize_resources: {e}")

optimize_thread = threading.Thread(target=test_optimize_resources, daemon=True)
optimize_thread.start()
optimize_thread.join()  # Chờ thread hoàn thành

# 3. Kiểm tra trạng thái các hàng đợi
try:
    print("Checking queue sizes...")
    print("Cloaking request queue size:", rm.cloaking_request_queue.qsize())
    print("Resource adjustment queue size:", rm.resource_adjustment_queue.qsize())
except Exception as e:
    print(f"Error checking queue sizes: {e}")

# 4. Kiểm tra discover_mining_processes
try:
    print("Testing discover_mining_processes...")
    rm.discover_mining_processes()
    print("Mining processes discovered successfully.")
except Exception as e:
    print(f"Error in discover_mining_processes: {e}")

# 5. Kiểm tra gather_metric_data_for_anomaly_detection
try:
    print("Testing gather_metric_data_for_anomaly_detection...")
    metrics = rm.gather_metric_data_for_anomaly_detection()
    print("Metric data gathered:", metrics)
except Exception as e:
    print(f"Error in gather_metric_data_for_anomaly_detection: {e}")

# 6. Kiểm tra allocate_resources_with_priority
try:
    print("Testing allocate_resources_with_priority...")
    rm.allocate_resources_with_priority()
    print("Resource allocation completed successfully.")
except Exception as e:
    print(f"Error in allocate_resources_with_priority: {e}")
