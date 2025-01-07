import os
import sys
import json
import logging
from threading import Semaphore, Lock, Thread, Event
from queue import PriorityQueue, Full
from ratelimiter import RateLimiter
import psutil
import time


class ThreadingManager:
    """
    Quản lý threading nâng cao:
    - Giới hạn số luồng đồng thời (Semaphore).
    - Giới hạn tần suất thực thi (RateLimiter).
    - Phân phối công việc thông minh (Priority Queue).
    - Điều chỉnh tài nguyên động (Dynamic Resource Adjustment).
    - Xử lý khi hệ thống bão hòa.
    """

    def __init__(self, max_cpu_threads: int, max_gpu_threads: int, 
                 cpu_rate_limit: int, gpu_rate_limit: int, cache_enabled: bool, logger: logging.Logger):
        self.logger = logger

        # Semaphore cho CPU và GPU
        self.cpu_semaphore = Semaphore(max_cpu_threads)
        self.gpu_semaphore = Semaphore(max_gpu_threads)
        self.logger.info(f"Semaphore thiết lập: {max_cpu_threads} luồng CPU, {max_gpu_threads} luồng GPU.")

        # RateLimiter
        self.cpu_rate_limiter = RateLimiter(max_calls=cpu_rate_limit, period=1)
        self.gpu_rate_limiter = RateLimiter(max_calls=gpu_rate_limit, period=1)
        self.logger.info(f"RateLimiter thiết lập: {cpu_rate_limit} nhiệm vụ/giây CPU, {gpu_rate_limit} nhiệm vụ/giây GPU.")

        # Caching
        self.cache_enabled = cache_enabled
        self.cache_lock = Lock()
        self.task_cache = {} if self.cache_enabled else None
        if self.cache_enabled:
            self.logger.info("Caching đã được bật.")

        # Hàng đợi ưu tiên cho công việc CPU và GPU
        self.cpu_task_queue = PriorityQueue(maxsize=100)  # Giới hạn 50 công việc
        self.gpu_task_queue = PriorityQueue(maxsize=50)

        # Dừng luồng
        self.stop_event = Event()

    def add_task(self, priority: int, task_id: int, task_type: str):
        """
        Thêm nhiệm vụ vào hàng đợi với mức ưu tiên.
        """
        queue = None
        if task_type == "CPU":
            queue = self.cpu_task_queue
        elif task_type == "GPU":
            queue = self.gpu_task_queue
        else:
            self.logger.error(f"Loại nhiệm vụ không hợp lệ: {task_type}")
            return

        try:
            queue.put_nowait((priority, task_id))
            self.logger.info(f"Thêm nhiệm vụ {task_id} vào {task_type} với ưu tiên {priority}.")
        except Full:
            self.logger.warning(f"Hàng đợi {task_type} đầy. Từ chối nhiệm vụ {task_id}.")


    def start(self, cpu_task_func, gpu_task_func):
        """
        Bắt đầu quản lý threading.
        """
        self.cpu_worker_thread = Thread(target=self._distribute_tasks, args=(cpu_task_func, "CPU"), daemon=True)
        self.gpu_worker_thread = Thread(target=self._distribute_tasks, args=(gpu_task_func, "GPU"), daemon=True)
        self.monitor_and_adjust_thread = Thread(target=self._monitor_and_adjust_resources, daemon=True)

        # Khởi động các luồng
        self.cpu_worker_thread.start()
        self.gpu_worker_thread.start()
        self.monitor_and_adjust_thread.start()

        self.logger.info("ThreadingManager khởi động thành công.")

    def stop(self):
        """
        Dừng toàn bộ quản lý threading.
        """
        self.logger.info("Dừng ThreadingManager...")
        self.stop_event.set()

        self.cpu_worker_thread.join()
        self.gpu_worker_thread.join()
        self.monitor_and_adjust_thread.join()
        self.logger.info("ThreadingManager đã dừng.")

    def _distribute_tasks(self, task_func, task_type: str):
        """
        Phân phối công việc từ hàng đợi.
        """
        queue = self.cpu_task_queue if task_type == "CPU" else self.gpu_task_queue
        semaphore = self.cpu_semaphore if task_type == "CPU" else self.gpu_semaphore
        rate_limiter = self.cpu_rate_limiter if task_type == "CPU" else self.gpu_rate_limiter

        while not self.stop_event.is_set():
            try:
                priority, task_id = queue.get(timeout=1)
                if task_id in self.task_cache:
                    self.logger.info(f"{task_type}: Task {task_id} đã xử lý trước đó. Bỏ qua.")
                    continue

                if not semaphore.acquire(timeout=5):
                    self.logger.warning(f"{task_type}: Semaphore timeout cho task {task_id}.")
                    continue

                with rate_limiter:
                    self.logger.info(f"{task_type}: Xử lý task {task_id}.")
                    task_func(task_id)

                if self.cache_enabled:
                    with self.cache_lock:
                        self.task_cache[task_id] = True

            except Exception as e:
                self.logger.error(f"{task_type}: Lỗi khi xử lý task {task_id}: {e}")
            finally:
                semaphore.release()
                queue.task_done()

    def _monitor_and_adjust_resources(self):
        """
        Giám sát và điều chỉnh tài nguyên động.
        Giá trị Semaphore được điều chỉnh tương đối, giảm 20% tài nguyên tối đa khi quá tải.
        """
        # Xác định giới hạn tối đa dựa trên tài nguyên hệ thống
        max_cpu_threads = psutil.cpu_count(logical=True)  # Số vCPU logic
        max_gpu_threads = 0

        try:
            import pynvml
            pynvml.nvmlInit()
            max_gpu_threads = pynvml.nvmlDeviceGetCount()  # Số GPU khả dụng
        except Exception as e:
            self.logger.warning(f"Lỗi khi xác định số GPU: {e}")

        self.logger.info(f"Giới hạn Semaphore tối đa: {max_cpu_threads} luồng CPU, {max_gpu_threads} luồng GPU")

        min_semaphore_limit = 1  # Luồng tối thiểu phải luôn >= 1

        while not self.stop_event.is_set():
            try:
                # Giám sát CPU
                cpu_usage = psutil.cpu_percent(interval=1)
                self.logger.info(f"CPU Usage: {cpu_usage}%")

                if cpu_usage > 85:
                    reduction = max(1, int(0.2 * max_cpu_threads))  # Giảm 20% tài nguyên
                    new_limit = max(min_semaphore_limit, self.cpu_semaphore._value - reduction)
                    self.cpu_semaphore._value = new_limit
                    self.logger.info(f"Giảm luồng CPU xuống {new_limit} (giảm {reduction})")
                elif cpu_usage < 60:
                    increment = max(1, int(0.2 * max_cpu_threads))  # Tăng 20% tài nguyên
                    new_limit = min(max_cpu_threads, self.cpu_semaphore._value + increment)
                    self.cpu_semaphore._value = new_limit
                    self.logger.info(f"Tăng luồng CPU lên {new_limit} (tăng {increment})")

                # Giám sát GPU
                try:
                    for gpu_index in range(max_gpu_threads):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                        gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        self.logger.info(f"GPU {gpu_index} Usage: {gpu_utilization}%")

                        if gpu_utilization > 85:
                            reduction = max(1, int(0.2 * max_gpu_threads))  # Giảm 20% tài nguyên
                            new_limit = max(min_semaphore_limit, self.gpu_semaphore._value - reduction)
                            self.gpu_semaphore._value = new_limit
                            self.logger.info(f"Giảm luồng GPU xuống {new_limit} (giảm {reduction})")
                        elif gpu_utilization < 60:
                            increment = max(1, int(0.2 * max_gpu_threads))  # Tăng 20% tài nguyên
                            new_limit = min(max_gpu_threads, self.gpu_semaphore._value + increment)
                            self.gpu_semaphore._value = new_limit
                            self.logger.info(f"Tăng luồng GPU lên {new_limit} (tăng {increment})")
                except Exception as e:
                    self.logger.warning(f"Lỗi khi giám sát GPU: {e}")

            except Exception as e:
                self.logger.error(f"Lỗi trong _monitor_and_adjust_resources: {e}")

            # Chu kỳ giám sát
            time.sleep(5)

def load_config():
    """
    Tải cấu hình từ tệp JSON.
    Trả về cấu hình dưới dạng dictionary.
    """
    try:
        # Lấy đường dẫn từ biến môi trường hoặc dùng giá trị mặc định
        config_dir = os.getenv("CONFIG_DIR", "/app/mining_environment/config")
        config_path = os.path.join(config_dir, "threading_config.json")

        # Kiểm tra xem tệp có tồn tại không
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Không tìm thấy tệp cấu hình tại: {config_path}")

        # Đọc và tải nội dung tệp JSON
        with open(config_path, "r") as f:
            return json.load(f)

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Lỗi: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Lỗi định dạng JSON trong tệp cấu hình: {e}")

def setup():
    """
    Cài đặt và chạy ThreadingManager.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ThreadingManager")

    try:
        # Tải cấu hình từ file config
        config = load_config()

        # Lấy tên tiến trình CPU và GPU từ cấu hình
        cpu_process_name = config["processes"]["CPU"]
        gpu_process_name = config["processes"]["GPU"]
        logger.info(f"Tiến trình CPU: {cpu_process_name}")
        logger.info(f"Tiến trình GPU: {gpu_process_name}")

        # Lấy cấu hình cho ThreadingManager từ config
        max_cpu_threads = config["max_cpu_threads"]
        max_gpu_threads = config["max_gpu_threads"]
        cpu_rate_limit = config["cpu_rate_limit"]
        gpu_rate_limit = config["gpu_rate_limit"]
        cache_enabled = config["cache_enabled"]

        # Khởi tạo ThreadingManager
        threading_manager = ThreadingManager(
            max_cpu_threads=max_cpu_threads,
            max_gpu_threads=max_gpu_threads,
            cpu_rate_limit=cpu_rate_limit,
            gpu_rate_limit=gpu_rate_limit,
            cache_enabled=cache_enabled,
            logger=logger
        )

        # Định nghĩa các tác vụ CPU và GPU
        def cpu_task(task_id):
            logger.info(f"[{cpu_process_name}] Đang xử lý nhiệm vụ CPU {task_id}")
            time.sleep(1)

        def gpu_task(task_id):
            logger.info(f"[{gpu_process_name}] Đang xử lý nhiệm vụ GPU {task_id}")
            time.sleep(2)

        # Thêm nhiệm vụ vào hàng đợi
        for i in range(10):
            threading_manager.add_task(priority=10 - i, task_id=i, task_type="CPU")
            threading_manager.add_task(priority=10 - i, task_id=i, task_type="GPU")

        # Chạy ThreadingManager
        try:
            threading_manager.start(cpu_task, gpu_task)
            time.sleep(30)  # Chạy trong 30 giây
        finally:
            threading_manager.stop()

    except FileNotFoundError:
        logger.error("Không tìm thấy tệp cấu hình. Vui lòng kiểm tra đường dẫn.")
    except KeyError as e:
        logger.error(f"Thiếu trường cấu hình quan trọng: {e}")
    except json.JSONDecodeError:
        logger.error("Tệp cấu hình không hợp lệ. Vui lòng kiểm tra định dạng JSON.")

if __name__ == "__main__":
    setup()
