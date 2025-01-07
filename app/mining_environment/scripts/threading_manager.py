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
        queue = self.cpu_task_queue if task_type == "CPU" else self.gpu_task_queue
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

                semaphore.acquire(timeout=5)  # Timeout nếu Semaphore bị nghẽn
                with rate_limiter:
                    self.logger.info(f"{task_type}: Xử lý task {task_id}.")
                    task_func(task_id)

                # Ghi lại vào cache nếu được bật
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
        """
        while not self.stop_event.is_set():
            try:
                # Giám sát CPU
                cpu_usage = psutil.cpu_percent(interval=1)
                self.logger.info(f"CPU Usage: {cpu_usage}%")

                if cpu_usage > 85:
                    new_limit = max(1, self.cpu_semaphore._value - 1)
                    self.cpu_semaphore._value = new_limit
                    self.logger.info(f"Giảm luồng CPU xuống {new_limit}")
                elif cpu_usage < 50:
                    new_limit = min(10, self.cpu_semaphore._value + 1)
                    self.cpu_semaphore._value = new_limit
                    self.logger.info(f"Tăng luồng CPU lên {new_limit}")

                # Giám sát GPU
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    self.logger.info(f"GPU Usage: {gpu_utilization}%")

                    if gpu_utilization > 85:
                        new_limit = max(1, self.gpu_semaphore._value - 1)
                        self.gpu_semaphore._value = new_limit
                        self.logger.info(f"Giảm luồng GPU xuống {new_limit}")
                    elif gpu_utilization < 50:
                        new_limit = min(10, self.gpu_semaphore._value + 1)
                        self.gpu_semaphore._value = new_limit
                        self.logger.info(f"Tăng luồng GPU lên {new_limit}")
                except Exception as e:
                    self.logger.warning(f"Lỗi khi giám sát GPU: {e}")

            except Exception as e:
                self.logger.error(f"Lỗi trong _monitor_and_adjust_resources: {e}")

            time.sleep(5)


def load_config():
    """
    Tải cấu hình từ tệp JSON.
    """
    config_dir = os.getenv("CONFIG_DIR", "/app/mining_environment/config")
    config_path = os.path.join(config_dir, "resource_config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def setup():
    """
    Cài đặt và chạy ThreadingManager.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ThreadingManager")

    config = load_config()

    threading_manager = ThreadingManager(
        max_cpu_threads=5,
        max_gpu_threads=2,
        cpu_rate_limit=25,
        gpu_rate_limit=20,
        cache_enabled=True,
        logger=logger
    )

    def cpu_task(task_id):
        logger.info(f"CPU Task {task_id} đang chạy.")
        time.sleep(1)

    def gpu_task(task_id):
        logger.info(f"GPU Task {task_id} đang chạy.")
        time.sleep(2)

    for i in range(10):
        threading_manager.add_task(priority=10 - i, task_id=i, task_type="CPU")
        threading_manager.add_task(priority=10 - i, task_id=i, task_type="GPU")

    try:
        threading_manager.start(cpu_task, gpu_task)
        time.sleep(30)
    finally:
        threading_manager.stop()


if __name__ == "__main__":
    setup()
