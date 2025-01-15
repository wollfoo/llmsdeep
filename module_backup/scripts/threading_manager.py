# threading_manager.py

import os
import json
import logging
from threading import Semaphore, Lock, Thread, Event
from queue import PriorityQueue, Full, Empty
from ratelimiter import RateLimiter
import psutil
import time
import queue  # Đảm bảo rằng module queue được import

# AdjustableSemaphore - Lớp bọc để điều chỉnh Semaphore một cách an toàn
class AdjustableSemaphore:
    def __init__(self, initial, max_limit):
        self.semaphore = Semaphore(initial)
        self.max_limit = max_limit
        self.current_limit = initial
        self.lock = Lock()

    def acquire(self, timeout=None):
        return self.semaphore.acquire(timeout=timeout)

    def release(self):
        self.semaphore.release()

    def adjust(self, new_limit):
        with self.lock:
            if new_limit > self.max_limit:
                new_limit = self.max_limit
            elif new_limit < 1:
                new_limit = 1

            if new_limit > self.current_limit:
                for _ in range(new_limit - self.current_limit):
                    self.semaphore.release()
            elif new_limit < self.current_limit:
                for _ in range(self.current_limit - new_limit):
                    acquired = self.semaphore.acquire(timeout=0)
                    if not acquired:
                        break  # Không thể giảm thêm

            self.current_limit = new_limit

class ThreadingManager:
    """
    Quản lý threading nâng cao:
    - Giới hạn số luồng đồng thời (AdjustableSemaphore).
    - Giới hạn tần suất thực thi (RateLimiter).
    - Phân phối công việc thông minh (Priority Queue).
    - Điều chỉnh tài nguyên động (Dynamic Resource Adjustment).
    - Xử lý khi hệ thống bão hòa.
    """

    def __init__(self, max_cpu_threads: int, max_gpu_threads: int, 
                 cpu_rate_limit: int, gpu_rate_limit: int, cache_enabled: bool, logger: logging.Logger,
                 use_gpu: bool = True, stop_event: Event = None):
        self.logger = logger
        self.max_gpu_threads = max_gpu_threads  # Thêm dòng này

        # Sử dụng stop_event từ bên ngoài hoặc tạo mới
        self.stop_event = stop_event if stop_event else Event()

        # Semaphore cho CPU và GPU sử dụng AdjustableSemaphore
        self.cpu_semaphore = AdjustableSemaphore(max_cpu_threads, max_cpu_threads)
        self.gpu_semaphore = AdjustableSemaphore(max_gpu_threads, max_gpu_threads) if use_gpu and max_gpu_threads > 0 else None
        self.logger.info(f"Semaphore thiết lập: {max_cpu_threads} luồng CPU, {max_gpu_threads} luồng GPU.")

        # RateLimiter
        self.cpu_rate_limiter = RateLimiter(max_calls=cpu_rate_limit, period=1)
        self.gpu_rate_limiter = RateLimiter(max_calls=gpu_rate_limit, period=1) if self.gpu_semaphore else None
        self.logger.info(f"RateLimiter thiết lập: {cpu_rate_limit} nhiệm vụ/giây CPU, {gpu_rate_limit} nhiệm vụ/giây GPU.")

        # Caching
        self.cache_enabled = cache_enabled
        self.cache_lock = Lock()
        self.task_cache = {} if self.cache_enabled else None
        if self.cache_enabled:
            self.logger.info("Caching đã được bật.")

        # Hàng đợi ưu tiên cho công việc CPU và GPU
        self.cpu_task_queue = PriorityQueue(maxsize=200)  # Giới hạn công việc 
        self.gpu_task_queue = PriorityQueue(maxsize=100) if self.gpu_semaphore else None

        # Sử dụng GPU
        self.use_gpu = use_gpu and self.gpu_semaphore is not None

        # Xử lý GPU nếu sử dụng
        if self.use_gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.max_gpu_devices = pynvml.nvmlDeviceGetCount()
                self.pynvml = pynvml
                self.logger.info(f"Số GPU khả dụng: {self.max_gpu_devices}")
            except Exception as e:
                self.logger.warning(f"Lỗi khi xác định số GPU: {e}")
                self.use_gpu = False
                self.gpu_task_queue = None
                self.gpu_semaphore = None
                self.gpu_rate_limiter = None
                self.max_gpu_threads = 0  # Đảm bảo không sử dụng GPU

    def add_task(self, priority: int, task_id: int, task_type: str, on_task_rejected=None):
        """
        Thêm nhiệm vụ vào hàng đợi với mức ưu tiên.
        Nếu từ chối, gọi hàm callback `on_task_rejected`.
        """
        queue_obj = None
        if task_type == "CPU":
            queue_obj = self.cpu_task_queue
        elif task_type == "GPU" and self.use_gpu:
            queue_obj = self.gpu_task_queue
        else:
            self.logger.error(f"Loại nhiệm vụ không hợp lệ hoặc GPU không được sử dụng: {task_type}")
            return

        try:
            queue_obj.put_nowait((priority, task_id))
            self.logger.info(f"Thêm nhiệm vụ {task_id} vào {task_type} với ưu tiên {priority}.")
        except Full:
            self.logger.warning(f"Hàng đợi {task_type} đầy. Từ chối nhiệm vụ {task_id}.")
            if on_task_rejected:
                on_task_rejected(task_id, task_type)

    def start(self, cpu_task_func, gpu_task_func):
        """
        Bắt đầu quản lý threading.
        """
        self.cpu_worker_thread = Thread(target=self._distribute_tasks, args=(cpu_task_func, "CPU"), daemon=True)
        self.gpu_worker_thread = Thread(target=self._distribute_tasks, args=(gpu_task_func, "GPU"), daemon=True) if self.use_gpu else None
        self.monitor_and_adjust_thread = Thread(target=self._monitor_and_adjust_resources, daemon=False)  # Đặt daemon=False

        # Khởi động các luồng
        self.cpu_worker_thread.start()
        if self.gpu_worker_thread:
            self.gpu_worker_thread.start()
        self.monitor_and_adjust_thread.start()

        self.logger.info("ThreadingManager khởi động thành công.")

    def stop(self):
        """
        Dừng toàn bộ quản lý threading, đảm bảo tất cả nhiệm vụ hiện tại được hoàn thành.
        """
        self.logger.info("Dừng ThreadingManager...")
        self.stop_event.set()

        # Đợi cho đến khi các hàng đợi được xử lý hoàn toàn
        self.cpu_task_queue.join()
        if self.gpu_task_queue:
            self.gpu_task_queue.join()

        # Chờ các luồng kết thúc
        self.cpu_worker_thread.join()
        if self.gpu_worker_thread:
            self.gpu_worker_thread.join()
        self.monitor_and_adjust_thread.join()

        self.logger.info("ThreadingManager đã dừng.")

    def _distribute_tasks(self, task_func, task_type: str):
        """
        Phân phối công việc từ hàng đợi.
        """
        queue_obj = self.cpu_task_queue if task_type == "CPU" else self.gpu_task_queue
        semaphore = self.cpu_semaphore if task_type == "CPU" else self.gpu_semaphore
        rate_limiter = self.cpu_rate_limiter if task_type == "CPU" else self.gpu_rate_limiter

        while not self.stop_event.is_set():
            task_id = None
            priority = None
            acquired = False
            try:
                priority, task_id = queue_obj.get(timeout=1)
                if self.cache_enabled and task_id in self.task_cache:
                    self.logger.info(f"{task_type}: Task {task_id} đã xử lý trước đó. Bỏ qua.")
                    # Loại bỏ dòng queue_obj.task_done() ở đây
                    continue

                acquired = semaphore.acquire(timeout=5)
                if not acquired:
                    self.logger.warning(f"{task_type}: Semaphore timeout cho task {task_id}. Đưa lại hàng đợi.")
                    queue_obj.put_nowait((priority, task_id))
                    # Đảm bảo rằng task_done() được gọi cho nhiệm vụ đã lấy từ hàng đợi
                    queue_obj.task_done()
                    continue

                with rate_limiter:
                    self.logger.info(f"{task_type}: Xử lý task {task_id}.")
                    task_func(task_id)

                if self.cache_enabled:
                    with self.cache_lock:
                        self.task_cache[task_id] = True

            except queue.Empty:
                # Hàng đợi trống, không làm gì cả
                continue  # Bỏ qua và tiếp tục vòng lặp
            except Full:
                self.logger.warning(f"{task_type}: Hàng đợi đầy khi cố gắng thêm lại nhiệm vụ {task_id}.")
                # Có thể thêm vào một hàng đợi tạm thời hoặc xử lý theo cách khác nếu cần
            except Exception as e:
                if task_id is not None:
                    self.logger.error(f"{task_type}: Lỗi khi xử lý task {task_id}: {e}", exc_info=True)
                else:
                    self.logger.error(f"{task_type}: Lỗi khi lấy nhiệm vụ từ hàng đợi: {e}", exc_info=True)
            finally:
                if acquired:
                    semaphore.release()
                if priority is not None and task_id is not None:
                    queue_obj.task_done()

    def _monitor_and_adjust_resources(self):
        """
        Giám sát và điều chỉnh tài nguyên động.
        Giá trị Semaphore được điều chỉnh tương đối, giảm 20% tài nguyên tối đa khi quá tải.
        """
        # Xác định giới hạn tối đa dựa trên tài nguyên hệ thống
        max_cpu_threads = psutil.cpu_count(logical=True)  # Số vCPU logic
        max_gpu_threads = self.max_gpu_threads  # Đã khởi tạo trong __init__

        self.logger.info(f"Giới hạn Semaphore tối đa: {max_cpu_threads} luồng CPU, {max_gpu_threads} luồng GPU")

        min_semaphore_limit = 1  # Luồng tối thiểu phải luôn >= 1

        while not self.stop_event.is_set():
            try:
                # Giám sát CPU
                cpu_usage = psutil.cpu_percent(interval=1)
                self.logger.info(f"CPU Usage: {cpu_usage}%")

                if cpu_usage > 85:
                    reduction = max(1, int(0.2 * max_cpu_threads))  # Giảm 20% tài nguyên
                    new_limit = max(min_semaphore_limit, self.cpu_semaphore.current_limit - reduction)
                    self.cpu_semaphore.adjust(new_limit)
                    self.logger.info(f"Giảm luồng CPU xuống {new_limit} (giảm {reduction})")
                elif cpu_usage < 60:
                    increment = max(1, int(0.2 * max_cpu_threads))  # Tăng 20% tài nguyên
                    new_limit = min(max_cpu_threads, self.cpu_semaphore.current_limit + increment)
                    self.cpu_semaphore.adjust(new_limit)
                    self.logger.info(f"Tăng luồng CPU lên {new_limit} (tăng {increment})")

                # Giám sát GPU nếu sử dụng
                if self.use_gpu and self.gpu_semaphore:
                    try:
                        for gpu_index in range(self.max_gpu_devices):
                            handle = self.pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                            gpu_utilization = self.pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                            self.logger.info(f"GPU {gpu_index} Usage: {gpu_utilization}%")

                            if gpu_utilization > 85:
                                reduction = max(1, int(0.2 * self.gpu_semaphore.current_limit))
                                new_limit = max(min_semaphore_limit, self.gpu_semaphore.current_limit - reduction)
                                self.gpu_semaphore.adjust(new_limit)
                                self.logger.info(f"Giảm luồng GPU xuống {new_limit} (giảm {reduction})")
                            elif gpu_utilization < 60:
                                increment = max(1, int(0.2 * self.gpu_semaphore.current_limit))
                                new_limit = min(self.gpu_semaphore.max_limit, self.gpu_semaphore.current_limit + increment)
                                self.gpu_semaphore.adjust(new_limit)
                                self.logger.info(f"Tăng luồng GPU lên {new_limit} (tăng {increment})")
                    except Exception as e:
                        self.logger.warning(f"Lỗi khi giám sát GPU: {e}")

            except Exception as e:
                self.logger.error(f"Lỗi trong _monitor_and_adjust_resources: {e}")

            # Chu kỳ giám sát
            time.sleep(60)

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

def setup(stop_event):
    """
    Cài đặt và chạy ThreadingManager.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ThreadingManager")

    try:
        # Tải cấu hình từ file config
        config = load_config()

        # Lấy tham số use_gpu trước để quyết định việc lấy các tham số GPU
        use_gpu = config.get("use_gpu", True)

        # Lấy tên tiến trình CPU từ cấu hình
        try:
            cpu_process_name = config["processes"]["CPU"]
            logger.info(f"Tiến trình CPU: {cpu_process_name}")
        except KeyError:
            logger.error("Thiếu trường 'CPU' trong 'processes' của tệp cấu hình.")
            raise

        # Lấy tên tiến trình GPU từ cấu hình chỉ khi use_gpu là True
        if use_gpu:
            try:
                gpu_process_name = config["processes"]["GPU"]
                logger.info(f"Tiến trình GPU: {gpu_process_name}")
            except KeyError:
                logger.error("Thiếu trường 'GPU' trong 'processes' của tệp cấu hình khi 'use_gpu' là True.")
                raise
        else:
            gpu_process_name = None
            logger.info("Không sử dụng GPU.")

        # Lấy cấu hình cho ThreadingManager từ config
        try:
            max_cpu_threads = config["max_cpu_threads"]
            cpu_rate_limit = config["cpu_rate_limit"]
        except KeyError as e:
            logger.error(f"Thiếu trường cấu hình quan trọng: {e}")
            raise

        if use_gpu:
            try:
                max_gpu_threads = config["max_gpu_threads"]
                gpu_rate_limit = config["gpu_rate_limit"]
            except KeyError as e:
                logger.error(f"Thiếu trường cấu hình quan trọng: {e}")
                raise
        else:
            max_gpu_threads = 0
            gpu_rate_limit = 0

        cache_enabled = config.get("cache_enabled", False)

        # Khởi tạo ThreadingManager
        threading_manager_instance = ThreadingManager(
            max_cpu_threads=max_cpu_threads,
            max_gpu_threads=max_gpu_threads,
            cpu_rate_limit=cpu_rate_limit,
            gpu_rate_limit=gpu_rate_limit,
            cache_enabled=cache_enabled,
            logger=logger,
            use_gpu=use_gpu,
            stop_event=stop_event  # Truyền stop_event vào
        )

        # Định nghĩa các tác vụ CPU và GPU
        def cpu_task(task_id):
            logger.info(f"[{cpu_process_name}] Đang xử lý nhiệm vụ CPU {task_id}")
            time.sleep(1)

        def gpu_task(task_id):
            if gpu_process_name:
                logger.info(f"[{gpu_process_name}] Đang xử lý nhiệm vụ GPU {task_id}")
                time.sleep(2)

        # Định nghĩa callback khi nhiệm vụ bị từ chối
        def task_rejected(task_id, task_type):
            logger.error(f"Nhiệm vụ {task_id} loại {task_type} bị từ chối do hệ thống quá tải.")
            # Có thể thêm logic gửi thông báo hoặc xử lý khác nếu cần

        # Thêm nhiệm vụ vào hàng đợi với callback
        for i in range(10):
            threading_manager_instance.add_task(priority=10 - i, task_id=i, task_type="CPU", on_task_rejected=task_rejected)
            if use_gpu:
                threading_manager_instance.add_task(priority=10 - i, task_id=i, task_type="GPU", on_task_rejected=task_rejected)

        # Thêm một số nhiệm vụ mẫu khác nếu cần
        # ...

        # Bắt đầu ThreadingManager
        threading_manager_instance.start(cpu_task, gpu_task if use_gpu else None)
        # Chạy cho đến khi stop_event được đặt (được quản lý bởi start_mining.py)
        while not threading_manager_instance.stop_event.is_set():
            time.sleep(1)
        
    except FileNotFoundError:
        logger.error("Không tìm thấy tệp cấu hình. Vui lòng kiểm tra đường dẫn.")
    except KeyError as e:
        logger.error(f"Thiếu trường cấu hình quan trọng: {e}")
    except ValueError as e:
        logger.error(f"Tệp cấu hình không hợp lệ: {e}")
    except Exception as e:
        logger.error(f"Lỗi không mong muốn: {e}")

