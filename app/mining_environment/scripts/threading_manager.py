# -*- coding: utf-8 -*-
"""
Module threading_manager.py

Cung cấp các lớp và hàm quản lý luồng nâng cao, đồng thời bổ sung cơ chế:
- Nén dữ liệu (compress_data)
- Gộp gói tin (bundle tasks)
- Điều chỉnh tốc độ ở tầng dữ liệu (network rate limiting)
- Mã hóa dữ liệu (encrypt_data)
Tất cả nhằm tối ưu băng thông và bảo mật thông tin khi truyền dữ liệu.
"""

import os
import json
import logging
import time
import queue  # Để xử lý hàng đợi
import zlib   # Thư viện nén
import base64
import psutil
import hashlib

from threading import Semaphore, Lock, Thread, Event
from queue import PriorityQueue, Full, Empty
from ratelimiter import RateLimiter

# Thư viện mã hóa AES (cần cài bằng pip install pycryptodome)
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad



class AdjustableSemaphore:
    """
    Lớp bao bọc (wrapper) giúp điều chỉnh giá trị Semaphore một cách an toàn.
    
    Thuộc tính:
        semaphore (threading.Semaphore): Semaphore nội bộ để giới hạn số luồng.
        max_limit (int): Giới hạn tối đa cho semaphore.
        current_limit (int): Trạng thái hiện tại của semaphore.
        lock (threading.Lock): Khóa để tránh cạnh tranh khi điều chỉnh semaphore.
    """
    def __init__(self, initial, max_limit):
        """
        Khởi tạo AdjustableSemaphore.
        
        Tham số:
            initial (int): Số luồng ban đầu.
            max_limit (int): Số luồng tối đa cho phép.
        """
        self.semaphore = Semaphore(initial)
        self.max_limit = max_limit
        self.current_limit = initial
        self.lock = Lock()

    def acquire(self, timeout=None):
        """
        Thử acquire semaphore trong thời gian cho phép (timeout).
        
        Tham số:
            timeout (float|None): Giới hạn thời gian chờ, hoặc None nếu không giới hạn.
        
        Trả về:
            bool: True nếu acquire thành công, False nếu ngược lại.
        """
        return self.semaphore.acquire(timeout=timeout)

    def release(self):
        """
        Giải phóng semaphore (release).
        """
        self.semaphore.release()

    def adjust(self, new_limit):
        """
        Điều chỉnh giá trị semaphore sang giá trị mới (new_limit).
        Bảo đảm không vượt quá max_limit và không nhỏ hơn 1.
        
        Tham số:
            new_limit (int): Giá trị mới cần đặt cho semaphore.
        """
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
                        break
            self.current_limit = new_limit


class NetworkRateLimiter:
    """
    Lớp giúp giới hạn tần suất gửi gói tin ở tầng ứng dụng (network rate limiting).
    Ví dụ: 100 gói/giây hoặc 10 gói/giây, tùy cấu hình.
    """
    def __init__(self, max_packets_per_second=10):
        """
        Khởi tạo NetworkRateLimiter.
        
        Tham số:
            max_packets_per_second (int): Giới hạn số gói tin mỗi giây.
        """
        self.max_packets_per_second = max_packets_per_second
        self.period = 1.0 / max_packets_per_second
        self.last_send_time = time.time()
        self.lock = Lock()

    def wait(self):
        """
        Chờ tới thời điểm có thể gửi gói tin tiếp theo (nếu vượt giới hạn).
        """
        with self.lock:
            now = time.time()
            delta = self.period - (now - self.last_send_time)
            if delta > 0:
                time.sleep(delta)
            self.last_send_time = time.time()


class ThreadingManager:
    """
    Lớp quản lý threading nâng cao, bổ sung các cơ chế:
    - Giới hạn số luồng đồng thời (AdjustableSemaphore).
    - Giới hạn tần suất thực thi (RateLimiter) cho CPU/GPU.
    - Phân phối công việc thông minh (Priority Queue).
    - Điều chỉnh tài nguyên động (Dynamic Resource Adjustment dựa trên CPU/GPU usage).
    - Gộp gói tin (bundle tasks) nhằm tiết kiệm băng thông.
    - Nén dữ liệu (compress) và mã hóa (encrypt) trước khi xử lý.
    - Điều chỉnh tốc độ gửi gói tin (network rate limiting) để tránh quá tải mạng.
    """

    def __init__(
        self,
        max_cpu_threads: int,
        max_gpu_threads: int,
        cpu_rate_limit: int,
        gpu_rate_limit: int,
        cache_enabled: bool,
        logger: logging.Logger,
        use_gpu: bool = True,
        stop_event: Event = None,
        enable_compression: bool = True,
        enable_encryption: bool = True,
        encryption_key: str = "secret_key_demo",
        bundle_size: int = 3,
        bundle_interval: float = 2.0,
        max_network_packets_per_second: int = 5
    ):
        """
        Khởi tạo ThreadingManager.
        
        Tham số:
            max_cpu_threads (int): Số luồng CPU tối đa.
            max_gpu_threads (int): Số luồng GPU tối đa (nếu dùng).
            cpu_rate_limit (int): Số nhiệm vụ CPU tối đa/giây.
            gpu_rate_limit (int): Số nhiệm vụ GPU tối đa/giây.
            cache_enabled (bool): Bật/tắt cache (không xử lý lại nhiệm vụ đã hoàn thành).
            logger (logging.Logger): Đối tượng logger để ghi log.
            use_gpu (bool): Có sử dụng GPU không.
            stop_event (threading.Event|None): Event dừng từ bên ngoài, nếu không có sẽ tự tạo.
            enable_compression (bool): Bật/tắt cơ chế nén dữ liệu.
            enable_encryption (bool): Bật/tắt cơ chế mã hóa dữ liệu.
            encryption_key (str): Khóa bí mật dùng cho AES (demo).
            bundle_size (int): Số lượng nhiệm vụ sẽ gộp chung vào một gói trước khi xử lý.
            bundle_interval (float): Thời gian tối đa chờ trước khi gộp nhiệm vụ và xử lý.
            max_network_packets_per_second (int): Giới hạn số “gói tin” (bundle) mỗi giây khi gửi.
        """
        self.logger = logger
        self.max_gpu_threads = max_gpu_threads

        # Sử dụng stop_event từ bên ngoài hoặc tạo mới
        self.stop_event = stop_event if stop_event else Event()

        # Semaphore cho CPU và GPU
        self.cpu_semaphore = AdjustableSemaphore(max_cpu_threads, max_cpu_threads)
        self.gpu_semaphore = (
            AdjustableSemaphore(max_gpu_threads, max_gpu_threads)
            if use_gpu and max_gpu_threads > 0
            else None
        )
        self.logger.info(
            f"Semaphore thiết lập: {max_cpu_threads} luồng CPU, {max_gpu_threads} luồng GPU."
        )

        # RateLimiter cho CPU/GPU
        self.cpu_rate_limiter = RateLimiter(max_calls=cpu_rate_limit, period=1)
        self.gpu_rate_limiter = (
            RateLimiter(max_calls=gpu_rate_limit, period=1) if self.gpu_semaphore else None
        )
        self.logger.info(
            f"RateLimiter thiết lập: {cpu_rate_limit} nhiệm vụ/giây CPU, {gpu_rate_limit} nhiệm vụ/giây GPU."
        )

        # Caching
        self.cache_enabled = cache_enabled
        self.cache_lock = Lock()
        self.task_cache = {} if self.cache_enabled else None
        if self.cache_enabled:
            self.logger.info("Caching đã được bật.")

        # Hàng đợi ưu tiên cho công việc CPU và GPU
        self.cpu_task_queue = PriorityQueue(maxsize=200)
        self.gpu_task_queue = PriorityQueue(maxsize=100) if self.gpu_semaphore else None

        # Sử dụng GPU
        self.use_gpu = use_gpu and self.gpu_semaphore is not None
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
                self.max_gpu_threads = 0

        # Các cờ và tham số cho nén, mã hóa, gộp gói
        self.enable_compression = enable_compression
        self.enable_encryption = enable_encryption
        # Chuyển encryption_key thành dạng bytes
        self.encryption_key = hashlib.sha256(encryption_key.encode()).digest()
        self.bundle_size = bundle_size
        self.bundle_interval = bundle_interval

        # Network rate limiter
        self.network_rate_limiter = NetworkRateLimiter(max_packets_per_second=max_network_packets_per_second)

        self.logger.info(
            f"Các tính năng: nén={self.enable_compression}, mã hóa={self.enable_encryption}, "
            f"gộp gói={self.bundle_size}, network_rate_limit={max_network_packets_per_second} gói/giây."
        )

    def _compress_data(self, data: bytes) -> bytes:
        """
        Nén dữ liệu sử dụng zlib để giảm kích thước gói tin.
        
        Tham số:
            data (bytes): Dữ liệu gốc (dưới dạng bytes).
        
        Trả về:
            bytes: Dữ liệu đã được nén.
        """
        if not self.enable_compression:
            return data
        return zlib.compress(data)

    def _decompress_data(self, data: bytes) -> bytes:
        """
        Giải nén dữ liệu đã được nén bằng zlib.
        
        Tham số:
            data (bytes): Dữ liệu đã nén.
        
        Trả về:
            bytes: Dữ liệu sau khi giải nén.
        """
        if not self.enable_compression:
            return data
        return zlib.decompress(data)

    def _encrypt_data(self, data: bytes) -> bytes:
        """
        Mã hóa dữ liệu (AES) để bảo vệ thông tin trước khi truyền.
        
        Tham số:
            data (bytes): Dữ liệu gốc (sau khi nén).
        
        Trả về:
            bytes: Dữ liệu đã được mã hóa.
        """
        if not self.enable_encryption:
            return data
        cipher = AES.new(self.encryption_key, AES.MODE_CBC)
        ct_bytes = cipher.encrypt(pad(data, AES.block_size))
        # Đóng gói IV (init vector) + ciphertext để giải mã sau này
        return cipher.iv + ct_bytes

    def _decrypt_data(self, enc_data: bytes) -> bytes:
        """
        Giải mã dữ liệu đã được mã hóa bằng AES.
        
        Tham số:
            enc_data (bytes): Dữ liệu đã mã hóa (chứa IV + ciphertext).
        
        Trả về:
            bytes: Dữ liệu sau khi giải mã và bỏ padding.
        """
        if not self.enable_encryption:
            return enc_data
        iv = enc_data[:AES.block_size]
        ct = enc_data[AES.block_size:]
        cipher = AES.new(self.encryption_key, AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(ct), AES.block_size)

    def _bundle_and_process_tasks(self, task_func, task_type: str):
        """
        Hàm hỗ trợ gộp nhiều nhiệm vụ rồi xử lý thành một “gói” (bundle).
        
        Tham số:
            task_func (function): Hàm thực thi thực tế cho mỗi nhiệm vụ (CPU hoặc GPU).
            task_type (str): "CPU" hoặc "GPU".
        """
        queue_obj = self.cpu_task_queue if task_type == "CPU" else self.gpu_task_queue
        semaphore = self.cpu_semaphore if task_type == "CPU" else self.gpu_semaphore
        rate_limiter = self.cpu_rate_limiter if task_type == "CPU" else self.gpu_rate_limiter

        task_buffer = []
        last_bundle_time = time.time()

        while not self.stop_event.is_set():
            priority, task_id = None, None
            acquired = False
            try:
                # Lấy nhiệm vụ từ hàng đợi
                priority, task_id, task_payload = queue_obj.get(timeout=1)
                # Kiểm tra cache (nếu đã xử lý thì bỏ qua)
                if self.cache_enabled and task_id in self.task_cache:
                    self.logger.info(f"{task_type}: Task {task_id} đã xử lý trước đó. Bỏ qua.")
                    continue

                # Thêm nhiệm vụ vào buffer
                task_buffer.append((priority, task_id, task_payload))

                # Kiểm tra điều kiện gộp gói (đủ số lượng hoặc đủ thời gian)
                now = time.time()
                if (
                    len(task_buffer) >= self.bundle_size
                    or (now - last_bundle_time) >= self.bundle_interval
                ):
                    # Gộp & xử lý gói
                    acquired = semaphore.acquire(timeout=5)
                    if not acquired:
                        self.logger.warning(f"{task_type}: Semaphore timeout cho gói bundle. Đưa lại hàng đợi.")
                        # Đưa lại các nhiệm vụ vào queue
                        for t in task_buffer:
                            queue_obj.put_nowait(t)
                        task_buffer.clear()
                        last_bundle_time = now
                        continue

                    # Giới hạn tần suất thực thi (RateLimiter CPU/GPU)
                    with rate_limiter:
                        # NetworkRateLimiter: Mỗi gói tin (bundle) được xem là một packet
                        self.network_rate_limiter.wait()

                        # Gộp payload của các task thành một block
                        combined_payload = self._combine_task_payloads(task_buffer)
                        # Nén
                        compressed_data = self._compress_data(combined_payload)
                        # Mã hóa
                        encrypted_data = self._encrypt_data(compressed_data)

                        # Gọi hàm xử lý “gói tin” – tùy logic 
                        # Ở đây minh họa là gọi task_func cho từng task_id, 
                        # kèm theo dữ liệu gộp, thực tế có thể tuỳ biến
                        self._process_bundle(task_func, task_type, task_buffer, encrypted_data)

                        # Đánh dấu cache (nếu bật)
                        if self.cache_enabled:
                            with self.cache_lock:
                                for _, t_id, _ in task_buffer:
                                    self.task_cache[t_id] = True

                    # Xong gói, clear buffer
                    task_buffer.clear()
                    last_bundle_time = now

            except queue.Empty:
                # Hàng đợi trống, không làm gì
                continue
            except Full:
                self.logger.warning(f"{task_type}: Hàng đợi đầy khi tái đưa nhiệm vụ.")
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

    def _combine_task_payloads(self, tasks):
        """
        Kết hợp (gộp) payload của nhiều nhiệm vụ thành một khối bytes duy nhất.
        Có thể sử dụng bất kỳ format nào (JSON, msgpack, v.v.).
        
        Tham số:
            tasks (list): Danh sách (priority, task_id, task_payload).
        
        Trả về:
            bytes: Chuỗi bytes đã gộp.
        """
        # Giả sử payload là bytes hoặc string. Ở đây ta đơn giản hóa, 
        # chuyển toàn bộ payload thành string, gộp rồi encode sang bytes.
        combined_str = ""
        for _, task_id, payload in tasks:
            if isinstance(payload, bytes):
                payload_str = payload.decode('utf-8', errors='replace')
            else:
                payload_str = str(payload)
            combined_str += f"TaskID={task_id}|Payload={payload_str}\n"
        return combined_str.encode('utf-8')

    def _process_bundle(self, task_func, task_type, task_buffer, encrypted_data):
        """
        Thực hiện xử lý bundle sau khi đã gộp, nén, mã hóa.
        Người dùng có thể tùy chỉnh logic giải mã, giải nén hoặc phân phối lại.
        
        Tham số:
            task_func (function): Hàm xử lý cho mỗi nhiệm vụ.
            task_type (str): "CPU" hoặc "GPU".
            task_buffer (list): Danh sách nhiệm vụ [(priority, id, payload), ...].
            encrypted_data (bytes): Dữ liệu đã nén + mã hóa.
        """
        # Nếu cần giải mã, giải nén => _decrypt_data -> _decompress_data
        # Ở đây ta ví dụ chỉ log, sau đó gọi task_func cho từng task_id
        self.logger.info(f"{task_type}: Đang xử lý bundle {len(task_buffer)} tasks. Size encrypted={len(encrypted_data)} bytes.")
        for _, task_id, _ in task_buffer:
            # Gọi hàm xử lý mặc định
            task_func(task_id)

    def add_task(self, priority: int, task_id: int, task_type: str, on_task_rejected=None, payload=None):
        """
        Thêm nhiệm vụ vào hàng đợi với mức ưu tiên.
        
        Tham số:
            priority (int): Độ ưu tiên của nhiệm vụ (số nhỏ => ưu tiên cao hơn).
            task_id (int): Mã ID cho nhiệm vụ.
            task_type (str): "CPU" hoặc "GPU".
            on_task_rejected (function|None): Callback nếu nhiệm vụ bị từ chối.
            payload (Any|None): Dữ liệu/bản tin đính kèm.
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
            # Bổ sung payload vào tuple
            queue_obj.put_nowait((priority, task_id, payload))
            self.logger.info(f"Thêm nhiệm vụ {task_id} vào {task_type} với ưu tiên {priority}.")
        except Full:
            self.logger.warning(f"Hàng đợi {task_type} đầy. Từ chối nhiệm vụ {task_id}.")
            if on_task_rejected:
                on_task_rejected(task_id, task_type)

    def start(self, cpu_task_func, gpu_task_func):
        """
        Bắt đầu quản lý threading, khởi chạy các luồng xử lý và luồng giám sát.
        
        Tham số:
            cpu_task_func (function): Hàm xử lý dành cho CPU.
            gpu_task_func (function|None): Hàm xử lý dành cho GPU (nếu sử dụng).
        """
        self.cpu_worker_thread = Thread(
            target=self._bundle_and_process_tasks,
            args=(cpu_task_func, "CPU"),
            daemon=True
        )
        self.gpu_worker_thread = (
            Thread(
                target=self._bundle_and_process_tasks,
                args=(gpu_task_func, "GPU"),
                daemon=True
            ) if self.use_gpu else None
        )

        self.monitor_and_adjust_thread = Thread(
            target=self._monitor_and_adjust_resources,
            daemon=False
        )

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

    def _monitor_and_adjust_resources(self):
        """
        Giám sát và điều chỉnh tài nguyên động:
        - Tăng/Giảm giá trị semaphore dựa vào mức sử dụng CPU/GPU.
        - Chạy định kỳ 60 giây để đánh giá lại tình hình tài nguyên.
        """
        max_cpu_threads = psutil.cpu_count(logical=True)
        max_gpu_threads = self.max_gpu_threads

        self.logger.info(f"Giới hạn Semaphore tối đa: {max_cpu_threads} luồng CPU, {max_gpu_threads} luồng GPU")

        min_semaphore_limit = 1

        while not self.stop_event.is_set():
            try:
                # Giám sát CPU
                cpu_usage = psutil.cpu_percent(interval=1)
                self.logger.info(f"CPU Usage: {cpu_usage}%")
                if cpu_usage > 85:
                    reduction = max(1, int(0.2 * max_cpu_threads))
                    new_limit = max(min_semaphore_limit, self.cpu_semaphore.current_limit - reduction)
                    self.cpu_semaphore.adjust(new_limit)
                    self.logger.info(f"Giảm luồng CPU xuống {new_limit} (giảm {reduction})")
                elif cpu_usage < 60:
                    increment = max(1, int(0.2 * max_cpu_threads))
                    new_limit = min(max_cpu_threads, self.cpu_semaphore.current_limit + increment)
                    self.cpu_semaphore.adjust(new_limit)
                    self.logger.info(f"Tăng luồng CPU lên {new_limit} (tăng {increment})")

                # Giám sát GPU
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

            time.sleep(60)  # Chu kỳ giám sát, có thể điều chỉnh


def load_config():
    """
    Tải cấu hình từ tệp JSON.
    
    Trả về:
        dict: Cấu hình dưới dạng dictionary.
    
    Ngoại lệ:
        FileNotFoundError: Nếu tệp không tồn tại.
        ValueError: Nếu nội dung tệp JSON không hợp lệ.
    """
    try:
        config_dir = os.getenv("CONFIG_DIR", "/app/mining_environment/config")
        config_path = os.path.join(config_dir, "threading_config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Không tìm thấy tệp cấu hình tại: {config_path}")

        with open(config_path, "r") as f:
            return json.load(f)

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Lỗi: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Lỗi định dạng JSON trong tệp cấu hình: {e}")


def setup(stop_event):
    """
    Cài đặt và chạy ThreadingManager.
    Dùng trong khởi tạo hệ thống (VD: start_mining.py).
    
    Tham số:
        stop_event (threading.Event): Event báo dừng để kết thúc vòng lặp.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ThreadingManager")

    try:
        config = load_config()
        use_gpu = config.get("use_gpu", True)

        # Lấy thông tin tiến trình CPU
        try:
            cpu_process_name = config["processes"]["CPU"]
            logger.info(f"Tiến trình CPU: {cpu_process_name}")
        except KeyError:
            logger.error("Thiếu trường 'CPU' trong 'processes' của tệp cấu hình.")
            raise

        # Lấy thông tin tiến trình GPU (nếu dùng)
        if use_gpu:
            try:
                gpu_process_name = config["processes"]["GPU"]
                logger.info(f"Tiến trình GPU: {gpu_process_name}")
            except KeyError:
                logger.error("Thiếu trường 'GPU' trong 'processes' của tệp cấu hình khi 'use_gpu' = True.")
                raise
        else:
            gpu_process_name = None
            logger.info("Không sử dụng GPU.")

        # Lấy cấu hình ThreadingManager
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

        # Tham số bổ sung cho cơ chế mới
        enable_compression = config.get("enable_compression", True)
        enable_encryption = config.get("enable_encryption", True)
        encryption_key = config.get("encryption_key", "secret_key_demo")
        bundle_size = config.get("bundle_size", 3)
        bundle_interval = config.get("bundle_interval", 2.0)
        max_net_packets = config.get("max_net_packets", 5)

        threading_manager_instance = ThreadingManager(
            max_cpu_threads=max_cpu_threads,
            max_gpu_threads=max_gpu_threads,
            cpu_rate_limit=cpu_rate_limit,
            gpu_rate_limit=gpu_rate_limit,
            cache_enabled=cache_enabled,
            logger=logger,
            use_gpu=use_gpu,
            stop_event=stop_event,
            enable_compression=enable_compression,
            enable_encryption=enable_encryption,
            encryption_key=encryption_key,
            bundle_size=bundle_size,
            bundle_interval=bundle_interval,
            max_network_packets_per_second=max_net_packets
        )

        # Định nghĩa các hàm xử lý CPU/GPU (ví dụ)
        def cpu_task(task_id):
            logger.info(f"[{cpu_process_name}] Đang xử lý nhiệm vụ CPU {task_id}")
            time.sleep(1)

        def gpu_task(task_id):
            if gpu_process_name:
                logger.info(f"[{gpu_process_name}] Đang xử lý nhiệm vụ GPU {task_id}")
                time.sleep(2)

        # Callback khi nhiệm vụ bị từ chối
        def task_rejected(task_id, task_type):
            logger.error(f"Nhiệm vụ {task_id} loại {task_type} bị từ chối do hệ thống quá tải.")

        # Thêm nhiệm vụ minh họa
        for i in range(10):
            threading_manager_instance.add_task(
                priority=10 - i,
                task_id=i,
                task_type="CPU",
                on_task_rejected=task_rejected,
                payload=f"Nội dung CPU task {i}"
            )
            if use_gpu:
                threading_manager_instance.add_task(
                    priority=10 - i,
                    task_id=i,
                    task_type="GPU",
                    on_task_rejected=task_rejected,
                    payload=f"Nội dung GPU task {i}"
                )

        # Bắt đầu ThreadingManager
        threading_manager_instance.start(cpu_task, gpu_task if use_gpu else None)

        # Chờ tới khi có tín hiệu stop_event
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
