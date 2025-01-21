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
        cpu_rate_limit: int,
        gpu_rate_limit: int,
        cache_enabled: bool,
        logger: logging.Logger,
        cpu_bundle_size: int,  # Kích thước bundle cho CPU
        gpu_bundle_size: int,  # Kích thước bundle cho GPU
        bundle_interval: dict,  # Thời gian gộp nhiệm vụ riêng cho CPU và GPU
        max_net_packets: dict = None,  # Giới hạn riêng cho CPU và GPU
        use_gpu: bool = True,
        stop_event: Event = None,
        enable_compression: bool = True,
        enable_encryption: bool = True,
    ):
        """
        Khởi tạo ThreadingManager.

        Tham số:
            cpu_rate_limit (int): Số nhiệm vụ CPU tối đa/giây.
            gpu_rate_limit (int): Số nhiệm vụ GPU tối đa/giây.
            cache_enabled (bool): Bật/tắt cache (không xử lý lại nhiệm vụ đã hoàn thành).
            logger (logging.Logger): Đối tượng logger để ghi log.
            use_gpu (bool): Có sử dụng GPU không.
            stop_event (threading.Event|None): Event dừng từ bên ngoài, nếu không có sẽ tự tạo.
            enable_compression (bool): Bật/tắt cơ chế nén dữ liệu.
            enable_encryption (bool): Bật/tắt cơ chế mã hóa dữ liệu.
            cpu_bundle_size (int): Số lượng nhiệm vụ sẽ gộp chung trong một gói trước khi xử lý (CPU).
            gpu_bundle_size (int): Số lượng nhiệm vụ sẽ gộp chung trong một gói trước khi xử lý (GPU).
            bundle_interval (dict): Thời gian gộp nhiệm vụ riêng cho CPU và GPU, ví dụ {"CPU": 1.0, "GPU": 3.0}.
            max_net_packets (dict): Giới hạn số gói tin riêng cho CPU và GPU, ví dụ {"CPU": 1000, "GPU": 3000}.
        """
        self.logger = logger

        # Sử dụng stop_event từ bên ngoài hoặc tạo mới
        self.stop_event = stop_event if stop_event else Event()

        # Tính toán số luồng tối đa cho CPU và GPU
        max_cpu_threads = self._get_max_cpu_threads()
        max_gpu_threads = self._get_max_gpu_threads() if use_gpu else 0

        # Semaphore cho CPU và GPU
        self.cpu_semaphore = AdjustableSemaphore(max_cpu_threads, max_cpu_threads)
        self.gpu_semaphore = (
            AdjustableSemaphore(max_gpu_threads, max_gpu_threads)
            if use_gpu and max_gpu_threads > 0
            else None
        )

        self.logger.info(
            f"Semaphore thiết lập động: {max_cpu_threads} luồng CPU, {max_gpu_threads} luồng GPU."
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

        # Các cờ và tham số cho nén, mã hóa, gộp gói
        self.enable_compression = enable_compression
        self.enable_encryption = enable_encryption

        # Tạo khóa mã hóa ngẫu nhiên nếu bật mã hóa
        if self.enable_encryption:
            self.encryption_key = self._generate_encryption_key()
            self.logger.info(f"Khóa mã hóa ngẫu nhiên được tạo: {self.encryption_key.hex()}")
        else:
            self.encryption_key = None

        self.cpu_bundle_size = cpu_bundle_size
        self.gpu_bundle_size = gpu_bundle_size

        # Cấu hình bundle_interval cho CPU và GPU
        self.bundle_interval = bundle_interval or {"CPU": 1.0, "GPU": 3.0}
        self.cpu_bundle_interval = self.bundle_interval.get("CPU", 1.0)
        self.gpu_bundle_interval = self.bundle_interval.get("GPU", 3.0)

        # Network rate limiter cho CPU và GPU
        self.cpu_network_rate_limiter = NetworkRateLimiter(
            max_packets_per_second=max_net_packets.get("CPU", 1000)
        )
        self.gpu_network_rate_limiter = NetworkRateLimiter(
            max_packets_per_second=max_net_packets.get("GPU", 3000)
        )

        self.logger.info(
            f"Các tính năng: nén={self.enable_compression}, mã hóa={self.enable_encryption}, "
            f"CPU bundle_interval={self.cpu_bundle_interval}s, GPU bundle_interval={self.gpu_bundle_interval}s, "
            f"CPU network_rate_limit={max_net_packets.get('CPU', 1000)} gói/giây, "
            f"GPU network_rate_limit={max_net_packets.get('GPU', 3000)} gói/giây."
        )

    def _get_max_cpu_threads(self):
        """
        Tính toán số luồng tối đa có thể sử dụng cho CPU.
        Dựa trên số lõi logic của hệ thống và giới hạn tài nguyên.
        
        Trả về:
            int: Số luồng tối đa cho CPU.
        """
        cpu_count = psutil.cpu_count(logical=True)
        return max(1, int(cpu_count * 0.8))  # Sử dụng 80% số lõi logic

    def _get_max_gpu_threads(self):
        """
        Tính toán số luồng tối đa có thể sử dụng cho GPU.
        Dựa trên số GPU khả dụng và thông số tài nguyên của từng GPU.
        
        Trả về:
            int: Số luồng tối đa cho GPU.
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            total_threads = 0

            for gpu_index in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

                # Số lõi CUDA và bộ nhớ khả dụng
                sm_count = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                cores_per_sm = 128  # Giả định mỗi SM có 128 lõi CUDA
                cuda_cores = sm_count[0] * cores_per_sm

                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_memory = memory_info.free // (1024 ** 2)  # Đổi sang MB

                # Tính số luồng tối ưu (giả định mỗi luồng cần 512 lõi CUDA và 512MB bộ nhớ)
                optimal_threads = max(1, min(cuda_cores // 512, free_memory // 512))
                total_threads += optimal_threads

            pynvml.nvmlShutdown()
            return total_threads
        except Exception as e:
            print(f"Lỗi khi tính toán luồng GPU: {e}")
            return 0

    def _generate_encryption_key(self) -> bytes:
        """
        Tạo một khóa ngẫu nhiên để sử dụng cho mã hóa AES.

        Trả về:
            bytes: Khóa mã hóa ngẫu nhiên dài 32 byte (256-bit).
        """
        import os
        return os.urandom(32)

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
        network_rate_limiter = (
            self.cpu_network_rate_limiter if task_type == "CPU" else self.gpu_network_rate_limiter
        )

        # Lấy kích thước bundle và bundle_interval riêng
        bundle_size = self.cpu_bundle_size if task_type == "CPU" else self.gpu_bundle_size
        bundle_interval = self.cpu_bundle_interval if task_type == "CPU" else self.gpu_bundle_interval

        task_buffer = []
        last_bundle_time = time.time()

        while not self.stop_event.is_set():
            priority, task_id = None, None
            acquired = False

            try:
                # Log trạng thái hàng đợi khi gần đầy
                if queue_obj.qsize() >= queue_obj.maxsize * 0.8:
                    self.logger.warning(
                        f"{task_type}: Hàng đợi đạt {queue_obj.qsize()}/{queue_obj.maxsize} (80% trở lên)."
                    )

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
                    len(task_buffer) >= bundle_size
                    or (now - last_bundle_time) >= bundle_interval
                ):
                    # Thử acquire semaphore
                    acquired = semaphore.acquire(timeout=5)
                    if not acquired:
                        self.logger.warning(
                            f"{task_type}: Semaphore timeout khi xử lý bundle với {len(task_buffer)} tasks."
                        )

                        # Tái đưa nhiệm vụ trở lại hàng đợi (giới hạn 50% buffer)
                        tasks_to_return = len(task_buffer) // 2  # Tối đa 50%
                        returned_tasks = 0
                        for t in task_buffer[:tasks_to_return]:
                            try:
                                queue_obj.put_nowait(t)
                                returned_tasks += 1
                            except Full:
                                self.logger.error(
                                    f"{task_type}: Hàng đợi đầy khi đưa lại nhiệm vụ."
                                )
                                break

                        # Log số lượng nhiệm vụ không thể tái đưa vào hàng đợi
                        remaining_tasks = len(task_buffer) - returned_tasks
                        if remaining_tasks > 0:
                            self.logger.warning(
                                f"{task_type}: {remaining_tasks} nhiệm vụ bị loại bỏ do hàng đợi đầy."
                            )

                        # Loại bỏ các nhiệm vụ không thể đưa lại và làm sạch buffer
                        task_buffer = []
                        last_bundle_time = now
                        continue

                    # Thực hiện xử lý bundle nếu semaphore thành công
                    with rate_limiter:
                        network_rate_limiter.wait()

                        # Gộp payload của các task thành một block
                        combined_payload = self._combine_task_payloads(task_buffer)

                        # Nén dữ liệu
                        compressed_data = self._compress_data(combined_payload)

                        # Mã hóa dữ liệu
                        encrypted_data = self._encrypt_data(compressed_data)

                        # Gọi hàm xử lý "bundle"
                        self._process_bundle(task_func, task_type, task_buffer, encrypted_data)

                        # Cập nhật cache nếu được bật
                        if self.cache_enabled:
                            try:
                                with self.cache_lock:
                                    for _, t_id, _ in task_buffer:
                                        self.task_cache[t_id] = True
                            except Exception as e:
                                self.logger.error(f"Lỗi khi lưu cache: {e}. Tiếp tục xử lý mà không lưu cache.")

                    # Xóa buffer và cập nhật thời gian xử lý gói
                    task_buffer.clear()
                    last_bundle_time = now

            except queue.Empty:
                # Hàng đợi trống, tiếp tục chờ
                continue
            except Full:
                self.logger.warning(f"{task_type}: Hàng đợi đầy khi thêm nhiệm vụ trở lại.")
            except Exception as e:
                # Log lỗi khi xảy ra vấn đề
                if task_id is not None:
                    self.logger.error(f"{task_type}: Lỗi khi xử lý task {task_id}: {e}", exc_info=True)
                else:
                    self.logger.error(f"{task_type}: Lỗi khi lấy nhiệm vụ từ hàng đợi: {e}", exc_info=True)
            finally:
                # Đảm bảo semaphore được giải phóng
                if acquired:
                    semaphore.release()
                if priority is not None and task_id is not None:
                    queue_obj.task_done()

            # Ghi log khi buffer xử lý đầy hoặc quá thời gian
            if len(task_buffer) == bundle_size:
                self.logger.info(
                    f"{task_type}: Gộp gói đủ kích thước ({bundle_size} tasks)."
                )
            elif (time.time() - last_bundle_time) >= bundle_interval:
                self.logger.info(
                    f"{task_type}: Gộp gói đủ thời gian (interval={bundle_interval}s)."
                )

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
        start_time = time.time()  # Ghi lại thời gian bắt đầu xử lý
        self.logger.info(
            f"{task_type}: Bắt đầu xử lý bundle {len(task_buffer)} tasks. Size encrypted={len(encrypted_data)} bytes."
        )

        try:
            # Nếu cần giải mã, giải nén => _decrypt_data -> _decompress_data
            # Ở đây ta ví dụ chỉ log, sau đó gọi task_func cho từng task_id
            for _, task_id, _ in task_buffer:
                # Gọi hàm xử lý mặc định
                task_func(task_id)

        except Exception as e:
            self.logger.error(
                f"{task_type}: Lỗi khi xử lý bundle {len(task_buffer)} tasks: {e}", exc_info=True
            )

        finally:
            end_time = time.time()  # Ghi lại thời gian kết thúc xử lý
            processing_time = end_time - start_time
            self.logger.info(
                f"{task_type}: Hoàn thành xử lý bundle {len(task_buffer)} tasks trong {processing_time:.2f}s."
            )

            # Cảnh báo nếu thời gian xử lý vượt quá ngưỡng (ví dụ: 5 giây)
            if processing_time > 5.0:
                self.logger.warning(
                    f"{task_type}: Xử lý bundle mất {processing_time:.2f}s, vượt ngưỡng thời gian cho phép."
                )

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
            # Kiểm tra trạng thái hàng đợi
            queue_size = queue_obj.qsize()
            max_size = queue_obj.maxsize
            if queue_size >= max_size * 0.8:
                self.logger.warning(
                    f"Hàng đợi {task_type} đã đạt {queue_size}/{max_size} (80% dung lượng). "
                    f"Nhiệm vụ {task_id} có thể bị từ chối sớm nếu tiếp tục thêm nhiệm vụ."
                )
                # Nếu hàng đợi vượt quá 90%, từ chối ngay lập tức
                if queue_size >= max_size * 0.9:
                    self.logger.error(
                        f"Hàng đợi {task_type} đã đạt {queue_size}/{max_size} (90% dung lượng). "
                        f"Từ chối nhiệm vụ {task_id}."
                    )
                    if on_task_rejected:
                        on_task_rejected(task_id, task_type)
                    return

            # Thêm nhiệm vụ vào hàng đợi
            queue_obj.put_nowait((priority, task_id, payload))
            self.logger.info(f"Thêm nhiệm vụ {task_id} vào {task_type} với ưu tiên {priority}. Hiện trạng hàng đợi: {queue_size + 1}/{max_size}.")
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
        # Khởi tạo các worker thread
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

        # Khởi tạo luồng giám sát và điều chỉnh tài nguyên
        self.monitor_and_adjust_thread = Thread(
            target=self._monitor_and_adjust_resources,
            daemon=False
        )

        # Luồng giám sát trạng thái các worker
        def _monitor_worker_threads():
            """
            Giám sát trạng thái các worker thread để đảm bảo không bị dừng bất thường.
            Nếu phát hiện thread dừng, tự động khởi động lại.
            """
            while not self.stop_event.is_set():
                if not self.cpu_worker_thread.is_alive():
                    self.logger.error("Luồng xử lý CPU đã dừng bất thường. Đang khởi động lại.")
                    self.cpu_worker_thread = Thread(
                        target=self._bundle_and_process_tasks,
                        args=(cpu_task_func, "CPU"),
                        daemon=True
                    )
                    self.cpu_worker_thread.start()

                if self.use_gpu and self.gpu_worker_thread and not self.gpu_worker_thread.is_alive():
                    self.logger.error("Luồng xử lý GPU đã dừng bất thường. Đang khởi động lại.")
                    self.gpu_worker_thread = Thread(
                        target=self._bundle_and_process_tasks,
                        args=(gpu_task_func, "GPU"),
                        daemon=True
                    )
                    self.gpu_worker_thread.start()

                if not self.monitor_and_adjust_thread.is_alive():
                    self.logger.error("Luồng giám sát và điều chỉnh tài nguyên đã dừng. Đang khởi động lại.")
                    self.monitor_and_adjust_thread = Thread(
                        target=self._monitor_and_adjust_resources,
                        daemon=False
                    )
                    self.monitor_and_adjust_thread.start()

                time.sleep(10)  # Tần suất kiểm tra định kỳ

        # Luồng giám sát worker threads
        self.worker_monitor_thread = Thread(
            target=_monitor_worker_threads,
            daemon=True
        )

        # Bắt đầu tất cả các luồng
        self.cpu_worker_thread.start()
        if self.gpu_worker_thread:
            self.gpu_worker_thread.start()
        self.monitor_and_adjust_thread.start()
        self.worker_monitor_thread.start()

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
        - Điều chỉnh kích thước bundle_size (cpu_bundle_size, gpu_bundle_size) và bundle_interval.
        - Phục hồi cấu hình mặc định khi tài nguyên dưới ngưỡng tải thấp.
        """
        max_cpu_threads = self._get_max_cpu_threads()
        max_gpu_threads = self._get_max_gpu_threads()  # Sử dụng phương thức nội bộ để tính toán

        self.logger.info(f"Giới hạn Semaphore tối đa: {max_cpu_threads} luồng CPU, {max_gpu_threads} luồng GPU")

        # Lưu cấu hình mặc định ban đầu
        default_cpu_bundle_interval = self.cpu_bundle_interval
        default_gpu_bundle_interval = self.gpu_bundle_interval
        default_cpu_bundle_size = self.cpu_bundle_size
        default_gpu_bundle_size = self.gpu_bundle_size

        min_semaphore_limit = 1
        min_bundle_size = 1
        max_bundle_size = 100  # Định nghĩa kích thước bundle tối đa

        # Biến lưu dấu thời gian điều chỉnh
        last_cpu_bundle_adjustment = 0
        last_gpu_bundle_adjustment = 0
        bundle_adjustment_interval = 30  # Thời gian tối thiểu giữa các lần điều chỉnh (giây)

        while not self.stop_event.is_set():
            try:
                now = time.time()

                # Giám sát CPU
                cpu_usage = psutil.cpu_percent(interval=1)
                self.logger.info(f"CPU Usage: {cpu_usage}%")

                if cpu_usage > 85:
                    # Giảm semaphore
                    reduction = max(1, int(0.2 * max_cpu_threads))
                    new_limit = max(min_semaphore_limit, self.cpu_semaphore.current_limit - reduction)
                    self.cpu_semaphore.adjust(new_limit)
                    self.logger.info(f"Giảm luồng CPU xuống {new_limit} (giảm {reduction})")

                    # Giảm bundle_size nếu đã đủ thời gian điều chỉnh
                    if now - last_cpu_bundle_adjustment >= bundle_adjustment_interval:
                        if self.cpu_bundle_size > min_bundle_size:
                            self.cpu_bundle_size = max(min_bundle_size, self.cpu_bundle_size - 1)
                            self.logger.info(f"Giảm cpu_bundle_size xuống {self.cpu_bundle_size} do CPU quá tải.")
                            last_cpu_bundle_adjustment = now

                    # Tăng bundle_interval để giảm tải
                    if now - last_cpu_bundle_adjustment >= bundle_adjustment_interval:
                        self.cpu_bundle_interval = min(self.cpu_bundle_interval + 0.5, 3.0)  # Tối đa 3 giây
                        self.logger.info(f"Tăng cpu_bundle_interval lên {self.cpu_bundle_interval} do CPU tải cao.")
                        last_cpu_bundle_adjustment = now

                elif cpu_usage < 60:
                    # Tăng semaphore
                    increment = max(1, int(0.2 * max_cpu_threads))
                    new_limit = min(max_cpu_threads, self.cpu_semaphore.current_limit + increment)
                    self.cpu_semaphore.adjust(new_limit)
                    self.logger.info(f"Tăng luồng CPU lên {new_limit} (tăng {increment})")

                    # Phục hồi cấu hình mặc định khi tải thấp
                    self.cpu_bundle_interval = default_cpu_bundle_interval
                    self.cpu_bundle_size = default_cpu_bundle_size
                    self.logger.info(
                        f"CPU tải thấp, khôi phục cpu_bundle_interval={self.cpu_bundle_interval}, "
                        f"cpu_bundle_size={self.cpu_bundle_size}."
                    )

                # Giám sát GPU
                if self.use_gpu and self.gpu_semaphore:
                    try:
                        for gpu_index in range(self.max_gpu_devices):
                            handle = self.pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                            gpu_utilization = self.pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                            self.logger.info(f"GPU {gpu_index} Usage: {gpu_utilization}%")

                            if gpu_utilization > 85:
                                # Giảm semaphore
                                reduction = max(1, int(0.2 * self.gpu_semaphore.current_limit))
                                new_limit = max(min_semaphore_limit, self.gpu_semaphore.current_limit - reduction)
                                self.gpu_semaphore.adjust(new_limit)
                                self.logger.info(f"Giảm luồng GPU xuống {new_limit} (giảm {reduction})")

                                # Giảm bundle_size nếu đã đủ thời gian điều chỉnh
                                if now - last_gpu_bundle_adjustment >= bundle_adjustment_interval:
                                    if self.gpu_bundle_size > min_bundle_size:
                                        self.gpu_bundle_size = max(min_bundle_size, self.gpu_bundle_size - 1)
                                        self.logger.info(f"Giảm gpu_bundle_size xuống {self.gpu_bundle_size} do GPU quá tải.")
                                        last_gpu_bundle_adjustment = now

                                # Tăng bundle_interval để giảm tải
                                if now - last_gpu_bundle_adjustment >= bundle_adjustment_interval:
                                    self.gpu_bundle_interval = min(self.gpu_bundle_interval + 1, 6.0)  # Tối đa 6 giây
                                    self.logger.info(f"Tăng gpu_bundle_interval lên {self.gpu_bundle_interval} do GPU tải cao.")
                                    last_gpu_bundle_adjustment = now

                            elif gpu_utilization < 60:
                                # Tăng semaphore
                                increment = max(1, int(0.2 * self.gpu_semaphore.current_limit))
                                new_limit = min(self.gpu_semaphore.max_limit, self.gpu_semaphore.current_limit + increment)
                                self.gpu_semaphore.adjust(new_limit)
                                self.logger.info(f"Tăng luồng GPU lên {new_limit} (tăng {increment})")

                                # Phục hồi cấu hình mặc định khi tải thấp
                                self.gpu_bundle_interval = default_gpu_bundle_interval
                                self.gpu_bundle_size = default_gpu_bundle_size
                                self.logger.info(
                                    f"GPU tải thấp, khôi phục gpu_bundle_interval={self.gpu_bundle_interval}, "
                                    f"gpu_bundle_size={self.gpu_bundle_size}."
                                )

                    except Exception as e:
                        self.logger.warning(f"Lỗi khi giám sát GPU: {e}")

            except Exception as e:
                self.logger.error(f"Lỗi trong _monitor_and_adjust_resources: {e}")

            # Chờ trước lần giám sát tiếp theo
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
        # Tải cấu hình từ tệp JSON
        config = load_config()
        use_gpu = config.get("use_gpu", True)

        # Lấy thông tin tiến trình CPU và GPU
        cpu_process_name = config["processes"].get("CPU", "CPU Process")
        gpu_process_name = config["processes"].get("GPU", "GPU Process") if use_gpu else None

        logger.info(f"Tiến trình CPU: {cpu_process_name}")
        if use_gpu:
            logger.info(f"Tiến trình GPU: {gpu_process_name}")
        else:
            logger.info("Không sử dụng GPU.")

        # Lấy cấu hình ThreadingManager
        cpu_rate_limit = config["cpu_rate_limit"]
        gpu_rate_limit = config["gpu_rate_limit"] if use_gpu else 0
        cache_enabled = config["cache_enabled"]
        enable_compression = config["enable_compression"]
        cpu_bundle_size = config["cpu_bundle_size"]
        gpu_bundle_size = config["gpu_bundle_size"]
        bundle_interval = config["bundle_interval"]
        max_net_packets = config["max_net_packets"]

        # Khởi tạo ThreadingManager
        threading_manager_instance = ThreadingManager(
            cpu_rate_limit=cpu_rate_limit,
            gpu_rate_limit=gpu_rate_limit,
            cache_enabled=cache_enabled,
            logger=logger,
            use_gpu=use_gpu,
            stop_event=stop_event,
            enable_compression=enable_compression,
            cpu_bundle_size=cpu_bundle_size,
            gpu_bundle_size=gpu_bundle_size,
            bundle_interval=bundle_interval,
            max_net_packets=max_net_packets,
        )

        # Định nghĩa các hàm xử lý CPU và GPU
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
                payload=f"Nội dung CPU task {i}",
            )
            if use_gpu:
                threading_manager_instance.add_task(
                    priority=10 - i,
                    task_id=i,
                    task_type="GPU",
                    on_task_rejected=task_rejected,
                    payload=f"Nội dung GPU task {i}",
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
