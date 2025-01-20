"""
Module event_bus.py

Cung cấp một Event Bus chạy theo mô hình đồng bộ (threading), hỗ trợ cơ chế Pub/Sub
để các thành phần trong hệ thống giao tiếp (publish/subcribe sự kiện).

Đã loại bỏ toàn bộ asyncio, await, và chuyển sang queue.Queue, threading.Lock để
tránh xung đột dữ liệu và đảm bảo an toàn trong môi trường đa luồng.
"""

import os
import sys
import logging
import threading
import time
import queue
from typing import Callable, Dict, List, Any
from pathlib import Path  # Sửa lỗi thiếu import



# Thêm đường dẫn tới thư mục chứa `logging_config.py`
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SCRIPT_DIR))

# Thiết lập đường dẫn tới thư mục logs
LOGS_DIR = Path(os.getenv('LOGS_DIR', '/app/mining_environment/logs'))
os.makedirs(LOGS_DIR, exist_ok=True)

from logging_config import setup_logging
logger = setup_logging('event_bus', LOGS_DIR / 'event_bus.log', 'INFO')



class EventBus:
    """
    EventBus đồng bộ (Synchronous + Threading), cho phép đăng ký callback
    cho từng loại sự kiện, và publish sự kiện đến các subscribers.

    Attributes:
        subscribers (Dict[str, List[Callable[[Any], None]]]):
            Bản đồ event_type -> danh sách hàm callback.
        queue (queue.Queue):
            Hàng đợi chứa các sự kiện (event_type, data) cần xử lý.
        lock (threading.Lock):
            Khóa dùng để bảo vệ truy cập cấu trúc dữ liệu (subscribers).
        retry_attempts (int):
            Số lần retry khi publish thất bại.
        retry_delay (float):
            Thời gian chờ giữa các lần retry (giây).
        _stop_flag (bool):
            Cờ để dừng vòng lặp lắng nghe sự kiện (start_listening).
        _listening_thread (threading.Thread):
            Thread chạy hàm lắng nghe sự kiện (nếu cần).
    """

    def __init__(self, queue_size: int = 1000, retry_attempts: int = 3, retry_delay: float = 1.0):
        """
        Khởi tạo EventBus đồng bộ với hàng đợi giới hạn kích thước.

        :param queue_size: Số lượng sự kiện tối đa có thể lưu trong queue.
        :param retry_attempts: Số lần thử lại khi publish thất bại.
        :param retry_delay: Thời gian chờ giữa mỗi lần retry (giây).
        """
        self.subscribers: Dict[str, List[Callable[[Any], None]]] = {}
        self.queue = queue.Queue(maxsize=queue_size)
        self.lock = threading.Lock()
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        self._stop_flag = False
        self._listening_thread: threading.Thread = None

    def subscribe(self, event_type: str, callback: Callable[[Any], None]) -> None:
        """
        Đăng ký (subscribe) một hàm callback cho loại sự kiện nhất định.

        :param event_type: Tên loại sự kiện (str).
        :param callback: Hàm đồng bộ hoặc đồng bộ-giả-lập (không sử dụng async) có chữ ký callback(data).
        """
        with self.lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(callback)
            logger.debug(f"Đã đăng ký callback cho event_type='{event_type}' => {callback}.")

    def publish(self, event_type: str, data: Any) -> None:
        """
        Phát sự kiện đến tất cả các subscriber đã đăng ký với event_type.

        :param event_type: Tên loại sự kiện.
        :param data: Dữ liệu (payload) kèm theo sự kiện.
        """
        logger.debug(f"Yêu cầu publish event_type='{event_type}' với data={data}.")
        retries = 0
        while retries < self.retry_attempts:
            try:
                with self.lock:
                    if event_type in self.subscribers:
                        callbacks = self.subscribers[event_type]
                        for cb in callbacks:
                            self._safe_execute(cb, data)
                # Publish thành công => break
                break
            except Exception as e:
                retries += 1
                logger.warning(
                    f"Lỗi khi publish event '{event_type}', retry {retries}/{self.retry_attempts}: {e}"
                )
                if retries < self.retry_attempts:
                    time.sleep(self.retry_delay)

    def _safe_execute(self, callback: Callable[[Any], None], data: Any) -> None:
        """
        Thực thi callback một cách an toàn, bắt ngoại lệ và log nếu xảy ra lỗi.

        :param callback: Hàm callback cần được gọi.
        :param data: Dữ liệu truyền vào callback.
        """
        try:
            callback(data)
        except Exception as e:
            logger.error(f"Lỗi trong callback '{callback}': {e}", exc_info=True)

    def enqueue_event(self, event_type: str, data: Any) -> None:
        """
        Đẩy một sự kiện (event_type, data) vào hàng đợi để xử lý sau.

        :param event_type: Loại sự kiện.
        :param data: Dữ liệu của sự kiện.
        """
        try:
            self.queue.put((event_type, data), block=False)
            logger.debug(f"Đã enqueue event='{event_type}' với data={data}.")
        except queue.Full:
            logger.warning(f"Hàng đợi sự kiện đã đầy, bỏ qua event_type='{event_type}'.")

    def start_listening(self) -> None:
        """
        Bắt đầu lắng nghe các sự kiện từ queue, xử lý chúng bằng publish().
        Hoạt động đồng bộ trong 1 thread riêng (nếu cần).
        """
        if self._listening_thread and self._listening_thread.is_alive():
            logger.warning("EventBus đã start_listening() trước đó => bỏ qua.")
            return

        self._stop_flag = False
        self._listening_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listening_thread.start()
        logger.info("EventBus bắt đầu lắng nghe sự kiện (trong thread).")

    def _listen_loop(self) -> None:
        """
        Vòng lặp nội bộ, chạy trong 1 thread, liên tục lấy sự kiện từ queue,
        gọi self.publish() để thông báo đến subscriber.
        """
        logger.debug("_listen_loop: bắt đầu chạy.")
        while not self._stop_flag:
            try:
                event_type, data = self.queue.get(timeout=0.5)
            except queue.Empty:
                # Không có sự kiện, thử lại
                continue

            try:
                self.publish(event_type, data)
            except Exception as e:
                logger.error(f"Lỗi khi xử lý sự kiện '{event_type}': {e}", exc_info=True)
            finally:
                self.queue.task_done()

        logger.debug("_listen_loop: đã thoát vòng lặp (stop_flag=True).")

    def stop(self) -> None:
        """
        Dừng EventBus => Báo _listen_loop dừng (stop_flag), chờ thread join (nếu cần).
        """
        logger.info("Đang dừng EventBus (đồng bộ)...")
        self._stop_flag = True
        if self._listening_thread and self._listening_thread.is_alive():
            self._listening_thread.join(timeout=2.0)
            logger.info("EventBus đã dừng lắng nghe (thread join).")
        else:
            logger.info("EventBus không cần dừng (chưa start_listening hoặc đã dừng).")
