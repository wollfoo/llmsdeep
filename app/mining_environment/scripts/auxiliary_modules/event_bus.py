import asyncio
from typing import Callable, Dict, List, Any


class EventBus:
    """
    Event Bus sử dụng asyncio hỗ trợ Pub/Sub với logic tránh deadlock và retry.
    """

    def __init__(self, queue_size: int = 1000, retry_attempts: int = 3, retry_delay: float = 1.0):
        """
        Khởi tạo EventBus với hàng đợi giới hạn kích thước.

        Args:
            queue_size (int): Kích thước tối đa của hàng đợi sự kiện.
            retry_attempts (int): Số lần retry khi publish thất bại.
            retry_delay (float): Thời gian chờ giữa các lần retry (giây).
        """
        self.subscribers: Dict[str, List[Callable[[Any], None]]] = {}
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self.lock = asyncio.Lock()
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

    async def subscribe(self, event_type: str, callback: Callable[[Any], None]):
        """
        Đăng ký một subscriber cho một loại sự kiện cụ thể.

        Args:
            event_type (str): Tên loại sự kiện.
            callback (Callable[[Any], None]): Hàm callback xử lý sự kiện.
        """
        async with self.lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(callback)

    async def _safe_execute(self, callback: Callable[[Any], None], data: Any):
        """
        Thực thi callback an toàn và xử lý ngoại lệ.

        Args:
            callback (Callable[[Any], None]): Hàm callback để thực thi.
            data (Any): Dữ liệu sự kiện được truyền vào callback.
        """
        try:
            await callback(data)
        except Exception as e:
            print(f"Lỗi trong callback {callback}: {e}")

    async def publish(self, event_type: str, data: Any):
        """
        Phát hành một sự kiện tới tất cả các subscriber đã đăng ký.

        Args:
            event_type (str): Tên loại sự kiện.
            data (Any): Dữ liệu kèm theo sự kiện.
        """
        retries = 0
        while retries < self.retry_attempts:
            try:
                async with self.lock:
                    if event_type in self.subscribers:
                        tasks = []
                        for callback in self.subscribers[event_type]:
                            print(f"Phát hành sự kiện '{event_type}' tới {callback}.")
                            tasks.append(asyncio.create_task(self._safe_execute(callback, data)))
                        await asyncio.gather(*tasks, return_exceptions=True)
                break  # Thoát vòng lặp nếu publish thành công
            except Exception as e:
                retries += 1
                print(f"Lỗi khi phát hành sự kiện '{event_type}', retry {retries}/{self.retry_attempts}: {e}")
                await asyncio.sleep(self.retry_delay)

    async def enqueue_event(self, event_type: str, data: Any):
        """
        Đẩy một sự kiện vào hàng đợi.

        Args:
            event_type (str): Tên loại sự kiện.
            data (Any): Dữ liệu kèm theo sự kiện.
        """
        try:
            await self.queue.put((event_type, data))
        except asyncio.QueueFull:
            print(f"Hàng đợi sự kiện đã đầy. Không thể thêm sự kiện '{event_type}'.")

    async def start_listening(self):
        """
        Lắng nghe các sự kiện từ queue và xử lý chúng.
        """
        try:
            while True:
                event_type, data = await self.queue.get()
                try:
                    await self.publish(event_type, data)
                except Exception as e:
                    print(f"Lỗi khi xử lý sự kiện '{event_type}': {e}")
                finally:
                    self.queue.task_done()
        except asyncio.CancelledError:
            print("EventBus đã dừng lắng nghe.")

    async def stop(self):
        """
        Dừng EventBus bằng cách hủy các coroutine lắng nghe.
        """
        print("Đang dừng EventBus...")
        # Không cần xử lý gì thêm ở đây, chỉ dừng lắng nghe.