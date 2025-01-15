# event_bus.py

import asyncio
from typing import Callable, Dict, List, Any

class EventBus:
    """
    Event Bus đơn giản sử dụng asyncio.Queue để hỗ trợ Pub/Sub.
    """
    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[Any], None]]] = {}
        self.queue: asyncio.Queue = asyncio.Queue()

    def subscribe(self, event_type: str, callback: Callable[[Any], None]):
        """
        Đăng ký một subscriber cho loại sự kiện cụ thể.
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    async def publish(self, event_type: str, data: Any):
        """
        Phát hành một sự kiện tới tất cả các subscriber đã đăng ký.
        """
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                await callback(data)

    async def start_listening(self):
        """
        Bắt đầu lắng nghe và xử lý các sự kiện từ queue.
        """
        while True:
            event_type, data = await self.queue.get()
            await self.publish(event_type, data)
            self.queue.task_done()
