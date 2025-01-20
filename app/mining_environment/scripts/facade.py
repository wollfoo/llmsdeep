"""
Module facade.py

Cung cấp lớp SystemFacade để quản lý khởi động/dừng đồng bộ (synchronous)
các module chính của hệ thống: ResourceManager, AnomalyDetector, SafeRestoreEvaluator.
Loại bỏ hoàn toàn async/await và đảm bảo tương thích với resource_manager.py,
anomaly_detector.py, anomaly_evaluator.py.
"""

import logging
import threading
from .resource_manager import ResourceManager
from .anomaly_detector import AnomalyDetector
from .anomaly_evaluator import SafeRestoreEvaluator
from .auxiliary_modules.event_bus import EventBus
from .auxiliary_modules.models import ConfigModel


class SystemFacade:
    """
    Lớp facade chịu trách nhiệm quản lý vòng đời (khởi động, dừng, khởi động lại)
    các module chính của hệ thống, gồm:
      - ResourceManager
      - AnomalyDetector
      - SafeRestoreEvaluator (nếu có start/stop).

    Hoạt động theo mô hình đồng bộ (không dùng asyncio).
    """

    def __init__(self,
                 config: ConfigModel,
                 event_bus: EventBus,
                 resource_logger: logging.Logger,
                 anomaly_logger: logging.Logger):
        """
        Khởi tạo SystemFacade với các cấu hình và logger tương ứng.

        :param config: Đối tượng ConfigModel chứa cấu hình hệ thống.
        :param event_bus: Đối tượng EventBus để giao tiếp giữa các module.
        :param resource_logger: Logger cho ResourceManager.
        :param anomaly_logger: Logger cho AnomalyDetector.
        :raises RuntimeError: Nếu khởi tạo ResourceManager hoặc AnomalyDetector thất bại.
        """
        self.config = config
        self.event_bus = event_bus
        self.resource_logger = resource_logger
        self.anomaly_logger = anomaly_logger

        # Khởi tạo ResourceManager (đồng bộ)
        try:
            self.resource_manager = ResourceManager(config, event_bus, self.resource_logger)
            if not self.resource_manager:
                raise RuntimeError("ResourceManager khởi tạo không thành công (None).")
            self.resource_logger.info("ResourceManager được khởi tạo thành công.")
        except Exception as e:
            self.resource_logger.error(f"Lỗi khi khởi tạo ResourceManager: {e}")
            raise RuntimeError("Không thể khởi tạo ResourceManager.") from e

        # Khởi tạo AnomalyDetector (đồng bộ)
        try:
            self.anomaly_detector = AnomalyDetector(config, event_bus, self.anomaly_logger, self.resource_manager)
            if not self.anomaly_detector:
                raise RuntimeError("AnomalyDetector khởi tạo không thành công (None).")
            self.anomaly_logger.info("AnomalyDetector được khởi tạo thành công.")
        except Exception as e:
            self.anomaly_logger.error(f"Lỗi khi khởi tạo AnomalyDetector: {e}")
            raise RuntimeError("Không thể khởi tạo AnomalyDetector.") from e

        # Khởi tạo SafeRestoreEvaluator (đồng bộ)
        try:
            self.safe_restore_evaluator = SafeRestoreEvaluator(config, self.resource_logger, self.resource_manager)
            if not hasattr(self.safe_restore_evaluator, 'start'):
                self.resource_logger.warning("SafeRestoreEvaluator không có phương thức start() => sẽ bỏ qua.")
            self.resource_logger.info("SafeRestoreEvaluator được khởi tạo thành công.")
        except Exception as e:
            self.resource_logger.error(f"Lỗi khi khởi tạo SafeRestoreEvaluator: {e}")
            raise RuntimeError("Không thể khởi tạo SafeRestoreEvaluator.") from e

    def start(self) -> None:
        """
        Khởi động đồng bộ các module trong hệ thống: ResourceManager, AnomalyDetector, SafeRestoreEvaluator (nếu có).
        """
        self.resource_logger.info("Bắt đầu khởi động các module trong SystemFacade...")

        # ResourceManager
        try:
            self.resource_manager.start()
            self.resource_logger.info("ResourceManager đã khởi động thành công (đồng bộ).")
        except Exception as e:
            self.resource_logger.error(f"Lỗi khi khởi động ResourceManager: {e}")

        # AnomalyDetector
        try:
            self.anomaly_detector.start()
            self.anomaly_logger.info("AnomalyDetector đã khởi động thành công (đồng bộ).")
        except Exception as e:
            self.anomaly_logger.error(f"Lỗi khi khởi động AnomalyDetector: {e}")

        # SafeRestoreEvaluator (nếu có hàm start)
        if hasattr(self.safe_restore_evaluator, 'start'):
            try:
                self.safe_restore_evaluator.start()
                self.resource_logger.info("SafeRestoreEvaluator đã khởi động thành công (đồng bộ).")
            except Exception as e:
                self.resource_logger.error(f"Lỗi khi khởi động SafeRestoreEvaluator: {e}")
        else:
            self.resource_logger.warning("Bỏ qua SafeRestoreEvaluator do không có phương thức start().")

    def stop(self) -> None:
        """
        Dừng đồng bộ tất cả các module: ResourceManager, AnomalyDetector, SafeRestoreEvaluator (nếu có).
        """
        self.resource_logger.info("Dừng các module trong SystemFacade...")

        # ResourceManager
        try:
            self.resource_manager.shutdown()
            self.resource_logger.info("ResourceManager đã được dừng (đồng bộ).")
        except Exception as e:
            self.resource_logger.error(f"Lỗi khi dừng ResourceManager: {e}")

        # AnomalyDetector
        try:
            self.anomaly_detector.stop()
            self.anomaly_logger.info("AnomalyDetector đã được dừng (đồng bộ).")
        except Exception as e:
            self.anomaly_logger.error(f"Lỗi khi dừng AnomalyDetector: {e}")

        # SafeRestoreEvaluator
        try:
            if hasattr(self.safe_restore_evaluator, 'stop'):
                self.safe_restore_evaluator.stop()
                self.resource_logger.info("SafeRestoreEvaluator đã được dừng (đồng bộ).")
            else:
                self.resource_logger.warning("SafeRestoreEvaluator không có phương thức stop(). Bỏ qua.")
        except Exception as e:
            self.resource_logger.error(f"Lỗi khi dừng SafeRestoreEvaluator: {e}")

    def handle_shutdown(self) -> None:
        """
        Xử lý sự kiện shutdown: dừng tất cả các module.
        """
        self.resource_logger.info("Nhận sự kiện shutdown, đang dừng các module...")
        self.stop()

    def restart(self) -> None:
        """
        Khởi động lại (dừng + start) các module trong hệ thống (đồng bộ).
        """
        self.resource_logger.info("Đang khởi động lại các module (đồng bộ)...")
        self.stop()
        self.start()

    def register_shutdown_event(self) -> None:
        """
        Đăng ký sự kiện 'shutdown' từ EventBus => gọi handle_shutdown() (đồng bộ).
        """
        def shutdown_handler(data):
            try:
                self.handle_shutdown()
            except Exception as e:
                self.resource_logger.error(f"Lỗi khi xử lý sự kiện shutdown: {e}")

        try:
            self.event_bus.subscribe('shutdown', shutdown_handler)
            self.resource_logger.info("Đã đăng ký sự kiện shutdown (đồng bộ).")
        except Exception as e:
            self.resource_logger.error(f"Lỗi khi đăng ký sự kiện shutdown: {e}")
