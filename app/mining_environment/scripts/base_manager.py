# base_manager.py
import torch
from pathlib import Path
import logging
from typing import Dict, Any

class BaseManager:
    def __init__(self, config: Dict[str, Any], model_path: Path, logger: logging.Logger):
        """
        Khởi tạo BaseManager với cấu hình đã được tải, đường dẫn tới mô hình AI và logger.

        Args:
            config (Dict[str, Any]): Cấu hình hệ thống.
            model_path (Path): Đường dẫn tới mô hình AI.
            logger (logging.Logger): Logger để ghi log.
        """
        self.config = config
        self.model_path = model_path
        self.logger = logger
        self.validate_config(self.config)
        self.model, self.device = self.load_model(self.model_path)

    def validate_config(self, config: Dict[str, Any]):
        """
        Xác thực cấu hình để đảm bảo rằng tất cả các khóa cần thiết đều tồn tại.

        Args:
            config (Dict[str, Any]): Cấu hình hệ thống.

        Raises:
            KeyError: Nếu thiếu bất kỳ khóa cấu hình nào.
        """
        required_keys = [
            "resource_allocation",
            "temperature_limits",
            "power_limits",
            "monitoring_parameters",
            "optimization_parameters",
            "cloak_strategies",
            "process_priority_map",
            "ai_driven_monitoring"
        ]
        for key in required_keys:
            if key not in config:
                self.logger.error(f"Missing configuration key: {key}")
                raise KeyError(f"Missing configuration key: {key}")
        # Add more detailed checks if necessary

    def load_model(self, model_path: Path):
        """
        Tải mô hình AI từ đường dẫn được chỉ định.

        Args:
            model_path (Path): Đường dẫn tới mô hình AI.

        Returns:
            Tuple[torch.nn.Module, torch.device]: Mô hình AI và thiết bị (CPU/GPU) mà mô hình được tải lên.

        Raises:
            FileNotFoundError: Nếu mô hình AI không tồn tại tại đường dẫn.
            Exception: Nếu có lỗi xảy ra trong quá trình tải mô hình.
        """
        if not model_path.exists():
            self.logger.error(f"AI model not found at: {model_path}")
            raise FileNotFoundError(f"AI model not found at: {model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = torch.load(model_path, map_location=device)
            model.eval()
            self.logger.info(f"Loaded AI model from {model_path}")
            return model, device
        except Exception as e:
            self.logger.error(f"Error loading AI model from {model_path}: {e}")
            raise e
