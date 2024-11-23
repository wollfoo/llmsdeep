# base_manager.py


from pathlib import Path
import logging
from typing import Dict, Any, Tuple
import sys

class BaseManager:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Khởi tạo BaseManager với cấu hình đã được tải và logger.

        Args:
            config (Dict[str, Any]): Cấu hình hệ thống.
            logger (logging.Logger): Logger để ghi log.
        """
        self.config = config
        self.logger = logger
        self.validate_config(self.config)

    def validate_config(self, config: Dict[str, Any]):
        """
        Xác thực cấu hình để đảm bảo rằng tất cả các khóa cần thiết đều tồn tại.

        Args:
            config (Dict[str, Any]): Cấu hình hệ thống.

        Raises:
            KeyError: Nếu thiếu bất kỳ khóa cấu hình nào.
            ValueError: Nếu bất kỳ giá trị cấu hình nào không hợp lệ.
        """
        required_keys = [
            "resource_allocation",
            "temperature_limits",
            "power_limits",
            "monitoring_parameters",
            "optimization_parameters",
            "cloak_strategies",
            "process_priority_map",
            "ai_driven_monitoring",
            "processes"
        ]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            error_message = f"Missing configuration keys: {', '.join(missing_keys)}"
            self.logger.error(error_message)
            raise KeyError(error_message)

        # Kiểm tra cấu hình cho từng tiến trình khai thác
        processes = config.get("processes", {})
        for proc_type, proc_name in processes.items():
            if proc_type not in ["CPU", "GPU"]:
                error_message = f"Unsupported process type: '{proc_type}'. Expected 'CPU' or 'GPU'."
                self.logger.error(error_message)
                raise KeyError(error_message)
            if not proc_name:
                error_message = f"Process name for '{proc_type}' cannot be empty."
                self.logger.error(error_message)
                raise ValueError(error_message)

        # Thêm các kiểm tra chi tiết hơn ở đây nếu cần

    # def load_model(self, model_path: Path) -> Tuple[torch.nn.Module, torch.device]:
    #     """
    #     Tải mô hình AI từ đường dẫn được chỉ định.

    #     Args:
    #         model_path (Path): Đường dẫn tới mô hình AI.

    #     Returns:
    #         Tuple[torch.nn.Module, torch.device]: Mô hình AI và thiết bị (CPU/GPU) mà mô hình được tải lên.

    #     Raises:
    #         FileNotFoundError: Nếu mô hình AI không tồn tại tại đường dẫn.
    #         Exception: Nếu có lỗi xảy ra trong quá trình tải mô hình.
    #     """
    #     if not model_path.exists():
    #         self.logger.error(f"AI model not found at: {model_path}")
    #         raise FileNotFoundError(f"AI model not found at: {model_path}")

    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     try:
    #         model = torch.load(model_path, map_location=device)
    #         model.eval()
    #         self.logger.info(f"Loaded AI model from '{model_path}' on {device}.")
    #         return model, device
    #     except Exception as e:
    #         self.logger.error(f"Error loading AI model from '{model_path}': {e}")
    #         raise e
