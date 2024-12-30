import unittest
import pynvml
from unittest.mock import MagicMock


class SharedResourceManager:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger or MagicMock()  # Sử dụng MagicMock nếu logger là None

    def adjust_gpu_power_limit(self, pid: int, power_limit: int, process_name: str):
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Lấy giới hạn năng lượng GPU
            min_limit, max_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
            print(f"Giới hạn năng lượng GPU (debug): {min_limit} - {max_limit}")
            print(f"Yêu cầu Power Limit (debug): {power_limit * 1000}")

            # Kiểm tra giới hạn hợp lệ
            if not (min_limit <= power_limit * 1000 <= max_limit):
                print("Ném ValueError (debug)")
                raise ValueError(
                    f"Power limit {power_limit}W không hợp lệ. "
                    f"Khoảng hợp lệ: {min_limit // 1000}W - {max_limit // 1000}W."
                )

            # Áp dụng giới hạn năng lượng
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, power_limit * 1000)
            self.logger.info(f"Set GPU power limit={power_limit}W cho {process_name} (PID={pid}).")
        except Exception as e:
            self.logger.error(f"Lỗi adjust_gpu_power_limit: {e}")
            raise
        finally:
            pynvml.nvmlShutdown()


class TestAdjustGPUPowerLimit(unittest.TestCase):
    def setUp(self):
        self.logger = MagicMock()
        self.resource_manager = SharedResourceManager(config={}, logger=self.logger)

    def test_valid_power_limit(self):
        try:
            self.resource_manager.adjust_gpu_power_limit(123, 150, "test_process")
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    def test_invalid_power_limit(self):
        # Kiểm tra ngoại lệ thủ công
        try:
            self.resource_manager.adjust_gpu_power_limit(123, 300, "test_process")
        except ValueError as e:
            # Xác minh ngoại lệ được ném ra đúng cách
            print(f"Raised ValueError (debug): {e}")
            self.assertIn("Power limit 300W không hợp lệ", str(e))
        else:
            # Nếu không có ngoại lệ nào được ném ra, bài kiểm tra sẽ thất bại
            self.fail("ValueError not raised for invalid power limit")


if __name__ == "__main__":
    unittest.main()
