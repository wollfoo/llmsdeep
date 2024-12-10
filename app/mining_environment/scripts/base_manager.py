#base_manager.py

from pathlib import Path
import logging
from typing import Dict, Any


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
        # Các khóa cần thiết chính
        required_keys = [
            "resource_allocation",
            "temperature_limits",
            "power_limits",
            "monitoring_parameters",
            "optimization_parameters",
            "cloak_strategies",
            "process_priority_map",
            "ai_driven_monitoring",
            "processes",
            "log_analytics",
            "alert_thresholds",
            "baseline_thresholds",
            "network_interface"
        ]

        # Kiểm tra các khóa chính
        self._validate_required_keys(config, required_keys)

        # Kiểm tra cấu hình cho từng phần
        self._validate_processes(config.get("processes", {}))
        self._validate_log_analytics(config.get("log_analytics", {}))
        self._validate_alert_thresholds(config.get("alert_thresholds", {}))
        self._validate_baseline_thresholds(config.get("baseline_thresholds", {}))
        self._validate_resource_allocation(config.get("resource_allocation", {}))
        self._validate_temperature_limits(config.get("temperature_limits", {}))
        self._validate_power_limits(config.get("power_limits", {}))
        self._validate_monitoring_parameters(config.get("monitoring_parameters", {}))
        self._validate_optimization_parameters(config.get("optimization_parameters", {}))
        self._validate_ai_driven_monitoring(config.get("ai_driven_monitoring", {}))
        self._validate_cloak_strategies(config.get("cloak_strategies", {}))
        self._validate_process_priority_map(config.get("process_priority_map", {}))
        self._validate_network_interface(config.get("network_interface", ""))

    def _validate_required_keys(self, config: Dict[str, Any], required_keys: list):
        """
        Kiểm tra sự tồn tại của các khóa chính.

        Args:
            config (Dict[str, Any]): Cấu hình cần kiểm tra.
            required_keys (list): Danh sách các khóa bắt buộc.

        Raises:
            KeyError: Nếu thiếu bất kỳ khóa nào.
        """
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            error_message = f"Thiếu các khóa cấu hình: {', '.join(missing_keys)}"
            self.logger.error(error_message)
            raise KeyError(error_message)

    def _validate_processes(self, processes: Dict[str, str]):
        """
        Kiểm tra cấu hình cho các tiến trình khai thác.

        Args:
            processes (Dict[str, str]): Danh sách tiến trình khai thác.

        Raises:
            KeyError: Nếu thiếu loại tiến trình hoặc tên tiến trình.
        """
        required_process_types = ["CPU", "GPU"]
        missing_process_types = [ptype for ptype in required_process_types if ptype not in processes]
        if missing_process_types:
            error_message = f"Thiếu cấu hình tiến trình cho: {', '.join(missing_process_types)}"
            self.logger.error(error_message)
            raise KeyError(error_message)
        for proc_type, proc_name in processes.items():
            if proc_type not in ["CPU", "GPU"]:
                error_message = f"Loại tiến trình không được hỗ trợ: '{proc_type}'. Chỉ hỗ trợ 'CPU' hoặc 'GPU'."
                self.logger.error(error_message)
                raise KeyError(error_message)
            if not proc_name:
                error_message = f"Tên tiến trình cho '{proc_type}' không được để trống."
                self.logger.error(error_message)
                raise ValueError(error_message)

    def _validate_log_analytics(self, log_analytics: Dict[str, Any]):
        """
        Kiểm tra cấu hình cho Log Analytics.

        Args:
            log_analytics (Dict[str, Any]): Cấu hình Log Analytics.

        Raises:
            ValueError: Nếu cấu hình không hợp lệ.
        """
        if log_analytics.get("enabled") and not log_analytics.get("queries"):
            raise ValueError("log_analytics.enabled là True nhưng không có queries được định nghĩa.")

    def _validate_alert_thresholds(self, alert_thresholds: Dict[str, Any]):
        """
        Kiểm tra cấu hình cho ngưỡng cảnh báo.

        Args:
            alert_thresholds (Dict[str, Any]): Cấu hình ngưỡng cảnh báo.

        Raises:
            ValueError: Nếu cấu hình không hợp lệ.
        """
        if not isinstance(alert_thresholds, dict) or not alert_thresholds:
            raise ValueError("alert_thresholds phải là một dictionary không trống.")

    def _validate_baseline_thresholds(self, baseline_thresholds: Dict[str, Any]):
        """
        Kiểm tra cấu hình baseline thresholds.

        Args:
            baseline_thresholds (Dict[str, Any]): Cấu hình baseline thresholds.

        Raises:
            KeyError: Nếu thiếu các khóa cần thiết.
        """
        required_baseline_keys = ["cpu_usage_percent", "ram_usage_percent", "gpu_usage_percent", "disk_io_usage_mbps", "network_usage_mbps"]
        missing_baseline_keys = [key for key in required_baseline_keys if key not in baseline_thresholds]
        if missing_baseline_keys:
            raise KeyError(f"Thiếu các khóa trong baseline_thresholds. Yêu cầu: {', '.join(missing_baseline_keys)}.")


    def _validate_resource_allocation(self, resource_allocation: Dict[str, Any]):
        """
        Kiểm tra cấu hình tài nguyên.

        Args:
            resource_allocation (Dict[str, Any]): Cấu hình tài nguyên.

        Raises:
            KeyError: Nếu thiếu các khóa cần thiết.
            ValueError: Nếu giá trị không hợp lệ.
        """
        if not isinstance(resource_allocation, dict):
            raise ValueError("resource_allocation phải là một dictionary.")

        # Định nghĩa các khóa con cần thiết cho từng tài nguyên
        required_resource_keys = {
            "ram": ["max_allocation_mb"],
            "gpu": ["max_usage_percent"],
            "disk_io": ["min_limit_mbps", "max_limit_mbps"],
            "network": ["bandwidth_limit_mbps"],
            "cache": ["limit_percent"]
        }

        # Kiểm tra sự tồn tại của các tài nguyên chính
        missing_resources = [res for res in required_resource_keys if res not in resource_allocation]
        if missing_resources:
            raise KeyError(f"resource_allocation phải định nghĩa cấu hình cho các tài nguyên: {', '.join(missing_resources)}.")

        # Kiểm tra sự tồn tại của các khóa con trong từng tài nguyên
        for resource, subkeys in required_resource_keys.items():
            missing_subkeys = [key for key in subkeys if key not in resource_allocation[resource]]
            if missing_subkeys:
                if len(missing_subkeys) == 1:
                    msg = f"resource_allocation['{resource}'] phải định nghĩa '{missing_subkeys[0]}'."
                else:
                    keys = "', '".join(missing_subkeys)
                    msg = f"resource_allocation['{resource}'] phải định nghĩa các khóa '{keys}'."
                raise KeyError(msg)


    def _validate_temperature_limits(self, temperature_limits: Dict[str, Any]):
        """
        Kiểm tra cấu hình giới hạn nhiệt độ.

        Args:
            temperature_limits (Dict[str, Any]): Cấu hình giới hạn nhiệt độ.

        Raises:
            KeyError: Nếu thiếu các khóa cần thiết.
        """
        required_temperature_keys = ["cpu_max_celsius", "gpu_max_celsius"]
        missing_temperature_keys = [key for key in required_temperature_keys if key not in temperature_limits]
        if missing_temperature_keys:
            raise KeyError(f"temperature_limits phải định nghĩa {', '.join(missing_temperature_keys)}.")

    def _validate_power_limits(self, power_limits: Dict[str, Any]):
        """
        Kiểm tra cấu hình giới hạn công suất.

        Args:
            power_limits (Dict[str, Any]): Cấu hình giới hạn công suất.

        Raises:
            KeyError: Nếu thiếu các khóa cần thiết.
        """
        if "per_device_power_watts" not in power_limits:
            raise KeyError("power_limits phải định nghĩa 'per_device_power_watts' cho CPU và GPU.")
        required_power_keys = ["cpu", "gpu"]
        missing_power_keys = [key for key in required_power_keys if key not in power_limits["per_device_power_watts"]]
        if missing_power_keys:
            raise KeyError(f"power_limits['per_device_power_watts'] phải định nghĩa {', '.join(missing_power_keys)}.")

    def _validate_monitoring_parameters(self, monitoring_parameters: Dict[str, Any]):
        """
        Kiểm tra cấu hình tham số giám sát.

        Args:
            monitoring_parameters (Dict[str, Any]): Cấu hình tham số giám sát.

        Raises:
            KeyError: Nếu thiếu các khóa cần thiết.
        """
        required_monitoring_keys = ["temperature_monitoring_interval_seconds", "power_monitoring_interval_seconds", "azure_monitor_interval_seconds", "optimization_interval_seconds"]
        missing_monitoring_keys = [key for key in required_monitoring_keys if key not in monitoring_parameters]
        if missing_monitoring_keys:
            raise KeyError(f"monitoring_parameters phải định nghĩa {', '.join(missing_monitoring_keys)}.")

    def _validate_optimization_parameters(self, optimization_parameters: Dict[str, Any]):
        """
        Kiểm tra cấu hình tham số tối ưu hóa.

        Args:
            optimization_parameters (Dict[str, Any]): Cấu hình tham số tối ưu hóa.

        Raises:
            KeyError: Nếu thiếu các khóa cần thiết.
        """
        required_optimization_keys = ["gpu_power_adjustment_step", "disk_io_limit_step_mbps"]
        missing_optimization_keys = [key for key in required_optimization_keys if key not in optimization_parameters]
        if missing_optimization_keys:
            raise KeyError(f"optimization_parameters phải định nghĩa {', '.join(missing_optimization_keys)}.")

    def _validate_ai_driven_monitoring(self, ai_driven_monitoring: Dict[str, Any]):
        """
        Kiểm tra cấu hình giám sát dựa trên AI.

        Args:
            ai_driven_monitoring (Dict[str, Any]): Cấu hình giám sát dựa trên AI.

        Raises:
            KeyError: Nếu thiếu các khóa cần thiết.
        """
        required_ai_keys = ["detection_interval_seconds", "cloak_activation_delay_seconds", "anomaly_cloaking_model"]
        missing_ai_keys = [key for key in required_ai_keys if key not in ai_driven_monitoring]
        if missing_ai_keys:
            raise KeyError(f"ai_driven_monitoring phải định nghĩa {', '.join(missing_ai_keys)}.")

        if "detection_threshold" not in ai_driven_monitoring["anomaly_cloaking_model"]:
            raise KeyError("ai_driven_monitoring['anomaly_cloaking_model'] phải định nghĩa 'detection_threshold'.")

    def _validate_cloak_strategies(self, cloak_strategies: Dict[str, Any]):
        """
        Kiểm tra cấu hình chiến lược cloaking.

        Args:
            cloak_strategies (Dict[str, Any]): Cấu hình chiến lược cloaking.

        Raises:
            ValueError: Nếu cấu hình không hợp lệ.
        """
        if not isinstance(cloak_strategies, dict) or not cloak_strategies:
            raise ValueError("cloak_strategies phải là một dictionary không trống.")

    def _validate_process_priority_map(self, process_priority_map: Dict[str, Any]):
        """
        Kiểm tra cấu hình bản đồ ưu tiên tiến trình.

        Args:
            process_priority_map (Dict[str, Any]): Cấu hình bản đồ ưu tiên tiến trình.

        Raises:
            ValueError: Nếu cấu hình không hợp lệ.
        """
        if not isinstance(process_priority_map, dict) or not process_priority_map:
            raise ValueError("process_priority_map phải là một dictionary không trống.")

    def _validate_network_interface(self, network_interface: str):
        """
        Kiểm tra cấu hình giao diện mạng.

        Args:
            network_interface (str): Tên giao diện mạng.

        Raises:
            ValueError: Nếu cấu hình không hợp lệ.
        """
        if not network_interface:
            raise ValueError("network_interface không được để trống.")
