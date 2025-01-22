# models.py

from pydantic import BaseModel, Field
from typing import Any, Dict

from pydantic import BaseModel
from typing import Any, Dict

class ConfigModel(BaseModel):
    # Các trường trong ConfigModel
    processes: Dict[str, str] = {}
    network_interface: str = "eth0"
    process_priority_map: Dict[str, int] = {}
    monitoring_parameters: Dict[str, Any] = {}
    temperature_limits: Dict[str, int] = {}
    power_limits: Dict[str, Any] = {}
    resource_allocation: Dict[str, Any] = {}
    baseline_thresholds: Dict[str, Any] = {}
    optimization_parameters: Dict[str, Any] = {}
    safe_restore_parameters: Dict[str, Any] = {}
    azure_integration: Dict[str, Any] = {}
    azure_anomaly_detector: Dict[str, Any] = {}
    azure_openai: Dict[str, Any] = {}
    server_config: Dict[str, Any] = {}
    optimization_goals: Dict[str, Any] = {}
    cloak_strategies: Dict[str, Any] = {}
    ai_driven_monitoring: Dict[str, Any] = {}
    log_analytics: Dict[str, Any] = {}
    alert_thresholds: Dict[str, Any] = {}

    # Thông số khác
    granularity: str = "minutely"

    class Config:
        extra = "allow"  # Cho phép các trường bổ sung trong dữ liệu

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi ConfigModel thành dictionary.
        """
        return self.dict(by_alias=True, exclude_unset=True)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Lấy giá trị từ ConfigModel như dictionary.
        """
        return self.dict().get(key, default)

    def copy(self) -> "ConfigModel":
        """
        Tạo bản sao ConfigModel.
        """
        return self.__class__(**self.dict())

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Cập nhật giá trị cho ConfigModel.
        """
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
    