# models.py

from pydantic import BaseModel, Field
from typing import Any, Dict

class ConfigModel(BaseModel):
    # Các trường trong ConfigModel
    processes: Dict[str, str]
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
    
    class Config:
        extra = "allow"  # Cho phép các trường bổ sung trong dữ liệu

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi ConfigModel thành dict, bao gồm cả các trường bổ sung.
        """
        return self.dict(by_alias=True, exclude_unset=True)
