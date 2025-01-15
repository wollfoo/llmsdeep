# models.py

from pydantic import BaseModel, Field
from typing import Any, Dict

class ConfigModel(BaseModel):
    processes: Dict[str, str]
    network_interface: str = "eth0"
    process_priority_map: Dict[str, int] = {}
    monitoring_parameters: Dict[str, Any] = {}
    temperature_limits: Dict[str, int] = {}
    power_limits: Dict[str, Any] = {}
    resource_allocation: Dict[str, Any] = {}
    baseline_thresholds: Dict[str, Any] = {}
