# tests/test_shared_resource_manager.py

import os
import sys
import logging
import pytest
import subprocess
import pynvml
import torch
from unittest.mock import patch, MagicMock, mock_open, call
from pathlib import Path
from queue import PriorityQueue, Empty, Queue
from threading import Lock, Event
from typing import Any, Dict, List



# Thiết lập biến môi trường TESTING=1
os.environ["TESTING"] = "1"

# Định nghĩa các thư mục cần thiết
APP_DIR = Path("/home/llmss/llmsdeep/app")
CONFIG_DIR = APP_DIR / "mining_environment" / "config"
MODELS_DIR = APP_DIR / "mining_environment" / "models"
SCRIPTS_DIR = APP_DIR / "mining_environment" / "scripts"

# Thêm APP_DIR vào sys.path
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from mining_environment.scripts.resource_manager import SharedResourceManager, MiningProcess

@pytest.fixture
def mock_logger():
    """Fixture để tạo mock logger."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.error = MagicMock()
    return logger

@pytest.fixture
def shared_resource_manager(mock_logger):
    """Fixture để tạo instance của SharedResourceManager với cấu hình hợp nhất và logger mock."""
    config = {
        "optimization_parameters": {
            "gpu_power_adjustment_step": 10,
            "disk_io_limit_step_mbps": 5
        },
        "resource_allocation": {
            "disk_io": {
                "min_limit_mbps": 10,
                "max_limit_mbps": 100
            },
            "ram": {
                "max_allocation_mb": 2048
            },
            "gpu": {
                "max_usage_percent": [50, 75, 100]
            },
            "network": {
                "bandwidth_limit_mbps": 100
            }
        },
        "processes": {
            "CPU": "cpu_miner",
            "GPU": "gpu_miner"
        }
    }
    return SharedResourceManager(config=config, logger=mock_logger)

@patch('mining_environment.scripts.resource_manager.SharedResourceManager.configure_network_interface', side_effect=Exception("Network Cloaking Exception"))
def test_apply_network_cloaking_exception(mock_configure_network, shared_resource_manager):
    """Kiểm thử phương thức apply_network_cloaking khi có ngoại lệ."""
    interface = "eth0"
    bandwidth_limit = 50.0
    process = MagicMock(spec=MiningProcess)
    process.name = "network_cloak_process"
    process.pid = 1717

    # Gọi phương thức và kiểm tra ngoại lệ
    with pytest.raises(Exception) as exc_info:
        shared_resource_manager.apply_network_cloaking(interface, bandwidth_limit, process)
    assert str(exc_info.value) == "Network Cloaking Exception"

    # Kiểm tra logger đã ghi log lỗi đúng cách
    shared_resource_manager.logger.error.assert_called_once_with(
        "Lỗi khi áp dụng cloaking mạng cho tiến trình network_cloak_process (PID: 1717): Network Cloaking Exception"
    )
