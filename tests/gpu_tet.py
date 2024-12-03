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

@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_cpu_frequency', side_effect=Exception("Restore CPU Frequency Error"))
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_cpu_threads')
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_ram_allocation')
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_gpu_power_limit')
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_disk_io_priority')
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.adjust_network_bandwidth')
@patch('mining_environment.scripts.resource_manager.SharedResourceManager.drop_caches')
def test_restore_resources_exception(
    mock_drop_caches, 
    mock_adjust_network_bw, 
    mock_adjust_disk_io, 
    mock_adjust_gpu_power, 
    mock_adjust_ram, 
    mock_adjust_cpu_threads, 
    mock_adjust_cpu_freq, 
    shared_resource_manager
):
    """Kiểm thử phương thức restore_resources khi có ngoại lệ trong quá trình khôi phục."""
    # Mock logger để kiểm tra
    shared_resource_manager.logger = MagicMock()

    process = MagicMock()
    process.pid = 2121
    process.name = "restore_process_exception"

    # Giả lập original_resource_limits
    shared_resource_manager.original_resource_limits = {
        process.pid: {
            'cpu_freq': 3000,
            'cpu_threads': 4,
            'ram_allocation_mb': 2048,
            'gpu_power_limit': 250,
            'ionice_class': 2,
            'network_bandwidth_limit_mbps': 100.0
        }
    }

    # Gọi phương thức restore_resources và kiểm tra ngoại lệ
    with pytest.raises(Exception) as exc_info:
        shared_resource_manager.restore_resources(process)

    # Kiểm tra ngoại lệ đúng loại và nội dung
    assert str(exc_info.value) == "Restore CPU Frequency Error"

    # Kiểm tra các hàm điều chỉnh được gọi đúng cách
    mock_adjust_cpu_freq.assert_called_once_with(process.pid, 3000, process.name)
    mock_adjust_cpu_threads.assert_called_once_with(process.pid, 4, process.name)
    mock_adjust_ram.assert_called_once_with(process.pid, 2048, process.name)
    mock_adjust_gpu_power.assert_called_once_with(process.pid, 250, process.name)
    mock_adjust_disk_io.assert_called_once_with(process.pid, 2, process.name)
    mock_adjust_network_bw.assert_called_once_with(process, 100.0)
    mock_drop_caches.assert_not_called()  # Không có yêu cầu drop_caches trong restore

    # Kiểm tra logger.error đã được gọi
    shared_resource_manager.logger.error.assert_called_once_with(
        f"Lỗi khi khôi phục tài nguyên cho tiến trình {process.name} (PID: {process.pid}): Restore CPU Frequency Error"
    )
