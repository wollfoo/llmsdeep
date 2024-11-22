# utils.py

import psutil
import functools
import logging
from time import sleep
from typing import Any, Dict, Optional, List
import pynvml

def retry(ExceptionToCheck, tries=4, delay=3, backoff=2):
    """
    Decorator to automatically retry a function if an exception occurs.

    :param ExceptionToCheck: The exception to check. May be a tuple of exceptions to check.
    :param tries: Number of times to try (not retry) before giving up.
    :param delay: Initial delay between retries in seconds.
    :param backoff: Multiplier applied to delay between retries.
    """
    def deco_retry(f):
        @functools.wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToCheck as e:
                    sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return f_retry
    return deco_retry

class MiningProcess:
    """
    Represents a mining process with resource usage metrics.
    """
    def __init__(self, pid: int, name: str, priority: int = 1, network_interface: str = 'eth0', logger: Optional[logging.Logger] = None):
        self.pid = pid
        self.name = name
        self.priority = priority  # Priority value (1 is lowest)
        self.cpu_usage = 0.0  # In percent
        self.gpu_usage = 0.0  # In percent
        self.memory_usage = 0.0  # In percent
        self.disk_io = 0.0  # In MB
        self.network_io = 0.0  # In MB since last update
        self.mark = pid % 65535  # Unique mark for networking, limited to 16 bits
        self.network_interface = network_interface
        self._prev_bytes_sent = None
        self._prev_bytes_recv = None
        self.logger = logger or logging.getLogger(__name__)

        # Initialize NVML if GPU usage measurement is required
        try:
            pynvml.nvmlInit()
            self.gpu_initialized = True
            self.logger.info("NVML initialized successfully for GPU usage measurement.")
        except pynvml.NVMLError as e:
            self.gpu_initialized = False
            self.logger.warning(f"NVML initialization failed: {e}. GPU usage measurement disabled for this process.")

    def update_resource_usage(self):
        """
        Update the resource usage metrics of the mining process.
        """
        try:
            proc = psutil.Process(self.pid)
            self.cpu_usage = proc.cpu_percent(interval=0.1)
            self.memory_usage = proc.memory_percent()

            # Update Disk I/O
            io_counters = proc.io_counters()
            self.disk_io = (io_counters.read_bytes + io_counters.write_bytes) / (1024 * 1024)  # Convert to MB

            # Update Network I/O
            net_io = psutil.net_io_counters(pernic=True)
            if self.network_interface in net_io:
                current_bytes_sent = net_io[self.network_interface].bytes_sent
                current_bytes_recv = net_io[self.network_interface].bytes_recv

                if self._prev_bytes_sent is not None and self._prev_bytes_recv is not None:
                    sent_diff = current_bytes_sent - self._prev_bytes_sent
                    recv_diff = current_bytes_recv - self._prev_bytes_recv
                    self.network_io = (sent_diff + recv_diff) / (1024 * 1024)  # MB
                else:
                    self.network_io = 0.0  # Initial measurement

                # Update previous bytes
                self._prev_bytes_sent = current_bytes_sent
                self._prev_bytes_recv = current_bytes_recv
            else:
                self.logger.warning(f"Network interface '{self.network_interface}' not found for process {self.name} (PID: {self.pid}).")
                self.network_io = 0.0

            # Update GPU Usage if initialized and applicable
            if self.gpu_initialized and self.is_gpu_process():
                self.gpu_usage = self.get_gpu_usage()
            else:
                self.gpu_usage = 0.0  # Not applicable
        except psutil.NoSuchProcess:
            self.logger.error(f"Process {self.name} (PID: {self.pid}) does not exist.")
            self.cpu_usage = self.memory_usage = self.disk_io = self.network_io = self.gpu_usage = 0.0
        except Exception as e:
            self.logger.error(f"Error updating resource usage for process {self.name} (PID: {self.pid}): {e}")
            self.cpu_usage = self.memory_usage = self.disk_io = self.network_io = self.gpu_usage = 0.0

    def is_gpu_process(self) -> bool:
        """
        Determine if the process utilizes GPU resources.
        This method can be enhanced based on specific criteria or configurations.
        """
        # Placeholder implementation: check if process name matches known GPU mining processes
        gpu_process_keywords = ['llmsengen', 'gpu_miner']  # Extend this list as needed
        return any(keyword in self.name.lower() for keyword in gpu_process_keywords)

    def get_gpu_usage(self) -> float:
        """
        Get GPU usage percentage for the process.
        Note: NVML does not provide per-process GPU utilization directly.
        This implementation estimates GPU usage based on total GPU memory usage.
        """
        try:
            total_gpu_memory = 0
            used_gpu_memory = 0
            for gpu_index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used = memory_info.used / (1024 ** 2)  # Convert to MB
                total = memory_info.total / (1024 ** 2)  # Convert to MB
                total_gpu_memory += total
                used_gpu_memory += used

            if total_gpu_memory == 0:
                return 0.0

            gpu_usage_percent = (used_gpu_memory / total_gpu_memory) * 100
            return gpu_usage_percent
        except pynvml.NVMLError as e:
            self.logger.error(f"NVML error while fetching GPU usage for process {self.name} (PID: {self.pid}): {e}")
            return 0.0
        except Exception as e:
            self.logger.error(f"Unexpected error while fetching GPU usage for process {self.name} (PID: {self.pid}): {e}")
            return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the MiningProcess attributes to a dictionary.
        """
        return {
            'pid': self.pid,
            'name': self.name,
            'priority': self.priority,
            'cpu_usage': self.cpu_usage,
            'gpu_usage': self.gpu_usage,
            'memory_usage': self.memory_usage,
            'disk_io': self.disk_io,
            'network_io': self.network_io,
            'mark': self.mark,
            'network_interface': self.network_interface
        }
