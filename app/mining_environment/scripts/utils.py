# utils.py

import psutil
import functools
from typing import Any, Dict

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
                except ExceptionToCheck:
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
    def __init__(self, pid: int, name: str, priority: int = 1, network_interface: str = 'eth0'):
        self.pid = pid
        self.name = name
        self.priority = priority  # Priority value (1 is lowest)
        self.cpu_usage = 0.0  # In percent
        self.gpu_usage = 0.0  # In percent (to be set externally)
        self.memory_usage = 0.0  # In percent
        self.disk_io = 0.0  # In MB
        self.network_io = 0.0  # In MB since last update
        self.mark = pid % 65535  # Unique mark for networking, limited to 16 bits
        self.network_interface = network_interface
        self._prev_bytes_sent = None
        self._prev_bytes_recv = None

    def update_resource_usage(self):
        """
        Update the resource usage metrics of the mining process.
        """
        try:
            proc = psutil.Process(self.pid)
            self.cpu_usage = proc.cpu_percent(interval=0.1)
            self.memory_usage = proc.memory_percent()
            io_counters = proc.io_counters()
            self.disk_io = (io_counters.read_bytes + io_counters.write_bytes) / (1024 * 1024)  # Convert to MB

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
                self.network_io = 0.0
        except psutil.NoSuchProcess:
            self.cpu_usage = self.memory_usage = self.disk_io = self.network_io = 0.0
        except Exception:
            self.cpu_usage = self.memory_usage = self.disk_io = self.network_io = 0.0
            # Let resource_manager.py handle logging of the exception

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
