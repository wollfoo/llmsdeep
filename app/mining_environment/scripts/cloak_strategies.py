# cloak_strategies.py

import subprocess
import psutil
import pynvml
from retrying import retry
from typing import Any, Dict
from auxiliary_modules.cgroup_manager import assign_process_to_cgroups

class CloakStrategy:
    """
    Base class for different cloaking strategies.
    """
    def apply(self, process: Any):
        raise NotImplementedError("The apply method must be implemented by subclasses.")

class CpuCloakStrategy(CloakStrategy):
    """
    Cloaking strategy for CPU.
    Throttles CPU frequency and reduces CPU threads.
    """
    def __init__(self, config: Dict[str, Any], logger):
        self.throttle_percentage = config.get('throttle_percentage', 20)
        self.freq_adjustment = config.get('frequency_adjustment_mhz', 2000)
        self.logger = logger

    @retry(Exception, tries=3, delay=2000, backoff=2)
    def apply(self, process: Any):
        try:
            assign_process_to_cgroups(process.pid, {'cpu_freq': self.freq_adjustment}, self.logger)
            self.logger.info(f"Throttled CPU frequency to {self.freq_adjustment}MHz ({self.throttle_percentage}% reduction) for process {process.name} (PID: {process.pid}).")
        except Exception as e:
            self.logger.error(f"Error throttling CPU for process {process.name} (PID: {process.pid}): {e}")
            raise

class GpuCloakStrategy(CloakStrategy):
    """
    Cloaking strategy for GPU.
    Throttles GPU power limit.
    """
    def __init__(self, config: Dict[str, Any], logger, gpu_initialized: bool):
        self.throttle_percentage = config.get('throttle_percentage', 20)
        self.logger = logger
        self.gpu_initialized = gpu_initialized

    @retry(Exception, tries=3, delay=2000, backoff=2)
    def apply(self, process: Any):
        if not self.gpu_initialized:
            self.logger.warning(f"GPU not initialized. Cannot apply GPU Cloaking for process {process.name} (PID: {process.pid}).")
            return
        try:
            GPU_COUNT = pynvml.nvmlDeviceGetCount()
            if GPU_COUNT == 0:
                self.logger.warning("No GPUs found on the system.")
                return
            gpu_index = process.pid % GPU_COUNT  # Distribute GPUs based on PID
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            current_power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            new_power_limit = int(current_power_limit * (1 - self.throttle_percentage / 100))
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, new_power_limit)
            self.logger.info(f"Throttled GPU {gpu_index} power limit to {new_power_limit}W ({self.throttle_percentage}% reduction) for process {process.name} (PID: {process.pid}).")
        except pynvml.NVMLError as e:
            self.logger.error(f"NVML error throttling GPU for process {process.name} (PID: {process.pid}): {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error throttling GPU for process {process.name} (PID: {process.pid}): {e}")
            raise

class NetworkCloakStrategy(CloakStrategy):
    """
    Cloaking strategy for Network.
    Reduces network bandwidth for a process.
    """
    def __init__(self, config: Dict[str, Any], logger):
        self.bandwidth_reduction_mbps = config.get('bandwidth_reduction_mbps', 10)
        self.network_interface = config.get('network_interface', 'eth0')
        self.logger = logger

    def get_primary_network_interface(self) -> str:
        try:
            output = subprocess.check_output(['ip', 'route']).decode()
            for line in output.splitlines():
                if line.startswith('default'):
                    return line.split()[4]
            return 'eth0'
        except Exception as e:
            self.logger.error(f"Error getting primary network interface: {e}")
            return 'eth0'

    @retry(Exception, tries=3, delay=2000, backoff=2)
    def apply(self, process: Any):
        try:
            self.logger.info(f"Using network interface: {self.network_interface} for process {process.name} (PID: {process.pid}).")

            # Check if HTB qdisc is already added
            existing_qdiscs = subprocess.check_output(['tc', 'qdisc', 'show', 'dev', self.network_interface]).decode()
            if 'htb' not in existing_qdiscs:
                subprocess.run([
                    'tc', 'qdisc', 'add', 'dev', self.network_interface, 'root', 'handle', '1:0', 'htb',
                    'default', '12'
                ], check=True)
                self.logger.info(f"Added HTB qdisc on {self.network_interface} for process {process.name} (PID: {process.pid}).")
            else:
                self.logger.info(f"HTB qdisc already exists on {self.network_interface} for process {process.name} (PID: {process.pid}).")
        except subprocess.CalledProcessError as e:
            if "already exists" in str(e):
                self.logger.info(f"HTB qdisc already exists on {self.network_interface} for process {process.name} (PID: {process.pid}).")
            else:
                self.logger.error(f"Error checking or adding HTB qdisc: {e}")
                raise
        except Exception as e:
            self.logger.error(f"Error setting up HTB qdisc: {e}")
            raise

        try:
            class_id = f'1:{process.mark}'
            bw_limit_mbps = self.bandwidth_reduction_mbps

            # Check if class already exists
            existing_classes = subprocess.check_output(['tc', 'class', 'show', 'dev', self.network_interface, 'parent', '1:0']).decode()
            if class_id not in existing_classes:
                subprocess.run([
                    'tc', 'class', 'add', 'dev', self.network_interface, 'parent', '1:0', 'classid', class_id,
                    'htb', 'rate', f"{bw_limit_mbps}mbit"
                ], check=True)
                self.logger.info(f"Added class {class_id} with rate {bw_limit_mbps} Mbps on {self.network_interface} for process {process.name} (PID: {process.pid}).")
            else:
                self.logger.info(f"Class {class_id} already exists on {self.network_interface} for process {process.name} (PID: {process.pid}).")
        except subprocess.CalledProcessError as e:
            if "already exists" in str(e):
                self.logger.info(f"Class {class_id} already exists on {self.network_interface} for process {process.name} (PID: {process.pid}).")
            else:
                self.logger.error(f"Error adding class {class_id}: {e}")
                raise
        except Exception as e:
            self.logger.error(f"Error setting up class {class_id}: {e}")
            raise

        try:
            # Apply iptables mark
            subprocess.run([
                'iptables', '-t', 'mangle', '-A', 'OUTPUT', '-p', 'tcp', '-m', 'owner', '--pid-owner', str(process.pid), '-j', 'MARK', '--set-mark', str(process.mark)
            ], check=True)
            self.logger.info(f"Marked packets for process {process.name} (PID: {process.pid}) with mark {process.mark}.")

            # Check if filter already exists
            existing_filters = subprocess.check_output(['tc', 'filter', 'show', 'dev', self.network_interface, 'parent', '1:0', 'protocol', 'ip']).decode()
            filter_exists = f'handle {process.mark} fw flowid {class_id}' in existing_filters
            if not filter_exists:
                subprocess.run([
                    'tc', 'filter', 'add', 'dev', self.network_interface, 'protocol', 'ip', 'parent',
                    '1:0', 'prio', '1', 'handle', str(process.mark), 'fw', 'flowid', class_id
                ], check=True)
                self.logger.info(f"Added filter for mark {process.mark} on {self.network_interface} to assign to class {class_id}.")
            else:
                self.logger.info(f"Filter for mark {process.mark} already exists on {self.network_interface} for process {process.name} (PID: {process.pid}).")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error adding filter for network bandwidth: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error setting up network bandwidth filter: {e}")
            raise

class DiskIoCloakStrategy(CloakStrategy):
    """
    Cloaking strategy for Disk I/O.
    Sets I/O throttling level for the process.
    """
    def __init__(self, config: Dict[str, Any], logger):
        self.io_throttling_level = config.get('io_throttling_level', 'idle')
        self.logger = logger

    @retry(Exception, tries=3, delay=2000, backoff=2)
    def apply(self, process: Any):
        try:
            existing_ionice = subprocess.check_output(['ionice', '-p', str(process.pid)], stderr=subprocess.STDOUT).decode()
            if 'idle' not in existing_ionice.lower():
                subprocess.run(['ionice', '-c', '3', '-p', str(process.pid)], check=True)
                self.logger.info(f"Set disk I/O throttling level to {self.io_throttling_level} for process {process.name} (PID: {process.pid}).")
            else:
                self.logger.info(f"Disk I/O throttling already applied for process {process.name} (PID: {process.pid}).")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error throttling Disk I/O for process {process.name} (PID: {process.pid}): {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error throttling Disk I/O for process {process.name} (PID: {process.pid}): {e}")
            raise

class CacheCloakStrategy(CloakStrategy):
    """
    Cloaking strategy for Cache.
    Reduces cache usage by dropping caches.
    Note: This affects the entire system and not just the specific process.
    """
    def __init__(self, config: Dict[str, Any], logger):
        self.cache_limit_percent = config.get('cache_limit_percent', 50)
        self.logger = logger

    @retry(Exception, tries=3, delay=2000, backoff=2)
    def apply(self, process: Any):
        try:
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3\n')  # Drop pagecache, dentries and inodes
            self.logger.info(f"Reduced cache usage by {self.cache_limit_percent}% by dropping caches for process {process.name} (PID: {process.pid}).")
        except PermissionError:
            self.logger.error(f"Insufficient permissions to drop caches. Cache throttling failed for process {process.name} (PID: {process.pid}).")
            raise
        except Exception as e:
            self.logger.error(f"Error throttling cache for process {process.name} (PID: {process.pid}): {e}")
            raise
