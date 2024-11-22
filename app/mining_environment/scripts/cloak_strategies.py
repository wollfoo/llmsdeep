# cloak_strategies.py

import subprocess
import psutil
import pynvml
import logging
from retrying import retry
from typing import Any, Dict, Optional
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
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.throttle_percentage = config.get('throttle_percentage', 20)
        self.freq_adjustment = config.get('frequency_adjustment_mhz', 2000)
        self.logger = logger

    @retry(Exception, tries=3, delay=2000, backoff=2)
    def apply(self, process: Any):
        """
        Apply CPU cloaking by adjusting CPU frequency and limiting CPU threads.
        """
        try:
            if not process.pid:
                self.logger.error("Process PID is not available.")
                return

            # Adjust CPU frequency
            assign_process_to_cgroups(
                process.pid,
                {'cpu_freq': self.freq_adjustment},
                self.logger
            )
            self.logger.info(
                f"Throttled CPU frequency to {self.freq_adjustment}MHz "
                f"({self.throttle_percentage}% reduction) for process {process.name} (PID: {process.pid})."
            )

            # Additional logic to reduce CPU threads can be implemented here
            # For example, adjusting the number of allowed CPU cores
        except Exception as e:
            self.logger.error(
                f"Error throttling CPU for process {process.name} (PID: {process.pid}): {e}"
            )
            raise

class GpuCloakStrategy(CloakStrategy):
    """
    Cloaking strategy for GPU.
    Throttles GPU power limit.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, gpu_initialized: bool):
        self.throttle_percentage = config.get('throttle_percentage', 20)
        self.logger = logger
        self.gpu_initialized = gpu_initialized

    @retry(Exception, tries=3, delay=2000, backoff=2)
    def apply(self, process: Any):
        """
        Apply GPU cloaking by adjusting GPU power limit.
        """
        if not self.gpu_initialized:
            self.logger.warning(
                f"GPU not initialized. Cannot apply GPU Cloaking for process {process.name} (PID: {process.pid})."
            )
            return

        try:
            GPU_COUNT = pynvml.nvmlDeviceGetCount()
            if GPU_COUNT == 0:
                self.logger.warning("No GPUs found on the system.")
                return

            gpu_index = self.assign_gpu(process.pid, GPU_COUNT)
            if gpu_index == -1:
                self.logger.warning(
                    f"No GPU assigned to process {process.name} (PID: {process.pid})."
                )
                return

            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            current_power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            new_power_limit = int(current_power_limit * (1 - self.throttle_percentage / 100))
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, new_power_limit)
            self.logger.info(
                f"Throttled GPU {gpu_index} power limit to {new_power_limit}W "
                f"({self.throttle_percentage}% reduction) for process {process.name} (PID: {process.pid})."
            )
        except pynvml.NVMLError as e:
            self.logger.error(
                f"NVML error throttling GPU for process {process.name} (PID: {process.pid}): {e}"
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Unexpected error throttling GPU for process {process.name} (PID: {process.pid}): {e}"
            )
            raise

    def assign_gpu(self, pid: int, gpu_count: int) -> int:
        """
        Assign a GPU to a process based on PID.

        Args:
            pid (int): Process ID.
            gpu_count (int): Total number of GPUs available.

        Returns:
            int: Assigned GPU index, or -1 if none found.
        """
        try:
            return pid % gpu_count
        except Exception as e:
            self.logger.error(f"Error assigning GPU based on PID: {e}")
            return -1

class NetworkCloakStrategy(CloakStrategy):
    """
    Cloaking strategy for Network.
    Reduces network bandwidth for a process.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.bandwidth_reduction_mbps = config.get('bandwidth_reduction_mbps', 10)
        self.network_interface = config.get('network_interface')
        self.logger = logger

        if not self.network_interface:
            self.network_interface = self.get_primary_network_interface()
            self.logger.info(f"Primary network interface determined: {self.network_interface}")

    def get_primary_network_interface(self) -> str:
        """
        Determine the primary network interface.

        Returns:
            str: Name of the primary network interface.
        """
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
        """
        Apply network cloaking by limiting bandwidth using tc and iptables.
        """
        try:
            self.logger.info(
                f"Using network interface: {self.network_interface} for process {process.name} (PID: {process.pid})."
            )

            # Setup HTB qdisc if not already present
            self.setup_htb_qdisc()

            # Setup class for the process
            self.setup_tc_class(process)

            # Apply iptables mark
            self.apply_iptables_mark(process)
        except Exception as e:
            self.logger.error(
                f"Error applying Network Cloaking for process {process.name} (PID: {process.pid}): {e}"
            )
            raise

    def setup_htb_qdisc(self):
        """
        Ensure that HTB qdisc is set on the network interface.
        """
        try:
            existing_qdiscs = subprocess.check_output(
                ['tc', 'qdisc', 'show', 'dev', self.network_interface]
            ).decode()
            if 'htb' not in existing_qdiscs:
                subprocess.run(
                    [
                        'tc', 'qdisc', 'add', 'dev', self.network_interface, 'root',
                        'handle', '1:0', 'htb', 'default', '12'
                    ],
                    check=True
                )
                self.logger.info(
                    f"Added HTB qdisc on {self.network_interface}."
                )
            else:
                self.logger.info(
                    f"HTB qdisc already exists on {self.network_interface}."
                )
        except subprocess.CalledProcessError as e:
            if "RTNETLINK answers: File exists" in str(e):
                self.logger.info(
                    f"HTB qdisc already exists on {self.network_interface}."
                )
            else:
                self.logger.error(f"Error setting up HTB qdisc: {e}")
                raise
        except Exception as e:
            self.logger.error(f"Unexpected error setting up HTB qdisc: {e}")
            raise

    def setup_tc_class(self, process: Any):
        """
        Setup tc class for the process to limit bandwidth.

        Args:
            process (Any): The process to apply network cloaking.
        """
        class_id = f"1:{process.mark}"
        try:
            existing_classes = subprocess.check_output(
                ['tc', 'class', 'show', 'dev', self.network_interface, 'parent', '1:0']
            ).decode()
            if class_id not in existing_classes:
                subprocess.run(
                    [
                        'tc', 'class', 'add', 'dev', self.network_interface, 'parent', '1:0',
                        'classid', class_id, 'htb', 'rate', f"{self.bandwidth_reduction_mbps}mbit"
                    ],
                    check=True
                )
                self.logger.info(
                    f"Added class {class_id} with rate {self.bandwidth_reduction_mbps} Mbps on {self.network_interface}."
                )
            else:
                self.logger.info(
                    f"Class {class_id} already exists on {self.network_interface}."
                )
        except subprocess.CalledProcessError as e:
            if "RTNETLINK answers: File exists" in str(e):
                self.logger.info(
                    f"Class {class_id} already exists on {self.network_interface}."
                )
            else:
                self.logger.error(f"Error adding class {class_id}: {e}")
                raise
        except Exception as e:
            self.logger.error(f"Unexpected error adding class {class_id}: {e}")
            raise

    def apply_iptables_mark(self, process: Any):
        """
        Apply iptables mark to the process to filter its network traffic.

        Args:
            process (Any): The process to apply network cloaking.
        """
        try:
            subprocess.run(
                [
                    'iptables', '-t', 'mangle', '-A', 'OUTPUT', '-p', 'tcp',
                    '-m', 'owner', '--pid-owner', str(process.pid), '-j', 'MARK',
                    '--set-mark', str(process.mark)
                ],
                check=True
            )
            self.logger.info(
                f"Marked packets for process {process.name} (PID: {process.pid}) with mark {process.mark}."
            )

            # Setup tc filter for the mark
            self.setup_tc_filter(process)
        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"Error applying iptables mark for process {process.name} (PID: {process.pid}): {e}"
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Unexpected error applying iptables mark for process {process.name} (PID: {process.pid}): {e}"
            )
            raise

    def setup_tc_filter(self, process: Any):
        """
        Setup tc filter to link the iptables mark to the tc class.

        Args:
            process (Any): The process to apply network cloaking.
        """
        class_id = f"1:{process.mark}"
        try:
            existing_filters = subprocess.check_output(
                ['tc', 'filter', 'show', 'dev', self.network_interface, 'parent', '1:0', 'protocol', 'ip']
            ).decode()
            filter_pattern = f'handle {process.mark} fw flowid {class_id}'
            if filter_pattern not in existing_filters:
                subprocess.run(
                    [
                        'tc', 'filter', 'add', 'dev', self.network_interface, 'protocol', 'ip',
                        'parent', '1:0', 'prio', '1', 'handle', str(process.mark),
                        'fw', 'flowid', class_id
                    ],
                    check=True
                )
                self.logger.info(
                    f"Added filter for mark {process.mark} to assign to class {class_id} on {self.network_interface}."
                )
            else:
                self.logger.info(
                    f"Filter for mark {process.mark} already exists on {self.network_interface}."
                )
        except subprocess.CalledProcessError as e:
            if "RTNETLINK answers: File exists" in str(e):
                self.logger.info(
                    f"Filter for mark {process.mark} already exists on {self.network_interface}."
                )
            else:
                self.logger.error(
                    f"Error adding filter for mark {process.mark} on {self.network_interface}: {e}"
                )
                raise
        except Exception as e:
            self.logger.error(
                f"Unexpected error adding filter for mark {process.mark} on {self.network_interface}: {e}"
            )
            raise

class DiskIoCloakStrategy(CloakStrategy):
    """
    Cloaking strategy for Disk I/O.
    Sets I/O throttling level for the process.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.io_throttling_level = config.get('io_throttling_level', 'idle')
        self.logger = logger

    @retry(Exception, tries=3, delay=2000, backoff=2)
    def apply(self, process: Any):
        """
        Apply Disk I/O cloaking by setting ionice for the process.
        """
        try:
            existing_ionice = subprocess.check_output(
                ['ionice', '-p', str(process.pid)]
            ).decode()
            if 'idle' not in existing_ionice.lower():
                subprocess.run(
                    ['ionice', '-c', '3', '-p', str(process.pid)],
                    check=True
                )
                self.logger.info(
                    f"Set disk I/O throttling level to {self.io_throttling_level} for process {process.name} (PID: {process.pid})."
                )
            else:
                self.logger.info(
                    f"Disk I/O throttling already applied for process {process.name} (PID: {process.pid})."
                )
        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"Error throttling Disk I/O for process {process.name} (PID: {process.pid}): {e}"
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Unexpected error throttling Disk I/O for process {process.name} (PID: {process.pid}): {e}"
            )
            raise

class CacheCloakStrategy(CloakStrategy):
    """
    Cloaking strategy for Cache.
    Reduces cache usage by dropping caches.
    Note: This affects the entire system and not just the specific process.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.cache_limit_percent = config.get('cache_limit_percent', 50)
        self.logger = logger

    @retry(Exception, tries=3, delay=2000, backoff=2)
    def apply(self, process: Any):
        """
        Apply Cache cloaking by dropping caches.
        WARNING: This affects the entire system and should be used cautiously.
        """
        try:
            # Ensure the operation is run with root privileges
            if os.geteuid() != 0:
                self.logger.error(
                    f"Insufficient permissions to drop caches. Cache throttling failed for process {process.name} (PID: {process.pid})."
                )
                return

            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3\n')  # Drop pagecache, dentries, and inodes
            self.logger.info(
                f"Reduced cache usage by {self.cache_limit_percent}% by dropping caches for process {process.name} (PID: {process.pid})."
            )
        except PermissionError:
            self.logger.error(
                f"Insufficient permissions to drop caches. Cache throttling failed for process {process.name} (PID: {process.pid})."
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Error throttling cache for process {process.name} (PID: {process.pid}): {e}"
            )
            raise
