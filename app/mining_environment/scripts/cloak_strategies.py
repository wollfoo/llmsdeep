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
    Throttles CPU frequency and reduces CPU threads if needed.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the CPU cloaking strategy with configuration and logger.

        """
        self.throttle_percentage = config.get('throttle_percentage', 20)
        if not isinstance(self.throttle_percentage, (int, float)) or not (0 <= self.throttle_percentage <= 100):
            logger.warning("Invalid throttle_percentage, defaulting to 20%.")
            self.throttle_percentage = 20
        
        self.freq_adjustment = config.get('frequency_adjustment_mhz', 2000)
        if not isinstance(self.freq_adjustment, int) or self.freq_adjustment <= 0:
            logger.warning("Invalid frequency_adjustment_mhz, defaulting to 2000MHz.")
            self.freq_adjustment = 2000
        
        self.logger = logger

    @retry(Exception, tries=3, delay=2, backoff=2)
    def apply(self, process: Any) -> Dict[str, Any]:
        try:
            if not getattr(process, 'pid', None):
                self.logger.error("Process PID is not available.")
                return {}

            if not hasattr(process, 'name') or not process.name:
                self.logger.warning("Process name is missing or empty.")
                process_name = "unknown"
            else:
                process_name = process.name

            adjustments = {
                'cpu_freq': self.freq_adjustment,
                'throttle_percentage': self.throttle_percentage,
            }
            self.logger.info(
                f"Prepared CPU throttling adjustments: freq={self.freq_adjustment}MHz "
                f"({self.throttle_percentage}% reduction) for process {process_name} (PID: {process.pid})."
            )
            return adjustments

        except Exception as e:
            self.logger.error(
                f"Error preparing CPU throttling for process {getattr(process, 'name', 'unknown')} "
                f"(PID: {getattr(process, 'pid', 'N/A')}): {e}"
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

    @retry(Exception, tries=3, delay=2, backoff=2)
    def apply(self, process: Any) -> Dict[str, Any]:
        """
        Apply GPU cloaking by throttling power limit.

        Args:
            process (Any): Process object with attributes 'pid' and 'name'.

        Returns:
            Dict[str, Any]: Adjustments including GPU index and new power limit.
        """
        if not self.gpu_initialized:
            self.logger.warning(
                f"GPU not initialized. Cannot prepare GPU throttling for process {getattr(process, 'name', 'unknown')} "
                f"(PID: {getattr(process, 'pid', 'N/A')})."
            )
            return {}

        try:
            if not hasattr(process, 'pid') or not hasattr(process, 'name'):
                self.logger.error("Process object is missing required attributes (pid, name).")
                return {}

            pynvml.nvmlInit()
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                if gpu_count == 0:
                    self.logger.warning("No GPUs found on the system.")
                    return {}

                gpu_index = self.assign_gpu(process.pid, gpu_count)
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
                    f"Prepared GPU throttling adjustments: GPU {gpu_index} power limit={new_power_limit} "
                    f"({self.throttle_percentage}% reduction) for process {process.name} (PID: {process.pid})."
                )
                return adjustments
            finally:
                pynvml.nvmlShutdown()

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
            gpu_count (int): Total number of GPUs.

        Returns:
            int: GPU index or -1 if assignment fails.
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
        try:
            output = subprocess.check_output(['ip', 'route']).decode()
            for line in output.splitlines():
                if line.startswith('default'):
                    return line.split()[4]
            return 'eth0'
        except Exception as e:
            self.logger.error(f"Error getting primary network interface: {e}")
            return 'eth0'

    @retry(Exception, tries=3, delay=2, backoff=2)
    def apply(self, process: Any) -> Dict[str, Any]:
        try:
            # [CHANGES] Kiểm tra process có 'mark' không, tránh lỗi KeyError
            mark_value = getattr(process, 'mark', None)
            adjustments = {
                'network_interface': self.network_interface,
                'bandwidth_limit_mbps': self.bandwidth_reduction_mbps,
            }
            # Nếu process có attribute 'mark', ta thêm vào
            if mark_value is not None:
                adjustments['process_mark'] = mark_value

            self.logger.info(
                f"Prepared Network throttling adjustments: interface={self.network_interface}, "
                f"bandwidth_limit={self.bandwidth_reduction_mbps}Mbps, mark={mark_value} "
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

    @retry(Exception, tries=3, delay=2, backoff=2)
    def apply(self, process: Any) -> Dict[str, Any]:
        try:
            ionice_class = '3' if self.io_throttling_level.lower() == 'idle' else '2'
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
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.cache_limit_percent = config.get('cache_limit_percent', 50)
        self.logger = logger

    @retry(Exception, tries=3, delay=2, backoff=2)
    def apply(self, process: Any) -> Dict[str, Any]:
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
    """
    Factory để tạo các instance của các chiến lược cloaking.
    """
    _strategies: Dict[str, Type[CloakStrategy]] = {
        'cpu': CpuCloakStrategy,
        'gpu': GpuCloakStrategy,
        'network': NetworkCloakStrategy,
        'disk_io': DiskIoCloakStrategy,
        'cache': CacheCloakStrategy
        # Thêm các chiến lược khác ở đây nếu cần
    }

    @staticmethod
    def create_strategy(strategy_name: str, config: Dict[str, Any],
                        logger: logging.Logger, gpu_initialized: bool = False
    ) -> Optional[CloakStrategy]:
        strategy_class = CloakStrategyFactory._strategies.get(strategy_name.lower())
        if strategy_class:
            if strategy_name.lower() == 'gpu':
                return strategy_class(config, logger, gpu_initialized)
            return strategy_class(config, logger)
        else:
            logger.warning(f"Không tìm thấy chiến lược cloaking: {strategy_name}")
            return None
