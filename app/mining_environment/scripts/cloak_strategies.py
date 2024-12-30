# cloak_strategies.py

import os
import subprocess
import psutil
import pynvml
import logging
from retrying import retry
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

class CloakStrategy(ABC):
    """
    Base class for different cloaking strategies.
    """
    @abstractmethod
    def apply(self, process: Any) -> Dict[str, Any]:
        """
        Apply the cloaking strategy to the given process.

        Args:
            process (Any): The process object.

        Returns:
            Dict[str, Any]: Adjustments to be applied by ResourceManager.
        """
        pass

class CpuCloakStrategy(CloakStrategy):
    """
    Cloaking strategy for CPU.
    Throttles CPU frequency and reduces CPU threads if needed.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.throttle_percentage = config.get('throttle_percentage', 20)
        if not isinstance(self.throttle_percentage, (int, float)) or not (0 <= self.throttle_percentage <= 100):
            logger.warning("Invalid throttle_percentage, defaulting to 20%.")
            self.throttle_percentage = 20

        self.freq_adjustment = config.get('frequency_adjustment_mhz', 2000)
        if not isinstance(self.freq_adjustment, int) or self.freq_adjustment <= 0:
            logger.warning("Invalid frequency_adjustment_mhz, defaulting to 2000MHz.")
            self.freq_adjustment = 2000

        self.logger = logger

    def apply(self, process: Any) -> Dict[str, Any]:
        """
        Apply CPU throttling to the given process.

        Args:
            process (Any): Process object with attributes 'pid' and 'name'.

        Returns:
            Dict[str, Any]: Adjustments including CPU frequency and throttle percentage.
        """
        if not hasattr(process, 'pid') or not hasattr(process, 'name') or not process.name:
            self.logger.error("Process object is missing required attributes (pid, name).")
            return {}

        try:
            adjustments = {
                'cpu_freq': self.freq_adjustment,
                'throttle_percentage': self.throttle_percentage,
            }
            self.logger.info(
                f"Prepared CPU throttling adjustments: freq={self.freq_adjustment}MHz "
                f"({self.throttle_percentage}% reduction) for process {process.name} (PID: {process.pid})."
            )

            # Apply adjustments
            self.apply_adjustments(process.pid, adjustments)
            return adjustments

        except Exception as e:
            self.logger.error(
                f"Error preparing CPU throttling for process {getattr(process, 'name', 'unknown')} "
                f"(PID: {getattr(process, 'pid', 'N/A')}): {e}"
            )
            raise

    def apply_adjustments(self, pid: int, adjustments: Dict[str, Any]):
        """
        Apply CPU frequency adjustments to the process.

        Args:
            pid (int): Process ID.
            adjustments (Dict[str, Any]): Adjustments dictionary.
        """
        try:
            cpu_freq = adjustments.get('cpu_freq')
            throttle_percentage = adjustments.get('throttle_percentage')

            self.logger.info(
                f"Applying CPU frequency={cpu_freq}MHz and throttle_percentage={throttle_percentage}% "
                f"to process with PID={pid}."
            )
            # Placeholder: Replace with actual system calls or API
        except Exception as e:
            self.logger.error(f"Error applying CPU adjustments to PID={pid}: {e}")

    def set_cpu_affinity(self, pid: int, cores: List[int]):
        """
        Set CPU affinity for the given process.

        Args:
            pid (int): Process ID.
            cores (List[int]): List of CPU cores to assign the process to.
        """
        try:
            if not cores or not isinstance(cores, list):
                self.logger.error("Invalid cores list for CPU affinity.")
                return

            cores_str = ",".join(map(str, cores))
            self.logger.info(f"Setting CPU affinity for PID={pid} to cores: {cores_str}")

            # Use taskset to set CPU affinity
            subprocess.run(['taskset', '-cp', cores_str, str(pid)], check=True)
            self.logger.info(f"Successfully set CPU affinity for PID={pid} to cores: {cores_str}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error setting CPU affinity with taskset for PID={pid}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in set_cpu_affinity for PID={pid}: {e}")

class GpuCloakStrategy(CloakStrategy):
    """
    Cloaking strategy for GPU.
    Throttles GPU power limit.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, gpu_initialized: bool):
        self.throttle_percentage = config.get('throttle_percentage', 20)
        self.logger = logger
        self.gpu_initialized = gpu_initialized

        # Log thông tin khởi tạo
        self.logger.debug(f"GpuCloakStrategy initialized with throttle_percentage={self.throttle_percentage}")

    def apply(self, process: Any) -> Dict[str, Any]:
        """
        Apply GPU throttling to the given process.

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
            # Kiểm tra các thuộc tính cần thiết của process
            if not hasattr(process, 'pid') or not hasattr(process, 'name'):
                self.logger.error("Process object is missing required attributes (pid, name).")
                return {}

            # Khởi tạo NVML
            pynvml.nvmlInit()
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                if gpu_count == 0:
                    self.logger.warning("No GPUs found on the system.")
                    return {}

                # Gán GPU cho process
                gpu_index = self.assign_gpu(process.pid, gpu_count)
                if gpu_index == -1:
                    self.logger.warning(
                        f"No GPU assigned to process {process.name} (PID: {process.pid})."
                    )
                    return {}

                # Lấy giới hạn nguồn hiện tại và tính giới hạn mới
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                current_power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                new_power_limit = int(current_power_limit * (1 - self.throttle_percentage / 100))

                adjustments = {
                    'gpu_index': gpu_index,
                    'gpu_power_limit': new_power_limit
                }

                # Log thông tin điều chỉnh
                self.logger.info(
                    f"Prepared GPU throttling adjustments: GPU {gpu_index} power limit={new_power_limit} "
                    f"({self.throttle_percentage}% reduction) for process {process.name} (PID: {process.pid})."
                )
                return adjustments
            finally:
                # Tắt NVML sau khi sử dụng
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
            # Gán GPU dựa trên PID modulo số lượng GPU
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
            if not self.network_interface:
                self.logger.warning("Failed to determine network interface. Defaulting to 'eth0'.")
                self.network_interface = "eth0"
            self.logger.info(f"Primary network interface determined: {self.network_interface}")

    def get_primary_network_interface(self) -> str:
        """
        Determine the primary network interface using `ip route`.

        Returns:
            str: The name of the primary network interface or 'eth0' as fallback.
        """
        try:
            output = subprocess.check_output(['ip', 'route']).decode()
            for line in output.splitlines():
                if line.startswith('default'):
                    return line.split()[4]
            self.logger.warning("No default route found. Falling back to 'eth0'.")
            return 'eth0'
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running 'ip route': {e}")
            return 'eth0'
        except Exception as e:
            self.logger.error(f"Unexpected error getting primary network interface: {e}")
            return 'eth0'

    def apply(self, process: Any) -> Dict[str, Any]:
        """
        Apply network throttling adjustments.

        Args:
            process (Any): The process object with attributes 'name', 'pid', and optionally 'mark'.

        Returns:
            Dict[str, Any]: Adjustments including network interface and bandwidth limit.
        """
        try:
            # Kiểm tra process hợp lệ
            if not hasattr(process, 'pid') or not hasattr(process, 'name'):
                raise ValueError("Process object is missing required attributes (pid, name).")

            if not isinstance(process, object):
                raise TypeError("Process must be an object with necessary attributes.")

            # Lấy giá trị mark nếu tồn tại
            mark_value = getattr(process, 'mark', None)

            adjustments = {
                'network_interface': self.network_interface,
                'bandwidth_limit_mbps': self.bandwidth_reduction_mbps,
            }

            # Nếu process có 'mark', thêm vào adjustments
            if mark_value is not None:
                adjustments['process_mark'] = mark_value

            self.logger.info(
                f"Prepared Network throttling adjustments: interface={self.network_interface}, "
                f"bandwidth_limit={self.bandwidth_reduction_mbps}Mbps, mark={mark_value} "
                f"for process {process.name} (PID: {process.pid})."
            )
            return adjustments
        except ValueError as e:
            self.logger.error(f"ValueError in apply: {e}")
            raise
        except TypeError as e:
            self.logger.error(f"TypeError in apply: {e}")
            raise
        except Exception as e:
            self.logger.error(
                f"Unexpected error preparing Network throttling for process {getattr(process, 'name', 'unknown')} "
                f"(PID: {getattr(process, 'pid', 'N/A')}): {e}"
            )
            raise

class DiskIoCloakStrategy(CloakStrategy):
    """
    Cloaking strategy for Disk I/O.
    Sets I/O throttling level for the process.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.io_throttling_level = config.get('io_throttling_level', 'idle').lower()
        if self.io_throttling_level not in {'idle', 'normal'}:
            logger.warning(f"Invalid io_throttling_level: {self.io_throttling_level}. Defaulting to 'idle'.")
            self.io_throttling_level = 'idle'
        self.logger = logger

    def apply(self, process: Any) -> Dict[str, Any]:
        """
        Apply Disk I/O throttling adjustments.

        Args:
            process (Any): Process object with attributes 'name' and 'pid'.

        Returns:
            Dict[str, Any]: Adjustments including ionice class.
        """
        try:
            # Kiểm tra process có các thuộc tính cần thiết
            if not hasattr(process, 'pid') or not hasattr(process, 'name'):
                raise ValueError("Process object is missing required attributes (pid, name).")

            # Xác định giá trị ionice_class
            ionice_class = '3' if self.io_throttling_level == 'idle' else '2'

            # Tạo adjustments
            adjustments = {
                'ionice_class': ionice_class
            }

            # Log thông tin
            self.logger.info(
                f"Prepared Disk I/O throttling adjustments: ionice_class={ionice_class} "
                f"for process {process.name} (PID: {process.pid})."
            )

            return adjustments
        except ValueError as e:
            self.logger.error(f"ValueError in apply: {e}")
            raise
        except Exception as e:
            self.logger.error(
                f"Error preparing Disk I/O throttling for process {getattr(process, 'name', 'unknown')} "
                f"(PID: {getattr(process, 'pid', 'N/A')}): {e}"
            )
            raise

class CacheCloakStrategy(CloakStrategy):
    """
    Cloaking strategy for Cache.
    Reduces cache usage by dropping caches.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.cache_limit_percent = config.get('cache_limit_percent', 50)
        if not (0 <= self.cache_limit_percent <= 100):
            logger.warning(
                f"Invalid cache_limit_percent: {self.cache_limit_percent}. Defaulting to 50%."
            )
            self.cache_limit_percent = 50
        self.logger = logger

    def apply(self, process: Any) -> Dict[str, Any]:
        """
        Apply Cache throttling adjustments.

        Args:
            process (Any): Process object with attributes 'name' and 'pid'.

        Returns:
            Dict[str, Any]: Adjustments including cache limit and drop cache flag.
        """
        try:
            # Kiểm tra process hợp lệ
            if not hasattr(process, 'pid') or not hasattr(process, 'name'):
                raise ValueError("Process object is missing required attributes (pid, name).")

            # Kiểm tra quyền
            if os.geteuid() != 0:
                self.logger.error(
                    f"Insufficient permissions to drop caches. Cache throttling failed for process {process.name} (PID: {process.pid})."
                )
                return {}

            # Tạo adjustments
            adjustments = {
                'drop_caches': True,
                'cache_limit_percent': self.cache_limit_percent
            }

            # Log thông tin
            self.logger.info(
                f"Prepared Cache throttling adjustments: drop_caches=True, "
                f"cache_limit_percent={self.cache_limit_percent}% "
                f"for process {process.name} (PID: {process.pid})."
            )

            return adjustments
        except ValueError as e:
            self.logger.error(f"ValueError in apply: {e}")
            raise
        except PermissionError:
            self.logger.error(
                f"Insufficient permissions to drop caches. Cache throttling failed for process {process.name} (PID: {process.pid})."
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Error preparing Cache throttling for process {getattr(process, 'name', 'unknown')} "
                f"(PID: {getattr(process, 'pid', 'N/A')}): {e}"
            )
            raise

class CloakStrategyFactory:
    """
    Factory to create instances of cloaking strategies.
    """
    _strategies: Dict[str, Type[CloakStrategy]] = {
        'cpu': CpuCloakStrategy,
        'gpu': GpuCloakStrategy,
        'network': NetworkCloakStrategy,
        'disk_io': DiskIoCloakStrategy,
        'cache': CacheCloakStrategy
    }

    @staticmethod
    def create_strategy(strategy_name: str, config: Dict[str, Any],
                        logger: logging.Logger, gpu_initialized: bool = False
    ) -> Optional[CloakStrategy]:
        """
        Create a cloaking strategy instance based on the strategy name.

        Args:
            strategy_name (str): Name of the strategy (e.g., 'cpu', 'gpu').
            config (Dict[str, Any]): Configuration for the strategy.
            logger (logging.Logger): Logger for the strategy.
            gpu_initialized (bool): GPU initialization status (used for GPU strategies).

        Returns:
            Optional[CloakStrategy]: An instance of the requested cloaking strategy, or None if not found.
        """
        strategy_class = CloakStrategyFactory._strategies.get(strategy_name.lower())

        # Check if the strategy is valid
        if strategy_class and issubclass(strategy_class, CloakStrategy):
            try:
                if strategy_name.lower() == 'gpu':
                    logger.debug(f"Creating GPU CloakStrategy: gpu_initialized={gpu_initialized}")
                    return strategy_class(config, logger, gpu_initialized)
                return strategy_class(config, logger)
            except Exception as e:
                logger.error(f"Error creating strategy '{strategy_name}': {e}")
                raise
        else:
            logger.warning(f"Không tìm thấy chiến lược cloaking: {strategy_name}")
            return None
