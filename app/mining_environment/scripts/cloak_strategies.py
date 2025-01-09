# cloak_strategies.py

import os
import subprocess
import psutil
import pynvml
import logging
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Type
from .utils import GPUManager
from .cgroup_manager import CgroupManager  # Import CgroupManager


class CloakStrategy(ABC):
    """
    Lớp cơ sở cho các chiến lược cloaking khác nhau.
    """

    @abstractmethod
    def apply(self, process: Any, cgroups: Dict[str, str]) -> None:
        """
        Áp dụng chiến lược cloaking cho tiến trình đã cho trong các cgroup được chỉ định.

        Args:
            process (Any): Đối tượng tiến trình.
            cgroups (Dict[str, str]): Dictionary chứa tên các cgroup cho các controller 
                                       (ví dụ: {'cpu': 'priority_cpu'}).
        """
        pass

    @abstractmethod
    def restore(self, process: Any, original_limits: Dict[str, Any]) -> None:
        """
        Khôi phục lại các giới hạn tài nguyên ban đầu cho tiến trình đã cho.

        Args:
            process (Any): Đối tượng tiến trình.
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

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, cgroup_manager: CgroupManager):
        """
        Khởi tạo CpuCloakStrategy với cấu hình, logger và CgroupManager.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking CPU.
            logger (logging.Logger): Logger để ghi log.
            cgroup_manager (CgroupManager): Instance của CgroupManager để thao tác với cgroup.
        """
        # Lấy cấu hình throttle_percentage với giá trị mặc định là 20%
        self.throttle_percentage = config.get('throttle_percentage', 20)
        if not isinstance(self.throttle_percentage, (int, float)) or not (0 <= self.throttle_percentage <= 100):
            logger.warning("Giá trị throttle_percentage không hợp lệ, mặc định 20%.")
            self.throttle_percentage = 20

        # Lấy cấu hình cpu_shares với giá trị mặc định là 1024 (nếu cần thiết)
        self.cpu_shares = config.get('cpu_shares', 1024)
        if not isinstance(self.cpu_shares, int) or self.cpu_shares <= 0:
            logger.warning("Giá trị cpu_shares không hợp lệ, mặc định 1024.")
            self.cpu_shares = 1024

        self.logger = logger
        self.cgroup_manager = cgroup_manager

    def apply(self, process: Any, cgroups: Dict[str, str]) -> None:
        """
        Áp dụng chiến lược cloaking CPU cho tiến trình đã cho trong cgroup đã chỉ định.

        Args:
            process (Any): Đối tượng tiến trình.
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

                # Thêm controller 'cpu' vào cgroup cha (giả sử 'root')
                parent_cgroup = "root"  # Thay đổi nếu cần
                success = self.cgroup_manager.add_cgroup_to_parent(parent_cgroup, cpu_cgroup)
                if not success:
                    self.logger.error(f"Không thể thêm cgroup CPU '{cpu_cgroup}' vào cgroup cha '{parent_cgroup}'.")
                    return

            # Tính toán quota CPU
            cpu_quota_us = self.calculate_cpu_quota()

            # Thiết lập quota CPU bằng cách sử dụng CgroupManager
            success = self.cgroup_manager.set_cpu_quota(cpu_cgroup, quota=cpu_quota_us, period=100000)
            if success:
                self.logger.info(
                    f"Đặt CPU quota là {cpu_quota_us}us cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cpu_cgroup}'."
                )
            else:
                self.logger.error(
                    f"Không thể đặt CPU quota là {cpu_quota_us}us cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cpu_cgroup}'."
                )
                return

            # Tối ưu hóa việc sử dụng cache bằng cách đặt độ ưu tiên tiến trình
            self.optimize_cache(pid)

            # Đặt affinity cho thread
            self.set_thread_affinity(pid, cgroups.get('cpuset'))

            # Gán tiến trình vào cgroup CPU
            success = self.cgroup_manager.assign_process_to_cgroup(pid, cpu_cgroup)
            if success:
                self.logger.info(f"Gán tiến trình PID={pid} vào cgroup '{cpu_cgroup}' cho CPU.")
            else:
                self.logger.error(f"Không thể gán tiến trình PID={pid} vào cgroup '{cpu_cgroup}' cho CPU.")

            adjustments = {
                'cpu_cloak': True
            }

            self.logger.info(
                f"Áp dụng cloaking CPU cho tiến trình {process_name} (PID: {pid}): throttle_percentage={self.throttle_percentage}%."
            )

        except Exception as e:
            self.logger.error(
                f"Lỗi khi áp dụng cloaking CPU cho tiến trình {getattr(process, 'name', 'unknown')} "
                f"(PID: {getattr(process, 'pid', 'N/A')}): {e}\n{traceback.format_exc()}"
            )
            raise

    def restore(self, process: Any, original_limits: Dict[str, Any]) -> None:
        """
        Khôi phục lại các giới hạn CPU ban đầu cho tiến trình đã cho.

        Args:
            process (Any): Đối tượng tiến trình.
            original_limits (Dict[str, Any]): Dictionary chứa các giới hạn CPU ban đầu.
        """
        try:
            pid = getattr(process, 'pid', None)
            process_name = getattr(process, 'name', 'unknown')

            if not pid:
                self.logger.error("Tiến trình không có PID. Không thể khôi phục CPU cloaking.")
                return

            cpu_cgroup = original_limits.get('cgroup')
            original_cpu_quota_us = original_limits.get('cpu_quota_us')

            if not cpu_cgroup:
                self.logger.error(f"Không có thông tin cgroup CPU để khôi phục cho tiến trình {process_name} (PID: {pid}).")
                return

            # Khôi phục quota CPU
            if original_cpu_quota_us:
                success = self.cgroup_manager.set_cpu_quota(cpu_cgroup, quota=original_cpu_quota_us, period=100000)
                if success:
                    self.logger.info(
                        f"Khôi phục CPU quota là {original_cpu_quota_us}us cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cpu_cgroup}'."
                    )
                else:
                    self.logger.error(
                        f"Không thể khôi phục CPU quota là {original_cpu_quota_us}us cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cpu_cgroup}'."
                    )

            # Khôi phục affinity CPU
            if original_limits.get('cpu_affinity') and cgroups.get('cpuset'):
                self.set_thread_affinity(pid, cgroups.get('cpuset'), original_limits.get('cpu_threads'))
                self.logger.info(
                    f"Khôi phục CPU affinity cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cgroups.get('cpuset')}'."
                )

            # Xóa cgroup CPU nếu cần
            self.cgroup_manager.delete_cgroup(cpu_cgroup)
            self.logger.info(f"Đã xóa cgroup CPU '{cpu_cgroup}' cho tiến trình {process_name} (PID: {pid}).")

        except Exception as e:
            self.logger.error(
                f"Lỗi khi khôi phục cloaking CPU cho tiến trình {process_name} (PID: {pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def get_process_info(self, process: Any) -> Tuple[int, str]:
        """
        Lấy PID và tên tiến trình từ đối tượng process.

        Args:
            process (Any): Đối tượng tiến trình.

        Returns:
            Tuple[int, str]: PID và tên tiến trình.
        """
        if isinstance(process, subprocess.Popen):
            pid = process.pid
            try:
                p = psutil.Process(pid)
                process_name = p.name()
            except psutil.NoSuchProcess:
                process_name = "unknown"
        else:
            if not hasattr(process, 'pid'):
                self.logger.error("Đối tượng tiến trình không có thuộc tính 'pid'.")
                raise AttributeError("Đối tượng tiến trình không có thuộc tính 'pid'.")
            pid = process.pid

            process_name = getattr(process, 'name', None)
            if not process_name:
                self.logger.warning(
                    f"Tiến trình PID={pid} không có thuộc tính 'name'. Cố gắng lấy tên qua psutil."
                )
                try:
                    p = psutil.Process(pid)
                    process_name = p.name()
                except psutil.NoSuchProcess:
                    process_name = "unknown"
                    self.logger.warning(
                        f"Tiến trình PID={pid} không tồn tại. Sử dụng process_name='unknown'."
                    )
        return pid, process_name

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

    def set_thread_affinity(self, pid: int, cpuset_cgroup: Optional[str], cpu_threads: Optional[str] = None) -> None:
        """
        Đặt affinity cho thread của tiến trình bằng cách cấu hình cgroup cpuset.

        Args:
            pid (int): PID của tiến trình.
            cpuset_cgroup (Optional[str]): Tên của cgroup cpuset.
            cpu_threads (Optional[str]): Số lượng threads CPU ban đầu để khôi phục (nếu có).
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

            # Đặt CPU affinity cho tiến trình
            p = psutil.Process(pid)
            p.cpu_affinity(available_cpus)
            self.logger.info(
                f"Đặt CPU affinity cho tiến trình PID {pid} vào các core {available_cpus} trong cgroup '{cpuset_cgroup}'."
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

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, gpu_initialized: bool, cgroup_manager: CgroupManager):
        """
        Khởi tạo GpuCloakStrategy với cấu hình, logger, trạng thái GPU và CgroupManager.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking GPU.
            logger (logging.Logger): Logger để ghi log.
            gpu_initialized (bool): Trạng thái khởi tạo GPU (sử dụng cho các chiến lược GPU).
            cgroup_manager (CgroupManager): Instance của CgroupManager để thao tác với cgroup.
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
        self.gpu_manager = GPUManager()
        self.gpu_initialized = gpu_initialized
        self.cgroup_manager = cgroup_manager

        # Không cần khởi tạo NVML tại đây
        if self.gpu_initialized:
            self.logger.info("GPUManager đã được khởi tạo thành công. Sẵn sàng áp dụng cloaking GPU.")
        else:
            self.logger.warning("GPUManager chưa được khởi tạo. Các chức năng cloaking GPU sẽ bị vô hiệu hóa.")

    def apply(self, process: Any, cgroups: Dict[str, str]) -> None:
        """
        Áp dụng chiến lược cloaking GPU cho tiến trình đã cho trong cgroup đã chỉ định.

        Args:
            process (Any): Đối tượng tiến trình.
            cgroups (Dict[str, str]): Dictionary chứa tên các cgroup cho các controller.
        """
        if not self.gpu_initialized:
            self.logger.warning(
                f"GPUManager chưa được khởi tạo. Không thể áp dụng cloaking GPU cho tiến trình "
                f"{getattr(process, 'name', 'unknown')} (PID: {getattr(process, 'pid', 'N/A')})."
            )
            return

        try:
            pid, process_name = self.get_process_info(process)
            gpu_count = self.gpu_manager.gpu_count

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

            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

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

            # Đặt power limit mới thông qua GPUManager
            success = self.gpu_manager.set_gpu_power_limit(gpu_index, desired_power_limit_w)
            if success:
                self.logger.info(
                    f"Đặt power limit GPU là {desired_power_limit_w}W cho tiến trình {process_name} (PID: {pid}) trên GPU {gpu_index}."
                )
            else:
                self.logger.error(
                    f"Không thể đặt power limit GPU cho tiến trình {process_name} (PID: {pid}) trên GPU {gpu_index}."
                )

            # Tùy chọn: Đặt xung nhịp GPU (SM và Memory)
            try:
                pynvml.nvmlDeviceSetApplicationsClocks(handle, self.target_mem_clock, self.target_sm_clock)
                self.logger.info(
                    f"Đặt xung nhịp GPU: SM={self.target_sm_clock}MHz, MEM={self.target_mem_clock}MHz "
                    f"cho tiến trình {process_name} (PID: {pid}) trên GPU {gpu_index}."
                )
            except pynvml.NVMLError as e:
                self.logger.warning(
                    f"Không thể đặt xung nhịp GPU cho GPU {gpu_index} trên tiến trình {process_name} (PID: {pid}): {e}"
                )

            # Gán tiến trình vào cgroup GPU nếu cần (ví dụ)
            gpu_cgroup = cgroups.get('gpu')
            if gpu_cgroup:
                try:
                    # Tạo và cấu hình cgroup GPU nếu chưa tồn tại
                    if not self.cgroup_manager.cgroup_exists(gpu_cgroup):
                        created = self.cgroup_manager.create_cgroup(gpu_cgroup)
                        if not created:
                            self.logger.error(f"Không thể tạo cgroup GPU '{gpu_cgroup}' cho tiến trình {process_name} (PID: {pid}).")
                            return

                        # Thêm controller 'gpu' vào cgroup cha (giả sử 'root_gpu')
                        parent_cgroup = "root_gpu"  # Thay đổi nếu cần
                        success = self.cgroup_manager.add_cgroup_to_parent(parent_cgroup, gpu_cgroup)
                        if not success:
                            self.logger.error(f"Không thể thêm cgroup GPU '{gpu_cgroup}' vào cgroup cha '{parent_cgroup}'.")
                            return

                    # Thiết lập các tham số GPU nếu cần (ví dụ: 'gpu.max')
                    # Giả sử 'gpu.max' là một tham số tùy chỉnh
                    gpu_max_value = str(desired_power_limit_w * 1000)  # Convert W to mW nếu cần
                    success = self.cgroup_manager.set_cgroup_parameter(gpu_cgroup, 'gpu.max', gpu_max_value)
                    if success:
                        self.logger.info(
                            f"Đặt 'gpu.max' là {gpu_max_value} cho cgroup GPU '{gpu_cgroup}'."
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
                except Exception as e:
                    self.logger.error(
                        f"Lỗi khi thao tác với cgroup GPU '{gpu_cgroup}' cho tiến trình {process_name} (PID: {pid}): {e}"
                    )
                    raise
            else:
                self.logger.warning(f"Không có cgroup GPU được cung cấp cho tiến trình {process_name} (PID: {pid}).")

        except Exception as e:
            self.logger.error(
                f"Lỗi khi áp dụng cloaking GPU cho tiến trình {getattr(process, 'name', 'unknown')} "
                f"(PID: {getattr(process, 'pid', 'N/A')}): {e}\n{traceback.format_exc()}"
            )
            raise

    def restore(self, process: Any, original_limits: Dict[str, Any]) -> None:
        """
        Khôi phục lại các giới hạn GPU ban đầu cho tiến trình đã cho.

        Args:
            process (Any): Đối tượng tiến trình.
            original_limits (Dict[str, Any]): Dictionary chứa các giới hạn GPU ban đầu.
        """
        try:
            pid = getattr(process, 'pid', None)
            process_name = getattr(process, 'name', 'unknown')

            if not pid:
                self.logger.error("Tiến trình không có PID. Không thể khôi phục GPU cloaking.")
                return

            gpu_cgroup = original_limits.get('cgroup')
            original_power_limit_w = original_limits.get('gpu_power_limit')
            original_sm_clock = original_limits.get('target_sm_clock')
            original_mem_clock = original_limits.get('target_mem_clock')

            if not gpu_cgroup:
                self.logger.error(f"Không có thông tin cgroup GPU để khôi phục cho tiến trình {process_name} (PID: {pid}).")
                return

            gpu_count = self.gpu_manager.gpu_count
            if gpu_count == 0:
                self.logger.warning("Không tìm thấy GPU nào trên hệ thống.")
                return

            gpu_index = self.assign_gpu(pid, gpu_count)
            if gpu_index == -1:
                self.logger.warning(
                    f"Không thể gán GPU cho tiến trình {process_name} (PID: {pid}) để khôi phục."
                )
                return

            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

            # Khôi phục power limit
            if original_power_limit_w:
                try:
                    pynvml.nvmlDeviceSetPowerManagementLimit(handle, original_power_limit_w * 1000)  # Convert W to mW
                    self.logger.info(
                        f"Khôi phục power limit GPU là {original_power_limit_w}W cho tiến trình {process_name} (PID: {pid}) trên GPU {gpu_index}."
                    )
                except pynvml.NVMLError as e:
                    self.logger.error(
                        f"Không thể khôi phục power limit GPU cho GPU {gpu_index} trên tiến trình {process_name} (PID: {pid}): {e}"
                    )

            # Khôi phục xung nhịp GPU (SM và Memory) nếu cần
            if original_sm_clock and original_mem_clock:
                try:
                    pynvml.nvmlDeviceSetApplicationsClocks(handle, original_mem_clock, original_sm_clock)
                    self.logger.info(
                        f"Khôi phục xung nhịp GPU: SM={original_sm_clock}MHz, MEM={original_mem_clock}MHz "
                        f"cho tiến trình {process_name} (PID: {pid}) trên GPU {gpu_index}."
                    )
                except pynvml.NVMLError as e:
                    self.logger.warning(
                        f"Không thể khôi phục xung nhịp GPU cho GPU {gpu_index} trên tiến trình {process_name} (PID: {pid}): {e}"
                    )

            # Khôi phục affinity CPU nếu cần
            if original_limits.get('cpu_affinity') and cgroups.get('cpuset'):
                self.set_thread_affinity(pid, cgroups.get('cpuset'), original_limits.get('cpu_threads'))
                self.logger.info(
                    f"Khôi phục CPU affinity cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cgroups.get('cpuset')}'."
                )

            # Xóa cgroup GPU nếu cần
            self.cgroup_manager.delete_cgroup(gpu_cgroup)
            self.logger.info(f"Đã xóa cgroup GPU '{gpu_cgroup}' cho tiến trình {process_name} (PID: {pid}).")

        except Exception as e:
            self.logger.error(
                f"Lỗi khi khôi phục cloaking GPU cho tiến trình {process_name} (PID: {pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def assign_gpu(self, pid: int, gpu_count: int) -> int:
        """
        Gán một GPU cho tiến trình dựa trên PID.

        Args:
            pid (int): ID của tiến trình.
            gpu_count (int): Tổng số GPU.

        Returns:
            int: Chỉ số GPU hoặc -1 nếu gán thất bại.
        """
        try:
            return pid % gpu_count
        except Exception as e:
            self.logger.error(f"Lỗi khi gán GPU dựa trên PID: {e}")
            return -1

    def set_thread_affinity(self, pid: int, cpuset_cgroup: Optional[str], cpu_threads: Optional[str] = None) -> None:
        """
        Đặt affinity cho thread của tiến trình bằng cách cấu hình cgroup cpuset.

        Args:
            pid (int): PID của tiến trình.
            cpuset_cgroup (Optional[str]): Tên của cgroup cpuset.
            cpu_threads (Optional[str]): Số lượng threads CPU ban đầu để khôi phục (nếu có).
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

            # Đặt CPU affinity cho tiến trình
            p = psutil.Process(pid)
            p.cpu_affinity(available_cpus)
            self.logger.info(
                f"Đặt CPU affinity cho tiến trình PID {pid} vào các core {available_cpus} trong cgroup '{cpuset_cgroup}'."
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


class NetworkCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking mạng:
      - Giảm băng thông mạng cho tiến trình.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, cgroup_manager: CgroupManager):
        """
        Khởi tạo NetworkCloakStrategy với cấu hình, logger và CgroupManager.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking Network.
            logger (logging.Logger): Logger để ghi log.
            cgroup_manager (CgroupManager): Instance của CgroupManager để thao tác với cgroup.
        """
        # Lấy cấu hình bandwidth_reduction_mbps với giá trị mặc định là 10Mbps
        self.bandwidth_reduction_mbps = config.get('bandwidth_reduction_mbps', 10)
        # Lấy cấu hình network_interface hoặc tự động xác định
        self.network_interface = config.get('network_interface')
        self.logger = logger
        self.cgroup_manager = cgroup_manager

        if not self.network_interface:
            self.network_interface = self.get_primary_network_interface()
            if not self.network_interface:
                self.logger.warning("Không thể xác định giao diện mạng. Mặc định là 'eth0'.")
                self.network_interface = "eth0"
            self.logger.info(f"Giao diện mạng chính xác định: {self.network_interface}")

    def apply(self, process: Any, cgroups: Dict[str, str]) -> None:
        """
        Áp dụng chiến lược cloaking mạng cho tiến trình đã cho trong cgroup đã chỉ định.

        Args:
            process (Any): Đối tượng tiến trình.
            cgroups (Dict[str, str]): Dictionary chứa tên các cgroup cho các controller.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Định nghĩa fwmark cho tiến trình cụ thể
            mark = pid % 32768  # fwmark phải < 65536
            self.logger.debug(f"Đặt fwmark={mark} cho tiến trình PID={pid}.")

            # Thêm quy tắc iptables để đánh dấu các gói tin từ PID này
            subprocess.run([
                'iptables', '-A', 'OUTPUT', '-m', 'owner', '--pid-owner', str(pid),
                '-j', 'MARK', '--set-mark', str(mark)
            ], check=True)
            self.logger.info(f"Đặt iptables MARK cho tiến trình {process_name} (PID: {pid}) với mark={mark}.")

            # Thêm tc filter để giới hạn băng thông dựa trên mark
            tc_cmd = [
                'tc', 'filter', 'add', 'dev', self.network_interface, 'protocol', 'ip',
                'parent', '1:0', 'prio', '1', 'handle', str(mark), 'fw', 'flowid', '1:1'
            ]
            subprocess.run(tc_cmd, check=True)
            self.logger.info(f"Thêm tc filter cho mark={mark} trên giao diện '{self.network_interface}'.")

            # Thêm tc qdisc để giới hạn băng thông
            tc_qdisc_cmd = [
                'tc', 'qdisc', 'add', 'dev', self.network_interface, 'parent', '1:1',
                'handle', '10:', 'htb', 'default', '12'
            ]
            subprocess.run(tc_qdisc_cmd, check=True)
            self.logger.info(f"Thêm tc qdisc cho mark={mark} trên giao diện '{self.network_interface}'.")

            # Thêm tc class để đặt tốc độ
            tc_class_cmd = [
                'tc', 'class', 'add', 'dev', self.network_interface, 'parent', '10:', 'classid', '10:1',
                'htb', 'rate', f'{self.bandwidth_reduction_mbps}mbit'
            ]
            subprocess.run(tc_class_cmd, check=True)
            self.logger.info(f"Đặt tc class rate là {self.bandwidth_reduction_mbps}mbit cho mark={mark}.")

            adjustments = {
                'network_interface': self.network_interface,
                'bandwidth_limit_mbps': self.bandwidth_reduction_mbps,
                'fwmark': mark
            }

            self.logger.info(
                f"Áp dụng cloaking mạng cho tiến trình {process_name} (PID: {pid}): "
                f"Giới hạn băng thông={self.bandwidth_reduction_mbps}Mbps trên giao diện '{self.network_interface}'."
            )

            # Gán tiến trình vào cgroup Network nếu cần (giả sử cgroup 'network')
            network_cgroup = cgroups.get('network')
            if network_cgroup:
                try:
                    # Tạo và cấu hình cgroup Network nếu chưa tồn tại
                    if not self.cgroup_manager.cgroup_exists(network_cgroup):
                        created = self.cgroup_manager.create_cgroup(network_cgroup)
                        if not created:
                            self.logger.error(f"Không thể tạo cgroup Network '{network_cgroup}' cho tiến trình {process_name} (PID: {pid}).")
                            return

                        # Thêm controller 'net_cls' vào cgroup cha (giả sử 'root_network')
                        parent_cgroup = "root_network"  # Thay đổi nếu cần
                        success = self.cgroup_manager.add_cgroup_to_parent(parent_cgroup, network_cgroup)
                        if not success:
                            self.logger.error(f"Không thể thêm cgroup Network '{network_cgroup}' vào cgroup cha '{parent_cgroup}'.")
                            return

                    # Thiết lập các tham số Network nếu cần (ví dụ: 'net_cls.classid')
                    # Giả sử 'net_cls.classid' là một tham số tùy chỉnh
                    classid_value = f"{mark}"
                    success = self.cgroup_manager.set_cgroup_parameter(network_cgroup, 'net_cls.classid', str(classid_value))
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
                except Exception as e:
                    self.logger.error(
                        f"Lỗi khi thao tác với cgroup Network '{network_cgroup}' cho tiến trình {process_name} (PID: {pid}): {e}"
                    )
                    raise
            else:
                self.logger.warning(f"Không có cgroup Network được cung cấp cho tiến trình {process_name} (PID: {pid}).")

        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"Lỗi khi áp dụng cloaking mạng cho tiến trình {getattr(process, 'name', 'unknown')} "
                f"(PID: {getattr(process, 'pid', 'N/A')}): {e}"
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Lỗi bất ngờ khi áp dụng cloaking mạng cho tiến trình {getattr(process, 'name', 'unknown')} "
                f"(PID: {getattr(process, 'pid', 'N/A')}): {e}\n{traceback.format_exc()}"
            )
            raise

    def restore(self, process: Any, original_limits: Dict[str, Any]) -> None:
        """
        Khôi phục lại các giới hạn mạng ban đầu cho tiến trình đã cho.

        Args:
            process (Any): Đối tượng tiến trình.
            original_limits (Dict[str, Any]): Dictionary chứa các giới hạn mạng ban đầu.
        """
        try:
            pid = getattr(process, 'pid', None)
            process_name = getattr(process, 'name', 'unknown')

            if not pid:
                self.logger.error("Tiến trình không có PID. Không thể khôi phục Network cloaking.")
                return

            network_cgroup = original_limits.get('cgroup')  # Giả sử bạn lưu trữ cgroup nếu cần
            mark = original_limits.get('fwmark')

            if not mark:
                self.logger.error(f"Không có fwmark để khôi phục cho tiến trình {process_name} (PID: {pid}).")
                return

            # Xóa quy tắc iptables
            subprocess.run([
                'iptables', '-D', 'OUTPUT', '-m', 'owner', '--pid-owner', str(pid),
                '-j', 'MARK', '--set-mark', str(mark)
            ], check=True)
            self.logger.info(f"Xóa iptables MARK cho tiến trình {process_name} (PID: {pid}) với mark={mark}.")

            # Xóa tc filter
            tc_filter_cmd = [
                'tc', 'filter', 'del', 'dev', self.network_interface, 'protocol', 'ip',
                'parent', '1:0', 'prio', '1', 'handle', str(mark), 'fw', 'flowid', '1:1'
            ]
            subprocess.run(tc_filter_cmd, check=True)
            self.logger.info(f"Xóa tc filter cho mark={mark} trên giao diện '{self.network_interface}'.")

            # Xóa tc class
            tc_class_cmd = [
                'tc', 'class', 'del', 'dev', self.network_interface, 'parent', '10:', 'classid', '10:1'
            ]
            subprocess.run(tc_class_cmd, check=True)
            self.logger.info(f"Xóa tc class '10:1' trên giao diện '{self.network_interface}'.")

            # Xóa tc qdisc
            tc_qdisc_cmd = [
                'tc', 'qdisc', 'del', 'dev', self.network_interface, 'parent', '1:1',
                'handle', '10:', 'htb', 'default', '12'
            ]
            subprocess.run(tc_qdisc_cmd, check=True)
            self.logger.info(f"Xóa tc qdisc cho mark={mark} trên giao diện '{self.network_interface}'.")

            # Khôi phục cgroup Network nếu cần
            if network_cgroup:
                self.cgroup_manager.delete_cgroup(network_cgroup, controllers='net_cls,net_prio')
                self.logger.info(f"Đã xóa cgroup Network '{network_cgroup}' cho tiến trình {process_name} (PID: {pid}).")

        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"Lỗi khi khôi phục cloaking mạng cho tiến trình {process_name} (PID: {pid}): {e}"
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Lỗi bất ngờ khi khôi phục cloaking mạng cho tiến trình {process_name} (PID: {pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def get_primary_network_interface(self) -> str:
        """
        Xác định giao diện mạng chính bằng cách sử dụng lệnh `ip route`.

        Returns:
            str: Tên của giao diện mạng chính hoặc 'eth0' nếu không xác định được.
        """
        try:
            output = subprocess.check_output(['ip', 'route']).decode()
            for line in output.splitlines():
                if line.startswith('default'):
                    return line.split()[4]
            self.logger.warning("Không tìm thấy default route. Mặc định là 'eth0'.")
            return 'eth0'
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi chạy lệnh 'ip route': {e}")
            return 'eth0'
        except Exception as e:
            self.logger.error(f"Lỗi bất ngờ khi xác định giao diện mạng chính: {e}")
            return 'eth0'


class DiskIoCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking Disk I/O:
      - Đặt mức độ throttling I/O bằng cách sử dụng cgroups v2 (io.weight).
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, cgroup_manager: CgroupManager):
        """
        Khởi tạo DiskIoCloakStrategy với cấu hình, logger và CgroupManager.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking Disk I/O.
            logger (logging.Logger): Logger để ghi log.
            cgroup_manager (CgroupManager): Instance của CgroupManager để thao tác với cgroup.
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

    def apply(self, process: Any, cgroups: Dict[str, str]) -> None:
        """
        Áp dụng chiến lược cloaking Disk I/O cho tiến trình đã cho trong cgroup đã chỉ định.

        Args:
            process (Any): Đối tượng tiến trình.
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

                # Thêm controller 'io' vào cgroup cha (giả sử 'root_io')
                parent_cgroup = "root_io"  # Thay đổi nếu cần
                success = self.cgroup_manager.add_cgroup_to_parent(parent_cgroup, io_cgroup)
                if not success:
                    self.logger.error(f"Không thể thêm cgroup I/O '{io_cgroup}' vào cgroup cha '{parent_cgroup}'.")
                    return

            # Thiết lập giới hạn I/O bằng cách sử dụng CgroupManager
            success = self.cgroup_manager.set_io_limit(io_cgroup, self.io_weight)
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

            adjustments = {
                'disk_io_cloak': True
            }

            self.logger.info(
                f"Áp dụng cloaking Disk I/O cho tiến trình {process_name} (PID: {pid}): io_weight={self.io_weight}."
            )

        except Exception as e:
            self.logger.error(
                f"Lỗi khi áp dụng cloaking Disk I/O cho tiến trình {getattr(process, 'name', 'unknown')} "
                f"(PID: {getattr(process, 'pid', 'N/A')}): {e}\n{traceback.format_exc()}"
            )
            raise

    def restore(self, process: Any, original_limits: Dict[str, Any]) -> None:
        """
        Khôi phục lại các giới hạn Disk I/O ban đầu cho tiến trình đã cho.

        Args:
            process (Any): Đối tượng tiến trình.
            original_limits (Dict[str, Any]): Dictionary chứa các giới hạn Disk I/O ban đầu.
        """
        try:
            pid = getattr(process, 'pid', None)
            process_name = getattr(process, 'name', 'unknown')

            if not pid:
                self.logger.error("Tiến trình không có PID. Không thể khôi phục Disk I/O cloaking.")
                return

            io_cgroup = original_limits.get('cgroup')
            original_io_weight = original_limits.get('io_weight')

            if not io_cgroup:
                self.logger.error(f"Không có thông tin cgroup I/O để khôi phục cho tiến trình {process_name} (PID: {pid}).")
                return

            # Khôi phục I/O weight
            if original_io_weight:
                success = self.cgroup_manager.set_io_limit(io_cgroup, original_io_weight)
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

        except Exception as e:
            self.logger.error(
                f"Lỗi khi khôi phục cloaking Disk I/O cho tiến trình {process_name} (PID: {pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def get_process_info(self, process: Any) -> Tuple[int, str]:
        """
        Lấy PID và tên tiến trình từ đối tượng process.

        Args:
            process (Any): Đối tượng tiến trình.

        Returns:
            Tuple[int, str]: PID và tên tiến trình.
        """
        if isinstance(process, subprocess.Popen):
            pid = process.pid
            try:
                p = psutil.Process(pid)
                process_name = p.name()
            except psutil.NoSuchProcess:
                process_name = "unknown"
        else:
            if not hasattr(process, 'pid'):
                self.logger.error("Đối tượng tiến trình không có thuộc tính 'pid'.")
                raise AttributeError("Đối tượng tiến trình không có thuộc tính 'pid'.")
            pid = process.pid

            process_name = getattr(process, 'name', None)
            if not process_name:
                self.logger.warning(
                    f"Tiến trình PID={pid} không có thuộc tính 'name'. Cố gắng lấy tên qua psutil."
                )
                try:
                    p = psutil.Process(pid)
                    process_name = p.name()
                except psutil.NoSuchProcess:
                    process_name = "unknown"
                    self.logger.warning(
                        f"Tiến trình PID={pid} không tồn tại. Sử dụng process_name='unknown'."
                    )
        return pid, process_name

    def get_total_cache(self) -> int:
        """
        Lấy tổng lượng cache hiện có trên hệ thống.

        Returns:
            int: Tổng cache tính bằng bytes.
        """
        # Ví dụ: giả sử tổng cache là 8GB
        return 8 * 1024 * 1024 * 1024  # 8GB in bytes


class CacheCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking Cache:
      - Giảm sử dụng cache bằng cách drop caches và giới hạn mức sử dụng cache thông qua cgroups.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, cgroup_manager: CgroupManager):
        """
        Khởi tạo CacheCloakStrategy với cấu hình, logger và CgroupManager.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking Cache.
            logger (logging.Logger): Logger để ghi log.
            cgroup_manager (CgroupManager): Instance của CgroupManager để thao tác với cgroup.
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

    def apply(self, process: Any, cgroups: Dict[str, str]) -> None:
        """
        Áp dụng chiến lược cloaking Cache cho tiến trình đã cho trong cgroup đã chỉ định.

        Args:
            process (Any): Đối tượng tiến trình.
            cgroups (Dict[str, str]): Dictionary chứa tên các cgroup cho các controller.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Đảm bảo tiến trình đang chạy với quyền đủ để drop caches
            if os.geteuid() != 0:
                self.logger.error(
                    f"Không đủ quyền để drop caches. Cloaking Cache thất bại cho tiến trình {process_name} (PID: {pid})."
                )
                return

            # Drop caches bằng cách ghi vào /proc/sys/vm/drop_caches
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3\n')
            self.logger.info(f"Đã drop caches để giảm sử dụng cache cho tiến trình {process_name} (PID: {pid}).")

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

                        # Thêm controller 'cache' vào cgroup cha (giả sử 'root_cache')
                        parent_cgroup = "root_cache"  # Thay đổi nếu cần
                        success = self.cgroup_manager.add_cgroup_to_parent(parent_cgroup, cache_cgroup)
                        if not success:
                            self.logger.error(f"Không thể thêm cgroup Cache '{cache_cgroup}' vào cgroup cha '{parent_cgroup}'.")
                            return

                    cache_limit_bytes = int(self.cache_limit_percent / 100 * self.get_total_cache())
                    success = self.cgroup_manager.set_cgroup_parameter(
                        cache_cgroup, 'cache.max', str(cache_limit_bytes)
                    )
                    if success:
                        self.logger.info(
                            f"Đặt giới hạn cache thành {self.cache_limit_percent}% ({cache_limit_bytes} bytes) cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cache_cgroup}'."
                        )
                    else:
                        self.logger.error(
                            f"Không thể đặt giới hạn cache cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cache_cgroup}'."
                        )
                except Exception as e:
                    self.logger.error(
                        f"Lỗi khi thao tác với cgroup Cache '{cache_cgroup}' cho tiến trình {process_name} (PID: {pid}): {e}"
                    )
                    raise
            else:
                self.logger.warning(f"Không có cgroup 'cache' được cung cấp cho tiến trình {process_name} (PID: {pid}).")

            adjustments = {
                'drop_caches': True,
                'cache_limit_percent': self.cache_limit_percent
            }

            self.logger.info(
                f"Áp dụng cloaking Cache cho tiến trình {process_name} (PID: {pid}): "
                f"drop_caches=True, cache_limit_percent={self.cache_limit_percent}%."
            )

        except PermissionError:
            self.logger.error(
                f"Không đủ quyền để drop caches. Cloaking Cache thất bại cho tiến trình {process_name} (PID: {pid})."
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Lỗi bất ngờ khi áp dụng cloaking Cache cho tiến trình {getattr(process, 'name', 'unknown')} "
                f"(PID: {getattr(process, 'pid', 'N/A')}): {e}\n{traceback.format_exc()}"
            )
            raise

    def restore(self, process: Any, original_limits: Dict[str, Any]) -> None:
        """
        Khôi phục lại các giới hạn Cache ban đầu cho tiến trình đã cho.

        Args:
            process (Any): Đối tượng tiến trình.
            original_limits (Dict[str, Any]): Dictionary chứa các giới hạn Cache ban đầu.
        """
        try:
            pid = getattr(process, 'pid', None)
            process_name = getattr(process, 'name', 'unknown')

            if not pid:
                self.logger.error("Tiến trình không có PID. Không thể khôi phục Cache cloaking.")
                return

            cache_cgroup = original_limits.get('cgroup')
            original_cache_limit_percent = original_limits.get('cache_limit_percent')

            if not cache_cgroup:
                self.logger.error(f"Không có thông tin cgroup Cache để khôi phục cho tiến trình {process_name} (PID: {pid}).")
                return

            # Khôi phục cache limit
            if original_cache_limit_percent is not None:
                cache_limit_bytes = int(original_cache_limit_percent / 100 * self.get_total_cache())
                success = self.cgroup_manager.set_cgroup_parameter(
                    cache_cgroup, 'cache.max', str(cache_limit_bytes)
                )
                if success:
                    self.logger.info(
                        f"Khôi phục giới hạn cache thành {original_cache_limit_percent}% ({cache_limit_bytes} bytes) cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cache_cgroup}'."
                    )
                else:
                    self.logger.error(
                        f"Không thể khôi phục giới hạn cache cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cache_cgroup}'."
                    )

            # Xóa cgroup Cache nếu cần
            self.cgroup_manager.delete_cgroup(cache_cgroup)
            self.logger.info(f"Đã xóa cgroup Cache '{cache_cgroup}' cho tiến trình {process_name} (PID: {pid}).")

        except Exception as e:
            self.logger.error(
                f"Lỗi khi khôi phục cloaking Cache cho tiến trình {process_name} (PID: {pid}): {e}\n{traceback.format_exc()}"
            )
            raise


class MemoryCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking Memory:
      - Giới hạn sử dụng bộ nhớ của tiến trình bằng cách sử dụng cgroups.
      - Giảm sử dụng bộ nhớ bằng cách drop caches nếu cần thiết.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, cgroup_manager: CgroupManager):
        """
        Khởi tạo MemoryCloakStrategy với cấu hình, logger và CgroupManager.

        Args:
            config (Dict[str, Any]): Cấu hình cho chiến lược cloaking Memory.
            logger (logging.Logger): Logger để ghi log.
            cgroup_manager (CgroupManager): Instance của CgroupManager để thao tác với cgroup.
        """
        # Lấy cấu hình cache_limit_percent với giá trị mặc định là 50%
        self.cache_limit_percent = config.get('cache_limit_percent', 50)
        if not isinstance(self.cache_limit_percent, (int, float)) or not (0 <= self.cache_limit_percent <= 100):
            logger.warning(
                f"Giá trị cache_limit_percent không hợp lệ: {self.cache_limit_percent}. Mặc định là 50%."
            )
            self.cache_limit_percent = 50

        self.logger = logger
        self.cgroup_manager = cgroup_manager

    def apply(self, process: Any, cgroups: Dict[str, str]) -> None:
        """
        Áp dụng chiến lược cloaking Memory cho tiến trình đã cho trong cgroup đã chỉ định.

        Args:
            process (Any): Đối tượng tiến trình.
            cgroups (Dict[str, str]): Dictionary chứa tên các cgroup cho các controller.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Đảm bảo tiến trình đang chạy với quyền đủ để drop caches
            if os.geteuid() != 0:
                self.logger.error(
                    f"Không đủ quyền để drop caches. Cloaking Memory thất bại cho tiến trình {process_name} (PID: {pid})."
                )
                return

            # Drop caches bằng cách ghi vào /proc/sys/vm/drop_caches
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3\n')
            self.logger.info(f"Đã drop caches để giảm sử dụng cache cho tiến trình {process_name} (PID: {pid}).")

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

                        # Thêm controller 'memory' vào cgroup cha (giả sử 'root_memory')
                        parent_cgroup = "root_memory"  # Thay đổi nếu cần
                        success = self.cgroup_manager.add_cgroup_to_parent(parent_cgroup, cache_cgroup)
                        if not success:
                            self.logger.error(f"Không thể thêm cgroup Cache '{cache_cgroup}' vào cgroup cha '{parent_cgroup}'.")
                            return

                    # Thiết lập giới hạn bộ nhớ
                    memory_limit_bytes = self.calculate_memory_limit()
                    success = self.cgroup_manager.set_memory_limit(cache_cgroup, memory_limit_bytes)
                    if success:
                        self.logger.info(
                            f"Đặt giới hạn bộ nhớ thành {self.cache_limit_percent}% ({memory_limit_bytes} bytes) cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cache_cgroup}'."
                        )
                    else:
                        self.logger.error(
                            f"Không thể đặt giới hạn bộ nhớ cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cache_cgroup}'."
                        )
                        return

                    # Gán tiến trình vào cgroup Cache
                    success = self.cgroup_manager.assign_process_to_cgroup(pid, cache_cgroup)
                    if success:
                        self.logger.info(f"Gán tiến trình PID={pid} vào cgroup '{cache_cgroup}' cho Memory.")
                    else:
                        self.logger.error(f"Không thể gán tiến trình PID={pid} vào cgroup '{cache_cgroup}' cho Memory.")

                except Exception as e:
                    self.logger.error(
                        f"Lỗi khi thao tác với cgroup Cache '{cache_cgroup}' cho tiến trình {process_name} (PID: {pid}): {e}"
                    )
                    raise
            else:
                self.logger.warning(f"Không có cgroup 'cache' được cung cấp cho tiến trình {process_name} (PID: {pid}).")

            adjustments = {
                'memory_cloak': True
            }

            self.logger.info(
                f"Áp dụng cloaking Memory cho tiến trình {process_name} (PID: {pid}): "
                f"drop_caches=True, cache_limit_percent={self.cache_limit_percent}%."
            )

        except PermissionError:
            self.logger.error(
                f"Không đủ quyền để drop caches. Cloaking Memory thất bại cho tiến trình {process_name} (PID: {pid})."
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Lỗi bất ngờ khi áp dụng cloaking Memory cho tiến trình {getattr(process, 'name', 'unknown')} "
                f"(PID: {getattr(process, 'pid', 'N/A')}): {e}\n{traceback.format_exc()}"
            )
            raise

    def restore(self, process: Any, original_limits: Dict[str, Any]) -> None:
        """
        Khôi phục lại các giới hạn Memory ban đầu cho tiến trình đã cho.

        Args:
            process (Any): Đối tượng tiến trình.
            original_limits (Dict[str, Any]): Dictionary chứa các giới hạn Memory ban đầu.
        """
        try:
            pid = getattr(process, 'pid', None)
            process_name = getattr(process, 'name', 'unknown')

            if not pid:
                self.logger.error("Tiến trình không có PID. Không thể khôi phục Memory cloaking.")
                return

            cache_cgroup = original_limits.get('cgroup')
            original_memory_limit_bytes = original_limits.get('memory_limit_bytes')

            if not cache_cgroup:
                self.logger.error(f"Không có thông tin cgroup Memory để khôi phục cho tiến trình {process_name} (PID: {pid}).")
                return

            # Khôi phục giới hạn bộ nhớ
            if original_memory_limit_bytes:
                success = self.cgroup_manager.set_memory_limit(cache_cgroup, original_memory_limit_bytes)
                if success:
                    self.logger.info(
                        f"Khôi phục giới hạn bộ nhớ thành {original_memory_limit_bytes} bytes cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cache_cgroup}'."
                    )
                else:
                    self.logger.error(
                        f"Không thể khôi phục giới hạn bộ nhớ cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cache_cgroup}'."
                    )

            # Xóa cgroup Memory nếu cần
            self.cgroup_manager.delete_cgroup(cache_cgroup)
            self.logger.info(f"Đã xóa cgroup Memory '{cache_cgroup}' cho tiến trình {process_name} (PID: {pid}).")

        except Exception as e:
            self.logger.error(
                f"Lỗi khi khôi phục cloaking Memory cho tiến trình {process_name} (PID: {pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def set_memory_limit(self, cgroup: str, limit_bytes: int) -> bool:
        """
        Đặt giới hạn bộ nhớ cho cgroup.

        Args:
            cgroup (str): Tên của cgroup.
            limit_bytes (int): Giới hạn bộ nhớ tính bằng bytes.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            memory_max_path = f"/sys/fs/cgroup/memory/{cgroup}/memory.max"
            with open(memory_max_path, 'w') as f:
                f.write(str(limit_bytes))
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi đặt memory.max cho cgroup '{cgroup}': {e}")
            return False

    def get_process_info(self, process: Any) -> Tuple[int, str]:
        """
        Lấy PID và tên tiến trình từ đối tượng process.

        Args:
            process (Any): Đối tượng tiến trình.

        Returns:
            Tuple[int, str]: PID và tên tiến trình.
        """
        if isinstance(process, psutil.Process):
            pid = process.pid
            process_name = process.name()
        else:
            if not hasattr(process, 'pid'):
                self.logger.error("Đối tượng tiến trình không có thuộc tính 'pid'.")
                raise AttributeError("Đối tượng tiến trình không có thuộc tính 'pid'.")
            pid = process.pid

            process_name = getattr(process, 'name', None)
            if not process_name:
                self.logger.warning(
                    f"Tiến trình PID={pid} không có thuộc tính 'name'. Cố gắng lấy tên qua psutil."
                )
                try:
                    p = psutil.Process(pid)
                    process_name = p.name()
                except psutil.NoSuchProcess:
                    process_name = "unknown"
                    self.logger.warning(
                        f"Tiến trình PID={pid} không tồn tại. Sử dụng process_name='unknown'."
                    )
        return pid, process_name

    def calculate_memory_limit(self) -> int:
        """
        Tính toán giới hạn bộ nhớ dựa trên cache_limit_percent.

        Returns:
            int: Giới hạn bộ nhớ tính bằng bytes.
        """
        total_memory_bytes = psutil.virtual_memory().total
        memory_limit_bytes = int((self.cache_limit_percent / 100) * total_memory_bytes)
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
        'memory': MemoryCloakStrategy  # Nếu bạn có chiến lược riêng cho memory
    }

    @staticmethod
    def create_strategy(strategy_name: str, config: Dict[str, Any],
                        logger: logging.Logger, cgroup_manager: CgroupManager, gpu_initialized: bool = False
    ) -> Optional[CloakStrategy]:
        """
        Tạo một instance của chiến lược cloaking dựa trên tên chiến lược.
        """
        strategy_class = CloakStrategyFactory._strategies.get(strategy_name.lower())

        if strategy_class and issubclass(strategy_class, CloakStrategy):
            try:
                if strategy_name.lower() == 'gpu':
                    logger.debug(f"Tạo GPU CloakStrategy: gpu_initialized={gpu_initialized}")
                    return strategy_class(config, logger, gpu_initialized, cgroup_manager)
                elif strategy_name.lower() in {'cpu', 'cache', 'network', 'disk_io', 'memory'}:
                    return strategy_class(config, logger, cgroup_manager)
                else:
                    # Các chiến lược khác có thể cần hoặc không cần CgroupManager
                    return strategy_class(config, logger, cgroup_manager)
            except Exception as e:
                logger.error(f"Lỗi khi tạo chiến lược '{strategy_name}': {e}")
                raise
        else:
            logger.warning(f"Không tìm thấy chiến lược cloaking: {strategy_name}")
            return None
