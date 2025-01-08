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
            cgroups (Dict[str, str]): Dictionary chứa tên các cgroup cho các controller (ví dụ: {'cpu': 'priority_cpu'}).
        """
        pass

class CpuCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking CPU:
      - Giới hạn sử dụng CPU thông qua cgroups (quota).
      - Tối ưu hóa việc sử dụng cache CPU.
      - Đặt affinity cho các thread vào các core CPU cụ thể.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        # Lấy cấu hình throttle_percentage với giá trị mặc định là 20%
        self.throttle_percentage = config.get('throttle_percentage', 20)
        if not isinstance(self.throttle_percentage, (int, float)) or not (0 <= self.throttle_percentage <= 100):
            logger.warning("Giá trị throttle_percentage không hợp lệ, mặc định 20%.")
            self.throttle_percentage = 20

        # Lấy cấu hình cpu_shares với giá trị mặc định là 1024
        self.cpu_shares = config.get('cpu_shares', 1024)
        if not isinstance(self.cpu_shares, int) or self.cpu_shares <= 0:
            logger.warning("Giá trị cpu_shares không hợp lệ, mặc định 1024.")
            self.cpu_shares = 1024

        self.logger = logger

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

            # Tính toán quota CPU
            cpu_quota_us = self.calculate_cpu_quota()

            # Thiết lập quota CPU bằng cách sử dụng cgroups
            subprocess.run(['cgset', '-r', f'cpu.cfs_quota_us={cpu_quota_us}', cpu_cgroup], check=True)
            self.logger.info(
                f"Đặt CPU quota là {cpu_quota_us}us cho tiến trình {process_name} (PID: {pid}) trong cgroup '{cpu_cgroup}'."
            )

            # Tối ưu hóa việc sử dụng cache bằng cách đặt độ ưu tiên tiến trình
            self.optimize_cache(pid)

            # Đặt affinity cho thread
            self.set_thread_affinity(pid, cgroups.get('cpuset'))

        except Exception as e:
            self.logger.error(
                f"Lỗi khi áp dụng cloaking CPU cho tiến trình {getattr(process, 'name', 'unknown')} "
                f"(PID: {getattr(process, 'pid', 'N/A')}): {e}\n{traceback.format_exc()}"
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

    def set_thread_affinity(self, pid: int, cpuset_cgroup: Optional[str]) -> None:
        """
        Đặt affinity cho thread của tiến trình bằng cách cấu hình cgroup cpuset.

        Args:
            pid (int): PID của tiến trình.
            cpuset_cgroup (Optional[str]): Tên của cgroup cpuset.
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

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, gpu_initialized: bool):
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
        self.gpu_initialized = self.gpu_manager.gpu_initialized

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

        except Exception as e:
            self.logger.error(
                f"Lỗi khi áp dụng cloaking GPU cho tiến trình {getattr(process, 'name', 'unknown')} "
                f"(PID: {getattr(process, 'pid', 'N/A')}): {e}\n{traceback.format_exc()}"
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

class NetworkCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking mạng:
      - Giảm băng thông mạng cho tiến trình.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        # Lấy cấu hình bandwidth_reduction_mbps với giá trị mặc định là 10Mbps
        self.bandwidth_reduction_mbps = config.get('bandwidth_reduction_mbps', 10)
        # Lấy cấu hình network_interface hoặc tự động xác định
        self.network_interface = config.get('network_interface')
        self.logger = logger

        if not self.network_interface:
            self.network_interface = self.get_primary_network_interface()
            if not self.network_interface:
                self.logger.warning("Không thể xác định giao diện mạng. Mặc định là 'eth0'.")
                self.network_interface = "eth0"
            self.logger.info(f"Giao diện mạng chính xác định: {self.network_interface}")

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

    def apply(self, process: Any, cgroups: Dict[str, str]) -> None:
        """
        Áp dụng chiến lược cloaking mạng cho tiến trình đã cho trong cgroup đã chỉ định.

        Args:
            process (Any): Đối tượng tiến trình.
            cgroups (Dict[str, str]): Dictionary chứa tên các cgroup cho các controller.
        """
        try:
            if not hasattr(process, 'pid') or not hasattr(process, 'name'):
                self.logger.error("Đối tượng tiến trình thiếu thuộc tính cần thiết (pid, name).")
                raise AttributeError("Đối tượng tiến trình thiếu thuộc tính cần thiết (pid, name).")

            pid = process.pid
            process_name = getattr(process, 'name', 'unknown')

            # Placeholder: Thực hiện giới hạn băng thông bằng cách sử dụng `tc` hoặc các công cụ tương tự
            # Ví dụ sử dụng `tc` để giới hạn băng thông
            # Lưu ý: Cần quyền root

            # Định nghĩa fwmark cho tiến trình cụ thể
            mark = pid % 32768  # fwmark phải < 65536
            self.logger.debug(f"Đặt fwmark={mark} cho tiến trình PID={pid}.")

            # Thêm quy tắc iptables để đánh dấu các gói tin từ PID này
            subprocess.run(['iptables', '-A', 'OUTPUT', '-m', 'owner', '--pid-owner', str(pid), '-j', 'MARK', '--set-mark', str(mark)], check=True)
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

class DiskIoCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking Disk I/O:
      - Đặt mức độ throttling I/O bằng cách sử dụng ionice.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        # Lấy cấu hình io_throttling_level với giá trị mặc định là 'idle'
        self.io_throttling_level = config.get('io_throttling_level', 'idle').lower()
        if self.io_throttling_level not in {'idle', 'best-effort', 'realtime'}:
            logger.warning(
                f"Giá trị io_throttling_level không hợp lệ: {self.io_throttling_level}. Mặc định là 'idle'."
            )
            self.io_throttling_level = 'idle'
        self.logger = logger

    def apply(self, process: Any, cgroups: Dict[str, str]) -> None:
        """
        Áp dụng chiến lược cloaking Disk I/O cho tiến trình đã cho trong cgroup đã chỉ định.

        Args:
            process (Any): Đối tượng tiến trình.
            cgroups (Dict[str, str]): Dictionary chứa tên các cgroup cho các controller.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Bản đồ io_throttling_level tới lớp ionice tương ứng
            io_class_map = {
                'idle': '3',
                'best-effort': '2',
                'realtime': '1'
            }
            ionice_class = io_class_map.get(self.io_throttling_level, '3')

            # Áp dụng lớp ionice cho tiến trình
            subprocess.run(['ionice', '-c', ionice_class, '-p', str(pid)], check=True)
            self.logger.info(
                f"Đặt ionice class={ionice_class} cho tiến trình {process_name} (PID: {pid})."
            )

        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"Lỗi khi áp dụng cloaking Disk I/O cho tiến trình {getattr(process, 'name', 'unknown')} "
                f"(PID: {getattr(process, 'pid', 'N/A')}): {e}"
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Lỗi bất ngờ khi áp dụng cloaking Disk I/O cho tiến trình {getattr(process, 'name', 'unknown')} "
                f"(PID: {getattr(process, 'pid', 'N/A')}): {e}\n{traceback.format_exc()}"
            )
            raise

class CacheCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking Cache:
      - Giảm sử dụng cache bằng cách drop caches.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        # Lấy cấu hình cache_limit_percent với giá trị mặc định là 50%
        self.cache_limit_percent = config.get('cache_limit_percent', 50)
        if not (0 <= self.cache_limit_percent <= 100):
            logger.warning(
                f"Giá trị cache_limit_percent không hợp lệ: {self.cache_limit_percent}. Mặc định là 50%."
            )
            self.cache_limit_percent = 50
        self.logger = logger

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
    }

    @staticmethod
    def create_strategy(strategy_name: str, config: Dict[str, Any],
                        logger: logging.Logger, gpu_initialized: bool = False
    ) -> Optional[CloakStrategy]:
        """
        Tạo một instance của chiến lược cloaking dựa trên tên chiến lược.

        Args:
            strategy_name (str): Tên của chiến lược (ví dụ: 'cpu', 'gpu').
            config (Dict[str, Any]): Cấu hình cho chiến lược.
            logger (logging.Logger): Logger cho chiến lược.
            gpu_initialized (bool): Trạng thái khởi tạo GPU (sử dụng cho các chiến lược GPU).

        Returns:
            Optional[CloakStrategy]: Một instance của chiến lược cloaking được yêu cầu, hoặc None nếu không tìm thấy.
        """
        strategy_class = CloakStrategyFactory._strategies.get(strategy_name.lower())

        if strategy_class and issubclass(strategy_class, CloakStrategy):
            try:
                if strategy_name.lower() == 'gpu':
                    logger.debug(f"Tạo GPU CloakStrategy: gpu_initialized={gpu_initialized}")
                    return strategy_class(config, logger, gpu_initialized)
                return strategy_class(config, logger)
            except Exception as e:
                logger.error(f"Lỗi khi tạo chiến lược '{strategy_name}': {e}")
                raise
        else:
            logger.warning(f"Không tìm thấy chiến lược cloaking: {strategy_name}")
            return None
