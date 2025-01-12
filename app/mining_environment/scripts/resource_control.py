# resource_control.py

import os
import uuid
import logging
import subprocess
from typing import Any, Dict, List, Optional, Tuple
import psutil
import pynvml  # NVIDIA Management Library

# Import GPUManager từ utils.py
# Đảm bảo rằng GPUManager được định nghĩa đầy đủ trong utils.py
from .utils import GPUManager  # Import GPUManager từ utils.py


class CPUResourceManager:
    """
    Quản lý tài nguyên CPU thông qua việc sử dụng cgroups để điều chỉnh giới hạn CPU, affinity và tối ưu hóa sử dụng CPU.
    """

    CGROUP_BASE_PATH = "/sys/fs/cgroup/cpu_cloak"

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ensure_cgroup_base()
        self.process_cgroup: Dict[int, str] = {}  # Mapping PID to cgroup name

    def ensure_cgroup_base(self):
        """
        Đảm bảo rằng thư mục cơ sở cho các cgroup cloaking CPU tồn tại.
        """
        try:
            if not os.path.exists(self.CGROUP_BASE_PATH):
                os.makedirs(self.CGROUP_BASE_PATH, exist_ok=True)
                self.logger.debug(f"Tạo thư mục cgroup cơ sở tại {self.CGROUP_BASE_PATH}.")
        except PermissionError:
            self.logger.error(f"Không đủ quyền để tạo thư mục cgroup tại {self.CGROUP_BASE_PATH}.")
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo thư mục cgroup tại {self.CGROUP_BASE_PATH}: {e}")

    def get_available_cpus(self) -> List[int]:
        """
        Lấy danh sách các core CPU có sẵn để đặt affinity.

        Returns:
            List[int]: Danh sách các core CPU.
        """
        try:
            cpu_count = psutil.cpu_count(logical=True)
            available_cpus = list(range(cpu_count))
            self.logger.debug(f"Available CPUs: {available_cpus}.")
            return available_cpus
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy danh sách CPU cores: {e}")
            return []

    def create_cgroup(self, pid: int, throttle_percentage: float) -> Optional[str]:
        """
        Tạo một cgroup mới cho tiến trình và thiết lập giới hạn CPU.

        Args:
            pid (int): PID của tiến trình.
            throttle_percentage (float): Phần trăm giới hạn sử dụng CPU (0-100).

        Returns:
            Optional[str]: Tên cgroup nếu thành công, None nếu thất bại.
        """
        try:
            if not (0 <= throttle_percentage <= 100):
                self.logger.error(f"throttle_percentage={throttle_percentage} không hợp lệ. Phải từ 0 đến 100.")
                return None

            # Tạo tên cgroup duy nhất
            cgroup_name = f"cpu_cloak_{uuid.uuid4().hex[:8]}"
            cgroup_path = os.path.join(self.CGROUP_BASE_PATH, cgroup_name)
            os.makedirs(cgroup_path, exist_ok=True)
            self.logger.debug(f"Tạo cgroup tại {cgroup_path} cho PID={pid}.")

            # Tính toán CPU quota dựa trên throttle_percentage
            # CPU quota được tính dựa trên 100000 us (100ms) mỗi 100ms
            # Nếu throttle_percentage là 50%, quota sẽ là 50000us mỗi 100000us
            cpu_period = 100000  # 100ms
            cpu_quota = int((throttle_percentage / 100) * cpu_period)
            cpu_quota = max(1000, cpu_quota)  # Đảm bảo quota tối thiểu

            with open(os.path.join(cgroup_path, "cpu.max"), "w") as f:
                f.write(f"{cpu_quota} {cpu_period}\n")
            self.logger.debug(f"Đặt CPU quota tại {cpu_quota}us cho cgroup {cgroup_name}.")

            # Thêm PID vào cgroup
            with open(os.path.join(cgroup_path, "cgroup.procs"), "w") as f:
                f.write(f"{pid}\n")
            self.logger.info(f"Thêm PID={pid} vào cgroup {cgroup_name} với throttle_percentage={throttle_percentage}%.")

            # Lưu mapping PID -> cgroup_name
            self.process_cgroup[pid] = cgroup_name

            return cgroup_name

        except PermissionError:
            self.logger.error(f"Không đủ quyền để tạo và cấu hình cgroup cho PID={pid}.")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo cgroup cho PID={pid}: {e}")
            return None

    def delete_cgroup(self, cgroup_name: str) -> bool:
        """
        Xóa một cgroup.

        Args:
            cgroup_name (str): Tên của cgroup cần xóa.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            cgroup_path = os.path.join(self.CGROUP_BASE_PATH, cgroup_name)
            # Đảm bảo không còn tiến trình nào trong cgroup
            procs_path = os.path.join(cgroup_path, "cgroup.procs")
            with open(procs_path, "r") as f:
                procs = f.read().strip()
                if procs:
                    self.logger.warning(f"Cgroup {cgroup_name} vẫn còn tiến trình PID={procs}. Không thể xóa.")
                    return False

            os.rmdir(cgroup_path)
            self.logger.info(f"Xóa cgroup {cgroup_name} thành công.")
            return True
        except FileNotFoundError:
            self.logger.warning(f"Cgroup {cgroup_name} không tồn tại khi cố gắng xóa.")
            return False
        except PermissionError:
            self.logger.error(f"Không đủ quyền để xóa cgroup {cgroup_name}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi xóa cgroup {cgroup_name}: {e}")
            return False

    def throttle_cpu_usage(self, pid: int, throttle_percentage: float) -> Optional[str]:
        """
        Giới hạn sử dụng CPU cho tiến trình bằng cách tạo một cgroup và thiết lập CPU quota.

        Args:
            pid (int): PID của tiến trình.
            throttle_percentage (float): Phần trăm giới hạn sử dụng CPU (0-100).

        Returns:
            Optional[str]: Tên cgroup nếu thành công, None nếu thất bại.
        """
        return self.create_cgroup(pid, throttle_percentage)

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục các thiết lập CPU cho tiến trình bằng cách xóa cgroup đã tạo.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            cgroup_name = self.process_cgroup.get(pid)
            if not cgroup_name:
                self.logger.warning(f"Không tìm thấy cgroup cho PID={pid} trong CPUResourceManager.")
                return False
            success = self.delete_cgroup(cgroup_name)
            if success:
                self.logger.info(f"Đã khôi phục CPU settings cho PID={pid}.")
                del self.process_cgroup[pid]
                return True
            else:
                self.logger.error(f"Không thể khôi phục CPU settings cho PID={pid}.")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục CPU settings cho PID={pid}: {e}")
            return False

    def set_cpu_affinity(self, pid: int, cores: List[int]) -> bool:
        """
        Đặt CPU affinity cho tiến trình.

        Args:
            pid (int): PID của tiến trình.
            cores (List[int]): Danh sách các core CPU để đặt affinity.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            process = psutil.Process(pid)
            process.cpu_affinity(cores)
            self.logger.debug(f"Đặt CPU affinity cho PID={pid} vào các core {cores}.")
            return True
        except psutil.NoSuchProcess:
            self.logger.error(f"Tiến trình PID={pid} không tồn tại khi đặt CPU affinity.")
            return False
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền để đặt CPU affinity cho PID={pid}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi đặt CPU affinity cho PID={pid}: {e}")
            return False

    def reset_cpu_affinity(self, pid: int) -> bool:
        """
        Khôi phục CPU affinity cho tiến trình về tất cả các core CPU có sẵn.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            available_cpus = self.get_available_cpus()
            return self.set_cpu_affinity(pid, available_cpus)
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục CPU affinity cho PID={pid}: {e}")
            return False

    def limit_cpu_for_external_processes(self, target_pids: List[int], throttle_percentage: float) -> bool:
        """
        Hạn chế CPU sử dụng cho các tiến trình bên ngoài mục tiêu.

        Args:
            target_pids (List[int]): Danh sách PID của tiến trình mục tiêu không bị hạn chế.
            throttle_percentage (float): Phần trăm throttle cho các tiến trình khác (0-100).

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            if not (0 <= throttle_percentage <= 100):
                self.logger.error(f"throttle_percentage={throttle_percentage} không hợp lệ. Phải từ 0 đến 100.")
                return False

            all_pids = [proc.pid for proc in psutil.process_iter(attrs=['pid'])]
            external_pids = set(all_pids) - set(target_pids)

            for pid in external_pids:
                cgroup_name = self.throttle_cpu_usage(pid, throttle_percentage)
                if cgroup_name:
                    self.logger.debug(f"Hạn chế CPU cho PID={pid} với cgroup={cgroup_name}.")
                else:
                    self.logger.warning(f"Không thể hạn chế CPU cho PID={pid}.")

            self.logger.info(f"Hạn chế CPU cho {len(external_pids)} tiến trình bên ngoài với throttle_percentage={throttle_percentage}%.")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi hạn chế CPU cho các tiến trình bên ngoài: {e}")
            return False

    def optimize_thread_scheduling(self, pid: int, target_cores: Optional[List[int]] = None) -> bool:
        """
        Tối ưu hóa scheduling cho các thread của tiến trình bằng cách đặt affinity.

        Args:
            pid (int): PID của tiến trình.
            target_cores (Optional[List[int]]): Danh sách các core CPU để đặt affinity. Nếu None, sử dụng tất cả các core CPU có sẵn.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            success = self.set_cpu_affinity(pid, target_cores or self.get_available_cpus())
            if success:
                self.logger.info(f"Tối ưu hóa scheduling cho PID={pid} thành công với target_cores={target_cores or self.get_available_cpus()}.")
            return success
        except Exception as e:
            self.logger.error(f"Lỗi khi tối ưu hóa scheduling cho PID={pid}: {e}")
            return False

    def optimize_cache_usage(self, pid: int) -> bool:
        """
        Tối ưu hóa việc sử dụng cache CPU bằng cách đặt độ ưu tiên tiến trình về mức bình thường.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            # Với cgroups, việc tối ưu hóa cache có thể liên quan đến việc đảm bảo tiến trình không bị throttled quá mức
            # Điều này đã được thực hiện thông qua throttle_cpu_usage
            # Ở đây, chúng ta chỉ đặt lại độ ưu tiên nếu cần
            # Tuy nhiên, do chúng ta đang sử dụng cgroups, việc điều chỉnh độ ưu tiên thông qua nice có thể không cần thiết
            # Do đó, phương thức này có thể được bỏ qua hoặc dùng để thực hiện các tối ưu hóa bổ sung
            # Ví dụ: Kiểm tra trạng thái cgroup và điều chỉnh nếu cần

            # Giả sử chúng ta không cần thực hiện thêm gì ở đây
            self.logger.debug(f"Tối ưu hóa cache cho PID={pid} đã được thực hiện thông qua cgroups.")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi tối ưu hóa cache cho PID={pid}: {e}")
            return False


class GPUResourceManager:
    """
    Quản lý tài nguyên GPU thông qua NVML và các công cụ NVIDIA khác.
    """

    def __init__(self, logger: logging.Logger, gpu_manager: GPUManager):
        """
        Khởi tạo GPUResourceManager với NVML và GPUManager.

        Args:
            logger (logging.Logger): Logger để ghi log các hoạt động và lỗi.
            gpu_manager (GPUManager): Quản lý GPU thông qua NVML.
        """
        self.logger = logger
        self.gpu_manager = gpu_manager
        self.gpu_initialized = False
        self.process_gpu_settings: Dict[int, Dict[int, Dict[str, Any]]] = {}  # Mapping PID -> GPU Index -> Settings

        try:
            self.gpu_manager.initialize()
            if self.gpu_manager.gpu_count > 0:
                self.gpu_initialized = True
                self.logger.info("GPUResourceManager đã được khởi tạo và có GPU sẵn sàng.")
            else:
                self.logger.warning("Không có GPU nào được phát hiện trên hệ thống.")
        except pynvml.NVMLError as error:
            self.logger.error(f"Lỗi khi khởi tạo NVML: {error}")
            self.gpu_initialized = False
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi khởi tạo GPUResourceManager: {e}")
            self.gpu_initialized = False

    def set_gpu_power_limit(self, pid: int, gpu_index: int, power_limit_w: int) -> bool:
        """
        Thiết lập power limit cho GPU và lưu lại thiết lập ban đầu.

        Args:
            pid (int): PID của tiến trình.
            gpu_index (int): Chỉ số GPU (bắt đầu từ 0).
            power_limit_w (int): Power limit tính bằng Watts.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        if not self.gpu_initialized:
            self.logger.error("GPU chưa được khởi tạo hoặc không có GPU nào trên hệ thống. Không thể đặt power limit.")
            return False

        if not (0 <= gpu_index < self.gpu_manager.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ. Có tổng cộng {self.gpu_manager.gpu_count} GPU.")
            return False

        if power_limit_w <= 0:
            self.logger.error("Giá trị power limit phải lớn hơn 0.")
            return False

        try:
            handle = self.gpu_manager.get_handle(gpu_index)
            power_limit_mw = power_limit_w * 1000  # Chuyển từ Watts sang milliWatts

            # Lấy power limit hiện tại để lưu lại
            current_power_limit_mw = self.gpu_manager.get_power_limit(handle)
            if current_power_limit_mw is not None:
                current_power_limit_w = current_power_limit_mw / 1000
                if pid not in self.process_gpu_settings:
                    self.process_gpu_settings[pid] = {}
                if gpu_index not in self.process_gpu_settings[pid]:
                    self.process_gpu_settings[pid][gpu_index] = {}
                self.process_gpu_settings[pid][gpu_index]['power_limit_w'] = current_power_limit_w

            self.gpu_manager.set_power_limit(handle, power_limit_mw)
            self.logger.debug(f"Đặt power limit cho GPU {gpu_index} là {power_limit_w}W cho PID={pid}.")
            return True
        except pynvml.NVMLError as error:
            self.logger.error(f"Lỗi NVML khi đặt power limit cho GPU {gpu_index}: {error}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi đặt power limit cho GPU {gpu_index}: {e}")
            return False

    def set_gpu_clocks(self, pid: int, gpu_index: int, sm_clock: int, mem_clock: int) -> bool:
        """
        Thiết lập xung nhịp GPU thông qua nvidia-smi và lưu lại thiết lập ban đầu.

        Args:
            pid (int): PID của tiến trình.
            gpu_index (int): Chỉ số GPU (bắt đầu từ 0).
            sm_clock (int): Xung nhịp SM GPU tính bằng MHz.
            mem_clock (int): Xung nhịp bộ nhớ GPU tính bằng MHz.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        if not self.gpu_initialized:
            self.logger.error("GPU chưa được khởi tạo hoặc không có GPU nào trên hệ thống. Không thể đặt xung nhịp.")
            return False

        if not (0 <= gpu_index < self.gpu_manager.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ. Có tổng cộng {self.gpu_manager.gpu_count} GPU.")
            return False

        if mem_clock <= 0 or sm_clock <= 0:
            self.logger.error("Giá trị xung nhịp phải lớn hơn 0.")
            return False

        try:
            handle = self.gpu_manager.get_handle(gpu_index)

            # Lấy xung nhịp hiện tại để lưu lại
            current_sm_clock = self.gpu_manager.get_current_sm_clock(handle)
            current_mem_clock = self.gpu_manager.get_current_mem_clock(handle)
            if pid not in self.process_gpu_settings:
                self.process_gpu_settings[pid] = {}
            if gpu_index not in self.process_gpu_settings[pid]:
                self.process_gpu_settings[pid][gpu_index] = {}
            self.process_gpu_settings[pid][gpu_index]['sm_clock_mhz'] = current_sm_clock
            self.process_gpu_settings[pid][gpu_index]['mem_clock_mhz'] = current_mem_clock

            # Thiết lập xung nhịp SM
            cmd_sm = [
                'nvidia-smi',
                '-i', str(gpu_index),
                '--lock-gpu-clocks=' + str(sm_clock)
            ]
            subprocess.run(cmd_sm, check=True)
            self.logger.debug(f"Đã thiết lập xung nhịp SM cho GPU {gpu_index} là {sm_clock}MHz cho PID={pid}.")

            # Thiết lập xung nhịp bộ nhớ
            cmd_mem = [
                'nvidia-smi',
                '-i', str(gpu_index),
                '--lock-memory-clocks=' + str(mem_clock)
            ]
            subprocess.run(cmd_mem, check=True)
            self.logger.debug(f"Đã thiết lập xung nhịp bộ nhớ cho GPU {gpu_index} là {mem_clock}MHz cho PID={pid}.")

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi thiết lập xung nhịp GPU {gpu_index} bằng nvidia-smi: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi thiết lập xung nhịp GPU {gpu_index}: {e}")
            return False

    def set_gpu_max_power(self, pid: int, gpu_index: int, gpu_max_mw: int) -> bool:
        """
        Thiết lập giới hạn power tối đa cho GPU thông qua NVML và lưu lại thiết lập ban đầu.

        Args:
            pid (int): PID của tiến trình.
            gpu_index (int): Chỉ số GPU (bắt đầu từ 0).
            gpu_max_mw (int): Giới hạn power tối đa tính bằng milliWatts.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        if not self.gpu_initialized:
            self.logger.error("GPU chưa được khởi tạo hoặc không có GPU nào trên hệ thống. Không thể đặt 'gpu_max'.")
            return False

        if not (0 <= gpu_index < self.gpu_manager.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ. Có tổng cộng {self.gpu_manager.gpu_count} GPU.")
            return False

        if gpu_max_mw <= 0:
            self.logger.error("Giới hạn power tối đa phải lớn hơn 0.")
            return False

        try:
            handle = self.gpu_manager.get_handle(gpu_index)
            # Lấy power limit hiện tại để lưu lại
            current_power_limit_mw = self.gpu_manager.get_power_limit(handle)
            if current_power_limit_mw is not None:
                current_power_limit_w = current_power_limit_mw / 1000
                if pid not in self.process_gpu_settings:
                    self.process_gpu_settings[pid] = {}
                if gpu_index not in self.process_gpu_settings[pid]:
                    self.process_gpu_settings[pid][gpu_index] = {}
                self.process_gpu_settings[pid][gpu_index]['power_limit_w'] = current_power_limit_w

            self.gpu_manager.set_power_limit(handle, gpu_max_mw)
            power_limit_w = gpu_max_mw / 1000  # Chuyển từ milliWatts sang Watts
            self.logger.debug(f"Đặt 'gpu_max' cho GPU {gpu_index} là {power_limit_w}W cho PID={pid}.")
            return True
        except pynvml.NVMLError as error:
            self.logger.error(f"Lỗi NVML khi đặt 'gpu_max' cho GPU {gpu_index}: {error}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi đặt 'gpu_max' cho GPU {gpu_index}: {e}")
            return False

    def get_gpu_power_limit(self, gpu_index: int) -> Optional[int]:
        """
        Lấy giới hạn power hiện tại của GPU.

        Args:
            gpu_index (int): Chỉ số GPU (bắt đầu từ 0).

        Returns:
            Optional[int]: Giới hạn power hiện tại tính bằng Watts hoặc None nếu lỗi.
        """
        if not self.gpu_initialized:
            self.logger.error("GPU chưa được khởi tạo hoặc không có GPU nào trên hệ thống. Không thể lấy power limit.")
            return None

        if not (0 <= gpu_index < self.gpu_manager.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ. Có tổng cộng {self.gpu_manager.gpu_count} GPU.")
            return None

        try:
            handle = self.gpu_manager.get_handle(gpu_index)
            power_limit_mw = self.gpu_manager.get_power_limit(handle)
            power_limit_w = power_limit_mw / 1000  # Chuyển từ milliWatts sang Watts
            self.logger.debug(f"Giới hạn power cho GPU {gpu_index} là {power_limit_w}W.")
            return power_limit_w
        except pynvml.NVMLError as error:
            self.logger.error(f"Lỗi NVML khi lấy power limit cho GPU {gpu_index}: {error}")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi lấy power limit cho GPU {gpu_index}: {e}")
            return None

    def get_gpu_temperature(self, gpu_index: int) -> Optional[float]:
        """
        Lấy nhiệt độ hiện tại của GPU.

        Args:
            gpu_index (int): Chỉ số GPU (bắt đầu từ 0).

        Returns:
            Optional[float]: Nhiệt độ GPU tính bằng °C hoặc None nếu lỗi.
        """
        if not self.gpu_initialized:
            self.logger.error("GPU chưa được khởi tạo hoặc không có GPU nào trên hệ thống. Không thể lấy nhiệt độ.")
            return None

        if not (0 <= gpu_index < self.gpu_manager.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ. Có tổng cộng {self.gpu_manager.gpu_count} GPU.")
            return None

        try:
            handle = self.gpu_manager.get_handle(gpu_index)
            temperature = self.gpu_manager.get_temperature(handle)
            self.logger.debug(f"Nhiệt độ GPU {gpu_index} là {temperature}°C.")
            return temperature
        except pynvml.NVMLError as error:
            self.logger.error(f"Lỗi NVML khi lấy nhiệt độ cho GPU {gpu_index}: {error}")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi lấy nhiệt độ cho GPU {gpu_index}: {e}")
            return None

    def get_gpu_utilization(self, gpu_index: int) -> Optional[Dict[str, float]]:
        """
        Lấy thông tin sử dụng GPU hiện tại.

        Args:
            gpu_index (int): Chỉ số GPU (bắt đầu từ 0).

        Returns:
            Optional[Dict[str, float]]: Dictionary chứa thông tin sử dụng GPU hoặc None nếu lỗi.
        """
        if not self.gpu_initialized:
            self.logger.error("GPU chưa được khởi tạo hoặc không có GPU nào trên hệ thống. Không thể lấy thông tin sử dụng GPU.")
            return None

        if not (0 <= gpu_index < self.gpu_manager.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ. Có tổng cộng {self.gpu_manager.gpu_count} GPU.")
            return None

        try:
            handle = self.gpu_manager.get_handle(gpu_index)
            utilization = self.gpu_manager.get_utilization(handle)
            self.logger.debug(f"Sử dụng GPU {gpu_index}: {utilization}")
            return utilization
        except pynvml.NVMLError as error:
            self.logger.error(f"Lỗi NVML khi lấy sử dụng GPU cho GPU {gpu_index}: {error}")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi lấy sử dụng GPU cho GPU {gpu_index}: {e}")
            return None

    def control_fan_speed(self, gpu_index: int, increase_percentage: float) -> bool:
        """
        Điều chỉnh tốc độ quạt GPU (nếu hỗ trợ).

        Args:
            gpu_index (int): Chỉ số GPU.
            increase_percentage (float): Tỷ lệ tăng tốc độ quạt (%).

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            # Sử dụng nvidia-settings để điều chỉnh tốc độ quạt
            cmd = [
                'nvidia-settings',
                '-a', f'[fan:{gpu_index}]/GPUFanControlState=1',
                '-a', f'[fan:{gpu_index}]/GPUTargetFanSpeed={int(increase_percentage)}'
            ]
            subprocess.run(cmd, check=True)
            self.logger.debug(f"Đã tăng tốc độ quạt GPU {gpu_index} lên {increase_percentage}%.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi điều chỉnh tốc độ quạt GPU {gpu_index}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi điều chỉnh tốc độ quạt GPU {gpu_index}: {e}")
            return False

    def limit_temperature(self, gpu_index: int, temperature_threshold: float, fan_speed_increase: float) -> bool:
        """
        Quản lý nhiệt độ GPU bằng cách điều chỉnh quạt, giới hạn power và xung nhịp.

        Args:
            gpu_index (int): Chỉ số GPU.
            temperature_threshold (float): Ngưỡng nhiệt độ tối đa (°C).
            fan_speed_increase (float): Tỷ lệ tăng tốc độ quạt (%) để làm mát.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            # Tăng tốc độ quạt để làm mát GPU
            success_fan = self.control_fan_speed(gpu_index, fan_speed_increase)
            if success_fan:
                self.logger.info(f"Tăng tốc độ quạt GPU {gpu_index} lên {fan_speed_increase}% để làm mát.")
            else:
                self.logger.warning(f"Không thể tăng tốc độ quạt GPU {gpu_index}. Kiểm tra hỗ trợ điều chỉnh quạt.")

            # Lấy nhiệt độ hiện tại của GPU
            current_temperature = self.get_gpu_temperature(gpu_index)
            if current_temperature is None:
                self.logger.warning(f"Không thể lấy nhiệt độ hiện tại cho GPU {gpu_index}.")
                return False

            if current_temperature > temperature_threshold:
                # Nhiệt độ vượt ngưỡng: Giảm power limit và xung nhịp
                self.logger.info(f"Nhiệt độ GPU {gpu_index} hiện tại {current_temperature}°C vượt ngưỡng {temperature_threshold}°C. Thực hiện cloaking.")

                excess_temperature = current_temperature - temperature_threshold

                # Định nghĩa các mức cloaking dựa trên mức độ vượt ngưỡng
                if excess_temperature <= 5:
                    throttle_percentage = 10  # Giảm power limit 10%
                    self.logger.debug(f"Mức độ vượt ngưỡng nhiệt độ nhẹ: {excess_temperature}°C. Giảm power limit 10%.")
                elif 5 < excess_temperature <= 10:
                    throttle_percentage = 20  # Giảm power limit 20%
                    self.logger.debug(f"Mức độ vượt ngưỡng nhiệt độ trung bình: {excess_temperature}°C. Giảm power limit 20%.")
                else:
                    throttle_percentage = 30  # Giảm power limit 30%
                    self.logger.debug(f"Mức độ vượt ngưỡng nhiệt độ nặng: {excess_temperature}°C. Giảm power limit 30%.")

                # Lấy power limit hiện tại
                current_power_limit_w = self.get_gpu_power_limit(gpu_index)
                if current_power_limit_w is None:
                    self.logger.warning(f"Không thể lấy power limit hiện tại cho GPU {gpu_index}.")
                    return False
                desired_power_limit_w = int(round(current_power_limit_w * (1 - throttle_percentage / 100)))

                # Giảm power limit
                success_power = self.set_gpu_power_limit(pid=None, gpu_index=gpu_index, power_limit_w=desired_power_limit_w)
                if success_power:
                    self.logger.info(f"Giảm power limit GPU {gpu_index} xuống {desired_power_limit_w}W để giảm nhiệt độ.")
                else:
                    self.logger.error(f"Không thể giảm power limit GPU {gpu_index} để giảm nhiệt độ.")

                # Giảm xung nhịp GPU để giảm nhiệt độ
                # Lấy xung nhịp hiện tại từ GPUManager
                current_sm_clock = self.gpu_manager.get_current_sm_clock(handle)
                current_mem_clock = self.gpu_manager.get_current_mem_clock(handle)

                # Đảm bảo không giảm xung nhịp dưới mức tối thiểu
                target_sm_clock = max(500, current_sm_clock - 100)   # Giảm xung nhịp SM xuống tối thiểu 500MHz
                target_mem_clock = max(300, current_mem_clock - 50)  # Giảm xung nhịp MEM xuống tối thiểu 300MHz

                # Giảm xung nhịp
                success_clocks = self.set_gpu_clocks(pid=None, gpu_index=gpu_index, sm_clock=target_sm_clock, mem_clock=target_mem_clock)
                if success_clocks:
                    self.logger.info(f"Giảm xung nhịp GPU {gpu_index} xuống SM={target_sm_clock}MHz, MEM={target_mem_clock}MHz để giảm nhiệt độ.")
                else:
                    self.logger.warning(f"Không thể giảm xung nhịp GPU {gpu_index} để giảm nhiệt độ.")

            elif current_temperature < temperature_threshold:
                # Nhiệt độ dưới ngưỡng: Tăng xung nhịp để cải thiện hiệu suất
                self.logger.info(f"Nhiệt độ GPU {gpu_index} hiện tại {current_temperature}°C dưới ngưỡng {temperature_threshold}°C. Tăng xung nhịp để cải thiện hiệu suất.")

                excess_cooling = temperature_threshold - current_temperature

                # Định nghĩa các mức tăng xung nhịp dựa trên mức độ dưới ngưỡng
                if excess_cooling <= 5:
                    boost_percentage = 10  # Tăng xung nhịp 10%
                    self.logger.debug(f"Mức độ dưới ngưỡng nhiệt độ nhẹ: {excess_cooling}°C. Tăng xung nhịp 10%.")
                elif 5 < excess_cooling <= 10:
                    boost_percentage = 20  # Tăng xung nhịp 20%
                    self.logger.debug(f"Mức độ dưới ngưỡng nhiệt độ trung bình: {excess_cooling}°C. Tăng xung nhịp 20%.")
                else:
                    boost_percentage = 30  # Tăng xung nhịp 30%
                    self.logger.debug(f"Mức độ dưới ngưỡng nhiệt độ nặng: {excess_cooling}°C. Tăng xung nhịp 30%.")

                # Tính toán xung nhịp mới
                desired_sm_clock = min(current_sm_clock + int(current_sm_clock * boost_percentage / 100), 1530)   # Không vượt quá Boost Clock
                desired_mem_clock = min(current_mem_clock + int(current_mem_clock * boost_percentage / 100), 877)  # Không vượt quá Memory Clock

                # Tăng xung nhịp
                success_clocks = self.set_gpu_clocks(pid=None, gpu_index=gpu_index, sm_clock=desired_sm_clock, mem_clock=desired_mem_clock)
                if success_clocks:
                    self.logger.info(f"Tăng xung nhịp GPU {gpu_index} lên SM={desired_sm_clock}MHz, MEM={desired_mem_clock}MHz để cải thiện hiệu suất.")
                else:
                    self.logger.warning(f"Không thể tăng xung nhịp GPU {gpu_index} để cải thiện hiệu suất.")

            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi giới hạn nhiệt độ GPU {gpu_index}: {e}")
            return False

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục các thiết lập GPU cho tiến trình bằng cách đặt lại power limit và xung nhịp cho tất cả các GPU mà tiến trình đang sử dụng.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            # Lấy các thiết lập ban đầu từ mapping
            pid_settings = self.process_gpu_settings.get(pid)
            if not pid_settings:
                self.logger.warning(f"Không tìm thấy thiết lập GPU ban đầu cho PID={pid}.")
                return False

            restored_all = True

            # Duyệt qua từng GPU đã được thiết lập cho PID này
            for gpu_index, settings in pid_settings.items():
                # Khôi phục power limit nếu có
                original_power_limit_w = settings.get('power_limit_w')
                if original_power_limit_w is not None:
                    success_power = self.set_gpu_power_limit(pid, gpu_index, int(original_power_limit_w))
                    if success_power:
                        self.logger.info(f"Đã khôi phục power limit GPU {gpu_index} về {original_power_limit_w}W cho PID={pid}.")
                    else:
                        self.logger.error(f"Không thể khôi phục power limit GPU {gpu_index} cho PID={pid}.")
                        restored_all = False

                # Khôi phục xung nhịp nếu có
                original_sm_clock = settings.get('sm_clock_mhz')
                original_mem_clock = settings.get('mem_clock_mhz')
                if original_sm_clock and original_mem_clock:
                    success_clocks = self.set_gpu_clocks(pid, gpu_index, int(original_sm_clock), int(original_mem_clock))
                    if success_clocks:
                        self.logger.info(f"Đã khôi phục xung nhịp GPU {gpu_index} về SM={original_sm_clock}MHz, MEM={original_mem_clock}MHz cho PID={pid}.")
                    else:
                        self.logger.error(f"Không thể khôi phục xung nhịp GPU {gpu_index} cho PID={pid}.")
                        restored_all = False

            # Xóa thiết lập từ mapping
            del self.process_gpu_settings[pid]
            self.logger.info(f"Đã khôi phục tất cả các thiết lập GPU cho PID={pid}.")
            return restored_all
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục GPU settings cho PID={pid}: {e}")
            return False


class NetworkResourceManager:
    """
    Quản lý tài nguyên mạng thông qua iptables và tc.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.process_marks: Dict[int, int] = {}  # Mapping PID to fwmark

    def mark_packets(self, pid: int, mark: int) -> bool:
        """
        Thêm quy tắc iptables để đánh dấu các gói tin từ PID này.

        Args:
            pid (int): PID của tiến trình.
            mark (int): fwmark để đặt.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            subprocess.run([
                'iptables', '-A', 'OUTPUT', '-m', 'owner', '--pid-owner', str(pid),
                '-j', 'MARK', '--set-mark', str(mark)
            ], check=True)
            self.logger.debug(f"Đặt iptables MARK cho PID={pid} với mark={mark}.")
            self.process_marks[pid] = mark
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi đặt iptables MARK cho PID={pid}: {e}")
            return False

    def unmark_packets(self, pid: int, mark: int) -> bool:
        """
        Xóa quy tắc iptables đánh dấu các gói tin từ PID này.

        Args:
            pid (int): PID của tiến trình.
            mark (int): fwmark đã đặt.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            subprocess.run([
                'iptables', '-D', 'OUTPUT', '-m', 'owner', '--pid-owner', str(pid),
                '-j', 'MARK', '--set-mark', str(mark)
            ], check=True)
            self.logger.debug(f"Xóa iptables MARK cho PID={pid} với mark={mark}.")
            if pid in self.process_marks:
                del self.process_marks[pid]
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi xóa iptables MARK cho PID={pid}: {e}")
            return False

    def limit_bandwidth(self, interface: str, mark: int, bandwidth_mbps: float) -> bool:
        """
        Giới hạn băng thông mạng dựa trên fwmark thông qua tc.

        Args:
            interface (str): Tên giao diện mạng.
            mark (int): fwmark để lọc.
            bandwidth_mbps (float): Giới hạn băng thông tính bằng Mbps.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            # Thêm qdisc nếu chưa tồn tại
            subprocess.run([
                'tc', 'qdisc', 'add', 'dev', interface, 'root', 'handle', '1:',
                'htb', 'default', '12'
            ], check=True)
            self.logger.debug(f"Đã thêm tc qdisc 'htb' cho giao diện '{interface}'.")

            # Thêm class để giới hạn băng thông
            subprocess.run([
                'tc', 'class', 'add', 'dev', interface, 'parent', '1:', 'classid', '1:1',
                'htb', 'rate', f'{bandwidth_mbps}mbit'
            ], check=True)
            self.logger.debug(f"Đã thêm tc class '1:1' với rate={bandwidth_mbps}mbit cho giao diện '{interface}'.")

            # Thêm filter để áp dụng giới hạn cho các gói tin có mark
            subprocess.run([
                'tc', 'filter', 'add', 'dev', interface, 'protocol', 'ip',
                'parent', '1:', 'prio', '1', 'handle', str(mark), 'fw', 'flowid', '1:1'
            ], check=True)
            self.logger.debug(f"Đã thêm tc filter cho mark={mark} trên giao diện '{interface}'.")

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi giới hạn băng thông mạng: {e}")
            return False

    def remove_bandwidth_limit(self, interface: str, mark: int) -> bool:
        """
        Xóa giới hạn băng thông mạng dựa trên fwmark thông qua tc.

        Args:
            interface (str): Tên giao diện mạng.
            mark (int): fwmark đã đặt.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            # Xóa filter
            subprocess.run([
                'tc', 'filter', 'del', 'dev', interface, 'protocol', 'ip',
                'parent', '1:', 'prio', '1', 'handle', str(mark), 'fw', 'flowid', '1:1'
            ], check=True)
            self.logger.debug(f"Đã xóa tc filter cho mark={mark} trên giao diện '{interface}'.")

            # Xóa class
            subprocess.run([
                'tc', 'class', 'del', 'dev', interface, 'parent', '1:', 'classid', '1:1'
            ], check=True)
            self.logger.debug(f"Đã xóa tc class '1:1' trên giao diện '{interface}'.")

            # Xóa qdisc nếu không còn class nào
            subprocess.run([
                'tc', 'qdisc', 'del', 'dev', interface, 'root', 'handle', '1:', 'htb'
            ], check=True)
            self.logger.debug(f"Đã xóa tc qdisc 'htb' cho giao diện '{interface}'.")

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi xóa giới hạn băng thông mạng: {e}")
            return False

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục các thiết lập mạng cho tiến trình bằng cách xóa giới hạn băng thông và unmark packets.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            mark = self.process_marks.get(pid)
            if not mark:
                self.logger.warning(f"Không tìm thấy mark cho PID={pid} trong NetworkResourceManager.")
                return False

            # Xác định giao diện mạng từ cấu hình hoặc mặc định
            # Giả sử cấu hình có sẵn, nếu không thì mặc định 'eth0'
            interface = 'eth0'  # Bạn có thể lấy từ cấu hình nếu cần

            # Xóa giới hạn băng thông
            success_bw = self.remove_bandwidth_limit(interface, mark)
            if success_bw:
                self.logger.info(f"Đã khôi phục giới hạn băng thông mạng cho PID={pid} với mark={mark} trên giao diện '{interface}'.")
            else:
                self.logger.error(f"Không thể khôi phục giới hạn băng thông mạng cho PID={pid}.")

            # Xóa iptables MARK
            success_unmark = self.unmark_packets(pid, mark)
            if success_unmark:
                self.logger.info(f"Đã xóa iptables MARK cho PID={pid} với mark={mark}.")
            else:
                self.logger.error(f"Không thể xóa iptables MARK cho PID={pid} với mark={mark}.")

            return success_bw and success_unmark
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục Network settings cho PID={pid}: {e}")
            return False


class DiskIOResourceManager:
    """
    Quản lý tài nguyên Disk I/O thông qua tc hoặc các cơ chế kiểm soát I/O khác.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.process_io_limits: Dict[int, float] = {}  # Mapping PID to I/O rate limit

    def limit_io(self, interface: str, rate_mbps: float) -> bool:
        """
        Giới hạn tốc độ I/O Disk sử dụng tc.

        Args:
            interface (str): Tên giao diện mạng.
            rate_mbps (float): Giới hạn tốc độ I/O tính bằng Mbps.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            # Thêm qdisc root nếu chưa tồn tại
            subprocess.run([
                'tc', 'qdisc', 'add', 'dev', interface, 'root', 'handle', '1:',
                'htb', 'default', '12'
            ], check=True)
            self.logger.debug(f"Đã thêm tc qdisc 'htb' cho giao diện '{interface}'.")

            # Thêm class để giới hạn I/O
            subprocess.run([
                'tc', 'class', 'add', 'dev', interface, 'parent', '1:', 'classid', '1:1',
                'htb', 'rate', f'{rate_mbps}mbit'
            ], check=True)
            self.logger.debug(f"Đã thêm tc class '1:1' với rate={rate_mbps}mbit cho giao diện '{interface}'.")

            # Lưu mapping PID -> rate_mbps nếu cần thiết
            # Trong ví dụ này, giả định rằng giới hạn I/O là toàn bộ cho giao diện
            # Nếu giới hạn theo PID, cần sử dụng thêm các filter và mapping khác

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi giới hạn Disk I/O: {e}")
            return False

    def remove_io_limit(self, interface: str) -> bool:
        """
        Xóa giới hạn I/O Disk sử dụng tc.

        Args:
            interface (str): Tên giao diện mạng.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            # Xóa qdisc root
            subprocess.run([
                'tc', 'qdisc', 'del', 'dev', interface, 'root', 'handle', '1:', 'htb'
            ], check=True)
            self.logger.debug(f"Đã xóa tc qdisc 'htb' cho giao diện '{interface}'.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi xóa giới hạn Disk I/O: {e}")
            return False

    def set_io_weight(self, pid: int, io_weight: int) -> bool:
        """
        Đặt trọng số I/O cho tiến trình.

        Args:
            pid (int): PID của tiến trình.
            io_weight (int): Trọng số I/O (1-1000).

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            # Đây là một ví dụ placeholder.
            # Thực tế cần sử dụng cgroups hoặc các cơ chế kiểm soát I/O khác để đặt trọng số I/O.
            self.logger.debug(f"Đặt trọng số I/O cho PID={pid} là {io_weight}. (Chưa triển khai)")
            # Triển khai thực tế ở đây nếu cần.
            self.process_io_limits[pid] = io_weight
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi đặt trọng số I/O cho PID={pid}: {e}")
            return False

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục giới hạn I/O Disk cho tiến trình bằng cách xóa giới hạn đã đặt.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            # Trong ví dụ này, giả định rằng giới hạn I/O là toàn bộ cho giao diện.
            # Nếu giới hạn theo PID, cần phải xác định rate_mbps đã đặt cho PID này.
            if pid in self.process_io_limits:
                del self.process_io_limits[pid]
                self.logger.info(f"Đã khôi phục Disk I/O settings cho PID={pid}.")
                return True
            else:
                self.logger.warning(f"Không tìm thấy thiết lập Disk I/O cho PID={pid}.")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục Disk I/O settings cho PID={pid}: {e}")
            return False


class CacheResourceManager:
    """
    Quản lý tài nguyên Cache thông qua việc drop caches.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def drop_caches(self) -> bool:
        """
        Drop caches bằng cách ghi vào /proc/sys/vm/drop_caches.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3\n')
            self.logger.debug("Đã drop caches thành công.")
            return True
        except PermissionError:
            self.logger.error("Không đủ quyền để drop caches.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi drop caches: {e}")
            return False

    def limit_cache_usage(self, cache_limit_percent: float) -> bool:
        """
        Giới hạn mức sử dụng cache bằng cách drop caches và các phương pháp khác nếu cần.

        Args:
            cache_limit_percent (float): Phần trăm giới hạn sử dụng cache.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            # Ví dụ: sử dụng drop_caches
            self.drop_caches()
            # Các biện pháp giới hạn cache khác có thể được thêm vào đây.
            self.logger.debug(f"Giới hạn mức sử dụng cache xuống còn {cache_limit_percent}%.")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi giới hạn sử dụng cache: {e}")
            return False

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục các thiết lập Cache cho tiến trình bằng cách đặt lại mức sử dụng cache.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            # Giả định rằng việc khôi phục cache là đặt lại mức sử dụng cache về 100%
            success = self.limit_cache_usage(100.0)
            if success:
                self.logger.info(f"Đã khôi phục Cache settings cho PID={pid} về 100%.")
            else:
                self.logger.error(f"Không thể khôi phục Cache settings cho PID={pid}.")
            return success
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục Cache settings cho PID={pid}: {e}")
            return False


class MemoryResourceManager:
    """
    Quản lý tài nguyên Memory thông qua việc sử dụng ulimit.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def set_memory_limit(self, pid: int, memory_limit_mb: int) -> bool:
        """
        Thiết lập giới hạn bộ nhớ cho tiến trình thông qua ulimit.

        Args:
            pid (int): PID của tiến trình.
            memory_limit_mb (int): Giới hạn bộ nhớ tính bằng MB.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            process = psutil.Process(pid)
            mem_bytes = memory_limit_mb * 1024 * 1024
            process.rlimit(psutil.RLIMIT_AS, (mem_bytes, mem_bytes))
            self.logger.debug(f"Đã đặt giới hạn bộ nhớ cho PID={pid} là {memory_limit_mb}MB.")
            return True
        except psutil.NoSuchProcess:
            self.logger.error(f"Tiến trình PID={pid} không tồn tại khi đặt giới hạn bộ nhớ.")
            return False
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền để đặt giới hạn bộ nhớ cho PID={pid}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi đặt giới hạn bộ nhớ cho PID={pid}: {e}")
            return False

    def get_memory_limit(self, pid: int) -> float:
        """
        Lấy giới hạn bộ nhớ đã thiết lập cho tiến trình cụ thể.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            float: Giới hạn bộ nhớ tính bằng bytes. Trả về float('inf') nếu không giới hạn hoặc lỗi.
        """
        try:
            process = psutil.Process(pid)
            mem_limit = process.rlimit(psutil.RLIMIT_AS)
            if mem_limit and mem_limit[1] != psutil.RLIM_INFINITY:
                self.logger.debug(f"Giới hạn bộ nhớ cho PID={pid} là {mem_limit[1]} bytes.")
                return float(mem_limit[1])
            else:
                self.logger.debug(f"Giới hạn bộ nhớ cho PID={pid} là không giới hạn.")
                return float('inf')
        except psutil.NoSuchProcess:
            self.logger.error(f"Tiến trình PID={pid} không tồn tại khi lấy giới hạn bộ nhớ.")
            return 0.0
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền để lấy giới hạn bộ nhớ cho PID={pid}.")
            return 0.0
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy giới hạn bộ nhớ cho PID={pid}: {e}")
            return 0.0

    def remove_memory_limit(self, pid: int) -> bool:
        """
        Khôi phục giới hạn bộ nhớ cho tiến trình về không giới hạn.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            process = psutil.Process(pid)
            process.rlimit(psutil.RLIMIT_AS, (psutil.RLIM_INFINITY, psutil.RLIM_INFINITY))
            self.logger.debug(f"Đã khôi phục giới hạn bộ nhớ cho PID={pid} về không giới hạn.")
            return True
        except psutil.NoSuchProcess:
            self.logger.error(f"Tiến trình PID={pid} không tồn tại khi khôi phục giới hạn bộ nhớ.")
            return False
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền để khôi phục giới hạn bộ nhớ cho PID={pid}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục giới hạn bộ nhớ cho PID={pid}: {e}")
            return False

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục giới hạn bộ nhớ cho tiến trình về không giới hạn.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        return self.remove_memory_limit(pid)


class NetworkResourceManager:
    """
    Quản lý tài nguyên mạng thông qua iptables và tc.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.process_marks: Dict[int, int] = {}  # Mapping PID to fwmark

    def mark_packets(self, pid: int, mark: int) -> bool:
        """
        Thêm quy tắc iptables để đánh dấu các gói tin từ PID này.

        Args:
            pid (int): PID của tiến trình.
            mark (int): fwmark để đặt.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            subprocess.run([
                'iptables', '-A', 'OUTPUT', '-m', 'owner', '--pid-owner', str(pid),
                '-j', 'MARK', '--set-mark', str(mark)
            ], check=True)
            self.logger.debug(f"Đặt iptables MARK cho PID={pid} với mark={mark}.")
            self.process_marks[pid] = mark
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi đặt iptables MARK cho PID={pid}: {e}")
            return False

    def unmark_packets(self, pid: int, mark: int) -> bool:
        """
        Xóa quy tắc iptables đánh dấu các gói tin từ PID này.

        Args:
            pid (int): PID của tiến trình.
            mark (int): fwmark đã đặt.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            subprocess.run([
                'iptables', '-D', 'OUTPUT', '-m', 'owner', '--pid-owner', str(pid),
                '-j', 'MARK', '--set-mark', str(mark)
            ], check=True)
            self.logger.debug(f"Xóa iptables MARK cho PID={pid} với mark={mark}.")
            if pid in self.process_marks:
                del self.process_marks[pid]
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi xóa iptables MARK cho PID={pid}: {e}")
            return False

    def limit_bandwidth(self, interface: str, mark: int, bandwidth_mbps: float) -> bool:
        """
        Giới hạn băng thông mạng dựa trên fwmark thông qua tc.

        Args:
            interface (str): Tên giao diện mạng.
            mark (int): fwmark để lọc.
            bandwidth_mbps (float): Giới hạn băng thông tính bằng Mbps.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            # Thêm qdisc nếu chưa tồn tại
            subprocess.run([
                'tc', 'qdisc', 'add', 'dev', interface, 'root', 'handle', '1:',
                'htb', 'default', '12'
            ], check=True)
            self.logger.debug(f"Đã thêm tc qdisc 'htb' cho giao diện '{interface}'.")

            # Thêm class để giới hạn băng thông
            subprocess.run([
                'tc', 'class', 'add', 'dev', interface, 'parent', '1:', 'classid', '1:1',
                'htb', 'rate', f'{bandwidth_mbps}mbit'
            ], check=True)
            self.logger.debug(f"Đã thêm tc class '1:1' với rate={bandwidth_mbps}mbit cho giao diện '{interface}'.")

            # Thêm filter để áp dụng giới hạn cho các gói tin có mark
            subprocess.run([
                'tc', 'filter', 'add', 'dev', interface, 'protocol', 'ip',
                'parent', '1:', 'prio', '1', 'handle', str(mark), 'fw', 'flowid', '1:1'
            ], check=True)
            self.logger.debug(f"Đã thêm tc filter cho mark={mark} trên giao diện '{interface}'.")

            self.process_marks[mark] = mark  # Giả định mapping mark -> mark

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi giới hạn băng thông mạng: {e}")
            return False

    def remove_bandwidth_limit(self, interface: str, mark: int) -> bool:
        """
        Xóa giới hạn băng thông mạng dựa trên fwmark thông qua tc.

        Args:
            interface (str): Tên giao diện mạng.
            mark (int): fwmark đã đặt.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            # Xóa filter
            subprocess.run([
                'tc', 'filter', 'del', 'dev', interface, 'protocol', 'ip',
                'parent', '1:', 'prio', '1', 'handle', str(mark), 'fw', 'flowid', '1:1'
            ], check=True)
            self.logger.debug(f"Đã xóa tc filter cho mark={mark} trên giao diện '{interface}'.")

            # Xóa class
            subprocess.run([
                'tc', 'class', 'del', 'dev', interface, 'parent', '1:', 'classid', '1:1'
            ], check=True)
            self.logger.debug(f"Đã xóa tc class '1:1' trên giao diện '{interface}'.")

            # Xóa qdisc nếu không còn class nào
            subprocess.run([
                'tc', 'qdisc', 'del', 'dev', interface, 'root', 'handle', '1:', 'htb'
            ], check=True)
            self.logger.debug(f"Đã xóa tc qdisc 'htb' cho giao diện '{interface}'.")

            # Xóa mapping mark
            if mark in self.process_marks:
                del self.process_marks[mark]

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi xóa giới hạn băng thông mạng: {e}")
            return False

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục các thiết lập mạng cho tiến trình bằng cách xóa giới hạn băng thông và unmark packets.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            mark = self.process_marks.get(pid)
            if not mark:
                self.logger.warning(f"Không tìm thấy mark cho PID={pid} trong NetworkResourceManager.")
                return False

            # Xác định giao diện mạng từ cấu hình hoặc mặc định
            # Giả định cấu hình được truyền từ nơi khác hoặc mặc định 'eth0'
            interface = 'eth0'  # Bạn có thể lấy từ cấu hình nếu cần

            # Xóa giới hạn băng thông
            success_bw = self.remove_bandwidth_limit(interface, mark)
            if success_bw:
                self.logger.info(f"Đã khôi phục giới hạn băng thông mạng cho PID={pid} với mark={mark} trên giao diện '{interface}'.")
            else:
                self.logger.error(f"Không thể khôi phục giới hạn băng thông mạng cho PID={pid}.")

            # Xóa iptables MARK
            success_unmark = self.unmark_packets(pid, mark)
            if success_unmark:
                self.logger.info(f"Đã xóa iptables MARK cho PID={pid} với mark={mark}.")
            else:
                self.logger.error(f"Không thể xóa iptables MARK cho PID={pid} với mark={mark}.")

            return success_bw and success_unmark
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục Network settings cho PID={pid}: {e}")
            return False


class DiskIOResourceManager:
    """
    Quản lý tài nguyên Disk I/O thông qua tc hoặc các cơ chế kiểm soát I/O khác.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.process_io_limits: Dict[int, float] = {}  # Mapping PID to I/O rate limit

    def limit_io(self, interface: str, rate_mbps: float) -> bool:
        """
        Giới hạn tốc độ I/O Disk sử dụng tc.

        Args:
            interface (str): Tên giao diện mạng.
            rate_mbps (float): Giới hạn tốc độ I/O tính bằng Mbps.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            # Thêm qdisc root nếu chưa tồn tại
            subprocess.run([
                'tc', 'qdisc', 'add', 'dev', interface, 'root', 'handle', '1:',
                'htb', 'default', '12'
            ], check=True)
            self.logger.debug(f"Đã thêm tc qdisc 'htb' cho giao diện '{interface}'.")

            # Thêm class để giới hạn I/O
            subprocess.run([
                'tc', 'class', 'add', 'dev', interface, 'parent', '1:', 'classid', '1:1',
                'htb', 'rate', f'{rate_mbps}mbit'
            ], check=True)
            self.logger.debug(f"Đã thêm tc class '1:1' với rate={rate_mbps}mbit cho giao diện '{interface}'.")

            # Lưu mapping PID -> rate_mbps nếu cần thiết
            # Trong ví dụ này, giả định rằng giới hạn I/O là toàn bộ cho giao diện
            # Nếu giới hạn theo PID, cần sử dụng thêm các filter và mapping khác

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi giới hạn Disk I/O: {e}")
            return False

    def remove_io_limit(self, interface: str) -> bool:
        """
        Xóa giới hạn I/O Disk sử dụng tc.

        Args:
            interface (str): Tên giao diện mạng.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            # Xóa qdisc root
            subprocess.run([
                'tc', 'qdisc', 'del', 'dev', interface, 'root', 'handle', '1:', 'htb'
            ], check=True)
            self.logger.debug(f"Đã xóa tc qdisc 'htb' cho giao diện '{interface}'.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi xóa giới hạn Disk I/O: {e}")
            return False

    def set_io_weight(self, pid: int, io_weight: int) -> bool:
        """
        Đặt trọng số I/O cho tiến trình.

        Args:
            pid (int): PID của tiến trình.
            io_weight (int): Trọng số I/O (1-1000).

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            # Đây là một ví dụ placeholder.
            # Thực tế cần sử dụng cgroups hoặc các cơ chế kiểm soát I/O khác để đặt trọng số I/O.
            self.logger.debug(f"Đặt trọng số I/O cho PID={pid} là {io_weight}. (Chưa triển khai)")
            # Triển khai thực tế ở đây nếu cần.
            self.process_io_limits[pid] = io_weight
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi đặt trọng số I/O cho PID={pid}: {e}")
            return False

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục giới hạn I/O Disk cho tiến trình bằng cách xóa giới hạn đã đặt.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            if pid in self.process_io_limits:
                del self.process_io_limits[pid]
                self.logger.info(f"Đã khôi phục Disk I/O settings cho PID={pid}.")
                return True
            else:
                self.logger.warning(f"Không tìm thấy thiết lập Disk I/O cho PID={pid}.")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục Disk I/O settings cho PID={pid}: {e}")
            return False


class CacheResourceManager:
    """
    Quản lý tài nguyên Cache thông qua việc drop caches.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.dropped_pids: List[int] = []  # Danh sách PID đã drop caches

    def drop_caches(self, pid: Optional[int] = None) -> bool:
        """
        Drop caches bằng cách ghi vào /proc/sys/vm/drop_caches.

        Args:
            pid (Optional[int], optional): PID của tiến trình nếu cần theo dõi. Defaults to None.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3\n')
            self.logger.debug("Đã drop caches thành công.")
            if pid:
                self.dropped_pids.append(pid)
            return True
        except PermissionError:
            self.logger.error("Không đủ quyền để drop caches.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi drop caches: {e}")
            return False

    def limit_cache_usage(self, cache_limit_percent: float, pid: Optional[int] = None) -> bool:
        """
        Giới hạn mức sử dụng cache bằng cách drop caches và các phương pháp khác nếu cần.

        Args:
            cache_limit_percent (float): Phần trăm giới hạn sử dụng cache.
            pid (Optional[int], optional): PID của tiến trình nếu cần theo dõi. Defaults to None.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            # Ví dụ: sử dụng drop_caches
            success = self.drop_caches(pid)
            if not success:
                return False

            # Các biện pháp giới hạn cache khác có thể được thêm vào đây.
            self.logger.debug(f"Giới hạn mức sử dụng cache xuống còn {cache_limit_percent}%.")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi giới hạn sử dụng cache: {e}")
            return False

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục các thiết lập Cache cho tiến trình bằng cách đặt lại mức sử dụng cache.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            # Giả định rằng việc khôi phục cache là đặt lại mức sử dụng cache về 100%
            success = self.limit_cache_usage(100.0)
            if success:
                self.logger.info(f"Đã khôi phục Cache settings cho PID={pid} về 100%.")
            else:
                self.logger.error(f"Không thể khôi phục Cache settings cho PID={pid}.")
            return success
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục Cache settings cho PID={pid}: {e}")
            return False


class MemoryResourceManager:
    """
    Quản lý tài nguyên Memory thông qua việc sử dụng ulimit.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def set_memory_limit(self, pid: int, memory_limit_mb: int) -> bool:
        """
        Thiết lập giới hạn bộ nhớ cho tiến trình thông qua ulimit.

        Args:
            pid (int): PID của tiến trình.
            memory_limit_mb (int): Giới hạn bộ nhớ tính bằng MB.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            process = psutil.Process(pid)
            mem_bytes = memory_limit_mb * 1024 * 1024
            process.rlimit(psutil.RLIMIT_AS, (mem_bytes, mem_bytes))
            self.logger.debug(f"Đã đặt giới hạn bộ nhớ cho PID={pid} là {memory_limit_mb}MB.")
            return True
        except psutil.NoSuchProcess:
            self.logger.error(f"Tiến trình PID={pid} không tồn tại khi đặt giới hạn bộ nhớ.")
            return False
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền để đặt giới hạn bộ nhớ cho PID={pid}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi đặt giới hạn bộ nhớ cho PID={pid}: {e}")
            return False

    def get_memory_limit(self, pid: int) -> float:
        """
        Lấy giới hạn bộ nhớ đã thiết lập cho tiến trình cụ thể.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            float: Giới hạn bộ nhớ tính bằng bytes. Trả về float('inf') nếu không giới hạn hoặc lỗi.
        """
        try:
            process = psutil.Process(pid)
            mem_limit = process.rlimit(psutil.RLIMIT_AS)
            if mem_limit and mem_limit[1] != psutil.RLIM_INFINITY:
                self.logger.debug(f"Giới hạn bộ nhớ cho PID={pid} là {mem_limit[1]} bytes.")
                return float(mem_limit[1])
            else:
                self.logger.debug(f"Giới hạn bộ nhớ cho PID={pid} là không giới hạn.")
                return float('inf')
        except psutil.NoSuchProcess:
            self.logger.error(f"Tiến trình PID={pid} không tồn tại khi lấy giới hạn bộ nhớ.")
            return 0.0
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền để lấy giới hạn bộ nhớ cho PID={pid}.")
            return 0.0
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy giới hạn bộ nhớ cho PID={pid}: {e}")
            return 0.0

    def remove_memory_limit(self, pid: int) -> bool:
        """
        Khôi phục giới hạn bộ nhớ cho tiến trình về không giới hạn.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            process = psutil.Process(pid)
            process.rlimit(psutil.RLIMIT_AS, (psutil.RLIM_INFINITY, psutil.RLIM_INFINITY))
            self.logger.debug(f"Đã khôi phục giới hạn bộ nhớ cho PID={pid} về không giới hạn.")
            return True
        except psutil.NoSuchProcess:
            self.logger.error(f"Tiến trình PID={pid} không tồn tại khi khôi phục giới hạn bộ nhớ.")
            return False
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền để khôi phục giới hạn bộ nhớ cho PID={pid}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục giới hạn bộ nhớ cho PID={pid}: {e}")
            return False

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục giới hạn bộ nhớ cho tiến trình về không giới hạn.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        return self.remove_memory_limit(pid)


class ResourceControlFactory:
    """
    Factory để tạo các instance của các resource manager.
    """

    @staticmethod
    def create_resource_managers(logger: logging.Logger) -> Dict[str, Any]:
        """
        Tạo và trả về một dictionary chứa các resource manager.

        Args:
            logger (logging.Logger): Logger để ghi log.

        Returns:
            Dict[str, Any]: Dictionary chứa các resource manager.
        """
        # Khởi tạo GPUManager
        gpu_manager = GPUManager()

        # Tạo các resource manager
        resource_managers = {
            'cpu': CPUResourceManager(logger),
            'gpu': GPUResourceManager(logger, gpu_manager),
            'network': NetworkResourceManager(logger),
            'io': DiskIOResourceManager(logger),
            'cache': CacheResourceManager(logger),
            'memory': MemoryResourceManager(logger)
        }

        return resource_managers
