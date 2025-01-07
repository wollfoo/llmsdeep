# cloak_strategies.py

import os
import subprocess
import psutil
import pynvml
import logging
import traceback
import time
from retrying import retry
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from threading import Semaphore, Thread
from ratelimiter import RateLimiter



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
    Chiến lược cloaking cho CPU:
      - Giới hạn sử dụng CPU bằng cgroup-tools (quota, affinity).
      - Giới hạn số luồng đồng thời (Semaphore).
      - Giới hạn tần suất thực thi (RateLimiter).
      - Tối ưu logic ứng dụng (caching) ở tầng ứng dụng.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        # Các biến cấu hình
        self.throttle_percentage = config.get('throttle_percentage', 20)
        if not isinstance(self.throttle_percentage, (int, float)) or not (0 <= self.throttle_percentage <= 100):
            logger.warning("Giá trị throttle_percentage không hợp lệ, mặc định 20%.")
            self.throttle_percentage = 20

        self.max_concurrent_threads = config.get('max_concurrent_threads', 4)
        if not isinstance(self.max_concurrent_threads, int) or self.max_concurrent_threads <= 0:
            logger.warning("Giá trị max_concurrent_threads không hợp lệ, mặc định 4.")
            self.max_concurrent_threads = 4
        self.thread_semaphore = Semaphore(self.max_concurrent_threads)

        self.max_calls_per_second = config.get('max_calls_per_second', 5)
        self.rate_limiter = RateLimiter(max_calls=self.max_calls_per_second, period=1)

        self.cache_enabled = config.get('cache_enabled', True)
        self.task_cache = {} if self.cache_enabled else None

        self.logger = logger

        # Lưu tên cgroup để thuận tiện cleanup về sau
        self.created_cgroups: List[str] = []

    def apply(self, process: Any) -> Dict[str, Any]:
        """
        Áp dụng throttling CPU (cgroup-tools) + throttling tầng ứng dụng cho tiến trình.
        """

        try:
            # Lấy PID
            if isinstance(process, subprocess.Popen):
                # Nếu là Popen, ta biết chắc process có thuộc tính .pid nhưng không có .name
                pid = process.pid
                try:
                    # Lấy tên process từ psutil
                    p = psutil.Process(pid)
                    process_name = p.name()
                except psutil.NoSuchProcess:
                    process_name = "unknown"
            else:
                # Ngược lại, nếu bạn có class custom, hoặc process thực sự có .name
                if not hasattr(process, 'pid'):
                    self.logger.error("Process không có pid.")
                    return {}
                pid = process.pid

                # Thử lấy thuộc tính name, nếu không có thì fallback sang psutil
                process_name = getattr(process, 'name', None)
                if not process_name:
                    self.logger.warning(
                        f"Process PID={pid} không có .name, đang thử lấy tên qua psutil..."
                    )
                    try:
                        p = psutil.Process(pid)
                        process_name = p.name()
                    except psutil.NoSuchProcess:
                        process_name = "unknown"
                        self.logger.warning(
                            f"Tiến trình PID={pid} không tồn tại. Sử dụng process_name='unknown'."
                        )

            # Thiết lập cgroups
            self.setup_cgroups(pid)

            adjustments = {
                'cpu_quota_us': self.calculate_cpu_quota(),
                'cpu_affinity': self.get_available_cpu_cores(),
                'max_concurrent_threads': self.max_concurrent_threads,
                'max_calls_per_second': self.max_calls_per_second,
                'cache_enabled': self.cache_enabled
            }

            self.logger.info(
                f"Áp dụng throttling CPU: quota={adjustments['cpu_quota_us']}us, "
                f"affinity={adjustments['cpu_affinity']}, "
                f"max_concurrent_threads={self.max_concurrent_threads}, "
                f"max_calls_per_second={self.max_calls_per_second}, "
                f"cache_enabled={self.cache_enabled} "
                f"cho tiến trình {process_name} (PID: {pid})."
            )
            return adjustments

        except Exception as e:
            self.logger.error(
                f"Lỗi khi chuẩn bị throttling CPU cho tiến trình {getattr(process, 'name', 'unknown')} "
                f"(PID: {getattr(process, 'pid', 'N/A')}): {e}\n{traceback.format_exc()}"
            )
            raise

    def setup_cgroups(self, pid: int):
        """
        Tạo và cấu hình cgroup (cpu, cpuset), gán quota + affinity cho PID.
        """
        try:
            cpu_cgroup = f"cpu_cloak_{pid}"
            cpuset_cgroup = f"cpuset_cloak_{pid}"

            # Tạo cgroup CPU
            if self.create_cgroup(cpu_cgroup, controller='cpu'):
                self.created_cgroups.append(f'cpu:/{cpu_cgroup}')

            # Tạo cgroup CPUSET
            if self.create_cgroup(cpuset_cgroup, controller='cpuset'):
                self.created_cgroups.append(f'cpuset:/{cpuset_cgroup}')

            # Thiết lập quota + affinity
            cpu_quota_us = self.calculate_cpu_quota()
            self.set_cpu_quota(cpu_cgroup, cpu_quota_us)

            available_cores = self.get_available_cpu_cores()
            self.set_cpu_affinity(cpuset_cgroup, available_cores)

            # Thiết lập cpuset.mems
            self.set_cpu_mems(cpuset_cgroup, [0])  # Giả sử hệ thống có một memory node

            # Gán PID
            self.assign_process_to_cgroup(cpu_cgroup, pid, controller='cpu')
            self.assign_process_to_cgroup(cpuset_cgroup, pid, controller='cpuset')

        except Exception as e:
            self.logger.error(f"Lỗi khi thiết lập cgroups cho PID {pid}: {e}")
            raise

    def set_cpu_mems(self, cgroup_name: str, mems: List[int]):
        """Thiết lập danh sách memory nodes cho cpuset cgroup."""
        try:
            mems_str = ",".join(map(str, mems))
            subprocess.run(['cgset', '-r', f'cpuset.mems={mems_str}', cgroup_name], check=True)
            self.logger.info(f"Đặt cpuset.mems={mems_str} cho cgroup cpuset '{cgroup_name}'.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi cgset cpuset.mems cho '{cgroup_name}': {e.stderr}")
            raise
        except Exception as e:
            self.logger.error(f"Lỗi set_cpu_mems: {e}")
            raise

    def create_cgroup(self, cgroup_name: str, controller: str) -> bool:
        """
        Tạo cgroup (nếu chưa tồn tại). Trả về True nếu tạo thành công hoặc đã tồn tại, False nếu lỗi.
        """
        try:
            # Kiểm tra cgroup
            if controller == 'cpu':
                check_cmd = ['cgget', '-n', '-r', 'cpu.cfs_quota_us', cgroup_name]
            elif controller == 'cpuset':
                check_cmd = ['cgget', '-n', '-r', 'cpuset.cpus', cgroup_name]
            else:
                self.logger.error(f"Controller không được hỗ trợ: {controller}")
                return False

            result = subprocess.run(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                # Chưa tồn tại => tạo
                subprocess.run(['cgcreate', '-g', f'{controller}:/'+cgroup_name], check=True)
                self.logger.info(f"Đã tạo cgroup '{cgroup_name}' cho controller '{controller}'.")
            else:
                self.logger.debug(f"Cgroup '{cgroup_name}' (controller '{controller}') đã tồn tại.")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi cgcreate/check cgroup '{cgroup_name}': {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi create_cgroup: {e}")
            return False

    def calculate_cpu_quota(self) -> int:
        """Tính quota CPU dựa trên self.throttle_percentage."""
        cpu_period_us = 100000  # 100ms
        return int(cpu_period_us * (1 - self.throttle_percentage / 100))

    def set_cpu_quota(self, cgroup_name: str, quota_us: int):
        """Thiết lập quota cho cgroup CPU."""
        try:
            subprocess.run(['cgset', '-r', f'cpu.cfs_quota_us={quota_us}', cgroup_name], check=True)
            self.logger.info(f"Đặt CPU quota={quota_us}us cho cgroup '{cgroup_name}'.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi cgset CPU quota cho cgroup '{cgroup_name}': {e.stderr}")
            raise

    def get_available_cpu_cores(self) -> List[int]:
        """Trả về danh sách core CPU (vd: [0,1,2,3])."""
        try:
            total_cores = psutil.cpu_count(logical=True)
            return list(range(total_cores))
        except Exception as e:
            self.logger.error(f"Lỗi get_available_cpu_cores: {e}")
            return []

    def set_cpu_affinity(self, cgroup_name: str, cores: List[int]):
        """Thiết lập danh sách core cho cpuset cgroup."""
        try:
            cores_str = ",".join(map(str, cores))
            subprocess.run(['cgset', '-r', f'cpuset.cpus={cores_str}', cgroup_name], check=True)
            self.logger.info(f"Đặt CPU affinity={cores_str} cho cgroup cpuset '{cgroup_name}'.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi cgset cpuset cho '{cgroup_name}': {e.stderr}")
            raise

    def assign_process_to_cgroup(self, cgroup_name: str, pid: int, controller: str):
        """Gán PID vào cgroup."""
        try:
            subprocess.run(['cgclassify', '-g', f'{controller}:/'+cgroup_name, str(pid)], check=True)
            self.logger.info(f"Đã gán PID={pid} vào cgroup '{cgroup_name}' controller={controller}.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi cgclassify gán PID {pid} cho cgroup '{cgroup_name}': {e.stderr}")
            raise

    # ----------------------------
    # Thêm hàm cleanup cgroup
    # ----------------------------
    def cleanup_cgroups(self):
        """
        Xóa các cgroup đã tạo (nếu tồn tại).
        Chỉ xóa những cgroup đã lưu trong self.created_cgroups.
        """
        for full_cgroup_path in self.created_cgroups:
            # full_cgroup_path thường dạng 'cpu:/cpu_cloak_XXXX'
            try:
                # cgdelete: cgdelete -g cpu:/cpu_cloak_XXXX
                subprocess.run(['cgdelete', '-g', full_cgroup_path], check=True)
                self.logger.info(f"Đã xóa cgroup '{full_cgroup_path}'.")
            except subprocess.CalledProcessError as e:
                # Nếu cgroup không tồn tại => log warning
                self.logger.warning(f"Không thể xóa cgroup '{full_cgroup_path}': {e.stderr}")
            except Exception as ex:
                self.logger.warning(f"Lỗi bất ngờ khi xóa cgroup '{full_cgroup_path}': {ex}")
        # Xóa danh sách để tránh xóa lại
        self.created_cgroups.clear()

    # ----------------------------
    # Rate limiting & logic caching
    # ----------------------------
    def acquire_thread_slot(self):
        self.logger.debug("Đang cố gắng chiếm 1 slot luồng.")
        self.thread_semaphore.acquire()
        self.logger.debug("Đã chiếm slot luồng thành công.")

    def release_thread_slot(self):
        self.thread_semaphore.release()
        self.logger.debug("Đã trả slot luồng.")

    def throttled_task(self, task_id: int, task_func):
        """
        Thực thi một nhiệm vụ kèm rate limit, caching, và số luồng đồng thời.
        """
        try:
            self.acquire_thread_slot()
            self.logger.info(f"Task {task_id} bắt đầu.")

            with self.rate_limiter:
                if self.cache_enabled and task_id in self.task_cache:
                    self.logger.info(f"Task {task_id} lấy từ cache.")
                    result = self.task_cache[task_id]
                else:
                    result = task_func(task_id)
                    if self.cache_enabled:
                        self.task_cache[task_id] = result
                        self.logger.info(f"Kết quả task {task_id} được cache.")

            self.logger.info(f"Task {task_id} hoàn thành.")
        except Exception as e:
            self.logger.error(f"Lỗi task {task_id}: {e}")
        finally:
            self.release_thread_slot()

    def run_tasks(self, tasks: List[int], task_func):
        """
        Chạy nhiều task với giới hạn luồng + rate limit.
        """
        threads = []
        for tid in tasks:
            t = Thread(target=self.throttled_task, args=(tid, task_func))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def clear_cache(self):
        """Xóa cache tác vụ."""
        if self.cache_enabled:
            self.task_cache.clear()
            self.logger.info("Đã xóa cache tác vụ.")

    def get_cached_task(self, task_id: int):
        """Lấy kết quả tác vụ trong cache (nếu có)."""
        if self.cache_enabled:
            return self.task_cache.get(task_id, None)
        return None

    def optimize_task_execution(self, task_id: int, task_func):
        """
        Thực thi tối ưu (caching, ...).
        """
        if self.cache_enabled and task_id in self.task_cache:
            self.logger.info(f"Task {task_id} lấy từ cache (optimize).")
            return self.task_cache[task_id]
        else:
            result = task_func(task_id)
            if self.cache_enabled:
                self.task_cache[task_id] = result
            return result


class GpuCloakStrategy(CloakStrategy):
    """
    Cloaking strategy for GPU.
    Throttles GPU power limit, and optionally adjusts GPU clocks & usage.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, gpu_initialized: bool):
        """
        Args:
            config (Dict[str, Any]): Configuration dictionary. 
                Possible keys (bên cạnh throttle_percentage):
                  - usage_threshold (int/float): Ngưỡng GPU usage (%) 
                                                  mà tại đó mới kích hoạt throttling.
                  - target_sm_clock (int): Xung nhịp SM (MHz) mong muốn.
                  - target_mem_clock (int): Xung nhịp bộ nhớ (MHz) mong muốn.
            logger (logging.Logger): Logger.
            gpu_initialized (bool): Cờ cho biết NVML đã init hay chưa.
        """
        # Mức giảm power limit (phần trăm)
        self.throttle_percentage = config.get('throttle_percentage', 20)
        if not isinstance(self.throttle_percentage, (int, float)) or not (0 <= self.throttle_percentage <= 100):
            logger.warning("Invalid throttle_percentage, defaulting to 20%.")
            self.throttle_percentage = 20

        # Mở rộng: ngưỡng usage, nếu GPU usage < ngưỡng => bỏ qua
        self.usage_threshold = config.get('usage_threshold', 80)
        if not isinstance(self.usage_threshold, (int, float)) or not (0 <= self.usage_threshold <= 100):
            logger.warning("Invalid usage_threshold, defaulting to 80%.")
            self.usage_threshold = 80

        # Mở rộng: thiết lập xung nhịp (nếu driver cho phép)
        self.target_sm_clock = config.get('target_sm_clock', 1200)   # ví dụ 1200MHz
        self.target_mem_clock = config.get('target_mem_clock', 800) # ví dụ 800MHz

        self.logger = logger
        self.gpu_initialized = gpu_initialized

        # Log thông tin khởi tạo
        self.logger.debug(
            f"GpuCloakStrategy initialized with throttle_percentage={self.throttle_percentage}%, "
            f"usage_threshold={self.usage_threshold}%, "
            f"target_sm_clock={self.target_sm_clock}MHz, target_mem_clock={self.target_mem_clock}MHz."
        )

    def apply(self, process: Any) -> Dict[str, Any]:
        """
        Apply GPU throttling to the given process, potentially adjusting power limit
        and GPU clocks, depending on usage_threshold.

        Args:
            process (Any): Process object with attributes 'pid' and 'name'.

        Returns:
            Dict[str, Any]: A dictionary of adjustments. E.g.:
                {
                    "gpu_index": int,
                    "gpu_power_limit": int,  # in Watts
                    "set_sm_clock": int,     # in MHz (optional)
                    "set_mem_clock": int     # in MHz (optional)
                }
        """
        if not self.gpu_initialized:
            self.logger.warning(
                f"GPU not initialized. Cannot prepare GPU throttling for process "
                f"{getattr(process, 'name', 'unknown')} (PID: {getattr(process, 'pid', 'N/A')})."
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

                # Lấy handle của GPU
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

                # Đọc GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu  # % usage
                mem_util = utilization.memory

                self.logger.info(
                    f"Current GPU usage for process {process.name} (PID: {process.pid}), "
                    f"GPU index={gpu_index}: gpu={gpu_util}%, mem={mem_util}%"
                )

                # Nếu usage thấp hơn ngưỡng => chưa cần throttle
                if gpu_util < self.usage_threshold:
                    self.logger.info(
                        f"GPU usage {gpu_util}% < usage_threshold={self.usage_threshold}%. "
                        "Skip additional throttling."
                    )
                    return {}

                # --- BẮT ĐẦU THROTTLE: Điều chỉnh power limit như cũ ---

                # Lấy giới hạn power hiện tại và constraints
                current_power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                min_limit_mw, max_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)

                # Chuyển sang W
                current_power_limit_w = current_power_limit_mw / 1000
                min_w = min_limit_mw / 1000
                max_w = max_limit_mw / 1000

                desired_power_limit_w = current_power_limit_w * (1 - self.throttle_percentage / 100)

                # Clamp power limit mới vào [min_w, max_w]
                if desired_power_limit_w < min_w:
                    self.logger.warning(
                        f"Desired GPU power limit {desired_power_limit_w}W < min_limit {min_w}W. Clamping to {min_w}W."
                    )
                    new_power_limit_w = min_w
                elif desired_power_limit_w > max_w:
                    self.logger.warning(
                        f"Desired GPU power limit {desired_power_limit_w}W > max_limit {max_w}W. Clamping to {max_w}W."
                    )
                    new_power_limit_w = max_w
                else:
                    new_power_limit_w = desired_power_limit_w

                # Làm tròn
                new_power_limit_w = int(round(new_power_limit_w))

                # --- MỞ RỘNG: Đặt xung nhịp GPU (SM, Memory) nếu driver cho phép ---
                #   Tùy Tesla V100 driver, có thể cần root, hoặc GPU in P0 state, v.v.
                try:
                    pynvml.nvmlDeviceSetApplicationsClocks(
                        handle, 
                        self.target_mem_clock,  # MHz
                        self.target_sm_clock    # MHz
                    )
                    self.logger.info(
                        f"Set GPU clocks => SM={self.target_sm_clock}MHz, MEM={self.target_mem_clock}MHz "
                        f"on GPU index={gpu_index} for PID={process.pid}"
                    )
                except pynvml.NVMLError as e:
                    # Nếu driver/hardware không cho phép => log warning
                    self.logger.warning(
                        f"Failed to set GPU clocks for GPU index={gpu_index}, PID={process.pid}: {e}"
                    )

                # Tạo dict adjustments
                adjustments = {
                    "gpu_index": gpu_index,
                    "gpu_power_limit": new_power_limit_w,  # in Watts
                    # Bổ sung thêm 2 key thông báo xung
                    "set_sm_clock": self.target_sm_clock,
                    "set_mem_clock": self.target_mem_clock
                }

                self.logger.info(
                    f"Prepared GPU throttling adjustments: GPU {gpu_index} => power limit={new_power_limit_w}W "
                    f"({self.throttle_percentage}% reduction), SM clock={self.target_sm_clock}MHz, "
                    f"Mem clock={self.target_mem_clock}MHz for process {process.name} (PID: {process.pid})."
                )
                return adjustments

            finally:
                # Tắt NVML sau khi sử dụng
                pynvml.nvmlShutdown()

        except pynvml.NVMLError as e:
            self.logger.error(
                f"NVML error preparing GPU throttling for process {process.name} (PID={process.pid}): {e}"
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Unexpected error preparing GPU throttling for process {process.name} (PID={process.pid}): {e}"
            )
            raise

    def assign_gpu(self, pid: int, gpu_count: int) -> int:
        """
        Assign a GPU to a process based on PID (default: pid % gpu_count).

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
