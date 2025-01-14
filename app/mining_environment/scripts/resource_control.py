# resource_control.py

import os
import uuid
import logging
import subprocess
from typing import Any, Dict, List, Optional, Tuple
import psutil
import pynvml  # NVIDIA Management Library
import asyncio

# Import GPUManager từ utils.py
from .utils import GPUManager  # Giả định GPUManager đã định nghĩa đầy đủ trong utils.py

###############################################################################
#                           CPU RESOURCE MANAGER                              #
###############################################################################
class CPUResourceManager:
    """
    Quản lý tài nguyên CPU sử dụng cgroups, affinity, và tối ưu hóa CPU.
    
    Trong mô hình event-driven:
      - Hệ thống (hoặc ResourceManager) sẽ gọi hàm throttle_cpu_usage(...) 
        khi có sự kiện cần giới hạn CPU (VD: quá nhiệt).
      - Gọi restore_resources(...) khi sự kiện “khôi phục CPU” xảy ra.
      - Không còn vòng lặp polling bên trong CPUResourceManager.
    """

    CGROUP_BASE_PATH = "/sys/fs/cgroup/cpu_cloak"

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        # Lưu thông tin PID -> cgroup
        self.process_cgroup: Dict[int, str] = {}

    async def ensure_cgroup_base(self) -> None:
        """
        Đảm bảo thư mục gốc cho cgroups CPU cloak tồn tại.
        Event-driven: thường gọi một lần khi khởi tạo ResourceManager.
        """
        try:
            if not os.path.exists(self.CGROUP_BASE_PATH):
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, os.makedirs, self.CGROUP_BASE_PATH, True, True)
                self.logger.debug(f"Tạo thư mục cgroup cơ sở tại {self.CGROUP_BASE_PATH}.")
        except PermissionError:
            self.logger.error(f"Không đủ quyền tạo thư mục cgroup tại {self.CGROUP_BASE_PATH}.")
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo thư mục cgroup tại {self.CGROUP_BASE_PATH}: {e}")

    def get_available_cpus(self) -> List[int]:
        """
        Lấy danh sách các core CPU để đặt affinity.
        Trong event-driven, hàm này được gọi khi thực sự cần thông tin core.
        """
        try:
            cpu_count = psutil.cpu_count(logical=True)
            available_cpus = list(range(cpu_count))
            self.logger.debug(f"Available CPUs: {available_cpus}.")
            return available_cpus
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy danh sách CPU cores: {e}")
            return []

    async def create_cgroup(self, pid: int, throttle_percentage: float) -> Optional[str]:
        """
        Tạo cgroup và thiết lập CPU quota (event-driven).
        """
        try:
            if not (0 <= throttle_percentage <= 100):
                self.logger.error(f"throttle_percentage={throttle_percentage} không hợp lệ (0-100).")
                return None

            cgroup_name = f"cpu_cloak_{uuid.uuid4().hex[:8]}"
            cgroup_path = os.path.join(self.CGROUP_BASE_PATH, cgroup_name)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, os.makedirs, cgroup_path, True, True)
            self.logger.debug(f"Tạo cgroup tại {cgroup_path} cho PID={pid}.")

            # Tính CPU quota (dựa trên throttle_percentage)
            cpu_period = 100000  # 100ms
            cpu_quota = int((throttle_percentage / 100) * cpu_period)
            cpu_quota = max(1000, cpu_quota)  # Tránh quota quá nhỏ

            with open(os.path.join(cgroup_path, "cpu.max"), "w") as f:
                f.write(f"{cpu_quota} {cpu_period}\n")
            self.logger.debug(f"Đặt CPU quota={cpu_quota}us cho cgroup {cgroup_name}.")

            # Gán PID vào cgroup
            with open(os.path.join(cgroup_path, "cgroup.procs"), "w") as f:
                f.write(f"{pid}\n")
            self.logger.info(
                f"Thêm PID={pid} vào cgroup {cgroup_name}, throttle_percentage={throttle_percentage}%."
            )

            self.process_cgroup[pid] = cgroup_name
            return cgroup_name

        except PermissionError:
            self.logger.error(f"Không đủ quyền tạo cgroup cho PID={pid}.")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo cgroup cho PID={pid}: {e}")
            return None

    async def delete_cgroup(self, cgroup_name: str) -> bool:
        """
        Xóa cgroup (event-driven).
        Gọi khi có sự kiện “khôi phục” CPU.
        """
        try:
            cgroup_path = os.path.join(self.CGROUP_BASE_PATH, cgroup_name)
            procs_path = os.path.join(cgroup_path, "cgroup.procs")

            with open(procs_path, "r") as f:
                procs = f.read().strip()
                if procs:
                    self.logger.warning(
                        f"Cgroup {cgroup_name} vẫn còn PID={procs}. Không thể xóa."
                    )
                    return False

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, os.rmdir, cgroup_path)
            self.logger.info(f"Xóa cgroup {cgroup_name} thành công.")
            return True
        except FileNotFoundError:
            self.logger.warning(f"Cgroup {cgroup_name} không tồn tại khi xóa.")
            return False
        except PermissionError:
            self.logger.error(f"Không đủ quyền để xóa cgroup {cgroup_name}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi xóa cgroup {cgroup_name}: {e}")
            return False

    ##########################################################################
    #                HÀM EVENT-DRIVEN: THROTTLE & RESTORE                    #
    ##########################################################################
    async def throttle_cpu_usage(self, pid: int, throttle_percentage: float) -> Optional[str]:
        """
        Giới hạn CPU cho PID thông qua cgroup (event-driven).
        Khi ResourceManager cần cloak CPU => gọi hàm này.
        """
        return await self.create_cgroup(pid, throttle_percentage)

    async def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục CPU bằng cách xóa cgroup (event-driven).
        Gọi khi ResourceManager cần “restore CPU” cho PID.
        """
        try:
            cgroup_name = self.process_cgroup.get(pid)
            if not cgroup_name:
                self.logger.warning(f"Không tìm thấy cgroup cho PID={pid} trong CPUResourceManager.")
                return False
            success = await self.delete_cgroup(cgroup_name)
            if success:
                self.logger.info(f"Khôi phục CPU cho PID={pid} thành công (xóa cgroup).")
                del self.process_cgroup[pid]
                return True
            else:
                self.logger.error(f"Không thể khôi phục CPU cho PID={pid}.")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục CPU cho PID={pid}: {e}")
            return False

    ##########################################################################
    #                   HÀM EVENT-DRIVEN KHÁC (AFFINITY,...)                 #
    ##########################################################################
    async def set_cpu_affinity(self, pid: int, cores: List[int]) -> bool:
        """
        Đặt CPU affinity (bất đồng bộ). 
        Gọi khi “cần tối ưu scheduling” hoặc “giới hạn core”.
        """
        try:
            process = psutil.Process(pid)
            await asyncio.get_event_loop().run_in_executor(None, process.cpu_affinity, cores)
            self.logger.debug(f"Đặt CPU affinity cho PID={pid} => {cores}.")
            return True
        except psutil.NoSuchProcess:
            self.logger.error(f"PID={pid} không tồn tại (set_cpu_affinity).")
            return False
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền set_cpu_affinity cho PID={pid}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi set_cpu_affinity cho PID={pid}: {e}")
            return False

    async def reset_cpu_affinity(self, pid: int) -> bool:
        """
        Khôi phục CPU affinity về tất cả core. 
        Event-driven: Gọi khi cần “hủy bỏ giới hạn cores”.
        """
        try:
            available_cpus = self.get_available_cpus()
            return await self.set_cpu_affinity(pid, available_cpus)
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục CPU affinity cho PID={pid}: {e}")
            return False

    async def limit_cpu_for_external_processes(self, target_pids: List[int], throttle_percentage: float) -> bool:
        """
        Giới hạn CPU cho các tiến trình “bên ngoài” (ngoài target_pids). 
        Gọi khi event “cần cloaking external”.
        """
        try:
            if not (0 <= throttle_percentage <= 100):
                self.logger.error(f"throttle_percentage={throttle_percentage} không hợp lệ (0-100).")
                return False

            all_pids = [proc.pid for proc in psutil.process_iter(attrs=['pid'])]
            external_pids = set(all_pids) - set(target_pids)

            tasks = []
            for pid in external_pids:
                tasks.append(self.throttle_cpu_usage(pid, throttle_percentage))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for pid_, result in zip(external_pids, results):
                if isinstance(result, Exception) or not result:
                    self.logger.warning(f"Không thể hạn chế CPU cho PID={pid_}.")

            self.logger.info(
                f"Hạn chế CPU cho {len(external_pids)} tiến trình outside => throttle={throttle_percentage}%."
            )
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi hạn chế CPU cho external processes: {e}")
            return False

    async def optimize_thread_scheduling(self, pid: int, target_cores: Optional[List[int]] = None) -> bool:
        """
        Tối ưu scheduling = đặt CPU affinity (event-driven).
        """
        try:
            success = await self.set_cpu_affinity(
                pid,
                target_cores or self.get_available_cpus()
            )
            if success:
                self.logger.info(
                    f"Đã tối ưu scheduling cho PID={pid}, cores={target_cores or self.get_available_cpus()}."
                )
            return success
        except Exception as e:
            self.logger.error(f"Lỗi optimize_thread_scheduling cho PID={pid}: {e}")
            return False

    async def optimize_cache_usage(self, pid: int) -> bool:
        """
        Tối ưu cache (event-driven). Mặc định: code hiện tại không thay đổi nhiều,
        do cgroups + throttle đã cover 1 phần.
        """
        try:
            # Không thực hiện thêm, placeholder cho tuỳ chỉnh sau này
            self.logger.debug(
                f"Tối ưu cache CPU (PID={pid}) hầu như đã được thực hiện qua cgroups."
            )
            return True
        except Exception as e:
            self.logger.error(f"Lỗi optimize_cache_usage cho PID={pid}: {e}")
            return False


###############################################################################
#                           GPU RESOURCE MANAGER                              #
###############################################################################
class GPUResourceManager:
    """
    Quản lý GPU thông qua NVML (event-driven).
    Các hàm set/get GPU power limit, clock,... được gọi khi cloak/restore GPU 
    (do ResourceManager hoặc CloakStrategy kích hoạt).
    """

    def __init__(self, logger: logging.Logger, gpu_manager: GPUManager):
        self.logger = logger
        self.gpu_manager = gpu_manager
        self.gpu_initialized = False
        # Lưu PID -> GPU Index -> {settings}
        self.process_gpu_settings: Dict[int, Dict[int, Dict[str, Any]]] = {}

    async def initialize(self) -> bool:
        """
        Khởi tạo NVML (event-driven). 
        Gọi một lần lúc ResourceControlFactory khởi tạo.
        """
        try:
            await asyncio.get_event_loop().run_in_executor(None, self.gpu_manager.initialize)
            if self.gpu_manager.gpu_count > 0:
                self.gpu_initialized = True
                self.logger.info("GPUResourceManager sẵn sàng, có GPU.")
            else:
                self.logger.warning("Không có GPU trên hệ thống.")
            return self.gpu_initialized
        except pynvml.NVMLError as error:
            self.logger.error(f"Lỗi khi init NVML: {error}")
            self.gpu_initialized = False
            return False
        except Exception as e:
            self.logger.error(f"Lỗi init GPUResourceManager: {e}")
            self.gpu_initialized = False
            return False

    async def set_gpu_power_limit(self, pid: int, gpu_index: int, power_limit_w: int) -> bool:
        """
        Đặt power limit GPU (event-driven).
        Gọi khi cloak GPU => reduce power, hoặc restore => trả lại power limit cũ.
        """
        if not self.gpu_initialized:
            self.logger.error("GPUResourceManager chưa init. Không thể set power limit.")
            return False
        if not (0 <= gpu_index < self.gpu_manager.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ.")
            return False
        if power_limit_w <= 0:
            self.logger.error("Power limit phải > 0.")
            return False

        try:
            handle = self.gpu_manager.get_handle(gpu_index)
            power_limit_mw = power_limit_w * 1000

            # Lưu lại power limit cũ
            current_power_limit_mw = await asyncio.get_event_loop().run_in_executor(
                None, self.gpu_manager.get_power_limit, handle
            )
            if current_power_limit_mw is not None:
                current_power_limit_w = current_power_limit_mw / 1000
                if pid not in self.process_gpu_settings:
                    self.process_gpu_settings[pid] = {}
                if gpu_index not in self.process_gpu_settings[pid]:
                    self.process_gpu_settings[pid][gpu_index] = {}
                self.process_gpu_settings[pid][gpu_index]['power_limit_w'] = current_power_limit_w

            # Thiết lập power limit mới
            await asyncio.get_event_loop().run_in_executor(
                None, self.gpu_manager.set_power_limit, handle, power_limit_mw
            )
            self.logger.debug(
                f"Set power limit={power_limit_w}W cho GPU={gpu_index}, PID={pid}."
            )
            return True
        except pynvml.NVMLError as error:
            self.logger.error(f"Lỗi NVML set power limit GPU={gpu_index}: {error}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi set power limit GPU={gpu_index}: {e}")
            return False

    async def set_gpu_clocks(self, pid: int, gpu_index: int, sm_clock: int, mem_clock: int) -> bool:
        """
        Đặt xung nhịp GPU qua `nvidia-smi` (event-driven).
        Gọi khi cloak GPU => hạ clock, hoặc restore => trả clock cũ.
        """
        if not self.gpu_initialized:
            self.logger.error("GPUResourceManager chưa init. Không thể set clocks.")
            return False
        if not (0 <= gpu_index < self.gpu_manager.gpu_count):
            self.logger.error(f"GPU index={gpu_index} không hợp lệ.")
            return False
        if mem_clock <= 0 or sm_clock <= 0:
            self.logger.error("Clock phải > 0.")
            return False

        try:
            handle = self.gpu_manager.get_handle(gpu_index)
            # Lấy xung nhịp hiện tại để lưu
            current_sm_clock = await asyncio.get_event_loop().run_in_executor(
                None, self.gpu_manager.get_current_sm_clock, handle
            )
            current_mem_clock = await asyncio.get_event_loop().run_in_executor(
                None, self.gpu_manager.get_current_mem_clock, handle
            )

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
            await asyncio.get_event_loop().run_in_executor(
                None, subprocess.run, cmd_sm, {'check': True}
            )
            self.logger.debug(f"Set SM clock GPU={gpu_index}={sm_clock}MHz cho PID={pid}.")

            # Thiết lập xung nhịp MEM
            cmd_mem = [
                'nvidia-smi',
                '-i', str(gpu_index),
                '--lock-memory-clocks=' + str(mem_clock)
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, subprocess.run, cmd_mem, {'check': True}
            )
            self.logger.debug(f"Set MEM clock GPU={gpu_index}={mem_clock}MHz cho PID={pid}.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi nvidia-smi set clocks GPU={gpu_index}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi set clocks GPU={gpu_index}: {e}")
            return False

    async def set_gpu_max_power(self, pid: int, gpu_index: int, gpu_max_mw: int) -> bool:
        """
        Thiết lập giới hạn power tối đa (event-driven).
        """
        if not self.gpu_initialized:
            self.logger.error("GPUResourceManager chưa init. Không thể set gpu_max.")
            return False
        if not (0 <= gpu_index < self.gpu_manager.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ.")
            return False
        if gpu_max_mw <= 0:
            self.logger.error("gpu_max_mw phải > 0.")
            return False

        try:
            handle = self.gpu_manager.get_handle(gpu_index)
            current_power_limit_mw = await asyncio.get_event_loop().run_in_executor(
                None, self.gpu_manager.get_power_limit, handle
            )
            if current_power_limit_mw is not None:
                current_power_limit_w = current_power_limit_mw / 1000
                if pid not in self.process_gpu_settings:
                    self.process_gpu_settings[pid] = {}
                if gpu_index not in self.process_gpu_settings[pid]:
                    self.process_gpu_settings[pid][gpu_index] = {}
                self.process_gpu_settings[pid][gpu_index]['power_limit_w'] = current_power_limit_w

            await asyncio.get_event_loop().run_in_executor(
                None, self.gpu_manager.set_power_limit, handle, gpu_max_mw
            )
            power_limit_w = gpu_max_mw / 1000
            self.logger.debug(f"Đặt gpu_max={power_limit_w}W cho GPU={gpu_index}, PID={pid}.")
            return True
        except pynvml.NVMLError as error:
            self.logger.error(f"Lỗi NVML set gpu_max GPU={gpu_index}: {error}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi set gpu_max GPU={gpu_index}: {e}")
            return False

    async def get_gpu_power_limit(self, gpu_index: int) -> Optional[int]:
        """
        Lấy power limit GPU (event-driven). 
        Thường được gọi khi cloak => tính tỉ lệ throttling.
        """
        if not self.gpu_initialized:
            self.logger.error("GPUResourceManager chưa init.")
            return None
        if not (0 <= gpu_index < self.gpu_manager.gpu_count):
            self.logger.error(f"GPU index={gpu_index} không hợp lệ.")
            return None

        try:
            handle = self.gpu_manager.get_handle(gpu_index)
            power_limit_mw = await asyncio.get_event_loop().run_in_executor(
                None, self.gpu_manager.get_power_limit, handle
            )
            if power_limit_mw is not None:
                power_limit_w = power_limit_mw / 1000
                self.logger.debug(f"GPU={gpu_index} power limit={power_limit_w}W.")
                return power_limit_w
            return None
        except Exception as e:
            self.logger.error(f"Lỗi get_gpu_power_limit GPU={gpu_index}: {e}")
            return None

    async def get_gpu_temperature(self, gpu_index: int) -> Optional[float]:
        """
        Lấy nhiệt độ GPU (event-driven).
        Dùng khi ResourceManager cần check temp => decide cloak.
        """
        if not self.gpu_initialized:
            self.logger.error("GPUResourceManager chưa init.")
            return None
        if not (0 <= gpu_index < self.gpu_manager.gpu_count):
            self.logger.error(f"GPU index={gpu_index} không hợp lệ.")
            return None

        try:
            handle = self.gpu_manager.get_handle(gpu_index)
            temperature = await asyncio.get_event_loop().run_in_executor(
                None, self.gpu_manager.get_temperature, handle
            )
            self.logger.debug(f"Nhiệt độ GPU={gpu_index}={temperature}°C.")
            return temperature
        except Exception as e:
            self.logger.error(f"Lỗi get_gpu_temperature GPU={gpu_index}: {e}")
            return None

    async def get_gpu_utilization(self, gpu_index: int) -> Optional[Dict[str, float]]:
        """
        Lấy info sử dụng GPU (event-driven).
        """
        if not self.gpu_initialized:
            self.logger.error("GPUResourceManager chưa init.")
            return None
        if not (0 <= gpu_index < self.gpu_manager.gpu_count):
            self.logger.error(f"GPU index={gpu_index} không hợp lệ.")
            return None

        try:
            handle = self.gpu_manager.get_handle(gpu_index)
            utilization = await asyncio.get_event_loop().run_in_executor(
                None, self.gpu_manager.get_utilization, handle
            )
            self.logger.debug(f"GPU={gpu_index} utilization={utilization}.")
            return utilization
        except Exception as e:
            self.logger.error(f"Lỗi get_gpu_utilization GPU={gpu_index}: {e}")
            return None

    async def control_fan_speed(self, gpu_index: int, increase_percentage: float) -> bool:
        """
        Điều chỉnh quạt (event-driven) - gọi khi cloak GPU => tăng tốc độ quạt để làm mát.
        """
        try:
            cmd = [
                'nvidia-settings',
                '-a', f'[fan:{gpu_index}]/GPUFanControlState=1',
                '-a', f'[fan:{gpu_index}]/GPUTargetFanSpeed={int(increase_percentage)}'
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, subprocess.run, cmd, {'check': True}
            )
            self.logger.debug(f"Tăng quạt GPU={gpu_index} => {increase_percentage}%.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi adjust fan GPU={gpu_index}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi control_fan_speed GPU={gpu_index}: {e}")
            return False

    async def limit_temperature(self, gpu_index: int, temperature_threshold: float, fan_speed_increase: float) -> bool:
        """
        Quản lý nhiệt độ GPU bằng cách điều chỉnh quạt, giới hạn power và xung nhịp.
        - Nếu nhiệt độ vượt ngưỡng => hạ power limit, hạ xung nhịp.
        - Nếu nhiệt độ dưới ngưỡng => tăng xung nhịp (boost), nhưng không vượt quá mức cho phép.
        - Nếu cần, bổ sung logic “giữ nguyên” khi nhiệt độ nằm trong khoảng an toàn.
    
        Args:
            gpu_index (int): Chỉ số GPU.
            temperature_threshold (float): Ngưỡng nhiệt độ tối đa (°C).
            fan_speed_increase (float): Tỷ lệ tăng tốc độ quạt (%) để làm mát.
    
        Returns:
            bool: True nếu tất cả thao tác thành công, False nếu xảy ra lỗi.
        """
        try:
            # 1) Tăng tốc độ quạt để làm mát (nếu được hỗ trợ).
            success_fan = await self.control_fan_speed(gpu_index, fan_speed_increase)
            if success_fan:
                self.logger.info(
                    f"Quạt GPU={gpu_index} đã tăng thêm {fan_speed_increase}% để hỗ trợ làm mát."
                )
            else:
                self.logger.warning(
                    f"Không thể tăng tốc độ quạt GPU={gpu_index}. Kiểm tra hỗ trợ điều chỉnh quạt."
                )

            # 2) Kiểm tra nhiệt độ hiện tại của GPU.
            current_temperature = await self.get_gpu_temperature(gpu_index)
            if current_temperature is None:
                self.logger.warning(f"Không thể lấy nhiệt độ GPU={gpu_index}.")
                return False

            # 3) So sánh với ngưỡng. Tùy thuộc vào “cao hơn” hay “thấp hơn” để quyết định hạ/tăng xung nhịp.
            if current_temperature > temperature_threshold:
                # Nhiệt độ vượt ngưỡng => giảm power limit, giảm xung nhịp.
                self.logger.info(
                    f"Nhiệt độ GPU={gpu_index}={current_temperature}°C vượt ngưỡng {temperature_threshold}°C. Thực hiện throttle."
                )
                # Tính mức độ vượt ngưỡng.
                excess_temp = current_temperature - temperature_threshold

                # Quy ước ví dụ: nhẹ (<=5°C), trung bình (5-10°C), nặng (>10°C).
                if excess_temp <= 5:
                    throttle_pct = 10  # Giảm 10% so với power limit hiện tại.
                    self.logger.debug(f"Mức độ vượt ngưỡng nhẹ, giảm power limit ~10%.")
                elif 5 < excess_temp <= 10:
                    throttle_pct = 20
                    self.logger.debug(f"Mức độ vượt ngưỡng trung bình, giảm power limit ~20%.")
                else:
                    throttle_pct = 30
                    self.logger.debug(f"Mức độ vượt ngưỡng cao, giảm power limit ~30%.")

                # Lấy power limit hiện tại.
                current_power_limit = await self.get_gpu_power_limit(gpu_index)
                if current_power_limit is None:
                    self.logger.warning(f"Không thể lấy power limit GPU={gpu_index}. Bỏ qua throttle.")
                    return False

                desired_power_limit = int(round(current_power_limit * (1 - throttle_pct / 100)))
                # Thay đổi power limit.
                success_pl = await self.set_gpu_power_limit(
                    pid=None,  # PID không cần thiết khi throttle toàn bộ GPU
                    gpu_index=gpu_index,
                    power_limit_w=desired_power_limit
                )
                if success_pl:
                    self.logger.info(
                        f"Giảm power limit GPU={gpu_index} xuống {desired_power_limit}W để giảm nhiệt độ."
                    )
                else:
                    self.logger.error(
                        f"Không thể giảm power limit GPU={gpu_index}."
                    )

                # Giảm xung nhịp GPU để hạ nhiệt độ.
                success_clocks = await self.set_gpu_clocks(
                    pid=None,
                    gpu_index=gpu_index,
                    sm_clock=max(500, current_sm_clock - 100),
                    mem_clock=max(300, current_mem_clock - 50)
                )
                if success_clocks:
                    self.logger.info(
                        f"Đã hạ xung nhịp GPU={gpu_index} xuống SM={max(500, current_sm_clock - 100)}MHz, MEM={max(300, current_mem_clock - 50)}MHz."
                    )
                else:
                    self.logger.warning(
                        f"Không thể hạ xung nhịp GPU={gpu_index}."
                    )

            elif current_temperature < temperature_threshold:
                # Nhiệt độ dưới ngưỡng => có thể “boost” GPU để tăng hiệu suất (tuỳ mục đích).
                self.logger.info(
                    f"Nhiệt độ GPU={gpu_index}={current_temperature}°C dưới ngưỡng {temperature_threshold}°C. Thử tăng xung nhịp."
                )
                diff_temp = temperature_threshold - current_temperature
                if diff_temp <= 5:
                    boost_pct = 10
                    self.logger.debug(f"Mức độ dưới ngưỡng nhẹ, tăng clock ~10%.")
                elif 5 < diff_temp <= 10:
                    boost_pct = 20
                    self.logger.debug(f"Mức độ dưới ngưỡng trung bình, tăng clock ~20%.")
                else:
                    boost_pct = 30
                    self.logger.debug(f"Mức độ dưới ngưỡng cao, tăng clock ~30%.")

                handle = self.gpu_manager.get_handle(gpu_index)
                current_sm_clock = await asyncio.get_event_loop().run_in_executor(
                    None, self.gpu_manager.get_current_sm_clock, handle
                )
                current_mem_clock = await asyncio.get_event_loop().run_in_executor(
                    None, self.gpu_manager.get_current_mem_clock, handle
                )

                if current_sm_clock is None or current_mem_clock is None:
                    self.logger.warning(f"Không thể lấy clock GPU={gpu_index}. Bỏ qua boost xung nhịp.")
                    return True

                # Giới hạn tối đa (ví dụ SM=1530MHz, MEM=877MHz) hoặc tuỳ GPU.
                new_sm_clock = min(
                    current_sm_clock + int(current_sm_clock * boost_pct / 100),
                    1530
                )
                new_mem_clock = min(
                    current_mem_clock + int(current_mem_clock * boost_pct / 100),
                    877
                )

                success_boost = await self.set_gpu_clocks(
                    pid=None,
                    gpu_index=gpu_index,
                    sm_clock=new_sm_clock,
                    mem_clock=new_mem_clock
                )
                if success_boost:
                    self.logger.info(
                        f"Đã tăng xung nhịp GPU={gpu_index} lên SM={new_sm_clock}MHz, MEM={new_mem_clock}MHz."
                    )
                else:
                    self.logger.warning(
                        f"Không thể boost xung nhịp GPU={gpu_index}."
                    )

            else:
                # Trường hợp nhiệt độ “xấp xỉ” bằng threshold (chưa vượt & chưa thấp hơn nhiều).
                # Có thể quyết định "không làm gì" hoặc logic tuỳ ý.
                self.logger.debug(
                    f"Nhiệt độ GPU={gpu_index} ~ {current_temperature}°C gần bằng threshold={temperature_threshold}°C. Không thay đổi."
                )

            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi điều khiển nhiệt độ GPU {gpu_index}: {str(e)}")
            return False
        
    async def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục GPU cho PID (event-driven).
        Trả về power limit, xung nhịp cũ...
        """
        try:
            pid_settings = self.process_gpu_settings.get(pid)
            if not pid_settings:
                self.logger.warning(f"Không tìm thấy GPU settings ban đầu cho PID={pid}.")
                return False

            restored_all = True
            for gpu_index, settings in pid_settings.items():
                # Khôi phục power limit
                original_power_limit_w = settings.get('power_limit_w')
                if original_power_limit_w is not None:
                    success_power = await self.set_gpu_power_limit(
                        pid, gpu_index, int(original_power_limit_w)
                    )
                    if success_power:
                        self.logger.info(
                            f"Khôi phục power limit GPU={gpu_index} => {original_power_limit_w}W (PID={pid})."
                        )
                    else:
                        self.logger.error(f"Không thể khôi phục power limit GPU={gpu_index}.")
                        restored_all = False

                # Khôi phục xung nhịp
                original_sm_clock = settings.get('sm_clock_mhz')
                original_mem_clock = settings.get('mem_clock_mhz')
                if original_sm_clock and original_mem_clock:
                    success_clocks = await self.set_gpu_clocks(
                        pid, gpu_index, int(original_sm_clock), int(original_mem_clock)
                    )
                    if success_clocks:
                        self.logger.info(
                            f"Khôi phục clock GPU={gpu_index} => SM={original_sm_clock}MHz, MEM={original_mem_clock}MHz (PID={pid})."
                        )
                    else:
                        self.logger.error(f"Không thể khôi phục clock GPU={gpu_index}.")
                        restored_all = False

            del self.process_gpu_settings[pid]
            self.logger.info(f"Đã khôi phục toàn bộ GPU settings cho PID={pid}.")
            return restored_all
        except Exception as e:
            self.logger.error(f"Lỗi restore_resources GPU cho PID={pid}: {e}")
            return False


###############################################################################
#                           NETWORK RESOURCE MANAGER                           #
###############################################################################
class NetworkResourceManager:
    """
    Quản lý tài nguyên mạng qua iptables + tc (event-driven).
    Gọi khi cloak network => “limit_bandwidth”, 
    hoặc restore => remove_bandwidth_limit, unmark_packets...
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.process_marks: Dict[int, int] = {}

    async def mark_packets(self, pid: int, mark: int) -> bool:
        """
        Thêm iptables rule (event-driven).
        """
        try:
            cmd = [
                'iptables', '-A', 'OUTPUT', '-m', 'owner',
                '--pid-owner', str(pid),
                '-j', 'MARK', '--set-mark', str(mark)
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, subprocess.run, cmd, {'check': True}
            )
            self.logger.debug(f"MARK iptables cho PID={pid}, mark={mark}.")
            self.process_marks[pid] = mark
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi iptables MARK PID={pid}: {e}")
            return False

    async def unmark_packets(self, pid: int, mark: int) -> bool:
        """
        Xoá iptables rule (event-driven).
        """
        try:
            cmd = [
                'iptables', '-D', 'OUTPUT', '-m', 'owner',
                '--pid-owner', str(pid),
                '-j', 'MARK', '--set-mark', str(mark)
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, subprocess.run, cmd, {'check': True}
            )
            self.logger.debug(f"Hủy MARK iptables cho PID={pid}, mark={mark}.")
            if pid in self.process_marks:
                del self.process_marks[pid]
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi iptables unMARK PID={pid}: {e}")
            return False

    async def limit_bandwidth(self, interface: str, mark: int, bandwidth_mbps: float) -> bool:
        """
        Giới hạn băng thông thông qua tc (event-driven).
        Gọi khi cloak network => giảm băng thông.
        """
        try:
            # Thêm qdisc root
            cmd_qdisc = [
                'tc', 'qdisc', 'add', 'dev', interface,
                'root', 'handle', '1:', 'htb', 'default', '12'
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, subprocess.run, cmd_qdisc, {'check': True}
            )
            self.logger.debug(f"Thêm tc qdisc 'htb' cho {interface}.")

            # Thêm class htb
            cmd_class = [
                'tc', 'class', 'add', 'dev', interface,
                'parent', '1:', 'classid', '1:1',
                'htb', 'rate', f'{bandwidth_mbps}mbit'
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, subprocess.run, cmd_class, {'check': True}
            )
            self.logger.debug(
                f"Thêm tc class '1:1' rate={bandwidth_mbps}mbit cho {interface}."
            )

            # Thêm filter áp dụng cho fwmark
            cmd_filter = [
                'tc', 'filter', 'add', 'dev', interface,
                'protocol', 'ip', 'parent', '1:', 'prio', '1',
                'handle', str(mark), 'fw', 'flowid', '1:1'
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, subprocess.run, cmd_filter, {'check': True}
            )
            self.logger.debug(f"Thêm tc filter mark={mark} trên {interface}.")

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi limit_bandwidth: {e}")
            return False

    async def remove_bandwidth_limit(self, interface: str, mark: int) -> bool:
        """
        Gỡ bỏ băng thông thông qua tc (event-driven).
        Gọi khi restore network => khôi phục.
        """
        try:
            # Xoá filter
            cmd_filter_del = [
                'tc', 'filter', 'del', 'dev', interface,
                'protocol', 'ip', 'parent', '1:', 'prio', '1',
                'handle', str(mark), 'fw', 'flowid', '1:1'
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, subprocess.run, cmd_filter_del, {'check': True}
            )
            self.logger.debug(f"Xóa tc filter mark={mark} trên {interface}.")

            # Xoá class
            cmd_class_del = [
                'tc', 'class', 'del', 'dev', interface,
                'parent', '1:', 'classid', '1:1'
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, subprocess.run, cmd_class_del, {'check': True}
            )
            self.logger.debug(f"Xóa tc class '1:1' trên {interface}.")

            # Xoá qdisc root
            cmd_qdisc_del = [
                'tc', 'qdisc', 'del', 'dev', interface,
                'root', 'handle', '1:', 'htb'
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, subprocess.run, cmd_qdisc_del, {'check': True}
            )
            self.logger.debug(f"Xóa tc qdisc 'htb' trên {interface}.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi remove_bandwidth_limit: {e}")
            return False

    async def limit_bandwidth_for_pid(self, pid: int, interface: str, bandwidth_mbps: float) -> bool:
        """
        Giới hạn băng thông cho PID cụ thể thông qua việc đánh dấu packets và giới hạn băng thông.
        """
        try:
            mark = pid  # Sử dụng PID làm mark, có thể điều chỉnh tùy nhu cầu
            success_mark = await self.mark_packets(pid, mark)
            if not success_mark:
                return False

            success_bw = await self.limit_bandwidth(interface, mark, bandwidth_mbps)
            if not success_bw:
                await self.unmark_packets(pid, mark)
                return False

            self.logger.info(f"Đã giới hạn băng thông cho PID={pid} trên interface={interface} => {bandwidth_mbps} Mbps.")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi limit_bandwidth_for_pid PID={pid}: {e}")
            return False

    async def remove_bandwidth_limit_for_pid(self, pid: int, interface: str) -> bool:
        """
        Gỡ bỏ giới hạn băng thông cho PID cụ thể.
        """
        try:
            mark = self.process_marks.get(pid)
            if not mark:
                self.logger.warning(f"Không tìm thấy mark PID={pid} trong NetworkResourceManager.")
                return False

            success_bw = await self.remove_bandwidth_limit(interface, mark)
            if not success_bw:
                return False

            success_unmark = await self.unmark_packets(pid, mark)
            if not success_unmark:
                return False

            self.logger.info(f"Đã gỡ bỏ giới hạn băng thông cho PID={pid} trên interface={interface}.")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi remove_bandwidth_limit_for_pid PID={pid}: {e}")
            return False


###############################################################################
#                      DISK I/O RESOURCE MANAGER                              #
###############################################################################
class DiskIOResourceManager:
    """
    Quản lý Disk I/O (event-driven).
    Khi cloak disk I/O => set_io_weight(pid, ...),
    Khi restore => remove giới hạn I/O.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.process_io_limits: Dict[int, float] = {}

    async def limit_io(self, interface: str, rate_mbps: float) -> bool:
        """
        Giới hạn Disk I/O (event-driven).
        Sử dụng cgroup v2 blkio hoặc tc.
        """
        try:
            # Sử dụng cgroup v2 blkio (ví dụ)
            # Tạo cgroup cho I/O
            cgroup_name = f"io_limit_{uuid.uuid4().hex[:8]}"
            cgroup_path = f"/sys/fs/cgroup/{cgroup_name}"
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, os.makedirs, cgroup_path, True, True)
            self.logger.debug(f"Tạo cgroup v2 blkio tại {cgroup_path}.")

            # Thiết lập giới hạn I/O
            # Ví dụ: giới hạn read/write bytes
            # Điều này phụ thuộc vào hệ thống và yêu cầu cụ thể
            # Đây là ví dụ giả định:
            blkio_max = int(rate_mbps * 125000)  # Convert Mbps to bytes/s
            with open(os.path.join(cgroup_path, "io.max"), "w") as f:
                f.write(f"* {blkio_max}\n")
            self.logger.debug(f"Đặt blkio max={blkio_max} bytes/s cho cgroup {cgroup_name}.")

            # Gán interface vào cgroup nếu cần
            # Tùy thuộc vào cách cấu hình mạng và I/O trên hệ thống

            # Lưu trữ cgroup cho quản lý sau này
            self.process_io_limits[interface] = blkio_max
            return True
        except PermissionError:
            self.logger.error(f"Không đủ quyền tạo cgroup v2 blkio tại {cgroup_path}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi limit_io Disk: {e}")
            return False

    async def remove_io_limit(self, interface: str) -> bool:
        """
        Gỡ bỏ Disk I/O limit (event-driven).
        """
        try:
            cgroup_max_path = f"/sys/fs/cgroup/{interface}/io.max"
            if os.path.exists(cgroup_max_path):
                with open(cgroup_max_path, "w") as f:
                    f.write("0 0\n")  # Không giới hạn
                self.logger.debug(f"Xóa giới hạn blkio tại {cgroup_max_path}.")
                del self.process_io_limits[interface]
                return True
            else:
                self.logger.warning(f"Cgroup v2 blkio tại {cgroup_max_path} không tồn tại khi xóa.")
                return False
        except PermissionError:
            self.logger.error(f"Không đủ quyền để xóa blkio limit tại {cgroup_max_path}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi remove_io_limit Disk: {e}")
            return False

    async def set_io_weight(self, pid: int, io_weight: int) -> bool:
        """
        Đặt trọng số I/O cho PID (event-driven).
        Sử dụng cgroup v2 hoặc ionice.
        """
        try:
            # Ví dụ sử dụng ionice để thiết lập ưu tiên I/O
            cmd = [
                'ionice', '-c', '2', '-n', str(io_weight), '-p', str(pid)
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, subprocess.run, cmd, {'check': True}
            )
            self.logger.debug(f"Set io_weight={io_weight} cho PID={pid} sử dụng ionice.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi ionice set_io_weight PID={pid}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi set_io_weight PID={pid}: {e}")
            return False

    async def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục Disk I/O (event-driven).
        Xoá cài đặt weight cũ.
        """
        try:
            # Khôi phục ionice cho PID
            cmd = [
                'ionice', '-c', '0', '-p', str(pid)
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, subprocess.run, cmd, {'check': True}
            )
            self.logger.info(f"Đã khôi phục Disk I/O cho PID={pid} sử dụng ionice.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi ionice restore_resources PID={pid}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi restore_resources Disk cho PID={pid}: {e}")
            return False


###############################################################################
#                       CACHE RESOURCE MANAGER                                #
###############################################################################
class CacheResourceManager:
    """
    Quản lý Cache (event-driven).
    Gọi drop_caches, limit_cache_usage khi cloak,
    Gọi restore_resources khi khôi phục.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.dropped_pids: List[int] = []

    async def drop_caches(self, pid: Optional[int] = None) -> bool:
        """
        Drop caches (event-driven).
        """
        try:
            cmd = ['sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches']
            await asyncio.get_event_loop().run_in_executor(
                None, subprocess.run, cmd, {'check': True}
            )
            self.logger.debug("Đã drop caches.")
            if pid:
                self.dropped_pids.append(pid)
            return True
        except subprocess.CalledProcessError:
            self.logger.error("Không đủ quyền drop_caches hoặc lệnh thất bại.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi drop_caches: {e}")
            return False

    async def limit_cache_usage(self, cache_limit_percent: float, pid: Optional[int] = None) -> bool:
        """
        Giới hạn cache (event-driven). Thí dụ: drop_caches + ...
        """
        try:
            success = await self.drop_caches(pid)
            if not success:
                return False
            self.logger.debug(f"Giới hạn cache => {cache_limit_percent}%.")
            # Thêm logic giới hạn cache nếu cần thiết
            return True
        except Exception as e:
            self.logger.error(f"Lỗi limit_cache_usage: {e}")
            return False

    async def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục cache => đặt cache_limit=100% (event-driven).
        """
        try:
            success = await self.limit_cache_usage(100.0, pid)
            if success:
                self.logger.info(f"Khôi phục Cache cho PID={pid} => 100%.")
            else:
                self.logger.error(f"Không thể khôi phục Cache cho PID={pid}.")
            return success
        except Exception as e:
            self.logger.error(f"Lỗi restore_resources Cache cho PID={pid}: {e}")
            return False


###############################################################################
#                       MEMORY RESOURCE MANAGER                               #
###############################################################################
class MemoryResourceManager:
    """
    Quản lý Memory qua ulimit/rlimit (event-driven).
    Gọi set_memory_limit khi cloak => hạ memory,
    Gọi remove_memory_limit khi restore.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    async def set_memory_limit(self, pid: int, memory_limit_mb: int) -> bool:
        """
        Đặt memory limit (event-driven).
        """
        try:
            process = psutil.Process(pid)
            mem_bytes = memory_limit_mb * 1024 * 1024
            await asyncio.get_event_loop().run_in_executor(
                None, process.rlimit, psutil.RLIMIT_AS, (mem_bytes, mem_bytes)
            )
            self.logger.debug(f"Đặt memory_limit={memory_limit_mb}MB cho PID={pid}.")
            return True
        except psutil.NoSuchProcess:
            self.logger.error(f"PID={pid} không tồn tại (set_memory_limit).")
            return False
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền set_memory_limit cho PID={pid}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi set_memory_limit cho PID={pid}: {e}")
            return False

    async def get_memory_limit(self, pid: int) -> float:
        """
        Lấy memory limit (event-driven).
        """
        try:
            process = psutil.Process(pid)
            mem_limit = process.rlimit(psutil.RLIMIT_AS)
            if mem_limit and mem_limit[1] != psutil.RLIM_INFINITY:
                self.logger.debug(f"Memory limit PID={pid}={mem_limit[1]} bytes.")
                return float(mem_limit[1])
            else:
                self.logger.debug(f"PID={pid} không giới hạn bộ nhớ.")
                return float('inf')
        except Exception as e:
            self.logger.error(f"Lỗi get_memory_limit PID={pid}: {e}")
            return 0.0

    async def remove_memory_limit(self, pid: int) -> bool:
        """
        Khôi phục memory => không giới hạn (event-driven).
        """
        try:
            process = psutil.Process(pid)
            await asyncio.get_event_loop().run_in_executor(
                None, process.rlimit,
                psutil.RLIMIT_AS,
                (psutil.RLIM_INFINITY, psutil.RLIM_INFINITY)
            )
            self.logger.debug(f"Khôi phục memory cho PID={pid} => không giới hạn.")
            return True
        except psutil.NoSuchProcess:
            self.logger.error(f"PID={pid} không tồn tại khi remove_memory_limit.")
            return False
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền remove_memory_limit cho PID={pid}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi remove_memory_limit cho PID={pid}: {e}")
            return False

    async def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục memory => remove_memory_limit (event-driven).
        """
        return await self.remove_memory_limit(pid)


###############################################################################
#                     RESOURCE CONTROL FACTORY                                #
###############################################################################
class ResourceControlFactory:
    """
    Factory tạo các resource manager (CPU, GPU, Network, Disk I/O, Cache, Memory).
    Event-driven: Mỗi manager sẽ được gọi khi cloak/restore.
    """

    @staticmethod
    async def create_resource_managers(logger: logging.Logger, gpu_manager: GPUManager) -> Dict[str, Any]:
        """
        Khởi tạo tất cả managers theo mô hình event-driven.
        Gọi 1 lần khi ResourceManager khởi tạo.
        
        Args:
            logger (logging.Logger): Logger để truyền cho các managers.
            gpu_manager (GPUManager): Instance của GPUManager đã được khởi tạo và cấu hình.

        Returns:
            Dict[str, Any]: Dictionary chứa các resource managers.
        """
        # CPU Manager
        cpu_manager = CPUResourceManager(logger)
        await cpu_manager.ensure_cgroup_base()

        # GPU Manager
        gpu_resource_manager = GPUResourceManager(logger, gpu_manager)
        await gpu_resource_manager.initialize()

        # Network Manager
        network_manager = NetworkResourceManager(logger)

        # Disk I/O Manager
        disk_io_manager = DiskIOResourceManager(logger)

        # Cache Manager
        cache_manager = CacheResourceManager(logger)

        # Memory Manager
        memory_manager = MemoryResourceManager(logger)

        resource_managers = {
            'cpu': cpu_manager,
            'gpu': gpu_resource_manager,
            'network': network_manager,
            'disk_io': disk_io_manager,
            'cache': cache_manager,
            'memory': memory_manager
        }
        return resource_managers
