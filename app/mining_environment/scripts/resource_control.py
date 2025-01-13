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
# Đảm bảo rằng GPUManager được định nghĩa đầy đủ trong utils.py
from .utils import GPUManager

###############################################################################
#                  CPU RESOURCE MANAGER (EVENT-DRIVEN)                       #
###############################################################################
class CPUResourceManager:
    """
    Quản lý tài nguyên CPU thông qua cgroups (giới hạn CPU, affinity) và tối ưu CPU.

    - Trong mô hình event-driven, các phương thức dưới đây được gọi khi:
      + CloakStrategy hoặc ResourceManager ra lệnh "giới hạn CPU" (event “limit CPU”).
      + ResourceManager ra lệnh "khôi phục CPU" (event “restore CPU”).
    - Không còn vòng lặp polling bên trong, toàn bộ logic được kích hoạt từ bên ngoài.
    """

    CGROUP_BASE_PATH = "/sys/fs/cgroup/cpu_cloak"

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.process_cgroup: Dict[int, str] = {}  # Mapping PID -> cgroup name

    async def ensure_cgroup_base(self) -> None:
        """
        Đảm bảo thư mục cgroup cơ sở tồn tại.
        Đây thường được gọi 1 lần khi khởi tạo ResourceManager (event “initialize”).
        """
        try:
            if not os.path.exists(self.CGROUP_BASE_PATH):
                # Sử dụng run_in_executor để tránh block event loop
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, os.makedirs, self.CGROUP_BASE_PATH, True, True)
                self.logger.debug(f"Tạo thư mục cgroup cơ sở tại {self.CGROUP_BASE_PATH}.")
        except PermissionError:
            self.logger.error(f"Không đủ quyền tạo thư mục cgroup tại {self.CGROUP_BASE_PATH}.")
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo thư mục cgroup tại {self.CGROUP_BASE_PATH}: {e}")

    def get_available_cpus(self) -> List[int]:
        """
        Lấy danh sách core CPU để đặt affinity.  
        (Phần này không polling, được gọi khi “apply” hay “restore” CPU.)
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
        Tạo cgroup mới cho PID và thiết lập giới hạn CPU (%).  
        Được gọi khi có event “giới hạn CPU” cho một tiến trình.
        """
        try:
            if not (0 <= throttle_percentage <= 100):
                self.logger.error(f"Giá trị throttle_percentage={throttle_percentage} không hợp lệ.")
                return None

            cgroup_name = f"cpu_cloak_{uuid.uuid4().hex[:8]}"
            cgroup_path = os.path.join(self.CGROUP_BASE_PATH, cgroup_name)

            # Tạo cgroup folder
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, os.makedirs, cgroup_path, True, True)
            self.logger.debug(f"Tạo cgroup tại {cgroup_path} cho PID={pid}.")

            # Tính CPU quota
            cpu_period = 100000  # 100ms
            cpu_quota = int((throttle_percentage / 100) * cpu_period)
            cpu_quota = max(1000, cpu_quota)  # Đảm bảo quota tối thiểu

            # Ghi giá trị quota
            with open(os.path.join(cgroup_path, "cpu.max"), "w") as f:
                f.write(f"{cpu_quota} {cpu_period}\n")
            self.logger.debug(f"Đặt CPU quota={cpu_quota}us cho cgroup {cgroup_name}.")

            # Thêm PID vào cgroup
            with open(os.path.join(cgroup_path, "cgroup.procs"), "w") as f:
                f.write(f"{pid}\n")
            self.logger.info(f"Thêm PID={pid} vào cgroup {cgroup_name} (CPU throttle={throttle_percentage}%).")

            # Lưu mapping PID -> cgroup_name
            self.process_cgroup[pid] = cgroup_name

            return cgroup_name

        except PermissionError:
            self.logger.error(f"Không đủ quyền tạo/cấu hình cgroup cho PID={pid}.")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo cgroup cho PID={pid}: {e}")
            return None

    async def delete_cgroup(self, cgroup_name: str) -> bool:
        """
        Xóa cgroup (gọi khi khôi phục CPU).  
        Event-driven: “restore CPU” => xóa cgroup cũ.
        """
        try:
            cgroup_path = os.path.join(self.CGROUP_BASE_PATH, cgroup_name)
            procs_path = os.path.join(cgroup_path, "cgroup.procs")

            with open(procs_path, "r") as f:
                procs = f.read().strip()
                if procs:
                    self.logger.warning(f"Cgroup {cgroup_name} vẫn còn tiến trình PID={procs}. Không thể xóa.")
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

    async def throttle_cpu_usage(self, pid: int, throttle_percentage: float) -> Optional[str]:
        """
        Giới hạn CPU cho tiến trình (tạo cgroup).  
        Gọi khi event “limit CPU for PID” xảy ra.
        """
        return await self.create_cgroup(pid, throttle_percentage)

    async def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục CPU cho PID (xóa cgroup).  
        Gọi khi event “restore CPU for PID” xảy ra.
        """
        try:
            cgroup_name = self.process_cgroup.get(pid)
            if not cgroup_name:
                self.logger.warning(f"Không tìm thấy cgroup cho PID={pid} trong CPUResourceManager.")
                return False

            success = await self.delete_cgroup(cgroup_name)
            if success:
                self.logger.info(f"Khôi phục CPU cho PID={pid} xong (đã xóa cgroup).")
                del self.process_cgroup[pid]
                return True
            else:
                self.logger.error(f"Không thể khôi phục CPU cho PID={pid}.")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục CPU settings cho PID={pid}: {e}")
            return False

    async def set_cpu_affinity(self, pid: int, cores: List[int]) -> bool:
        """
        Đặt CPU affinity cho PID (được gọi khi cần tối ưu / thay đổi scheduling).  
        Sự kiện: “apply CPU affinity”.
        """
        try:
            process = psutil.Process(pid)
            await asyncio.get_event_loop().run_in_executor(None, process.cpu_affinity, cores)
            self.logger.debug(f"Đặt CPU affinity PID={pid} vào core {cores}.")
            return True
        except psutil.NoSuchProcess:
            self.logger.error(f"Tiến trình PID={pid} không tồn tại (set CPU affinity).")
            return False
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền đặt CPU affinity cho PID={pid}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi đặt CPU affinity cho PID={pid}: {e}")
            return False

    async def reset_cpu_affinity(self, pid: int) -> bool:
        """
        Khôi phục CPU affinity về tất cả core.  
        Sự kiện: “reset CPU affinity”.
        """
        try:
            available_cpus = self.get_available_cpus()
            return await self.set_cpu_affinity(pid, available_cpus)
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục CPU affinity cho PID={pid}: {e}")
            return False

    async def limit_cpu_for_external_processes(self, target_pids: List[int], throttle_percentage: float) -> bool:
        """
        Giới hạn CPU cho tiến trình ngoài (tránh “polling” bằng cách được gọi lúc cloaking CPU).  
        Sự kiện: “limit external CPU usage”.
        """
        try:
            if not (0 <= throttle_percentage <= 100):
                self.logger.error(f"Giá trị throttle_percentage={throttle_percentage} không hợp lệ.")
                return False

            all_pids = [proc.pid for proc in psutil.process_iter(attrs=['pid'])]
            external_pids = set(all_pids) - set(target_pids)

            tasks = []
            for pid in external_pids:
                tasks.append(self.throttle_cpu_usage(pid, throttle_percentage))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for pid, result in zip(external_pids, results):
                if isinstance(result, Exception) or not result:
                    self.logger.warning(f"Không thể hạn chế CPU cho PID={pid}.")
            self.logger.info(f"Giới hạn CPU cho {len(external_pids)} PID bên ngoài (throttle={throttle_percentage}%).")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi hạn chế CPU cho tiến trình ngoài: {e}")
            return False

    async def optimize_thread_scheduling(self, pid: int, target_cores: Optional[List[int]] = None) -> bool:
        """
        Tối ưu scheduling (đặt affinity).  
        Gọi khi “optimize CPU scheduling event”.
        """
        try:
            success = await self.set_cpu_affinity(pid, target_cores or self.get_available_cpus())
            if success:
                self.logger.info(f"Tối ưu scheduling PID={pid} (cores={target_cores}).")
            return success
        except Exception as e:
            self.logger.error(f"Lỗi tối ưu scheduling PID={pid}: {e}")
            return False

    async def optimize_cache_usage(self, pid: int) -> bool:
        """
        Tối ưu cache CPU, ví dụ set priority/nice, v.v.  
        Hiện tại chủ yếu rely cgroup => Ít xử lý.  
        """
        try:
            self.logger.debug(f"Tối ưu cache PID={pid} qua cgroups (mô phỏng).")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi tối ưu cache PID={pid}: {e}")
            return False


###############################################################################
#                 GPU RESOURCE MANAGER (EVENT-DRIVEN)                        #
###############################################################################
class GPUResourceManager:
    """
    Quản lý tài nguyên GPU qua NVML.  
    - Mô hình event-driven: Gọi phương thức khi “apply GPU cloak” hoặc “restore GPU”.
    - Lưu trữ settings ban đầu của PID để khôi phục.
    """

    def __init__(self, logger: logging.Logger, gpu_manager: GPUManager):
        self.logger = logger
        self.gpu_manager = gpu_manager
        self.gpu_initialized = False
        # Lưu {pid: {gpu_index: {key: val}}} (power_limit_w, sm_clock_mhz, mem_clock_mhz, ...)
        self.process_gpu_settings: Dict[int, Dict[int, Dict[str, Any]]] = {}

    async def initialize(self) -> bool:
        """
        Khởi tạo NVML (event “initialize GPU”).  
        Chỉ thực hiện 1 lần khi ResourceControlFactory gọi.
        """
        try:
            await asyncio.get_event_loop().run_in_executor(None, self.gpu_manager.initialize)
            if self.gpu_manager.gpu_count > 0:
                self.gpu_initialized = True
                self.logger.info("GPUResourceManager: đã khởi tạo, GPU sẵn sàng.")
            else:
                self.logger.warning("GPUResourceManager: Không phát hiện GPU nào.")
            return self.gpu_initialized
        except pynvml.NVMLError as error:
            self.logger.error(f"Lỗi NVML khi khởi tạo: {error}")
            self.gpu_initialized = False
            return False
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi khởi tạo GPUResourceManager: {e}")
            self.gpu_initialized = False
            return False

    async def set_gpu_power_limit(self, pid: Optional[int], gpu_index: int, power_limit_w: int) -> bool:
        """
        Đặt power limit GPU. Lưu power limit cũ (nếu pid!=None) để khôi phục.  
        Gọi khi “apply GPU cloak” => “set power limit” / “restore GPU cloak”.
        """
        if not self.gpu_initialized:
            self.logger.error("GPUResourceManager: Chưa khởi tạo hoặc không có GPU.")
            return False
        if not (0 <= gpu_index < self.gpu_manager.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ.")
            return False
        if power_limit_w <= 0:
            self.logger.error("Giá trị power_limit_w phải > 0.")
            return False

        try:
            handle = self.gpu_manager.get_handle(gpu_index)
            # Lấy power limit hiện tại
            current_power_limit_mw = await asyncio.get_event_loop().run_in_executor(
                None, self.gpu_manager.get_power_limit, handle
            )
            if current_power_limit_mw is not None and pid is not None:
                current_power_limit_w = current_power_limit_mw / 1000
                if pid not in self.process_gpu_settings:
                    self.process_gpu_settings[pid] = {}
                if gpu_index not in self.process_gpu_settings[pid]:
                    self.process_gpu_settings[pid][gpu_index] = {}
                self.process_gpu_settings[pid][gpu_index]['power_limit_w'] = current_power_limit_w

            # Đặt power limit mới
            power_limit_mw = power_limit_w * 1000
            await asyncio.get_event_loop().run_in_executor(None, self.gpu_manager.set_power_limit, handle, power_limit_mw)
            self.logger.debug(f"Đặt power limit GPU {gpu_index}={power_limit_w}W (pid={pid}).")
            return True
        except pynvml.NVMLError as error:
            self.logger.error(f"Lỗi NVML khi set_power_limit GPU {gpu_index}: {error}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi set_power_limit GPU {gpu_index}: {e}")
            return False

    async def set_gpu_clocks(self, pid: Optional[int], gpu_index: int, sm_clock: int, mem_clock: int) -> bool:
        """
        Đặt xung nhịp SM/MEM. Lưu lại xung nhịp cũ nếu pid!=None.  
        Gọi khi event “apply GPU cloak” => “set clocks”.
        """
        if not self.gpu_initialized:
            self.logger.error("GPUResourceManager: Chưa khởi tạo.")
            return False
        if not (0 <= gpu_index < self.gpu_manager.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ.")
            return False
        if sm_clock <= 0 or mem_clock <= 0:
            self.logger.error("sm_clock/mem_clock phải > 0.")
            return False

        try:
            handle = self.gpu_manager.get_handle(gpu_index)
            # Lấy xung nhịp hiện tại
            current_sm_clock = await asyncio.get_event_loop().run_in_executor(None, self.gpu_manager.get_current_sm_clock, handle)
            current_mem_clock = await asyncio.get_event_loop().run_in_executor(None, self.gpu_manager.get_current_mem_clock, handle)
            if pid is not None:
                if pid not in self.process_gpu_settings:
                    self.process_gpu_settings[pid] = {}
                if gpu_index not in self.process_gpu_settings[pid]:
                    self.process_gpu_settings[pid][gpu_index] = {}
                self.process_gpu_settings[pid][gpu_index]['sm_clock_mhz'] = current_sm_clock
                self.process_gpu_settings[pid][gpu_index]['mem_clock_mhz'] = current_mem_clock

            # Gọi nvidia-smi lock
            cmd_sm = ['nvidia-smi', '-i', str(gpu_index), f'--lock-gpu-clocks={sm_clock}']
            await asyncio.get_event_loop().run_in_executor(None, subprocess.run, cmd_sm, {'check': True})
            self.logger.debug(f"Set SM clock GPU={gpu_index}={sm_clock}MHz (pid={pid}).")

            cmd_mem = ['nvidia-smi', '-i', str(gpu_index), f'--lock-memory-clocks={mem_clock}']
            await asyncio.get_event_loop().run_in_executor(None, subprocess.run, cmd_mem, {'check': True})
            self.logger.debug(f"Set MEM clock GPU={gpu_index}={mem_clock}MHz (pid={pid}).")

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi nvidia-smi khi set GPU clocks: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi set_gpu_clocks GPU={gpu_index}: {e}")
            return False

    async def get_gpu_power_limit(self, gpu_index: int) -> Optional[int]:
        """
        Lấy power limit (Watts).  
        Gọi khi cloak_strategies hoặc ResourceManager cần biết.
        """
        if not self.gpu_initialized:
            self.logger.error("GPUResourceManager: Chưa khởi tạo.")
            return None
        if not (0 <= gpu_index < self.gpu_manager.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ.")
            return None

        try:
            handle = self.gpu_manager.get_handle(gpu_index)
            power_limit_mw = await asyncio.get_event_loop().run_in_executor(None, self.gpu_manager.get_power_limit, handle)
            power_limit_w = power_limit_mw / 1000
            self.logger.debug(f"GPU {gpu_index} power limit={power_limit_w}W.")
            return power_limit_w
        except Exception as e:
            self.logger.error(f"Lỗi get_gpu_power_limit GPU={gpu_index}: {e}")
            return None

    async def get_gpu_temperature(self, gpu_index: int) -> Optional[float]:
        """
        Lấy nhiệt độ GPU (°C).  
        Gọi khi cloak_strategies/ResourceManager cần check temp.
        """
        if not self.gpu_initialized:
            self.logger.error("GPUResourceManager: Chưa khởi tạo.")
            return None
        if not (0 <= gpu_index < self.gpu_manager.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ.")
            return None
        try:
            handle = self.gpu_manager.get_handle(gpu_index)
            temp_c = await asyncio.get_event_loop().run_in_executor(None, self.gpu_manager.get_temperature, handle)
            self.logger.debug(f"Nhiệt độ GPU {gpu_index}={temp_c}°C.")
            return temp_c
        except Exception as e:
            self.logger.error(f"Lỗi get_gpu_temperature GPU={gpu_index}: {e}")
            return None

    async def get_gpu_utilization(self, gpu_index: int) -> Optional[Dict[str, float]]:
        """
        Lấy thông tin usage GPU.  
        Gọi khi cloak_strategies/ResourceManager cần usage.
        """
        if not self.gpu_initialized:
            self.logger.error("GPUResourceManager: Chưa khởi tạo.")
            return None
        if not (0 <= gpu_index < self.gpu_manager.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ.")
            return None

        try:
            handle = self.gpu_manager.get_handle(gpu_index)
            utilization = await asyncio.get_event_loop().run_in_executor(None, self.gpu_manager.get_utilization, handle)
            self.logger.debug(f"Sử dụng GPU {gpu_index}: {utilization}")
            return utilization
        except Exception as e:
            self.logger.error(f"Lỗi get_gpu_utilization GPU={gpu_index}: {e}")
            return None

    async def control_fan_speed(self, gpu_index: int, increase_percentage: float) -> bool:
        """
        Điều chỉnh tốc độ quạt qua nvidia-settings (nếu GPU hỗ trợ).  
        Gọi khi event “limit temperature GPU”.
        """
        try:
            cmd = [
                'nvidia-settings',
                '-a', f'[fan:{gpu_index}]/GPUFanControlState=1',
                '-a', f'[fan:{gpu_index}]/GPUTargetFanSpeed={int(increase_percentage)}'
            ]
            await asyncio.get_event_loop().run_in_executor(None, subprocess.run, cmd, {'check': True})
            self.logger.debug(f"Tăng tốc độ quạt GPU {gpu_index} lên {increase_percentage}%.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi nvidia-settings khi điều chỉnh quạt GPU {gpu_index}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi điều chỉnh quạt GPU {gpu_index}: {e}")
            return False

    async def limit_temperature(self, gpu_index: int, temperature_threshold: float, fan_speed_increase: float) -> bool:
        """
        Quản lý nhiệt độ GPU (tăng quạt, giảm power, giảm clocks) nếu > threshold.  
        Gọi khi cloak_strategies/ResourceManager cần “limit GPU temp event”.
        """
        try:
            success_fan = await self.control_fan_speed(gpu_index, fan_speed_increase)
            if not success_fan:
                self.logger.warning(f"Không thể tăng quạt GPU {gpu_index}, có thể GPU không hỗ trợ?")

            current_temp = await self.get_gpu_temperature(gpu_index)
            if current_temp is None:
                self.logger.warning(f"Không thể lấy nhiệt độ GPU {gpu_index}.")
                return False

            # Nếu cao hơn threshold => Cloaking GPU
            if current_temp > temperature_threshold:
                self.logger.info(f"Nhiệt độ GPU {gpu_index}={current_temp}°C vượt ngưỡng {temperature_threshold}°C.")
                # Logic giảm xung nhịp, giảm power
                # ...
            else:
                self.logger.info(f"Nhiệt độ GPU {gpu_index}={current_temp}°C dưới ngưỡng => có thể boost xung nhịp.")

            return True
        except Exception as e:
            self.logger.error(f"Lỗi limit_temperature GPU {gpu_index}: {e}")
            return False

    async def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục GPU (power limit, clocks) cho PID.  
        Được gọi khi event “restore GPU for PID”.
        """
        try:
            pid_settings = self.process_gpu_settings.get(pid)
            if not pid_settings:
                self.logger.warning(f"Không tìm thấy GPU settings ban đầu cho PID={pid}.")
                return False

            all_ok = True
            for gpu_index, settings in pid_settings.items():
                # Power limit
                orig_power_w = settings.get('power_limit_w')
                if orig_power_w is not None:
                    ok_power = await self.set_gpu_power_limit(pid, gpu_index, int(orig_power_w))
                    if ok_power:
                        self.logger.info(f"Khôi phục power limit GPU {gpu_index}={orig_power_w}W (PID={pid}).")
                    else:
                        self.logger.error(f"Không thể khôi phục power limit GPU {gpu_index} (PID={pid}).")
                        all_ok = False

                # Xung nhịp
                orig_sm_clock = settings.get('sm_clock_mhz')
                orig_mem_clock = settings.get('mem_clock_mhz')
                if orig_sm_clock and orig_mem_clock:
                    ok_clocks = await self.set_gpu_clocks(pid, gpu_index, int(orig_sm_clock), int(orig_mem_clock))
                    if ok_clocks:
                        self.logger.info(f"Khôi phục xung nhịp GPU {gpu_index}, SM={orig_sm_clock}MHz, MEM={orig_mem_clock}MHz (PID={pid}).")
                    else:
                        self.logger.warning(f"Không thể khôi phục xung nhịp GPU {gpu_index} (PID={pid}).")
                        all_ok = False

            del self.process_gpu_settings[pid]
            return all_ok
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục GPU PID={pid}: {e}")
            return False


###############################################################################
#               NETWORK RESOURCE MANAGER (EVENT-DRIVEN)                       #
###############################################################################
class NetworkResourceManager:
    """
    Quản lý tài nguyên mạng qua iptables + tc.  
    - Event-driven: Gọi khi “limit network bandwidth” hoặc “restore network” từ CloakStrategy/ResourceManager.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.process_marks: Dict[int, int] = {}  # PID -> fwmark

    async def mark_packets(self, pid: int, mark: int) -> bool:
        """
        Đánh dấu gói tin bằng iptables (event “apply network cloak”).
        """
        try:
            cmd = [
                'iptables', '-A', 'OUTPUT', '-m', 'owner', '--pid-owner', str(pid),
                '-j', 'MARK', '--set-mark', str(mark)
            ]
            await asyncio.get_event_loop().run_in_executor(None, subprocess.run, cmd, {'check': True})
            self.logger.debug(f"MARK iptables cho PID={pid}, mark={mark}.")
            self.process_marks[pid] = mark
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi iptables MARK cho PID={pid}: {e}")
            return False

    async def unmark_packets(self, pid: int, mark: int) -> bool:
        """
        Gỡ đánh dấu gói tin iptables (event “restore network cloak”).
        """
        try:
            cmd = [
                'iptables', '-D', 'OUTPUT', '-m', 'owner', '--pid-owner', str(pid),
                '-j', 'MARK', '--set-mark', str(mark)
            ]
            await asyncio.get_event_loop().run_in_executor(None, subprocess.run, cmd, {'check': True})
            self.logger.debug(f"Gỡ MARK iptables PID={pid}, mark={mark}.")
            if pid in self.process_marks:
                del self.process_marks[pid]
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi gỡ iptables MARK cho PID={pid}: {e}")
            return False

    async def limit_bandwidth(self, interface: str, mark: int, bandwidth_mbps: float) -> bool:
        """
        Giới hạn băng thông (tc) cho fwmark.  
        Gọi khi event “limit network bandwidth”.
        """
        try:
            # Tạo qdisc gốc
            cmd_qdisc = [
                'tc', 'qdisc', 'add', 'dev', interface, 'root', 'handle', '1:',
                'htb', 'default', '12'
            ]
            await asyncio.get_event_loop().run_in_executor(None, subprocess.run, cmd_qdisc, {'check': True})
            self.logger.debug(f"tc qdisc htb cho {interface}.")

            # Tạo class
            cmd_class = [
                'tc', 'class', 'add', 'dev', interface, 'parent', '1:', 'classid', '1:1',
                'htb', 'rate', f'{bandwidth_mbps}mbit'
            ]
            await asyncio.get_event_loop().run_in_executor(None, subprocess.run, cmd_class, {'check': True})
            self.logger.debug(f"tc class '1:1' rate={bandwidth_mbps}mbit cho {interface}.")

            # Thêm filter
            cmd_filter = [
                'tc', 'filter', 'add', 'dev', interface, 'protocol', 'ip',
                'parent', '1:', 'prio', '1', 'handle', str(mark), 'fw', 'flowid', '1:1'
            ]
            await asyncio.get_event_loop().run_in_executor(None, subprocess.run, cmd_filter, {'check': True})
            self.logger.debug(f"tc filter mark={mark} cho {interface}.")

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi limit_bandwidth: {e}")
            return False

    async def remove_bandwidth_limit(self, interface: str, mark: int) -> bool:
        """
        Gỡ bỏ giới hạn băng thông (event “restore network cloak”).
        """
        try:
            # Xóa filter
            cmd_filter_del = [
                'tc', 'filter', 'del', 'dev', interface, 'protocol', 'ip',
                'parent', '1:', 'prio', '1', 'handle', str(mark), 'fw', 'flowid', '1:1'
            ]
            await asyncio.get_event_loop().run_in_executor(None, subprocess.run, cmd_filter_del, {'check': True})
            self.logger.debug(f"Đã xóa filter mark={mark} cho {interface}.")

            # Xóa class
            cmd_class_del = [
                'tc', 'class', 'del', 'dev', interface, 'parent', '1:', 'classid', '1:1'
            ]
            await asyncio.get_event_loop().run_in_executor(None, subprocess.run, cmd_class_del, {'check': True})
            self.logger.debug(f"Đã xóa class '1:1' cho {interface}.")

            # Xóa qdisc
            cmd_qdisc_del = [
                'tc', 'qdisc', 'del', 'dev', interface, 'root', 'handle', '1:', 'htb'
            ]
            await asyncio.get_event_loop().run_in_executor(None, subprocess.run, cmd_qdisc_del, {'check': True})
            self.logger.debug(f"Đã xóa qdisc 'htb' cho {interface}.")

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi remove_bandwidth_limit: {e}")
            return False

    async def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục tài nguyên mạng cho PID (gỡ băng thông, unmark).  
        Gọi khi event “restore network cloak”.
        """
        try:
            mark = self.process_marks.get(pid)
            if not mark:
                self.logger.warning(f"Không tìm thấy mark cho PID={pid} trong NetworkResourceManager.")
                return False

            interface = 'eth0'  # Cứng, hoặc lấy từ config
            success_bw = await self.remove_bandwidth_limit(interface, mark)
            if success_bw:
                self.logger.info(f"Khôi phục băng thông cho PID={pid}, mark={mark}.")
            else:
                self.logger.error(f"Không thể khôi phục băng thông PID={pid}.")

            success_unmark = await self.unmark_packets(pid, mark)
            if success_unmark:
                self.logger.info(f"Đã xóa iptables MARK cho PID={pid}, mark={mark}.")
            else:
                self.logger.error(f"Không thể xóa iptables MARK PID={pid}, mark={mark}.")

            return success_bw and success_unmark
        except Exception as e:
            self.logger.error(f"Lỗi restore mạng cho PID={pid}: {e}")
            return False


###############################################################################
#             DISK IO RESOURCE MANAGER (EVENT-DRIVEN)                         #
###############################################################################
class DiskIOResourceManager:
    """
    Quản lý Disk I/O.  
    Event-driven: Gọi khi “limit disk IO” hoặc “restore disk IO” từ CloakStrategy/ResourceManager.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.process_io_limits: Dict[int, float] = {}  # PID -> I/O rate limit

    async def limit_io(self, interface: str, rate_mbps: float) -> bool:
        """
        Giới hạn tốc độ I/O (tc) - ví dụ placeholder.  
        Event-driven: “apply disk IO limit”.
        """
        try:
            # Tạo qdisc root
            cmd_qdisc = [
                'tc', 'qdisc', 'add', 'dev', interface, 'root', 'handle', '1:',
                'htb', 'default', '12'
            ]
            await asyncio.get_event_loop().run_in_executor(None, subprocess.run, cmd_qdisc, {'check': True})
            self.logger.debug(f"Thêm tc qdisc 'htb' cho {interface}.")

            # Thêm class
            cmd_class = [
                'tc', 'class', 'add', 'dev', interface, 'parent', '1:', 'classid', '1:1',
                'htb', 'rate', f'{rate_mbps}mbit'
            ]
            await asyncio.get_event_loop().run_in_executor(None, subprocess.run, cmd_class, {'check': True})
            self.logger.debug(f"Thêm tc class '1:1', rate={rate_mbps}mbit cho {interface}.")

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi giới hạn Disk I/O: {e}")
            return False

    async def remove_io_limit(self, interface: str) -> bool:
        """
        Gỡ bỏ giới hạn I/O (event “restore disk IO limit”).
        """
        try:
            cmd_qdisc_del = [
                'tc', 'qdisc', 'del', 'dev', interface, 'root', 'handle', '1:', 'htb'
            ]
            await asyncio.get_event_loop().run_in_executor(None, subprocess.run, cmd_qdisc_del, {'check': True})
            self.logger.debug(f"Đã xóa tc qdisc 'htb' cho {interface}.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi xóa giới hạn Disk I/O: {e}")
            return False

    async def set_io_weight(self, pid: int, io_weight: int) -> bool:
        """
        Đặt trọng số I/O cho PID qua cgroups/ionice (mô phỏng).  
        Event-driven: “apply disk IO cloak”.
        """
        try:
            self.logger.debug(f"[DiskIO] Set io_weight PID={pid}={io_weight} (chưa triển khai cgroup).")
            self.process_io_limits[pid] = io_weight
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi set I/O weight PID={pid}: {e}")
            return False

    async def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục Disk I/O cho PID (xóa giới hạn).  
        Event-driven: “restore disk IO cloak”.
        """
        try:
            if pid in self.process_io_limits:
                del self.process_io_limits[pid]
                self.logger.info(f"Đã khôi phục Disk I/O cho PID={pid}.")
                return True
            else:
                self.logger.warning(f"Không tìm thấy Disk I/O limit cho PID={pid}.")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi khôi phục Disk I/O PID={pid}: {e}")
            return False


###############################################################################
#            CACHE RESOURCE MANAGER (EVENT-DRIVEN)                            #
###############################################################################
class CacheResourceManager:
    """
    Quản lý cache (drop caches, v.v.).  
    Event-driven: “apply cache cloak” hoặc “restore cache cloak”.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.dropped_pids: List[int] = []

    async def drop_caches(self, pid: Optional[int] = None) -> bool:
        """
        Gọi 'echo 3 > /proc/sys/vm/drop_caches' để xóa pagecache, dentries, inodes.  
        Chỉ thực thi khi event “drop caches” được kích hoạt.
        """
        try:
            cmd = ['sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches']
            await asyncio.get_event_loop().run_in_executor(None, subprocess.run, cmd, {'check': True})
            self.logger.debug("Đã drop caches.")
            if pid:
                self.dropped_pids.append(pid)
            return True
        except subprocess.CalledProcessError:
            self.logger.error("Không đủ quyền hoặc drop caches không thành công.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi drop caches: {e}")
            return False

    async def limit_cache_usage(self, cache_limit_percent: float, pid: Optional[int] = None) -> bool:
        """
        Giới hạn cache => tạm thời dùng drop caches + (có thể cgroups cache).  
        Event-driven: “apply cache cloak”.
        """
        try:
            success = await self.drop_caches(pid)
            if not success:
                return False

            self.logger.debug(f"Giới hạn cache={cache_limit_percent}%. (Mô phỏng cgroups cache)")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi limit cache usage: {e}")
            return False

    async def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục cache => gỡ bỏ giới hạn, đặt lại 100%.  
        Event-driven: “restore cache cloak”.
        """
        try:
            # Giả định đặt lại 100%
            success = await self.limit_cache_usage(100.0, pid)
            if success:
                self.logger.info(f"Khôi phục cache 100% cho PID={pid}.")
            else:
                self.logger.error(f"Không thể khôi phục cache PID={pid}.")
            return success
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục cache PID={pid}: {e}")
            return False


###############################################################################
#           MEMORY RESOURCE MANAGER (EVENT-DRIVEN)                            #
###############################################################################
class MemoryResourceManager:
    """
    Quản lý bộ nhớ (ulimit / cgroups).  
    Event-driven: “apply memory cloak” => set_memory_limit(pid, X), “restore memory cloak” => remove_memory_limit(pid).
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    async def set_memory_limit(self, pid: int, memory_limit_mb: int) -> bool:
        """
        Đặt giới hạn bộ nhớ cho PID (ulimit).  
        """
        try:
            process = psutil.Process(pid)
            mem_bytes = memory_limit_mb * 1024 * 1024
            await asyncio.get_event_loop().run_in_executor(
                None, process.rlimit, psutil.RLIMIT_AS, (mem_bytes, mem_bytes)
            )
            self.logger.debug(f"Đặt memory limit PID={pid}={memory_limit_mb}MB.")
            return True
        except psutil.NoSuchProcess:
            self.logger.error(f"Tiến trình PID={pid} không tồn tại (set memory limit).")
            return False
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền đặt memory limit PID={pid}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi set memory limit PID={pid}: {e}")
            return False

    async def get_memory_limit(self, pid: int) -> float:
        """
        Lấy giới hạn bộ nhớ đã đặt.  
        Gọi khi cần kiểm tra (event “check memory limit”).
        """
        try:
            process = psutil.Process(pid)
            mem_limit = process.rlimit(psutil.RLIMIT_AS)
            if mem_limit and mem_limit[1] != psutil.RLIM_INFINITY:
                self.logger.debug(f"PID={pid} memory limit={mem_limit[1]} bytes.")
                return float(mem_limit[1])
            else:
                self.logger.debug(f"PID={pid} không giới hạn bộ nhớ.")
                return float('inf')
        except psutil.NoSuchProcess:
            self.logger.error(f"Tiến trình PID={pid} không tồn tại (get memory limit).")
            return 0.0
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền get memory limit PID={pid}.")
            return 0.0
        except Exception as e:
            self.logger.error(f"Lỗi khi get memory limit PID={pid}: {e}")
            return 0.0

    async def remove_memory_limit(self, pid: int) -> bool:
        """
        Khôi phục lại memory limit về không giới hạn.  
        Event “restore memory”.
        """
        try:
            process = psutil.Process(pid)
            await asyncio.get_event_loop().run_in_executor(
                None, process.rlimit, psutil.RLIMIT_AS, (psutil.RLIM_INFINITY, psutil.RLIM_INFINITY)
            )
            self.logger.debug(f"Khôi phục memory limit PID={pid} => không giới hạn.")
            return True
        except psutil.NoSuchProcess:
            self.logger.error(f"Tiến trình PID={pid} không tồn tại (remove memory limit).")
            return False
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền khôi phục memory limit PID={pid}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi remove memory limit PID={pid}: {e}")
            return False

    async def restore_resources(self, pid: int) -> bool:
        """Khôi phục memory (event-driven)."""
        return await self.remove_memory_limit(pid)


###############################################################################
#         RESOURCE CONTROL FACTORY (EVENT-DRIVEN INTEGRATION)                 #
###############################################################################
class ResourceControlFactory:
    """
    Factory tạo resource managers (CPU/GPU/Network/DiskIO/Cache/Memory).
    - Trong event-driven architecture, ta gọi create_resource_managers() 1 lần
      để có dictionary “manager_name -> manager_instance”,
      sau đó ResourceManager/AnomalyDetector/triggers sẽ gọi chúng.
    """

    @staticmethod
    async def create_resource_managers(logger: logging.Logger) -> Dict[str, Any]:
        """
        Tạo và trả về dict { 'cpu':..., 'gpu':..., 'network':..., 'disk_io':..., 'cache':..., 'memory':... }.
        Event “initialize resource managers”.
        """
        # Tạo GPUManager
        gpu_manager = GPUManager()

        # Tạo CPUResourceManager
        cpu_manager = CPUResourceManager(logger)
        await cpu_manager.ensure_cgroup_base()

        # GPUResourceManager
        gpu_resource_manager = GPUResourceManager(logger, gpu_manager)
        await gpu_resource_manager.initialize()

        # NetworkResourceManager
        network_manager = NetworkResourceManager(logger)

        # DiskIOResourceManager
        disk_io_manager = DiskIOResourceManager(logger)

        # CacheResourceManager
        cache_manager = CacheResourceManager(logger)

        # MemoryResourceManager
        memory_manager = MemoryResourceManager(logger)

        return {
            'cpu': cpu_manager,
            'gpu': gpu_resource_manager,
            'network': network_manager,
            'disk_io': disk_io_manager,
            'cache': cache_manager,
            'memory': memory_manager
        }
