# resource_control.py

import os
import uuid
import logging
import subprocess
from typing import Any, Dict, List, Optional, Tuple
import psutil
import pynvml  # NVIDIA Management Library
import asyncio


###############################################################################
#                           CPU RESOURCE MANAGER                              #
###############################################################################
class CPUResourceManager:
    """
    Quản lý tài nguyên CPU sử dụng cgroups, affinity, và tối ưu hóa CPU (event-driven).
    """

    CGROUP_BASE_PATH = "/sys/fs/cgroup/cpu_cloak"

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.logger = logger
        self.config = config
        # Lưu thông tin PID -> cgroup
        self.process_cgroup: Dict[int, str] = {}

    async def ensure_cgroup_base(self) -> None:
        """
        Đảm bảo thư mục gốc cho cgroups CPU cloak tồn tại.
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
        """
        try:
            cpu_count = psutil.cpu_count(logical=True)
            available_cpus = list(range(cpu_count))
            self.logger.debug(f"Available CPUs: {available_cpus}.")
            return available_cpus
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy danh sách CPU cores: {e}")
            return []

    async def throttle_cpu_usage(self, pid: int, throttle_percentage: float) -> Optional[str]:
        """
        Giới hạn CPU cho PID thông qua cgroup.
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
            cpu_quota = max(1000, cpu_quota)  # tránh quota quá nhỏ

            with open(os.path.join(cgroup_path, "cpu.max"), "w") as f:
                f.write(f"{cpu_quota} {cpu_period}\n")
            self.logger.debug(f"Đặt CPU quota={cpu_quota}us cho cgroup {cgroup_name}.")

            # Gán PID vào cgroup
            with open(os.path.join(cgroup_path, "cgroup.procs"), "w") as f:
                f.write(f"{pid}\n")
            self.logger.info(
                f"Thêm PID={pid} vào cgroup {cgroup_name}, throttle={throttle_percentage}%."
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

    async def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục CPU bằng cách xóa cgroup (event-driven).
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

    async def optimize_thread_scheduling(self, pid: int, cores: Optional[List[int]] = None) -> bool:
        """
        Đặt CPU affinity (bất đồng bộ).
        """
        try:
            process = psutil.Process(pid)
            target_cores = cores or self.get_available_cpus()
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, process.cpu_affinity, target_cores)
            self.logger.debug(f"Đặt CPU affinity cho PID={pid} => {target_cores}.")
            return True
        except psutil.NoSuchProcess:
            self.logger.error(f"PID={pid} không tồn tại (optimize_thread_scheduling).")
            return False
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền set_cpu_affinity cho PID={pid}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi optimize_thread_scheduling cho PID={pid}: {e}")
            return False

    async def optimize_cache_usage(self, pid: int) -> bool:
        """
        Tối ưu cache (event-driven). Tạm thời chưa có logic cụ thể -> log.
        """
        try:
            self.logger.debug(
                f"Tối ưu cache CPU (PID={pid}): cgroups + throttle đã cover 1 phần."
            )
            return True
        except Exception as e:
            self.logger.error(f"Lỗi optimize_cache_usage cho PID={pid}: {e}")
            return False

    async def limit_cpu_for_external_processes(self, target_pids: List[int], throttle_percentage: float) -> bool:
        """
        Giới hạn CPU cho các tiến trình “bên ngoài” (ngoài target_pids).
        """
        try:
            if not (0 <= throttle_percentage <= 100):
                self.logger.error(f"throttle_percentage={throttle_percentage} không hợp lệ (0-100).")
                return False

            all_pids = [proc.pid for proc in psutil.process_iter(attrs=['pid'])]
            external_pids = set(all_pids) - set(target_pids)

            # Tạo cgroup cho mỗi PID external
            results = []
            for pid_ in external_pids:
                result = await self.throttle_cpu_usage(pid_, throttle_percentage)
                if not result:
                    self.logger.warning(f"Không thể hạn chế CPU cho PID={pid_}.")
                else:
                    results.append(pid_)

            self.logger.info(
                f"Hạn chế CPU cho {len(results)} tiến trình outside => throttle={throttle_percentage}%."
            )
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi hạn chế CPU cho external processes: {e}")
            return False


###############################################################################
#                           GPU RESOURCE MANAGER                              #
###############################################################################
class GPUResourceManager:
    """
    Quản lý GPU thông qua pynvml (event-driven).
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.logger = logger
        self.config = config
        self.gpu_initialized = False
        # Lưu PID -> GPU Index -> {settings}
        self.process_gpu_settings: Dict[int, Dict[int, Dict[str, Any]]] = {}

    async def initialize_nvml(self) -> bool:
        """
        Khởi tạo pynvml nếu chưa được khởi tạo.
        """
        try:
            pynvml.nvmlInit()
            self.logger.info("pynvml đã được khởi tạo.")
            self.gpu_initialized = True
            return True
        except pynvml.NVMLError as error:
            self.logger.error(f"Lỗi khi khởi tạo pynvml: {error}")
            self.gpu_initialized = False
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi tạo pynvml: {e}")
            self.gpu_initialized = False
            return False

    def is_nvml_initialized(self) -> bool:
        return self.gpu_initialized

    async def get_gpu_count(self) -> int:
        """Trả về số lượng GPU khả dụng."""
        if not self.gpu_initialized:
            return 0
        try:
            return pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError:
            return 0

    async def get_handle(self, gpu_index: int):
        """Lấy handle GPU."""
        if not self.gpu_initialized:
            return None
        try:
            return pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except pynvml.NVMLError:
            return None

    async def get_gpu_power_limit(self, gpu_index: int) -> Optional[int]:
        """
        Lấy power limit GPU (W).
        """
        if not self.gpu_initialized:
            return None
        try:
            handle = await self.get_handle(gpu_index)
            if not handle:
                return None
            limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            return int(limit_mw // 1000)  # convert mW -> W
        except Exception as e:
            self.logger.error(f"Lỗi get_gpu_power_limit GPU={gpu_index}: {e}")
            return None

    async def set_gpu_power_limit(self, pid: int, gpu_index: int, power_limit_w: int) -> bool:
        """
        Đặt power limit GPU (W).
        """
        if not self.gpu_initialized:
            self.logger.error("GPUResourceManager chưa init. Không thể set power limit.")
            return False
        try:
            handle = await self.get_handle(gpu_index)
            if not handle or power_limit_w <= 0:
                return False

            # Lưu power limit cũ
            current_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            current_w = current_mw // 1000
            if pid not in self.process_gpu_settings:
                self.process_gpu_settings[pid] = {}
            if gpu_index not in self.process_gpu_settings[pid]:
                self.process_gpu_settings[pid][gpu_index] = {}
            self.process_gpu_settings[pid][gpu_index]['power_limit_w'] = current_w

            new_limit_mw = power_limit_w * 1000
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, new_limit_mw)
            self.logger.debug(f"Set power limit={power_limit_w}W cho GPU={gpu_index}, PID={pid}.")
            return True
        except pynvml.NVMLError as error:
            self.logger.error(f"Lỗi NVML set power limit GPU={gpu_index}: {error}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi set power limit GPU={gpu_index}: {e}")
            return False

    async def set_gpu_clocks(self, pid: int, gpu_index: int, sm_clock: int, mem_clock: int) -> bool:
        """
        Đặt xung nhịp GPU SM và MEM (MHz) bằng nvidia-smi lock.
        """
        if not self.gpu_initialized:
            self.logger.error("GPUResourceManager chưa init. Không thể set clocks.")
            return False
        try:
            handle = await self.get_handle(gpu_index)
            if not handle or sm_clock <= 0 or mem_clock <= 0:
                return False

            # Lưu xung nhịp hiện tại
            current_sm_clock = pynvml.nvmlDeviceGetClock(handle, pynvml.NVML_CLOCK_SM)
            current_mem_clock = pynvml.nvmlDeviceGetClock(handle, pynvml.NVML_CLOCK_MEM)

            if pid not in self.process_gpu_settings:
                self.process_gpu_settings[pid] = {}
            if gpu_index not in self.process_gpu_settings[pid]:
                self.process_gpu_settings[pid][gpu_index] = {}
            self.process_gpu_settings[pid][gpu_index]['sm_clock_mhz'] = current_sm_clock
            self.process_gpu_settings[pid][gpu_index]['mem_clock_mhz'] = current_mem_clock

            loop = asyncio.get_event_loop()
            cmd_sm = [
                'nvidia-smi',
                '-i', str(gpu_index),
                '--lock-gpu-clocks=' + str(sm_clock)
            ]
            await loop.run_in_executor(None, subprocess.run, cmd_sm, {'check': True})
            self.logger.debug(f"Set SM clock={sm_clock}MHz cho GPU={gpu_index}, PID={pid}.")

            cmd_mem = [
                'nvidia-smi',
                '-i', str(gpu_index),
                '--lock-memory-clocks=' + str(mem_clock)
            ]
            await loop.run_in_executor(None, subprocess.run, cmd_mem, {'check': True})
            self.logger.debug(f"Set MEM clock={mem_clock}MHz cho GPU={gpu_index}, PID={pid}.")

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi nvidia-smi set clocks GPU={gpu_index}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi set clocks GPU={gpu_index}: {e}")
            return False

    async def limit_temperature(self, gpu_index: int, temperature_threshold: float, fan_speed_increase: float) -> bool:
        """
        Quản lý nhiệt độ GPU bằng cách điều chỉnh quạt, giới hạn power và xung nhịp.
        - Nếu nhiệt độ vượt ngưỡng => hạ power limit, hạ xung nhịp.
        - Nếu nhiệt độ dưới ngưỡng => boost xung nhịp (nếu muốn),
        - Nếu nhiệt độ nằm gần threshold => có thể bỏ qua hoặc logic tùy ý.

        Args:
            gpu_index (int): Chỉ số GPU.
            temperature_threshold (float): Ngưỡng nhiệt độ tối đa (°C).
            fan_speed_increase (float): Tỷ lệ tăng tốc độ quạt (%) để làm mát.

        Returns:
            bool: True nếu tất cả thao tác thành công, False nếu xảy ra lỗi.
        """
        try:
            # 1) Kiểm tra đã init NVML chưa
            if not self.gpu_initialized:
                self.logger.error("GPUResourceManager chưa init. Không thể limit_temperature.")
                return False

            # 2) Tăng tốc độ quạt (nếu GPU driver hỗ trợ)
            success_fan = await self.control_fan_speed(gpu_index, fan_speed_increase)
            if success_fan:
                self.logger.info(f"Quạt GPU={gpu_index} tăng thêm {fan_speed_increase}%.")
            else:
                self.logger.warning(f"Không thể điều chỉnh quạt GPU={gpu_index}. Có thể driver không hỗ trợ.")

            # 3) Kiểm tra nhiệt độ hiện tại
            current_temperature = await self.get_gpu_temperature(gpu_index)
            if current_temperature is None:
                self.logger.warning(f"Không thể lấy nhiệt độ GPU={gpu_index}. Bỏ qua điều chỉnh nhiệt.")
                return False

            # 4) Lấy xung nhịp hiện tại (để hạ hoặc boost)
            current_sm_clock = None
            current_mem_clock = None

            try:
                handle = await self.get_handle(gpu_index)
                if handle:
                    loop = asyncio.get_event_loop()
                    # Lấy SM clock
                    current_sm_clock = await loop.run_in_executor(
                        None, pynvml.nvmlDeviceGetClock, handle, pynvml.NVML_CLOCK_SM
                    )
                    # Lấy MEM clock
                    current_mem_clock = await loop.run_in_executor(
                        None, pynvml.nvmlDeviceGetClock, handle, pynvml.NVML_CLOCK_MEM
                    )
            except Exception as e:
                self.logger.error(f"Không thể lấy current clocks GPU={gpu_index}: {e}")
                return False

            if current_sm_clock is None or current_mem_clock is None:
                self.logger.warning(f"current_sm_clock hoặc current_mem_clock = None, GPU={gpu_index}.")
                return False

            # 5) So sánh nhiệt độ với threshold
            if current_temperature > temperature_threshold:
                # ***** Trường hợp nhiệt độ cao => throttle power + hạ xung nhịp *****
                self.logger.info(
                    f"Nhiệt độ GPU={gpu_index}={current_temperature}°C vượt {temperature_threshold}°C => throttle."
                )
                # Xác định mức độ “vượt”
                excess_temp = current_temperature - temperature_threshold
                if excess_temp <= 5:
                    throttle_pct = 10
                elif excess_temp <= 10:
                    throttle_pct = 20
                else:
                    throttle_pct = 30
                self.logger.debug(f"excess_temp={excess_temp}°C => throttle_pct={throttle_pct}%")

                # Lấy power limit hiện tại
                current_power_limit = await self.get_gpu_power_limit(gpu_index)
                if current_power_limit is None:
                    self.logger.warning(f"Không thể get power_limit GPU={gpu_index}. Bỏ qua throttle.")
                    return False

                desired_power_limit = int(round(current_power_limit * (1 - throttle_pct / 100)))
                # Giới hạn power limit
                success_pl = await self.set_gpu_power_limit(pid=None, gpu_index=gpu_index, power_limit_w=desired_power_limit)
                if success_pl:
                    self.logger.info(
                        f"Giảm power limit GPU={gpu_index} => {desired_power_limit}W (origin={current_power_limit}W)."
                    )
                else:
                    self.logger.error(f"Không thể set power limit GPU={gpu_index}.")

                # Hạ xung nhịp (VD: SM -100, MEM -50)
                new_sm_clock = max(500, current_sm_clock - 100)
                new_mem_clock = max(300, current_mem_clock - 50)
                success_clocks = await self.set_gpu_clocks(pid=None, gpu_index=gpu_index,
                                                        sm_clock=new_sm_clock, mem_clock=new_mem_clock)
                if success_clocks:
                    self.logger.info(
                        f"Hạ xung nhịp GPU={gpu_index}: SM={new_sm_clock}MHz, MEM={new_mem_clock}MHz."
                    )
                else:
                    self.logger.warning(f"Không thể hạ xung nhịp GPU={gpu_index}.")

            elif current_temperature < temperature_threshold:
                # ***** Trường hợp nhiệt độ thấp => boost (tuỳ ý) *****
                self.logger.info(
                    f"Nhiệt độ GPU={gpu_index}={current_temperature}°C < {temperature_threshold}°C => có thể boost."
                )
                diff_temp = temperature_threshold - current_temperature
                if diff_temp <= 5:
                    boost_pct = 10
                elif diff_temp <= 10:
                    boost_pct = 20
                else:
                    boost_pct = 30
                self.logger.debug(f"diff_temp={diff_temp}°C => boost_pct={boost_pct}%")

                # Tính clock mới
                # Giới hạn SM=1530, MEM=877 (ví dụ) hoặc tuỳ GPU
                new_sm_clock = min(current_sm_clock + int(current_sm_clock * boost_pct / 100), 1530)
                new_mem_clock = min(current_mem_clock + int(current_mem_clock * boost_pct / 100), 877)

                success_boost = await self.set_gpu_clocks(pid=None, gpu_index=gpu_index,
                                                        sm_clock=new_sm_clock, mem_clock=new_mem_clock)
                if success_boost:
                    self.logger.info(
                        f"Tăng xung nhịp GPU={gpu_index} => SM={new_sm_clock}MHz, MEM={new_mem_clock}MHz."
                    )
                else:
                    self.logger.warning(f"Không thể boost xung nhịp GPU={gpu_index}.")
            else:
                # ***** Trường hợp nhiệt độ ~ threshold => có thể bỏ qua hoặc logic khác *****
                self.logger.debug(
                    f"Nhiệt độ GPU={gpu_index}={current_temperature}°C xấp xỉ threshold={temperature_threshold}°C => không điều chỉnh."
                )

            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi điều khiển nhiệt độ GPU={gpu_index}: {e}")
            return False

    async def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục GPU cho PID (event-driven): power limit, clocks...
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
                    ok = await self.set_gpu_power_limit(pid, gpu_index, int(original_power_limit_w))
                    if ok:
                        self.logger.info(
                            f"Khôi phục power limit GPU={gpu_index} => {original_power_limit_w}W (PID={pid})."
                        )
                    else:
                        self.logger.error(f"Không thể khôi phục power limit GPU={gpu_index}.")
                        restored_all = False

                # Khôi phục xung nhịp
                original_sm_clock = settings.get('sm_clock_mhz')
                original_mem_clock = settings.get('mem_clock_mhz')
                if original_sm_clock is not None and original_mem_clock is not None:
                    ok_clocks = await self.set_gpu_clocks(pid, gpu_index, original_sm_clock, original_mem_clock)
                    if ok_clocks:
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
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.logger = logger
        self.config = config
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
        except Exception as e:
            self.logger.error(f"Lỗi mark_packets PID={pid}: {e}")
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
        except Exception as e:
            self.logger.error(f"Lỗi unmark_packets PID={pid}: {e}")
            return False

    async def limit_bandwidth(self, interface: str, mark: int, bandwidth_mbps: float) -> bool:
        """
        Giới hạn băng thông thông qua tc (event-driven).
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
            self.logger.debug(f"Thêm tc class '1:1' rate={bandwidth_mbps}mbit cho {interface}.")

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
        Giới hạn băng thông cho PID cụ thể thông qua packets mark + tc.
        """
        try:
            mark = pid % 32768
            success_mark = await self.mark_packets(pid, mark)
            if not success_mark:
                return False

            success_bw = await self.limit_bandwidth(interface, mark, bandwidth_mbps)
            if not success_bw:
                await self.unmark_packets(pid, mark)
                return False

            self.logger.info(f"Giới hạn băng thông PID={pid} => {bandwidth_mbps}Mbps trên interface={interface}.")
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
                self.logger.warning(f"Không tìm thấy mark cho PID={pid} trong NetworkResourceManager.")
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
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.logger = logger
        self.config = config
        self.process_io_limits: Dict[int, float] = {}

    async def set_io_weight(self, pid: int, io_weight: int) -> bool:
        """
        Đặt trọng số I/O cho PID (ionice).
        """
        try:
            cmd = [
                'ionice', '-c', '2', '-n', str(io_weight), '-p', str(pid)
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, subprocess.run, cmd, {'check': True}
            )
            self.logger.debug(f"Set io_weight={io_weight} cho PID={pid} qua ionice.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi ionice set_io_weight PID={pid}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi set_io_weight PID={pid}: {e}")
            return False

    async def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục Disk I/O => set io_weight=0 (class=0).
        """
        try:
            cmd = [
                'ionice', '-c', '0', '-p', str(pid)
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, subprocess.run, cmd, {'check': True}
            )
            self.logger.info(f"Khôi phục Disk I/O cho PID={pid} (ionice class=0).")
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
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.logger = logger
        self.config = config
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
        Giới hạn cache => Tối giản: drop caches + log.
        """
        try:
            success = await self.drop_caches(pid)
            if not success:
                return False
            self.logger.debug(f"Giới hạn cache => {cache_limit_percent}%. (chưa có logic chi tiết)")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi limit_cache_usage: {e}")
            return False

    async def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục cache => limit_cache_usage(100).
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
    Quản lý Memory qua psutil rlimit (event-driven).
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.logger = logger
        self.config = config

    async def set_memory_limit(self, pid: int, memory_limit_mb: int) -> bool:
        """
        Đặt memory limit (MB).
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
        Lấy memory limit (bytes).
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
        Khôi phục memory => không giới hạn.
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
        Khôi phục memory => remove_memory_limit.
        """
        return await self.remove_memory_limit(pid)


###############################################################################
#                     RESOURCE CONTROL FACTORY                                # 
###############################################################################
class ResourceControlFactory:
    """
    Factory tạo các resource manager (CPU, GPU, Network, Disk I/O, Cache, Memory).
    """

    @staticmethod
    async def create_resource_managers(config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
        """
        Khởi tạo tất cả resource managers theo mô hình event-driven.

        Args:
            config (Dict[str, Any]): Cấu hình của ResourceManager (JSON).
            logger (logging.Logger): Logger.

        Returns:
            Dict[str, Any]: Dictionary chứa các resource managers.
        """
        resource_managers = {}

        # Danh sách các lớp manager cần khởi tạo
        manager_classes = {
            'cpu': CPUResourceManager,
            'gpu': GPUResourceManager,
            'network': NetworkResourceManager,
            'disk_io': DiskIOResourceManager,
            'cache': CacheResourceManager,
            'memory': MemoryResourceManager,
        }

        for name, manager_class in manager_classes.items():
            try:
                logger.info(f"Đang khởi tạo {name} manager...")
                manager_instance = manager_class(config, logger)

                # Các bước khởi tạo đặc biệt (nếu có)
                if name == 'cpu':
                    await manager_instance.ensure_cgroup_base()
                elif name == 'gpu':
                    await manager_instance.initialize_nvml()
                # network / disk_io / cache / memory => không có setup_* methods

                resource_managers[name] = manager_instance
                logger.info(f"{name.capitalize()} manager đã được khởi tạo thành công.")
            except Exception as e:
                logger.error(f"Lỗi khi khởi tạo {name} manager: {e}", exc_info=True)

        if not resource_managers:
            logger.error("Không có resource managers nào được khởi tạo.")
            raise RuntimeError("Tất cả resource managers đều khởi tạo thất bại.")

        logger.info("Tất cả resource managers đã được khởi tạo.")
        return resource_managers
