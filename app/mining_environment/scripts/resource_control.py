# resource_control.py

import os
import uuid
import logging
import subprocess
from typing import Any, Dict, List, Optional
import psutil
import pynvml

###############################################################################
#                           CPU RESOURCE MANAGER                              #
###############################################################################
class CPUResourceManager:
    """
    Quản lý tài nguyên CPU sử dụng cgroups, affinity, và tối ưu hóa CPU theo mô hình đồng bộ.

    Attributes:
        logger (logging.Logger): Logger để ghi log.
        config (Dict[str, Any]): Cấu hình cho CPU resource manager.
        CGROUP_BASE_PATH (str): Đường dẫn gốc cho cgroup CPU cloaking.
        process_cgroup (Dict[int, str]): Bản đồ PID -> tên cgroup.
    """

    CGROUP_BASE_PATH = "/sys/fs/cgroup/cpu_cloak"

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Khởi tạo CPUResourceManager.

        :param config: Cấu hình CPU Resource Manager (dict).
        :param logger: Đối tượng Logger.
        """
        self.logger = logger
        self.config = config
        self.process_cgroup: Dict[int, str] = {}

        # Đảm bảo thư mục gốc cho cgroups CPU cloak.
        self.ensure_cgroup_base()

    def ensure_cgroup_base(self) -> None:
        """
        Đảm bảo thư mục gốc cho cgroups CPU cloak tồn tại. Đồng bộ.
        """
        try:
            if not os.path.exists(self.CGROUP_BASE_PATH):
                os.makedirs(self.CGROUP_BASE_PATH, exist_ok=True)
                self.logger.debug(f"Tạo thư mục cgroup cơ sở tại {self.CGROUP_BASE_PATH}.")
        except PermissionError:
            self.logger.error(f"Không đủ quyền tạo thư mục cgroup tại {self.CGROUP_BASE_PATH}.")
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo thư mục cgroup tại {self.CGROUP_BASE_PATH}: {e}")

    def get_available_cpus(self) -> List[int]:
        """
        Lấy danh sách các core CPU để đặt affinity.

        :return: Danh sách số hiệu core CPU (List[int]).
        """
        try:
            cpu_count = psutil.cpu_count(logical=True)
            available_cpus = list(range(cpu_count))
            self.logger.debug(f"Available CPUs: {available_cpus}.")
            return available_cpus
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy danh sách CPU cores: {e}")
            return []

    def throttle_cpu_usage(self, pid: int, throttle_percentage: float) -> Optional[str]:
        """
        Giới hạn CPU cho PID thông qua cgroup, đồng bộ.

        :param pid: PID của tiến trình cần giới hạn.
        :param throttle_percentage: Tỷ lệ giới hạn CPU (0-100).
        :return: Tên cgroup (str) nếu tạo thành công, None nếu thất bại.
        """
        try:
            if not (0 <= throttle_percentage <= 100):
                self.logger.error(f"throttle_percentage={throttle_percentage} không hợp lệ (0-100).")
                return None

            cgroup_name = f"cpu_cloak_{uuid.uuid4().hex[:8]}"
            cgroup_path = os.path.join(self.CGROUP_BASE_PATH, cgroup_name)
            os.makedirs(cgroup_path, exist_ok=True)
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

    def delete_cgroup(self, cgroup_name: str) -> bool:
        """
        Xóa cgroup (đồng bộ).

        :param cgroup_name: Tên cgroup cần xóa.
        :return: True nếu xóa thành công, False nếu thất bại.
        """
        try:
            cgroup_path = os.path.join(self.CGROUP_BASE_PATH, cgroup_name)
            procs_path = os.path.join(cgroup_path, "cgroup.procs")

            if os.path.exists(procs_path):
                with open(procs_path, "r") as f:
                    procs = f.read().strip()
                    if procs:
                        self.logger.warning(
                            f"Cgroup {cgroup_name} vẫn còn PID={procs}. Không thể xóa."
                        )
                        return False

            os.rmdir(cgroup_path)
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

    def optimize_thread_scheduling(self, pid: int, cores: Optional[List[int]] = None) -> bool:
        """
        Đặt CPU affinity cho tiến trình (đồng bộ).

        :param pid: PID của tiến trình.
        :param cores: Danh sách core CPU (nếu None => dùng tất cả).
        :return: True nếu thành công, False nếu thất bại.
        """
        try:
            process = psutil.Process(pid)
            target_cores = cores or self.get_available_cpus()
            process.cpu_affinity(target_cores)
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

    def optimize_cache_usage(self, pid: int) -> bool:
        """
        Tối ưu cache CPU (đồng bộ).
        Hiện tại chỉ log, vì logic cụ thể chưa được xác định.
        """
        try:
            self.logger.debug(
                f"Tối ưu cache CPU (PID={pid}): cgroups + throttle đã cover 1 phần."
            )
            return True
        except Exception as e:
            self.logger.error(f"Lỗi optimize_cache_usage cho PID={pid}: {e}")
            return False

    def limit_cpu_for_external_processes(self, target_pids: List[int], throttle_percentage: float) -> bool:
        """
        Giới hạn CPU cho các tiến trình “bên ngoài” (ngoài target_pids), đồng bộ.

        :param target_pids: Danh sách các PID không bị ảnh hưởng.
        :param throttle_percentage: Tỷ lệ giới hạn CPU cho tiến trình bên ngoài (0-100).
        :return: True nếu thực thi không có lỗi (dù có pid bị lỗi riêng), False nếu xảy ra lỗi tổng quát.
        """
        try:
            if not (0 <= throttle_percentage <= 100):
                self.logger.error(f"throttle_percentage={throttle_percentage} không hợp lệ (0-100).")
                return False

            all_pids = [proc.pid for proc in psutil.process_iter(attrs=['pid'])]
            external_pids = set(all_pids) - set(target_pids)

            results = []
            for pid_ in external_pids:
                result = self.throttle_cpu_usage(pid_, throttle_percentage)
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

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục CPU bằng cách xóa cgroup (đồng bộ).

        :param pid: PID của tiến trình cần khôi phục.
        :return: True nếu thành công, False nếu thất bại.
        """
        try:
            cgroup_name = self.process_cgroup.get(pid)
            if not cgroup_name:
                self.logger.warning(f"Không tìm thấy cgroup cho PID={pid} trong CPUResourceManager.")
                return False
            success = self.delete_cgroup(cgroup_name)
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


###############################################################################
#                           GPU RESOURCE MANAGER                              #
###############################################################################
class GPUResourceManager:
    """
    Quản lý GPU thông qua pynvml (đồng bộ).

    Attributes:
        logger (logging.Logger): Logger để ghi log.
        config (Dict[str, Any]): Cấu hình GPU Resource Manager.
        gpu_initialized (bool): Cờ đánh dấu NVML đã khởi tạo hay chưa.
        process_gpu_settings (Dict[int, Dict[int, Dict[str, Any]]]): Lưu PID -> GPU Index -> settings.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Khởi tạo GPUResourceManager.

        :param config: Cấu hình GPU Resource Manager (dict).
        :param logger: Đối tượng Logger.
        """
        self.logger = logger
        self.config = config
        self.gpu_initialized = False
        self.process_gpu_settings: Dict[int, Dict[int, Dict[str, Any]]] = {}

        # Tự động khởi tạo NVML
        self.initialize_nvml()

    def initialize_nvml(self) -> bool:
        """
        Khởi tạo pynvml (đồng bộ).

        :return: True nếu khởi tạo thành công, False nếu thất bại.
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
        """
        Kiểm tra NVML đã được khởi tạo hay chưa.

        :return: True nếu NVML đã khởi tạo, False nếu chưa.
        """
        return self.gpu_initialized

    def get_gpu_count(self) -> int:
        """
        Lấy số lượng GPU (đồng bộ).

        :return: Số GPU (int).
        """
        if not self.gpu_initialized:
            return 0
        try:
            return pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError:
            return 0

    def get_handle(self, gpu_index: int):
        """
        Lấy handle của GPU theo chỉ số (đồng bộ).

        :param gpu_index: Chỉ số GPU.
        :return: Handle thiết bị GPU, hoặc None nếu lỗi.
        """
        if not self.gpu_initialized:
            return None
        try:
            return pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except pynvml.NVMLError:
            return None

    def get_gpu_power_limit(self, gpu_index: int) -> Optional[int]:
        """
        Trả về power limit (W) của GPU (đồng bộ).

        :param gpu_index: Chỉ số GPU.
        :return: Power limit (int) hoặc None nếu lỗi.
        """
        if not self.gpu_initialized:
            return None
        try:
            handle = self.get_handle(gpu_index)
            if not handle:
                return None
            limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            return int(limit_mw // 1000)  # convert mW -> W
        except Exception as e:
            self.logger.error(f"Lỗi get_gpu_power_limit GPU={gpu_index}: {e}")
            return None

    def set_gpu_power_limit(self, pid: Optional[int], gpu_index: int, power_limit_w: int) -> bool:
        """
        Đặt power limit cho GPU (đồng bộ).

        :param pid: PID cần quản lý, có thể None nếu áp dụng chung.
        :param gpu_index: Chỉ số GPU.
        :param power_limit_w: Power limit cần đặt (W).
        :return: True nếu thành công, False nếu thất bại.
        """
        if not self.gpu_initialized:
            self.logger.error("GPUResourceManager chưa init. Không thể set power limit.")
            return False
        try:
            handle = self.get_handle(gpu_index)
            if not handle or power_limit_w <= 0:
                return False

            # Lưu power limit cũ
            current_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            current_w = current_mw // 1000
            if pid is not None:
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

    def set_gpu_clocks(self, pid: Optional[int], gpu_index: int, sm_clock: int, mem_clock: int) -> bool:
        """
        Đặt xung nhịp GPU (đồng bộ) thông qua nvidia-smi commands.

        :param pid: PID cần quản lý, có thể None nếu áp dụng chung.
        :param gpu_index: Chỉ số GPU.
        :param sm_clock: Mức SM clock (MHz).
        :param mem_clock: Mức Memory clock (MHz).
        :return: True nếu thành công, False nếu thất bại.
        """
        if not self.gpu_initialized:
            self.logger.error("GPUResourceManager chưa init. Không thể set clocks.")
            return False
        try:
            handle = self.get_handle(gpu_index)
            if not handle or sm_clock <= 0 or mem_clock <= 0:
                return False

            # Lấy SM/MEM clock hiện tại
            current_sm_clock = pynvml.nvmlDeviceGetClock(handle, pynvml.NVML_CLOCK_SM)
            current_mem_clock = pynvml.nvmlDeviceGetClock(handle, pynvml.NVML_CLOCK_MEM)
            if pid is not None:
                if pid not in self.process_gpu_settings:
                    self.process_gpu_settings[pid] = {}
                if gpu_index not in self.process_gpu_settings[pid]:
                    self.process_gpu_settings[pid][gpu_index] = {}
                self.process_gpu_settings[pid][gpu_index]['sm_clock_mhz'] = current_sm_clock
                self.process_gpu_settings[pid][gpu_index]['mem_clock_mhz'] = current_mem_clock

            # Set SM clock
            cmd_sm = [
                'nvidia-smi',
                '-i', str(gpu_index),
                '--lock-gpu-clocks=' + str(sm_clock)
            ]
            subprocess.run(cmd_sm, check=True)
            self.logger.debug(f"Set SM clock={sm_clock}MHz cho GPU={gpu_index}, PID={pid}.")

            # Set MEM clock
            cmd_mem = [
                'nvidia-smi',
                '-i', str(gpu_index),
                '--lock-memory-clocks=' + str(mem_clock)
            ]
            subprocess.run(cmd_mem, check=True)
            self.logger.debug(f"Set MEM clock={mem_clock}MHz cho GPU={gpu_index}, PID={pid}.")

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi nvidia-smi set clocks GPU={gpu_index}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi set clocks GPU={gpu_index}: {e}")
            return False

    def limit_temperature(self, gpu_index: int, temperature_threshold: float, fan_speed_increase: float) -> bool:
        """
        Hạn chế nhiệt độ GPU bằng cách điều chỉnh quạt, power limit, xung nhịp... (đồng bộ).

        :param gpu_index: Chỉ số GPU cần hạn chế.
        :param temperature_threshold: Ngưỡng nhiệt độ (°C).
        :param fan_speed_increase: Tỷ lệ tăng fan speed (giả định).
        :return: True nếu thành công, False nếu thất bại.
        """
        try:
            if not self.gpu_initialized:
                self.logger.error("GPUResourceManager chưa init. Không thể limit_temperature.")
                return False

            # 1) Tăng tốc độ quạt (nếu driver hỗ trợ)
            success_fan = self.control_fan_speed(gpu_index, fan_speed_increase)
            if success_fan:
                self.logger.info(f"Quạt GPU={gpu_index} tăng thêm {fan_speed_increase}%.")
            else:
                self.logger.warning(f"Không thể điều chỉnh quạt GPU={gpu_index}.")

            # 2) Kiểm tra nhiệt độ hiện tại
            current_temperature = self.get_gpu_temperature(gpu_index)
            if current_temperature is None:
                self.logger.warning(f"Không thể lấy nhiệt độ GPU={gpu_index}. Bỏ qua điều chỉnh.")
                return False

            # 3) Lấy xung nhịp hiện tại
            handle = self.get_handle(gpu_index)
            if not handle:
                return False
            try:
                current_sm_clock = pynvml.nvmlDeviceGetClock(handle, pynvml.NVML_CLOCK_SM)
                current_mem_clock = pynvml.nvmlDeviceGetClock(handle, pynvml.NVML_CLOCK_MEM)
            except Exception as ex:
                self.logger.error(f"Không thể lấy current clocks GPU={gpu_index}: {ex}")
                return False

            # 4) So sánh nhiệt độ với threshold
            if current_temperature <= temperature_threshold:
                self.logger.info(
                    f"Nhiệt độ GPU={gpu_index}={current_temperature}°C vượt {temperature_threshold}°C => throttle."
                )
                # Tính mức độ throttle
                excess_temp = current_temperature - temperature_threshold
                if excess_temp <= 5:
                    throttle_pct = 10
                elif excess_temp <= 10:
                    throttle_pct = 20
                else:
                    throttle_pct = 30
                self.logger.debug(f"excess_temp={excess_temp}°C => throttle_pct={throttle_pct}%")

                # Giới hạn power
                current_power_limit = self.get_gpu_power_limit(gpu_index)
                if current_power_limit is None:
                    self.logger.warning(f"Không thể get power_limit GPU={gpu_index}. Bỏ qua throttle.")
                    return False
                desired_power_limit = int(round(current_power_limit * (1 - throttle_pct / 100)))
                success_pl = self.set_gpu_power_limit(None, gpu_index, desired_power_limit)
                if success_pl:
                    self.logger.info(
                        f"Giảm power limit GPU={gpu_index} => {desired_power_limit}W (origin={current_power_limit}W)."
                    )
                else:
                    self.logger.error(f"Không thể set power limit GPU={gpu_index}.")

                # Hạ xung nhịp
                new_sm_clock = max(500, current_sm_clock - 100)
                new_mem_clock = max(300, current_mem_clock - 50)
                success_clocks = self.set_gpu_clocks(None, gpu_index, new_sm_clock, new_mem_clock)
                if success_clocks:
                    self.logger.info(
                        f"Hạ xung nhịp GPU={gpu_index}: SM={new_sm_clock}MHz, MEM={new_mem_clock}MHz."
                    )
                else:
                    self.logger.warning(f"Không thể hạ xung nhịp GPU={gpu_index}.")

            elif current_temperature < temperature_threshold:
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

                # Boost clock
                new_sm_clock = min(current_sm_clock + int(current_sm_clock * boost_pct / 100), 1245)
                new_mem_clock = min(current_mem_clock + int(current_mem_clock * boost_pct / 100), 877)

                success_boost = self.set_gpu_clocks(None, gpu_index, new_sm_clock, new_mem_clock)
                if success_boost:
                    self.logger.info(
                        f"Tăng xung nhịp GPU={gpu_index} => SM={new_sm_clock}MHz, MEM={new_mem_clock}MHz."
                    )
                else:
                    self.logger.warning(f"Không thể boost xung nhịp GPU={gpu_index}.")
            else:
                self.logger.debug(
                    f"Nhiệt độ GPU={gpu_index}={current_temperature}°C xấp xỉ threshold={temperature_threshold}°C => không điều chỉnh."
                )
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi điều khiển nhiệt độ GPU={gpu_index}: {e}")
            return False

    def get_gpu_temperature(self, gpu_index: int) -> Optional[float]:
        """
        Lấy nhiệt độ GPU (đồng bộ).

        :param gpu_index: Chỉ số GPU.
        :return: Nhiệt độ GPU (float) hoặc None nếu lỗi.
        """
        try:
            if not self.gpu_initialized:
                return None
            handle = self.get_handle(gpu_index)
            if not handle:
                return None
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            return float(temp)
        except Exception as e:
            self.logger.error(f"Lỗi get_gpu_temperature GPU={gpu_index}: {e}")
            return None

    def control_fan_speed(self, gpu_index: int, increase_percentage: float) -> bool:
        """
        Điều chỉnh quạt GPU bằng nvidia-settings (đồng bộ). Tuỳ driver hỗ trợ.

        :param gpu_index: Chỉ số GPU.
        :param increase_percentage: Mức tăng quạt (giả lập).
        :return: True nếu thành công, False nếu thất bại.
        """
        try:
            cmd = [
                'nvidia-settings',
                '-a', f'[fan:{gpu_index}]/GPUFanControlState=1',
                '-a', f'[fan:{gpu_index}]/GPUTargetFanSpeed={int(increase_percentage)}'
            ]
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi điều chỉnh quạt GPU {gpu_index}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi điều chỉnh quạt GPU {gpu_index}: {e}")
            return False

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục GPU cho PID: power limit, clocks... (đồng bộ).

        :param pid: PID tiến trình cần khôi phục GPU.
        :return: True nếu khôi phục thành công, False nếu thất bại.
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
                    ok = self.set_gpu_power_limit(pid, gpu_index, int(original_power_limit_w))
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
                    ok_clocks = self.set_gpu_clocks(pid, gpu_index, original_sm_clock, original_mem_clock)
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
    Quản lý tài nguyên mạng qua iptables + tc (đồng bộ).

    Attributes:
        logger (logging.Logger): Logger để ghi log.
        config (Dict[str, Any]): Cấu hình Network Resource Manager.
        process_marks (Dict[int, int]): Bản đồ PID -> mark iptables.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Khởi tạo NetworkResourceManager.

        :param config: Cấu hình network (dict).
        :param logger: Logger.
        """
        self.logger = logger
        self.config = config
        self.process_marks: Dict[int, int] = {}

    def mark_packets(self, pid: int, mark: int) -> bool:
        """
        Thêm iptables rule để đánh dấu (MARK) gói tin, đồng bộ.

        :param pid: PID tiến trình.
        :param mark: Giá trị mark.
        :return: True nếu thành công, False nếu thất bại.
        """
        try:
            cmd = [
                'iptables', '-A', 'OUTPUT', '-m', 'owner',
                '--pid-owner', str(pid),
                '-j', 'MARK', '--set-mark', str(mark)
            ]
            subprocess.run(cmd, check=True)
            self.logger.debug(f"MARK iptables cho PID={pid}, mark={mark}.")
            self.process_marks[pid] = mark
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi iptables MARK PID={pid}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi mark_packets PID={pid}: {e}")
            return False

    def unmark_packets(self, pid: int, mark: int) -> bool:
        """
        Xoá iptables rule (đồng bộ).

        :param pid: PID tiến trình.
        :param mark: Giá trị mark cần xóa.
        :return: True nếu thành công, False nếu thất bại.
        """
        try:
            cmd = [
                'iptables', '-D', 'OUTPUT', '-m', 'owner',
                '--pid-owner', str(pid),
                '-j', 'MARK', '--set-mark', str(mark)
            ]
            subprocess.run(cmd, check=True)
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

    def limit_bandwidth(self, interface: str, mark: int, bandwidth_mbps: float) -> bool:
        """
        Giới hạn băng thông thông qua tc (đồng bộ).

        :param interface: Tên interface (vd: eth0).
        :param mark: Giá trị mark iptables.
        :param bandwidth_mbps: Tốc độ (Mbps) để giới hạn.
        :return: True nếu thành công, False nếu thất bại.
        """
        try:
            cmd_qdisc = [
                'tc', 'qdisc', 'add', 'dev', interface,
                'root', 'handle', '1:', 'htb', 'default', '12'
            ]
            subprocess.run(cmd_qdisc, check=True)
            self.logger.debug(f"Thêm tc qdisc 'htb' cho {interface}.")

            cmd_class = [
                'tc', 'class', 'add', 'dev', interface,
                'parent', '1:', 'classid', '1:1',
                'htb', 'rate', f'{bandwidth_mbps}mbit'
            ]
            subprocess.run(cmd_class, check=True)
            self.logger.debug(f"Thêm tc class '1:1' rate={bandwidth_mbps}mbit cho {interface}.")

            cmd_filter = [
                'tc', 'filter', 'add', 'dev', interface,
                'protocol', 'ip', 'parent', '1:', 'prio', '1',
                'handle', str(mark), 'fw', 'flowid', '1:1'
            ]
            subprocess.run(cmd_filter, check=True)
            self.logger.debug(f"Thêm tc filter mark={mark} trên {interface}.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi limit_bandwidth: {e}")
            return False

    def remove_bandwidth_limit(self, interface: str, mark: int) -> bool:
        """
        Gỡ bỏ giới hạn băng thông thông qua tc (đồng bộ).

        :param interface: Tên interface (vd: eth0).
        :param mark: Giá trị mark iptables.
        :return: True nếu thành công, False nếu thất bại.
        """
        try:
            cmd_filter_del = [
                'tc', 'filter', 'del', 'dev', interface,
                'protocol', 'ip', 'parent', '1:', 'prio', '1',
                'handle', str(mark), 'fw', 'flowid', '1:1'
            ]
            subprocess.run(cmd_filter_del, check=True)
            self.logger.debug(f"Xóa tc filter mark={mark} trên {interface}.")

            cmd_class_del = [
                'tc', 'class', 'del', 'dev', interface,
                'parent', '1:', 'classid', '1:1'
            ]
            subprocess.run(cmd_class_del, check=True)
            self.logger.debug(f"Xóa tc class '1:1' trên {interface}.")

            cmd_qdisc_del = [
                'tc', 'qdisc', 'del', 'dev', interface,
                'root', 'handle', '1:', 'htb'
            ]
            subprocess.run(cmd_qdisc_del, check=True)
            self.logger.debug(f"Xóa tc qdisc 'htb' trên {interface}.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi remove_bandwidth_limit: {e}")
            return False


###############################################################################
#                      DISK I/O RESOURCE MANAGER                              #
###############################################################################
class DiskIOResourceManager:
    """
    Quản lý Disk I/O (đồng bộ) qua ionice hoặc cgroup I/O.

    Attributes:
        logger (logging.Logger): Logger để ghi log.
        config (Dict[str, Any]): Cấu hình Disk I/O Resource Manager.
        process_io_limits (Dict[int, float]): Lưu PID -> giá trị io_weight hoặc giới hạn I/O.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Khởi tạo DiskIOResourceManager.

        :param config: Cấu hình Disk I/O (dict).
        :param logger: Logger.
        """
        self.logger = logger
        self.config = config
        self.process_io_limits: Dict[int, float] = {}

    def set_io_weight(self, pid: int, io_weight: int) -> bool:
        """
        Đặt trọng số I/O cho PID (ionice) - đồng bộ.

        :param pid: PID cần giới hạn.
        :param io_weight: Mức io_weight (1-1000).
        :return: True nếu thành công, False nếu lỗi.
        """
        try:
            cmd = ['ionice', '-c', '2', '-n', str(io_weight), '-p', str(pid)]
            subprocess.run(cmd, check=True)
            self.logger.debug(f"Set io_weight={io_weight} cho PID={pid} qua ionice.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi ionice set_io_weight PID={pid}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi set_io_weight PID={pid}: {e}")
            return False

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục Disk I/O => set ionice class=0 (best effort) - đồng bộ.

        :param pid: PID cần khôi phục Disk I/O.
        :return: True nếu thành công, False nếu lỗi.
        """
        try:
            cmd = ['ionice', '-c', '0', '-p', str(pid)]
            subprocess.run(cmd, check=True)
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
    Quản lý Cache (đồng bộ).

    Attributes:
        logger (logging.Logger): Logger để ghi log.
        config (Dict[str, Any]): Cấu hình Cache Resource Manager.
        dropped_pids (List[int]): Lưu danh sách PID từng được drop cache.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Khởi tạo CacheResourceManager.

        :param config: Cấu hình Cache (dict).
        :param logger: Logger.
        """
        self.logger = logger
        self.config = config
        self.dropped_pids: List[int] = []

    def drop_caches(self, pid: Optional[int] = None) -> bool:
        """
        Drop caches (đồng bộ).

        :param pid: PID liên quan (nếu muốn lưu thêm vào dropped_pids).
        :return: True nếu thành công, False nếu lỗi.
        """
        try:
            cmd = ['sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches']
            subprocess.run(cmd, check=True)
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

    def limit_cache_usage(self, cache_limit_percent: float, pid: Optional[int] = None) -> bool:
        """
        Giới hạn cache => Tối giản: drop caches + log (đồng bộ).
        Chưa có cơ chế kernel-level limit caches.

        :param cache_limit_percent: Tỷ lệ cache limit (0-100).
        :param pid: PID nếu muốn lưu info, mặc định=None.
        :return: True nếu thành công, False nếu lỗi.
        """
        try:
            success = self.drop_caches(pid)
            if not success:
                return False
            self.logger.debug(f"Giới hạn cache => {cache_limit_percent}%. (chưa có logic chi tiết)")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi limit_cache_usage: {e}")
            return False

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục cache => limit_cache_usage(100) (đồng bộ).

        :param pid: PID cần khôi phục cache.
        :return: True nếu thành công, False nếu lỗi.
        """
        try:
            success = self.limit_cache_usage(100.0, pid)
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
    Quản lý Memory qua psutil rlimit (đồng bộ).

    Attributes:
        logger (logging.Logger): Logger để ghi log.
        config (Dict[str, Any]): Cấu hình Memory Resource Manager.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Khởi tạo MemoryResourceManager.

        :param config: Cấu hình Memory (dict).
        :param logger: Logger.
        """
        self.logger = logger
        self.config = config

    def set_memory_limit(self, pid: int, memory_limit_mb: int) -> bool:
        """
        Đặt memory limit (MB) cho tiến trình (đồng bộ).

        :param pid: PID cần giới hạn bộ nhớ.
        :param memory_limit_mb: Giới hạn bộ nhớ (MB).
        :return: True nếu thành công, False nếu lỗi.
        """
        try:
            process = psutil.Process(pid)
            mem_bytes = memory_limit_mb * 1024 * 1024
            process.rlimit(psutil.RLIMIT_AS, (mem_bytes, mem_bytes))
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

    def get_memory_limit(self, pid: int) -> float:
        """
        Lấy memory limit (bytes) cho tiến trình (đồng bộ).

        :param pid: PID cần kiểm tra limit.
        :return: Giá trị memory limit (bytes), hoặc 0.0 nếu lỗi.
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

    def remove_memory_limit(self, pid: int) -> bool:
        """
        Khôi phục memory => không giới hạn (đồng bộ).

        :param pid: PID cần bỏ giới hạn.
        :return: True nếu thành công, False nếu lỗi.
        """
        try:
            process = psutil.Process(pid)
            process.rlimit(psutil.RLIMIT_AS, (psutil.RLIM_INFINITY, psutil.RLIM_INFINITY))
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

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục memory => remove_memory_limit (đồng bộ).

        :param pid: PID cần khôi phục memory.
        :return: True nếu thành công, False nếu lỗi.
        """
        return self.remove_memory_limit(pid)


###############################################################################
#                     RESOURCE CONTROL FACTORY                                #
###############################################################################
class ResourceControlFactory:
    """
    Factory tạo các resource manager (CPU, GPU, Network, Disk I/O, Cache, Memory) theo mô hình đồng bộ.
    """

    @staticmethod
    def create_resource_managers(config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
        """
        Khởi tạo tất cả resource managers theo mô hình đồng bộ.

        :param config: Cấu hình ResourceManager (dict).
        :param logger: Logger dùng để ghi log.
        :return: Dictionary chứa các resource managers.
        """
        resource_managers = {}
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
                resource_managers[name] = manager_instance
                logger.info(f"{name.capitalize()} manager đã được khởi tạo thành công.")
            except Exception as e:
                logger.error(f"Lỗi khi khởi tạo {name} manager: {e}", exc_info=True)

        if not resource_managers:
            logger.error("Không có resource managers nào được khởi tạo.")
            raise RuntimeError("Tất cả resource managers đều khởi tạo thất bại.")

        logger.info("Tất cả resource managers đã được khởi tạo.")
        return resource_managers
