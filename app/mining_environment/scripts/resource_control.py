# resource_control.py

import logging
import subprocess
import os
from typing import Any, Dict, Optional, Tuple
from .utils import GPUManager  # Import GPUManager từ utils.py
from .cgroup_manager import CgroupManager  # Import CgroupManager từ cgroup_manager.py




class CPUResourceManager:
    """
    Quản lý tài nguyên CPU thông qua cgroups.
    """

    def __init__(self, logger: logging.Logger, cgroup_manager: CgroupManager):
        self.logger = logger
        self.cgroup_manager = cgroup_manager


    def set_cpu_quota(self, cgroup_name: str, quota: int, period: int = 100000) -> bool:
        """
        Thiết lập CPU quota cho cgroup.

        Args:
            cgroup_name (str): Tên của cgroup.
            quota (int): Giá trị CPU quota.
            period (int, optional): Số microseconds CPU period. Defaults to 100000.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            cpu_max_path = f"/sys/fs/cgroup/{cgroup_name}/cpu.max"
            with open(cpu_max_path, 'w') as f:
                f.write(f"{quota} {period}")
            self.logger.debug(f"Đặt CPU quota cho cgroup '{cgroup_name}': {quota}us/{period}us.")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi đặt CPU quota cho cgroup '{cgroup_name}': {e}")
            return False
        
    def delete_cgroup(self, cgroup_name: str) -> bool:
        """
        Xóa cgroup CPU sử dụng CgroupManager.
        """
        return self.cgroup_manager.delete_cgroup(cgroup_name)

class GPUResourceManager:
    """
    Quản lý tài nguyên GPU thông qua NVML (NVIDIA Management Library) và nvidia-smi.
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

    def set_gpu_power_limit(self, gpu_index: int, power_limit_w: int) -> bool:
        """
        Thiết lập power limit cho GPU.

        Args:
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
            self.gpu_manager.set_power_limit(handle, power_limit_mw)
            self.logger.debug(f"Đặt power limit cho GPU {gpu_index} là {power_limit_w}W.")
            return True
        except pynvml.NVMLError as error:
            self.logger.error(f"Lỗi NVML khi đặt power limit cho GPU {gpu_index}: {error}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi đặt power limit cho GPU {gpu_index}: {e}")
            return False

    def set_gpu_clocks(self, gpu_index: int, sm_clock: int, mem_clock: int) -> bool:
        """
        Thiết lập xung nhịp GPU thông qua nvidia-smi.

        Args:
            gpu_index (int): Chỉ số GPU (bắt đầu từ 0).
            mem_clock (int): Xung nhịp bộ nhớ GPU tính bằng MHz.
            sm_clock (int): Xung nhịp SM GPU tính bằng MHz.

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
            # Thiết lập xung nhịp SM
            cmd_sm = [
                'nvidia-smi',
                '-i', str(gpu_index),
                '--lock-gpu-clocks=' + str(sm_clock)
            ]
            subprocess.run(cmd_sm, check=True)
            self.logger.debug(f"Đã thiết lập xung nhịp SM cho GPU {gpu_index} là {sm_clock}MHz.")

            # Thiết lập xung nhịp bộ nhớ
            cmd_mem = [
                'nvidia-smi',
                '-i', str(gpu_index),
                '--lock-memory-clocks=' + str(mem_clock)
            ]
            subprocess.run(cmd_mem, check=True)
            self.logger.debug(f"Đã thiết lập xung nhịp bộ nhớ cho GPU {gpu_index} là {mem_clock}MHz.")

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi thiết lập xung nhịp GPU {gpu_index} bằng nvidia-smi: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi thiết lập xung nhịp GPU {gpu_index}: {e}")
            return False

    def set_gpu_max_power(self, gpu_index: int, gpu_max_mw: int) -> bool:
        """
        Thiết lập giới hạn power tối đa cho GPU thông qua NVML.

        Args:
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
            self.gpu_manager.set_power_limit(handle, gpu_max_mw)
            power_limit_w = gpu_max_mw / 1000  # Chuyển từ milliWatts sang Watts
            self.logger.debug(f"Đặt 'gpu_max' cho GPU {gpu_index} là {power_limit_w}W.")
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
            util = self.gpu_manager.get_utilization(handle)
            self.logger.debug(f"Sử dụng GPU {gpu_index}: {util}")
            return util
        except pynvml.NVMLError as error:
            self.logger.error(f"Lỗi NVML khi lấy sử dụng GPU cho GPU {gpu_index}: {error}")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi lấy sử dụng GPU cho GPU {gpu_index}: {e}")
            return None

    def get_gpu_temperature(self, gpu_index: int) -> Optional[int]:
        """
        Lấy nhiệt độ hiện tại của GPU.

        Args:
            gpu_index (int): Chỉ số GPU (bắt đầu từ 0).

        Returns:
            Optional[int]: Nhiệt độ GPU tính bằng độ C hoặc None nếu lỗi.
        """
        if not self.gpu_initialized:
            self.logger.error("GPU chưa được khởi tạo hoặc không có GPU nào trên hệ thống. Không thể lấy nhiệt độ GPU.")
            return None

        if not (0 <= gpu_index < self.gpu_manager.gpu_count):
            self.logger.error(f"GPU index {gpu_index} không hợp lệ. Có tổng cộng {self.gpu_manager.gpu_count} GPU.")
            return None

        try:
            handle = self.gpu_manager.get_handle(gpu_index)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            self.logger.debug(f"Nhiệt độ GPU {gpu_index}: {temperature}°C")
            return temperature
        except pynvml.NVMLError as error:
            self.logger.error(f"Lỗi NVML khi lấy nhiệt độ GPU {gpu_index}: {error}")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi lấy nhiệt độ GPU {gpu_index}: {e}")
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
            # Lưu ý: Cần cài đặt nvidia-settings và GPU phải hỗ trợ điều chỉnh quạt
            cmd = [
                'nvidia-settings',
                '-a', f'[gpu:{gpu_index}]/GPUFanControlState=1',
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
        Thực hiện các hành động để quản lý nhiệt độ GPU.
        - Luôn luôn tăng tốc độ quạt để làm mát.
        - Nếu nhiệt độ vượt ngưỡng, giảm power limit và xung nhịp.
        - Nếu nhiệt độ dưới ngưỡng, tăng xung nhịp để cải thiện hiệu suất.
        
        Args:
            gpu_index (int): Chỉ số GPU.
            temperature_threshold (float): Ngưỡng nhiệt độ tối đa (°C).
            fan_speed_increase (float): Tỷ lệ tăng tốc độ quạt (%) khi cần thiết.
        
        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            # Luôn luôn tăng tốc độ quạt để làm mát GPU
            success_fan = self.control_fan_speed(gpu_index, fan_speed_increase)
            if success_fan:
                self.logger.info(
                    f"Tăng tốc độ quạt GPU {gpu_index} lên {fan_speed_increase}% để làm mát."
                )
            else:
                self.logger.warning(
                    f"Không thể tăng tốc độ quạt GPU {gpu_index}. Cần kiểm tra hỗ trợ điều chỉnh quạt."
                )
            
            # Lấy nhiệt độ hiện tại của GPU
            current_temperature = self.get_gpu_temperature(gpu_index)
            if current_temperature is None:
                self.logger.warning(f"Không thể lấy nhiệt độ hiện tại cho GPU {gpu_index}.")
                return False

            if current_temperature > temperature_threshold:
                # **Nhiệt độ vượt ngưỡng: Giảm power limit và xung nhịp**
                self.logger.info(f"Nhiệt độ GPU {gpu_index} hiện tại {current_temperature}°C vượt ngưỡng {temperature_threshold}°C. Thực hiện cloaking.")
                
                # Tính toán mức giảm power limit dựa trên mức vượt ngưỡng
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
                desired_power_limit_w = self.get_gpu_power_limit(gpu_index)
                if desired_power_limit_w is None:
                    self.logger.warning(f"Không thể lấy power limit hiện tại cho GPU {gpu_index}.")
                    return False
                desired_power_limit_w = int(round(desired_power_limit_w * (1 - throttle_percentage / 100)))

                # Giảm power limit
                success_power = self.set_gpu_power_limit(gpu_index, desired_power_limit_w)
                if success_power:
                    self.logger.info(
                        f"Giảm power limit GPU {gpu_index} xuống {desired_power_limit_w}W để giảm nhiệt độ."
                    )
                else:
                    self.logger.error(
                        f"Không thể giảm power limit GPU {gpu_index} để giảm nhiệt độ."
                    )

                # Giảm xung nhịp GPU để giảm nhiệt độ
                target_sm_clock = max(500, self.gpu_manager.target_sm_clock - 100)   # Giảm xung nhịp SM xuống tối thiểu 500MHz
                target_mem_clock = max(300, self.gpu_manager.target_mem_clock - 50)  # Giảm xung nhịp MEM xuống tối thiểu 300MHz
                success_clocks = self.set_gpu_clocks(gpu_index, target_sm_clock, target_mem_clock)
                if success_clocks:
                    self.logger.info(
                        f"Giảm xung nhịp GPU {gpu_index} xuống SM={target_sm_clock}MHz, MEM={target_mem_clock}MHz để giảm nhiệt độ."
                    )
                else:
                    self.logger.warning(
                        f"Không thể giảm xung nhịp GPU {gpu_index} để giảm nhiệt độ."
                    )

            elif current_temperature < temperature_threshold:
                # **Nhiệt độ dưới ngưỡng: Tăng xung nhịp**
                self.logger.info(f"Nhiệt độ GPU {gpu_index} hiện tại {current_temperature}°C dưới ngưỡng {temperature_threshold}°C. Tăng xung nhịp để cải thiện hiệu suất.")
                
                # Tính toán mức tăng xung nhịp
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

                # Lấy xung nhịp hiện tại
                current_sm_clock = self.gpu_manager.target_sm_clock
                current_mem_clock = self.gpu_manager.target_mem_clock

                # Tính toán xung nhịp mới
                desired_sm_clock = min(current_sm_clock + int(current_sm_clock * boost_percentage / 100), 1530)  # Không vượt quá Boost Clock
                desired_mem_clock = min(current_mem_clock + int(current_mem_clock * boost_percentage / 100), 877)  # Không vượt quá Memory Clock

                # Tăng xung nhịp
                success_clocks = self.set_gpu_clocks(gpu_index, desired_sm_clock, desired_mem_clock)
                if success_clocks:
                    self.logger.info(
                        f"Tăng xung nhịp GPU {gpu_index} lên SM={desired_sm_clock}MHz, MEM={desired_mem_clock}MHz để cải thiện hiệu suất."
                    )
                else:
                    self.logger.warning(
                        f"Không thể tăng xung nhịp GPU {gpu_index} để cải thiện hiệu suất."
                    )

            return True

        except Exception as e:
            self.logger.error(f"Lỗi khi giới hạn nhiệt độ GPU {gpu_index}: {e}")
            return False

    def cleanup(self):
        """
        Giải phóng tài nguyên NVML khi không còn cần thiết.
        """
        try:
            self.gpu_manager.shutdown()
            self.logger.info("Đã shutdown NVML thành công.")
        except pynvml.NVMLError as error:
            self.logger.error(f"Lỗi NVML khi shutdown: {error}")
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi shutdown NVML: {e}")

class NetworkResourceManager:
    """
    Quản lý tài nguyên mạng thông qua iptables và tc.
    """

    def __init__(self, logger: logging.Logger, cgroup_manager: CgroupManager):
        self.logger = logger
        self.cgroup_manager = cgroup_manager

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

    def set_classid(self, cgroup_name: str, classid: int) -> bool:
        """
        Đặt classid cho cgroup Network thông qua cgroup parameter.

        Args:
            cgroup_name (str): Tên của cgroup Network.
            classid (int): Classid để đặt.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            classid_path = f"/sys/fs/cgroup/{cgroup_name}/net_cls.classid"
            with open(classid_path, 'w') as f:
                f.write(str(classid))
            self.logger.debug(f"Đặt 'net_cls.classid' cho cgroup '{cgroup_name}' là {classid}.")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi đặt 'net_cls.classid' cho cgroup '{cgroup_name}': {e}")
            return False

    def restore_classid(self, cgroup_name: str) -> bool:
        """
        Khôi phục classid cho cgroup Network bằng cách xóa giá trị.

        Args:
            cgroup_name (str): Tên của cgroup Network.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            classid_path = f"/sys/fs/cgroup/{cgroup_name}/net_cls.classid"
            with open(classid_path, 'w') as f:
                f.write('0')
            self.logger.debug(f"Khôi phục 'net_cls.classid' cho cgroup '{cgroup_name}' thành 0.")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục 'net_cls.classid' cho cgroup '{cgroup_name}': {e}")
            return False

    def delete_cgroup(self, cgroup_name: str) -> bool:
        """
        Xóa cgroup Network sử dụng CgroupManager.
        """
        return self.cgroup_manager.delete_cgroup(cgroup_name)
    
class DiskIOResourceManager:
    """
    Quản lý tài nguyên Disk I/O thông qua cgroups.
    """

    def __init__(self, logger: logging.Logger, cgroup_manager: CgroupManager):
        self.logger = logger
        self.cgroup_manager = cgroup_manager

    def set_io_weight(self, cgroup_name: str, io_weight: int) -> bool:
        """
        Thiết lập I/O weight cho cgroup.

        Args:
            cgroup_name (str): Tên của cgroup.
            io_weight (int): Trọng số I/O (1-1000).

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            io_weight_path = f"/sys/fs/cgroup/{cgroup_name}/io.weight"
            with open(io_weight_path, 'w') as f:
                f.write(str(io_weight))
            self.logger.debug(f"Đặt I/O weight cho cgroup '{cgroup_name}' là {io_weight}.")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi đặt I/O weight cho cgroup '{cgroup_name}': {e}")
            return False

    def delete_cgroup(self, cgroup_name: str) -> bool:
        """
        Xóa cgroup Disk I/O sử dụng CgroupManager.
        """
        return self.cgroup_manager.delete_cgroup(cgroup_name)
    
class CacheResourceManager:
    """
    Quản lý tài nguyên Cache thông qua cgroups.
    """

    def __init__(self, logger: logging.Logger, cgroup_manager: CgroupManager):
        self.logger = logger
        self.cgroup_manager = cgroup_manager

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

    def set_cache_limit(self, cgroup_name: str, cache_limit_bytes: int) -> bool:
        """
        Thiết lập giới hạn cache cho cgroup.

        Args:
            cgroup_name (str): Tên của cgroup.
            cache_limit_bytes (int): Giới hạn cache tính bằng bytes.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            cache_max_path = f"/sys/fs/cgroup/{cgroup_name}/cache.max"
            with open(cache_max_path, 'w') as f:
                f.write(str(cache_limit_bytes))
            self.logger.debug(f"Đặt 'cache.max' cho cgroup '{cgroup_name}' là {cache_limit_bytes} bytes.")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi đặt 'cache.max' cho cgroup '{cgroup_name}': {e}")
            return False

    def delete_cgroup(self, cgroup_name: str) -> bool:
        """
        Xóa cgroup Cache sử dụng CgroupManager.
        """
        return self.cgroup_manager.delete_cgroup(cgroup_name)
    
class MemoryResourceManager:
    """
    Quản lý tài nguyên Memory thông qua cgroups.
    """

    def __init__(self, logger: logging.Logger, cgroup_manager: CgroupManager):
        self.logger = logger
        self.cgroup_manager = cgroup_manager
        self.memory_limits = {}  # Optional: Dùng để lưu trữ giới hạn bộ nhớ đã thiết lập theo PID

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

    def set_memory_limit(self, cgroup_name: str, memory_limit_bytes: int) -> bool:
        """
        Thiết lập giới hạn bộ nhớ cho cgroup.

        Args:
            cgroup_name (str): Tên của cgroup.
            memory_limit_bytes (int): Giới hạn bộ nhớ tính bằng bytes.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            memory_max_path = f"/sys/fs/cgroup/{cgroup_name}/memory.max"
            with open(memory_max_path, 'w') as f:
                f.write(str(memory_limit_bytes))
            self.logger.debug(f"Đặt 'memory.max' cho cgroup '{cgroup_name}' là {memory_limit_bytes} bytes.")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi đặt 'memory.max' cho cgroup '{cgroup_name}': {e}")
            return False

    def get_memory_limit(self, cgroup_name: str) -> float:
        """
        Lấy giới hạn bộ nhớ đã thiết lập cho cgroup cụ thể.

        Args:
            cgroup_name (str): Tên của cgroup.

        Returns:
            float: Giới hạn bộ nhớ tính bằng bytes. Trả về 0.0 nếu không thành công.
        """
        try:
            memory_max_path = f"/sys/fs/cgroup/{cgroup_name}/memory.max"
            with open(memory_max_path, 'r') as f:
                content = f.read().strip()
                if content == "max":
                    self.logger.debug(f"Giới hạn bộ nhớ cho cgroup '{cgroup_name}' là 'max' (inf).")
                    return float('inf')
                memory_limit = float(content)
                self.logger.debug(f"Giới hạn bộ nhớ cho cgroup '{cgroup_name}' là {memory_limit} bytes.")
                return memory_limit
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy 'memory.max' từ cgroup '{cgroup_name}': {e}")
            return 0.0

    def delete_cgroup(self, cgroup_name: str) -> bool:
        """
        Xóa cgroup Memory sử dụng CgroupManager.
        """
        return self.cgroup_manager.delete_cgroup(cgroup_name)

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

        # Khởi tạo CgroupManager cho các resource manager khác (không bao gồm GPUResourceManager)
        cgroup_manager = CgroupManager(logger)

        # Tạo các resource manager cơ bản
        resource_managers = {
            'cpu': CPUResourceManager(logger, cgroup_manager),
            'network': NetworkResourceManager(logger, cgroup_manager),
            'io': DiskIOResourceManager(logger, cgroup_manager),
            'cache': CacheResourceManager(logger, cgroup_manager),
            'memory': MemoryResourceManager(logger, cgroup_manager),
            'gpu': GPUResourceManager(logger, gpu_manager)  # Luôn khởi tạo GPUResourceManager
        }

        # Lớp GPUResourceManager tự xử lý việc có GPU hay không và không phụ thuộc vào cgroups v2
        # Do đó, không cần kiểm tra sự có mặt của controller 'gpu'

        return resource_managers
