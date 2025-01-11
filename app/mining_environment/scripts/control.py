# resource_control.py

import logging
import subprocess
import os
from typing import Any, Dict, List, Optional, Tuple

import psutil
import pynvml  # NVIDIA Management Library

from .utils import GPUManager  # Import GPUManager từ utils.py


class CPUResourceManager:
    """
    Quản lý tài nguyên CPU thông qua việc điều chỉnh độ ưu tiên tiến trình và affinity.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def set_cpu_priority(self, pid: int, priority: int) -> bool:
        """
        Thiết lập độ ưu tiên CPU cho tiến trình.

        Args:
            pid (int): PID của tiến trình.
            priority (int): Giá trị độ ưu tiên (từ -20 đến 19, với -20 là ưu tiên cao nhất).

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            process = psutil.Process(pid)
            process.nice(priority)
            self.logger.debug(f"Đặt độ ưu tiên CPU cho PID={pid} thành {priority}.")
            return True
        except psutil.NoSuchProcess:
            self.logger.error(f"Tiến trình PID={pid} không tồn tại khi đặt độ ưu tiên CPU.")
            return False
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền để đặt độ ưu tiên CPU cho PID={pid}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi đặt độ ưu tiên CPU cho PID={pid}: {e}")
            return False

    def delete_resource_limits(self, pid: int) -> bool:
        """
        Khôi phục độ ưu tiên CPU cho tiến trình về mức bình thường.

        Args:
            pid (int): PID của tiến trình.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        try:
            process = psutil.Process(pid)
            process.nice(psutil.NORMAL_PRIORITY_CLASS)
            self.logger.debug(f"Đã khôi phục độ ưu tiên CPU cho PID={pid}.")
            return True
        except psutil.NoSuchProcess:
            self.logger.error(f"Tiến trình PID={pid} không tồn tại khi khôi phục CPU.")
            return False
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền để khôi phục CPU cho PID={pid}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục CPU cho PID={pid}: {e}")
            return False

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
            utilization = self.gpu_manager.get_utilization(handle)
            self.logger.debug(f"Sử dụng GPU {gpu_index}: {utilization}")
            return utilization
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
                success_power = self.set_gpu_power_limit(gpu_index, desired_power_limit_w)
                if success_power:
                    self.logger.info(f"Giảm power limit GPU {gpu_index} xuống {desired_power_limit_w}W để giảm nhiệt độ.")
                else:
                    self.logger.error(f"Không thể giảm power limit GPU {gpu_index} để giảm nhiệt độ.")

                # Giảm xung nhịp GPU để giảm nhiệt độ
                target_sm_clock = max(500, self.gpu_manager.target_sm_clock - 100)   # Giảm xung nhịp SM xuống tối thiểu 500MHz
                target_mem_clock = max(300, self.gpu_manager.target_mem_clock - 50)  # Giảm xung nhịp MEM xuống tối thiểu 300MHz
                success_clocks = self.set_gpu_clocks(gpu_index, target_sm_clock, target_mem_clock)
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

                # Lấy xung nhịp hiện tại
                current_sm_clock = self.gpu_manager.target_sm_clock
                current_mem_clock = self.gpu_manager.target_mem_clock

                # Tính toán xung nhịp mới
                desired_sm_clock = min(current_sm_clock + int(current_sm_clock * boost_percentage / 100), 1530)  # Không vượt quá Boost Clock
                desired_mem_clock = min(current_mem_clock + int(current_mem_clock * boost_percentage / 100), 877)  # Không vượt quá Memory Clock

                # Tăng xung nhịp
                success_clocks = self.set_gpu_clocks(gpu_index, desired_sm_clock, desired_mem_clock)
                if success_clocks:
                    self.logger.info(f"Tăng xung nhịp GPU {gpu_index} lên SM={desired_sm_clock}MHz, MEM={desired_mem_clock}MHz để cải thiện hiệu suất.")
                else:
                    self.logger.warning(f"Không thể tăng xung nhịp GPU {gpu_index} để cải thiện hiệu suất.")

            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi giới hạn nhiệt độ GPU {gpu_index}: {e}")
            return False
        

    # Các lớp quản lý tài nguyên khác sẽ được giữ nguyên hoặc thêm các phương thức cần thiết.


class NetworkResourceManager:
    """
    Quản lý tài nguyên mạng thông qua iptables và tc.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

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


class DiskIOResourceManager:
    """
    Quản lý tài nguyên Disk I/O thông qua tc hoặc các cơ chế kiểm soát I/O khác.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

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
            # Thêm qdisc root
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
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi đặt trọng số I/O cho PID={pid}: {e}")
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
            float: Giới hạn bộ nhớ tính bằng bytes. Trả về 0.0 nếu không thành công.
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
