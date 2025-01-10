# resource_control.py

import logging
import subprocess
import os
from typing import Any, Dict
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
    Quản lý tài nguyên GPU thông qua NVML và cgroups.
    """

    def __init__(self, logger: logging.Logger, gpu_manager: GPUManager, cgroup_manager: CgroupManager):
        self.logger = logger
        self.gpu_manager = gpu_manager
        self.cgroup_manager = cgroup_manager
        if self.gpu_manager.gpu_count > 0:
            self.gpu_initialized = True
        else:
            self.gpu_initialized = False
            self.logger.warning("Không có GPU nào được phát hiện trên hệ thống.")

    def set_gpu_power_limit(self, gpu_index: int, power_limit_w: int) -> bool:
        """
        Thiết lập power limit cho GPU.

        Args:
            gpu_index (int): Chỉ số GPU.
            power_limit_w (int): Power limit tính bằng Watts.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        if not self.gpu_initialized:
            self.logger.error("GPU chưa được khởi tạo. Không thể đặt power limit.")
            return False
        try:
            handle = self.gpu_manager.get_handle(gpu_index)
            power_limit_mw = power_limit_w * 1000
            self.gpu_manager.set_power_limit(handle, power_limit_mw)
            self.logger.debug(f"Đặt power limit cho GPU {gpu_index} là {power_limit_w}W.")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi đặt power limit cho GPU {gpu_index}: {e}")
            return False

    def set_gpu_clocks(self, gpu_index: int, mem_clock: int, sm_clock: int) -> bool:
        """
        Thiết lập xung nhịp GPU.

        Args:
            gpu_index (int): Chỉ số GPU.
            mem_clock (int): Xung nhịp bộ nhớ GPU tính bằng MHz.
            sm_clock (int): Xung nhịp SM GPU tính bằng MHz.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        if not self.gpu_initialized:
            self.logger.error("GPU chưa được khởi tạo. Không thể đặt xung nhịp.")
            return False
        try:
            handle = self.gpu_manager.get_handle(gpu_index)
            self.gpu_manager.set_clocks(handle, mem_clock, sm_clock)
            self.logger.debug(f"Đặt xung nhịp GPU {gpu_index}: SM={sm_clock}MHz, MEM={mem_clock}MHz.")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi đặt xung nhịp GPU {gpu_index}: {e}")
            return False

    def set_gpu_max(self, cgroup_name: str, gpu_max_mw: int) -> bool:
        """
        Thiết lập giới hạn GPU max cho cgroup thông qua cgroup parameter.

        Args:
            cgroup_name (str): Tên của cgroup.
            gpu_max_mw (int): Giới hạn GPU max tính bằng mW.

        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        if not self.gpu_initialized:
            self.logger.error("GPU chưa được khởi tạo. Không thể đặt 'gpu.max'.")
            return False
        try:
            gpu_max_path = f"/sys/fs/cgroup/{cgroup_name}/gpu.max"
            with open(gpu_max_path, 'w') as f:
                f.write(str(gpu_max_mw))
            self.logger.debug(f"Đặt 'gpu.max' cho cgroup '{cgroup_name}' là {gpu_max_mw} mW.")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi đặt 'gpu.max' cho cgroup '{cgroup_name}': {e}")
            return False
        
    def delete_cgroup(self, cgroup_name: str) -> bool:
        """
        Xóa cgroup GPU sử dụng CgroupManager.
        """
        return self.cgroup_manager.delete_cgroup(cgroup_name)


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
        gpu_manager = GPUManager()
        cgroup_manager = CgroupManager(logger)  # Khởi tạo CgroupManager

        resource_managers = {
            'cpu': CPUResourceManager(logger, cgroup_manager),  # Truyền cgroup_manager vào
            'gpu': GPUResourceManager(logger, gpu_manager, cgroup_manager),
            'network': NetworkResourceManager(logger, cgroup_manager),
            'io': DiskIOResourceManager(logger, cgroup_manager),
            'cache': CacheResourceManager(logger, cgroup_manager),
            'memory': MemoryResourceManager(logger, cgroup_manager)
        }
        return resource_managers
