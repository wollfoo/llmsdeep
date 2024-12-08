# auxiliary_modules/cgroup_manager.py

import os
import sys
import subprocess
from pathlib import Path
import psutil
import logging
from typing import Dict, Any, Tuple


# Thêm đường dẫn tới thư mục chứa `logging_config.py`

SCRIPT_DIR = Path(__file__).resolve().parent.parent  
sys.path.append(str(SCRIPT_DIR))  

# Define configuration directories (assumed to be set in environment variables)
LOGS_DIR = Path(os.getenv('LOGS_DIR', '/app/mining_environment/logs'))

from logging_config import setup_logging

# Setup logger for cgroup_manager
cgroup_manager_logger = setup_logging('cgroup_manager', LOGS_DIR / 'cgroup_manager.log', 'INFO')


def setup_cgroups(resource_config: Dict[str, Any], logger: logging.Logger):
    """
    Thiết lập cgroups cho các tài nguyên như CPU, RAM, và Disk I/O dựa trên cấu hình.

    Args:
        resource_config (dict): Cấu hình tài nguyên từ tệp cấu hình.
        logger (Logger): Đối tượng logger để ghi log.
    """
    try:
        # Thiết lập cgroups CPU
        setup_cpu_cgroup(resource_config, logger)

        # Thiết lập cgroups RAM
        setup_ram_cgroup(resource_config, logger)

        # Thiết lập cgroups Disk I/O
        setup_disk_io_cgroup(resource_config, logger)

        logger.info("Tất cả các cgroups đã được thiết lập thành công.")
    except Exception as e:
        logger.error(f"Lỗi khi thiết lập cgroups: {e}")
        sys.exit(1)


def setup_cpu_cgroup(resource_config: Dict[str, Any], logger: logging.Logger):
    """
    Thiết lập cgroups CPU dựa trên cấu hình.

    Args:
        resource_config (dict): Cấu hình tài nguyên.
        logger (Logger): Đối tượng logger.
    """
    try:
        cpu_group_path = "/sys/fs/cgroup/cpu/mining_group"
        os.makedirs(cpu_group_path, exist_ok=True)

        cpu_shares = resource_config['resource_allocation']['cpu'].get('cpu_shares', 768)
        cpu_quota = resource_config['resource_allocation']['cpu'].get('cpu_quota', -1)  # -1 means no limit

        with open(os.path.join(cpu_group_path, "cpu.shares"), 'w') as f:
            f.write(str(cpu_shares) + "\n")
        logger.info(f"Cgroups CPU đã được thiết lập với cpu.shares={cpu_shares}.")

        if cpu_quota != -1:
            with open(os.path.join(cpu_group_path, "cpu.cfs_quota_us"), 'w') as f:
                f.write(str(cpu_quota) + "\n")
            logger.info(f"Cgroups CPU đã được thiết lập với cpu.cfs_quota_us={cpu_quota}μs.")
        else:
            logger.info("Không thiết lập cpu.cfs_quota_us cho cgroups CPU.")
    except PermissionError:
        logger.error("Không có quyền để thiết lập cgroups CPU. Vui lòng chạy script với quyền root.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Lỗi khi thiết lập cgroups CPU: {e}")
        raise


def setup_ram_cgroup(resource_config: Dict[str, Any], logger: logging.Logger):
    """
    Thiết lập cgroups RAM dựa trên cấu hình.

    Args:
        resource_config (dict): Cấu hình tài nguyên.
        logger (Logger): Đối tượng logger.
    """
    try:
        ram_group_path = "/sys/fs/cgroup/memory/mining_group"
        os.makedirs(ram_group_path, exist_ok=True)

        ram_limit_mb = resource_config['resource_allocation']['ram'].get('max_allocation_mb', 1024)
        ram_limit_bytes = ram_limit_mb * 1024 * 1024

        with open(os.path.join(ram_group_path, "memory.limit_in_bytes"), 'w') as f:
            f.write(str(ram_limit_bytes) + "\n")
        logger.info(f"Cgroups RAM đã được thiết lập với memory.limit_in_bytes={ram_limit_bytes} bytes ({ram_limit_mb} MB).")
    except PermissionError:
        logger.error("Không có quyền để thiết lập cgroups RAM. Vui lòng chạy script với quyền root.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Lỗi khi thiết lập cgroups RAM: {e}")
        raise


def setup_disk_io_cgroup(resource_config: Dict[str, Any], logger: logging.Logger):
    """
    Thiết lập cgroups Disk I/O dựa trên cấu hình.

    Args:
        resource_config (dict): Cấu hình tài nguyên.
        logger (Logger): Đối tượng logger.
    """
    try:
        disk_io_group_path = "/sys/fs/cgroup/blkio/mining_group"
        os.makedirs(disk_io_group_path, exist_ok=True)

        # Lấy MAJ:MIN của thiết bị lưu trữ chính
        major, minor = get_primary_disk_device(logger)

        read_limit_mbps = resource_config['resource_allocation']['disk_io'].get('read_limit_mbps', 10)
        write_limit_mbps = resource_config['resource_allocation']['disk_io'].get('write_limit_mbps', 10)
        read_limit_bytes = mbps_to_bytes(read_limit_mbps)
        write_limit_bytes = mbps_to_bytes(write_limit_mbps)

        # Thiết lập giới hạn đọc
        blkio_read_path = os.path.join(disk_io_group_path, "blkio.throttle.read_bps_device")
        with open(blkio_read_path, 'w') as f:
            f.write(f"{major}:{minor} {read_limit_bytes}\n")
        logger.info(f"Cgroups Disk I/O đã được thiết lập với read_limit={read_limit_mbps} Mbps trên thiết bị {major}:{minor}.")

        # Thiết lập giới hạn ghi
        blkio_write_path = os.path.join(disk_io_group_path, "blkio.throttle.write_bps_device")
        with open(blkio_write_path, 'w') as f:
            f.write(f"{major}:{minor} {write_limit_bytes}\n")
        logger.info(f"Cgroups Disk I/O đã được thiết lập với write_limit={write_limit_mbps} Mbps trên thiết bị {major}:{minor}.")
    except PermissionError:
        logger.error("Không có quyền để thiết lập cgroups Disk I/O. Vui lòng chạy script với quyền root.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Lỗi khi thiết lập cgroups Disk I/O: {e}")
        raise


def assign_process_to_cgroups(pid: int, resource_dict: Dict[str, Any], process_name: str, logger: logging.Logger):
    """
    Gán tiến trình với PID vào các cgroups và thiết lập giới hạn tài nguyên cụ thể.

    Args:
        pid (int): PID của tiến trình cần gán.
        resource_dict (dict): Các giới hạn tài nguyên.
        process_name (str): Tên của tiến trình.
        logger (Logger): Đối tượng logger để ghi log.
    """
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        logger.warning(f"Không tìm thấy tiến trình PID: {pid} để gán cgroups.")
        return
    except Exception as e:
        logger.error(f"Lỗi khi truy cập tiến trình PID {pid}: {e}")
        return

    try:
        # Gán CPU cores
        if 'cpu_threads' in resource_dict:
            cpu_threads = resource_dict['cpu_threads']
            assign_cpu_cgroup(proc, cpu_threads, logger)

        # Gán RAM limit
        if 'memory' in resource_dict:
            memory_limit_mb = resource_dict['memory']
            assign_ram_cgroup(proc, memory_limit_mb, logger)

        # Gán Disk I/O limit
        if 'disk_io_limit_mbps' in resource_dict:
            disk_io_limit_mbps = resource_dict['disk_io_limit_mbps']
            assign_disk_io_cgroup(proc, disk_io_limit_mbps, logger)

        # Gán CPU frequency nếu có
        if 'cpu_freq' in resource_dict:
            cpu_freq_mhz = resource_dict['cpu_freq']
            assign_cpu_freq(proc, cpu_freq_mhz, logger)

        # Thêm PID vào các cgroups
        add_pid_to_cgroups(pid, logger)

    except PermissionError:
        logger.error("Không có quyền để gán tiến trình vào cgroups. Vui lòng chạy script với quyền root.")
    except Exception as e:
        logger.error(f"Lỗi khi gán tiến trình PID {pid} vào cgroups: {e}")


def assign_cpu_cgroup(proc: psutil.Process, cpu_threads: int, logger: logging.Logger):
    """
    Gán tiến trình vào cgroups CPU với số lượng threads cụ thể.

    Args:
        proc (psutil.Process): Đối tượng tiến trình.
        cpu_threads (int): Số lượng threads CPU cần gán.
        logger (Logger): Đối tượng logger.
    """
    try:
        total_cpu_cores = psutil.cpu_count(logical=True)
        if cpu_threads > total_cpu_cores:
            cpu_threads = total_cpu_cores
            logger.warning(f"Số lượng threads CPU vượt quá số lõi CPU có sẵn. Đã giảm xuống {cpu_threads}.")

        # Phân bổ các lõi CPU dựa trên số lượng threads
        available_cores = list(range(cpu_threads))
        proc.cpu_affinity(available_cores)
        logger.info(f"Đã gán tiến trình PID {proc.pid} vào CPU cores: {available_cores}")
    except Exception as e:
        logger.error(f"Lỗi khi gán CPU cores cho tiến trình PID {proc.pid}: {e}")
        sys.exit(1)  # Thêm dòng này để kết thúc chương trình với mã lỗi 1


def assign_ram_cgroup(proc: psutil.Process, memory_limit_mb: int, logger: logging.Logger):
    """
    Thiết lập giới hạn RAM cho tiến trình trong cgroups RAM.

    Args:
        proc (psutil.Process): Đối tượng tiến trình.
        memory_limit_mb (int): Giới hạn RAM trong MB.
        logger (Logger): Đối tượng logger.
    """
    try:
        ram_group_path = "/sys/fs/cgroup/memory/mining_group"
        memory_limit_bytes = memory_limit_mb * 1024 * 1024

        with open(os.path.join(ram_group_path, "memory.limit_in_bytes"), 'w') as f:
            f.write(str(memory_limit_bytes) + "\n")
        logger.info(f"Đã thiết lập giới hạn RAM {memory_limit_mb}MB cho tiến trình PID: {proc.pid}")
    except Exception as e:
        logger.error(f"Lỗi khi thiết lập RAM cho tiến trình PID {proc.pid}: {e}")


def assign_disk_io_cgroup(proc: psutil.Process, disk_io_limit_mbps: float, logger: logging.Logger):
    """
    Thiết lập giới hạn Disk I/O cho tiến trình trong cgroups Disk I/O.

    Args:
        proc (psutil.Process): Đối tượng tiến trình.
        disk_io_limit_mbps (float): Giới hạn Disk I/O trong Mbps.
        logger (Logger): Đối tượng logger.
    """
    try:
        disk_io_group_path = "/sys/fs/cgroup/blkio/mining_group"
        major, minor = get_primary_disk_device(logger)

        read_limit_bytes = mbps_to_bytes(disk_io_limit_mbps)
        write_limit_bytes = mbps_to_bytes(disk_io_limit_mbps)

        # Cập nhật giới hạn đọc
        blkio_read_path = os.path.join(disk_io_group_path, "blkio.throttle.read_bps_device")
        with open(blkio_read_path, 'w') as f:
            f.write(f"{major}:{minor} {read_limit_bytes}\n")
        logger.info(f"Đã thiết lập Disk I/O read limit {disk_io_limit_mbps} Mbps cho tiến trình PID: {proc.pid}")

        # Cập nhật giới hạn ghi
        blkio_write_path = os.path.join(disk_io_group_path, "blkio.throttle.write_bps_device")
        with open(blkio_write_path, 'w') as f:
            f.write(f"{major}:{minor} {write_limit_bytes}\n")
        logger.info(f"Đã thiết lập Disk I/O write limit {disk_io_limit_mbps} Mbps cho tiến trình PID: {proc.pid}")
    except Exception as e:
        logger.error(f"Lỗi khi thiết lập Disk I/O cho tiến trình PID {proc.pid}: {e}")
        sys.exit(1)  # Thêm dòng này để kết thúc chương trình với mã lỗi 1

def assign_cpu_freq(proc: psutil.Process, cpu_freq_mhz: int, logger: logging.Logger):
    """
    Thiết lập tần số CPU cho tiến trình.

    Args:
        proc (psutil.Process): Đối tượng tiến trình.
        cpu_freq_mhz (int): Tần số CPU trong MHz.
        logger (Logger): Đối tượng logger.
    """
    try:
        # Phương pháp thiết lập tần số CPU có thể khác nhau tùy hệ thống.
        # Đây là một ví dụ sử dụng cpufrequtils
        subprocess.run(['cpufreq-set', '-p', '-u', f'{cpu_freq_mhz}MHz'], check=True)
        logger.info(f"Đã thiết lập tần số CPU thành {cpu_freq_mhz}MHz cho tiến trình PID: {proc.pid}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi thiết lập tần số CPU cho tiến trình PID {proc.pid}: {e}")
    except FileNotFoundError:
        logger.error("cpufreq-set không được cài đặt trên hệ thống.")
    except Exception as e:
        logger.error(f"Lỗi khi thiết lập tần số CPU cho tiến trình PID {proc.pid}: {e}")


def add_pid_to_cgroups(pid: int, logger: logging.Logger):
    """
    Thêm PID vào các cgroups CPU, RAM, và Disk I/O.

    Args:
        pid (int): PID của tiến trình.
        logger (Logger): Đối tượng logger.
    """
    try:
        cgroup_paths = [
            "/sys/fs/cgroup/cpu/mining_group",
            "/sys/fs/cgroup/memory/mining_group",
            "/sys/fs/cgroup/blkio/mining_group"
        ]

        for cgroup_path in cgroup_paths:
            with open(os.path.join(cgroup_path, "cgroup.procs"), 'a') as f:
                f.write(f"{pid}\n")
            logger.info(f"Đã thêm PID {pid} vào cgroup {cgroup_path}.")
    except PermissionError:
        logger.error("Không có quyền để thêm PID vào cgroups. Vui lòng chạy script với quyền root.")
    except Exception as e:
        logger.error(f"Lỗi khi thêm PID {pid} vào cgroups: {e}")


def get_primary_disk_device(logger: logging.Logger) -> Tuple[int, int]:
    """
    Tự động phát hiện thiết bị lưu trữ chính (primary disk device).

    Returns:
        tuple: (major, minor) của thiết bị lưu trữ chính.

    Raises:
        SystemExit: Nếu không thể xác định thiết bị lưu trữ chính.
    """
    try:
        # Lấy MAJ:MIN của thiết bị lưu trữ chính
        output = subprocess.check_output(['lsblk', '-ndo', 'MAJ:MIN', '/']).decode().strip()
        major, minor = map(int, output.split(':'))
        logger.debug(f"Thiết bị lưu trữ chính: MAJ={major}, MIN={minor}")
        return major, minor
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi chạy lệnh lsblk để phát hiện thiết bị lưu trữ chính: {e}")
        sys.exit(1)
    except ValueError:
        logger.error("Dữ liệu trả về từ lsblk không hợp lệ.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Lỗi khi phát hiện thiết bị lưu trữ chính: {e}")
        sys.exit(1)


def mbps_to_bytes(mbps: float) -> int:
    """
    Chuyển đổi Mbps sang Bytes/s.

    Args:
        mbps (float): Số Mbps.

    Returns:
        int: Số Bytes/s tương ứng.
    """
    return int(mbps * 125_000)  # 1 Mbps = 125,000 Bytes/s
