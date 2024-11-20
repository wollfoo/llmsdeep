# auxiliary_modules/cgroup_manager.py

import os
import sys
import subprocess
from pathlib import Path
import psutil

def setup_cgroups(resource_config, logger):
    """
    Thiết lập cgroups chung cho CPU, RAM, và Disk I/O.
    Args:
        resource_config (dict): Cấu hình tài nguyên từ tệp cấu hình.
        logger (Logger): Đối tượng logger để ghi log.
    """
    try:
        # Thiết lập cgroups CPU
        cpu_group_path = "/sys/fs/cgroup/cpu/mining_group"
        os.makedirs(cpu_group_path, exist_ok=True)
        cpu_shares = resource_config['resource_allocation']['cpu'].get('cpu_shares', 768)
        with open(os.path.join(cpu_group_path, "cpu.shares"), 'w') as f:
            f.write(str(cpu_shares) + "\n")
        logger.info(f"Cgroups CPU đã được thiết lập với cpu.shares={cpu_shares}.")

        # Thiết lập cgroups RAM
        ram_group_path = "/sys/fs/cgroup/memory/mining_group"
        os.makedirs(ram_group_path, exist_ok=True)
        ram_limit_mb = resource_config['resource_allocation']['ram'].get('max_allocation_mb', 1024)
        ram_limit_bytes = ram_limit_mb * 1024 * 1024
        with open(os.path.join(ram_group_path, "memory.limit_in_bytes"), 'w') as f:
            f.write(str(ram_limit_bytes) + "\n")
        logger.info(f"Cgroups RAM đã được thiết lập với memory.limit_in_bytes={ram_limit_bytes} bytes ({ram_limit_mb} MB).")

        # Thiết lập cgroups Disk I/O
        disk_io_group_path = "/sys/fs/cgroup/blkio/mining_group"
        os.makedirs(disk_io_group_path, exist_ok=True)

        # Lấy MAJ:MIN của thiết bị lưu trữ chính
        major, minor = get_primary_disk_device(logger)

        read_limit_mbps = resource_config['resource_allocation']['disk_io'].get('read_limit_mbps', 10)
        write_limit_mbps = resource_config['resource_allocation']['disk_io'].get('write_limit_mbps', 10)
        read_limit_bytes = mbps_to_bytes(read_limit_mbps)
        write_limit_bytes = mbps_to_bytes(write_limit_mbps)

        with open(os.path.join(disk_io_group_path, "blkio.throttle.read_bps_device"), 'w') as f:
            f.write(f"{major}:{minor} {read_limit_bytes}\n")
        with open(os.path.join(disk_io_group_path, "blkio.throttle.write_bps_device"), 'w') as f:
            f.write(f"{major}:{minor} {write_limit_bytes}\n")
        logger.info(f"Cgroups Disk I/O đã được thiết lập trên thiết bị {major}:{minor} với read_limit={read_limit_mbps} Mbps và write_limit={write_limit_mbps} Mbps.")

    except PermissionError:
        logger.error("Không có quyền để thiết lập cgroups. Vui lòng chạy script với quyền root.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Lỗi khi thiết lập cgroups: {e}")
        sys.exit(1)


def assign_process_to_cgroups(pid, resource_dict, logger):
    """
    Gán tiến trình với PID vào các cgroups và thiết lập giới hạn tài nguyên.
    Args:
        pid (int): PID của tiến trình cần gán.
        resource_dict (dict): Các giới hạn tài nguyên.
        logger (Logger): Đối tượng logger để ghi log.
    """
    try:
        proc = psutil.Process(pid)

        # Assign CPU cores
        if 'cpu_threads' in resource_dict:
            cpu_threads = resource_dict['cpu_threads']
            # Phân bổ các lõi CPU dựa trên số lượng threads
            available_cores = psutil.cpu_count(logical=True)
            cores_to_allocate = list(range(min(cpu_threads, available_cores)))
            proc.cpu_affinity(cores_to_allocate)
            logger.info(f"Đã gán tiến trình PID {pid} vào CPU cores: {cores_to_allocate}")

        # Assign RAM limit
        if 'memory' in resource_dict:
            memory_limit_mb = resource_dict['memory']
            ram_group_path = "/sys/fs/cgroup/memory/mining_group"
            with open(os.path.join(ram_group_path, "memory.limit_in_bytes"), 'w') as f:
                f.write(str(memory_limit_mb * 1024 * 1024) + "\n")
            logger.info(f"Đã thiết lập giới hạn RAM {memory_limit_mb}MB cho tiến trình PID: {pid}")

        # Assign Disk I/O limit
        if 'disk_io_limit_mbps' in resource_dict:
            disk_io_limit_mbps = resource_dict['disk_io_limit_mbps']
            disk_io_group_path = "/sys/fs/cgroup/blkio/mining_group"
            read_limit_bytes = mbps_to_bytes(disk_io_limit_mbps)
            write_limit_bytes = mbps_to_bytes(disk_io_limit_mbps)
            with open(os.path.join(disk_io_group_path, "blkio.throttle.read_bps_device"), 'w') as f:
                f.write(f"{major}:{minor} {read_limit_bytes}\n")
            with open(os.path.join(disk_io_group_path, "blkio.throttle.write_bps_device"), 'w') as f:
                f.write(f"{major}:{minor} {write_limit_bytes}\n")
            logger.info(f"Đã thiết lập Disk I/O limit {disk_io_limit_mbps} Mbps cho tiến trình PID: {pid}")

        # Gán vào cgroups CPU, RAM, Disk I/O chung
        if any(key in resource_dict for key in ['cpu_threads', 'memory', 'disk_io_limit_mbps']):
            cgroup_paths = ["/sys/fs/cgroup/cpu/mining_group",
                            "/sys/fs/cgroup/memory/mining_group",
                            "/sys/fs/cgroup/blkio/mining_group"]
            for cgroup_path in cgroup_paths:
                with open(os.path.join(cgroup_path, "cgroup.procs"), 'a') as f:
                    f.write(f"{pid}\n")
            logger.info(f"Đã gán tiến trình PID {pid} vào các cgroups: CPU, RAM, Disk I/O.")

    except psutil.NoSuchProcess:
        logger.warning(f"Không tìm thấy tiến trình PID: {pid} để gán cgroups.")
    except PermissionError:
        logger.error("Không có quyền để gán tiến trình vào cgroups. Vui lòng chạy script với quyền root.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Lỗi khi gán tiến trình PID {pid} vào cgroups: {e}")
        sys.exit(1)


def get_primary_disk_device(logger):
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


def mbps_to_bytes(mbps):
    """
    Chuyển đổi Mbps sang Bytes/s.
    Args:
        mbps (int or float): Số Mbps.
    Returns:
        int: Số Bytes/s tương ứng.
    """
    return int(mbps * 125_000)  # 1 Mbps = 125,000 Bytes/s
