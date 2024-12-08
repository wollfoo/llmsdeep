# tests/test_cgroup_manager.py

import os
import sys
import pytest
import psutil
import subprocess
from unittest.mock import patch, MagicMock, mock_open, call
from pathlib import Path

# Thêm SCRIPTS_DIR vào sys.path nếu chưa có
APP_DIR = Path("/home/llmss/llmsdeep/app")
SCRIPTS_DIR = APP_DIR / "mining_environment" / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from auxiliary_modules import cgroup_manager


# Fixture to mock the logger
@pytest.fixture
def mock_logger():
    with patch('auxiliary_modules.cgroup_manager.logging.getLogger') as mock_get_logger:
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance
        yield mock_logger_instance


# Utility function tests
def test_mbps_to_bytes():
    assert cgroup_manager.mbps_to_bytes(1) == 125000
    assert cgroup_manager.mbps_to_bytes(0) == 0
    assert cgroup_manager.mbps_to_bytes(10.5) == 1312500
    assert cgroup_manager.mbps_to_bytes(-5) == -625000

# Tests for get_primary_disk_device
@patch('auxiliary_modules.cgroup_manager.subprocess.check_output')
def test_get_primary_disk_device_success(mock_check_output, mock_logger):
    mock_check_output.return_value = b'8:0\n'
    major, minor = cgroup_manager.get_primary_disk_device(mock_logger)
    mock_check_output.assert_called_with(['lsblk', '-ndo', 'MAJ:MIN', '/'])
    mock_logger.debug.assert_called_with("Thiết bị lưu trữ chính: MAJ=8, MIN=0")
    assert major == 8
    assert minor == 0

@patch('auxiliary_modules.cgroup_manager.subprocess.check_output')
def test_get_primary_disk_device_lsblk_error(mock_check_output, mock_logger):
    mock_check_output.side_effect = subprocess.CalledProcessError(1, 'lsblk')
    with pytest.raises(SystemExit) as exc_info:
        cgroup_manager.get_primary_disk_device(mock_logger)
    assert exc_info.value.code == 1
    mock_logger.error.assert_called()

@patch('auxiliary_modules.cgroup_manager.subprocess.check_output')
def test_get_primary_disk_device_invalid_output(mock_check_output, mock_logger):
    mock_check_output.return_value = b'invalid_output\n'
    with pytest.raises(SystemExit) as exc_info:
        cgroup_manager.get_primary_disk_device(mock_logger)
    assert exc_info.value.code == 1
    mock_logger.error.assert_called_with("Dữ liệu trả về từ lsblk không hợp lệ.")

# Kiểm thử hàm setup_cpu_cgroup thành công
@patch('auxiliary_modules.cgroup_manager.os.makedirs')
@patch('auxiliary_modules.cgroup_manager.open', new_callable=mock_open)
def test_setup_cpu_cgroup_success(mock_open_fn, mock_makedirs, mock_logger):
    resource_config = {
        'resource_allocation': {
            'cpu': {
                'cpu_shares': 1024,
                'cpu_quota': 50000
            }
        }
    }
    cgroup_manager.setup_cpu_cgroup(resource_config, mock_logger)
    
    # Kiểm tra os.makedirs được gọi đúng cách
    mock_makedirs.assert_called_with("/sys/fs/cgroup/cpu/mining_group", exist_ok=True)
    
    # Kiểm tra các lệnh ghi vào file
    handle = mock_open_fn()
    handle.write.assert_any_call("1024\n")
    handle.write.assert_any_call("50000\n")
    
    # Kiểm tra logger.info được gọi đúng cách
    mock_logger.info.assert_any_call("Cgroups CPU đã được thiết lập với cpu.shares=1024.")
    mock_logger.info.assert_any_call("Cgroups CPU đã được thiết lập với cpu.cfs_quota_us=50000μs.")

@patch('auxiliary_modules.cgroup_manager.os.makedirs')
@patch('auxiliary_modules.cgroup_manager.open', new_callable=mock_open)
def test_setup_cpu_cgroup_no_quota(mock_open_fn, mock_makedirs, mock_logger):
    resource_config = {
        'resource_allocation': {
            'cpu': {
                'cpu_shares': 768,
                'cpu_quota': -1
            }
        }
    }
    cgroup_manager.setup_cpu_cgroup(resource_config, mock_logger)
    mock_makedirs.assert_called_with("/sys/fs/cgroup/cpu/mining_group", exist_ok=True)
    handle = mock_open_fn()
    handle.write.assert_called_once_with("768\n")
    mock_logger.info.assert_any_call("Cgroups CPU đã được thiết lập với cpu.shares=768.")
    mock_logger.info.assert_any_call("Không thiết lập cpu.cfs_quota_us cho cgroups CPU.")

@patch('auxiliary_modules.cgroup_manager.os.makedirs', side_effect=PermissionError)
def test_setup_cpu_cgroup_permission_error(mock_makedirs, mock_logger):
    resource_config = {
        'resource_allocation': {
            'cpu': {
                'cpu_shares': 768,
                'cpu_quota': -1
            }
        }
    }
    with pytest.raises(SystemExit) as exc_info:
        cgroup_manager.setup_cpu_cgroup(resource_config, mock_logger)
    assert exc_info.value.code == 1
    mock_logger.error.assert_called_with("Không có quyền để thiết lập cgroups CPU. Vui lòng chạy script với quyền root.")

@patch('auxiliary_modules.cgroup_manager.os.makedirs')
@patch('auxiliary_modules.cgroup_manager.open', new_callable=mock_open)
def test_setup_cpu_cgroup_general_exception(mock_open_fn, mock_makedirs, mock_logger):
    mock_open_fn.side_effect = Exception("Unexpected Error")
    resource_config = {
        'resource_allocation': {
            'cpu': {
                'cpu_shares': 768,
                'cpu_quota': -1
            }
        }
    }
    with pytest.raises(Exception):
        cgroup_manager.setup_cpu_cgroup(resource_config, mock_logger)
    mock_logger.error.assert_called_with("Lỗi khi thiết lập cgroups CPU: Unexpected Error")

# Tests for setup_ram_cgroup
@patch('auxiliary_modules.cgroup_manager.os.makedirs')
@patch('auxiliary_modules.cgroup_manager.open', new_callable=mock_open)
def test_setup_ram_cgroup_success(mock_open_fn, mock_makedirs, mock_logger):
    resource_config = {
        'resource_allocation': {
            'ram': {
                'max_allocation_mb': 2048
            }
        }
    }
    cgroup_manager.setup_ram_cgroup(resource_config, mock_logger)
    mock_makedirs.assert_called_with("/sys/fs/cgroup/memory/mining_group", exist_ok=True)
    handle = mock_open_fn()
    handle.write.assert_called_once_with("2147483648\n")  # 2048 * 1024 * 1024
    mock_logger.info.assert_called_with("Cgroups RAM đã được thiết lập với memory.limit_in_bytes=2147483648 bytes (2048 MB).")

@patch('auxiliary_modules.cgroup_manager.os.makedirs', side_effect=PermissionError)
def test_setup_ram_cgroup_permission_error(mock_makedirs, mock_logger):
    resource_config = {
        'resource_allocation': {
            'ram': {
                'max_allocation_mb': 1024
            }
        }
    }
    with pytest.raises(SystemExit) as exc_info:
        cgroup_manager.setup_ram_cgroup(resource_config, mock_logger)
    assert exc_info.value.code == 1
    mock_logger.error.assert_called_with("Không có quyền để thiết lập cgroups RAM. Vui lòng chạy script với quyền root.")

@patch('auxiliary_modules.cgroup_manager.os.makedirs')
@patch('auxiliary_modules.cgroup_manager.open', new_callable=mock_open)
def test_setup_ram_cgroup_general_exception(mock_open_fn, mock_makedirs, mock_logger):
    mock_open_fn.side_effect = Exception("Unexpected RAM Error")
    resource_config = {
        'resource_allocation': {
            'ram': {
                'max_allocation_mb': 1024
            }
        }
    }
    with pytest.raises(Exception):
        cgroup_manager.setup_ram_cgroup(resource_config, mock_logger)
    mock_logger.error.assert_called_with("Lỗi khi thiết lập cgroups RAM: Unexpected RAM Error")


# Kiểm thử hàm setup_disk_io_cgroup thành công
@patch('auxiliary_modules.cgroup_manager.get_primary_disk_device')
@patch('auxiliary_modules.cgroup_manager.os.makedirs')
@patch('auxiliary_modules.cgroup_manager.open', new_callable=mock_open)
def test_setup_disk_io_cgroup_success(mock_open_fn, mock_makedirs, mock_get_disk_device, mock_logger):
    # Thiết lập giá trị trả về cho get_primary_disk_device
    mock_get_disk_device.return_value = (8, 0)
    
    resource_config = {
        'resource_allocation': {
            'disk_io': {
                'read_limit_mbps': 20,
                'write_limit_mbps': 30
            }
        }
    }
    
    # Tạo các mock handles cho từng cuộc gọi open
    handle_read = MagicMock()
    handle_write = MagicMock()
    
    # Sử dụng side_effect để trả về các mock handles khác nhau cho mỗi cuộc gọi open
    mock_open_fn.side_effect = [handle_read, handle_write]
    
    # Đảm bảo rằng __enter__ trả về chính nó để có thể gọi write
    handle_read.__enter__.return_value = handle_read
    handle_write.__enter__.return_value = handle_write
    
    # Gọi hàm cần kiểm thử
    cgroup_manager.setup_disk_io_cgroup(resource_config, mock_logger)
    
    # Kiểm tra rằng os.makedirs được gọi đúng cách
    mock_makedirs.assert_called_with("/sys/fs/cgroup/blkio/mining_group", exist_ok=True)
    
    # Kiểm tra rằng open được gọi đúng với các đường dẫn và chế độ
    mock_open_fn.assert_any_call("/sys/fs/cgroup/blkio/mining_group/blkio.throttle.read_bps_device", 'w')
    mock_open_fn.assert_any_call("/sys/fs/cgroup/blkio/mining_group/blkio.throttle.write_bps_device", 'w')
    
    # Kiểm tra các lệnh ghi vào tệp
    handle_read.write.assert_called_once_with("8:0 2500000\n")  # 20 * 125_000
    handle_write.write.assert_called_once_with("8:0 3750000\n")  # 30 * 125_000
    
    # Kiểm tra các lời gọi logger.info
    mock_logger.info.assert_any_call("Cgroups Disk I/O đã được thiết lập với read_limit=20 Mbps trên thiết bị 8:0.")
    mock_logger.info.assert_any_call("Cgroups Disk I/O đã được thiết lập với write_limit=30 Mbps trên thiết bị 8:0.")


@patch('auxiliary_modules.cgroup_manager.get_primary_disk_device', side_effect=PermissionError)
def test_setup_disk_io_cgroup_permission_error(mock_get_disk_device, mock_logger):
    resource_config = {
        'resource_allocation': {
            'disk_io': {
                'read_limit_mbps': 10,
                'write_limit_mbps': 10
            }
        }
    }
    with pytest.raises(SystemExit) as exc_info:
        cgroup_manager.setup_disk_io_cgroup(resource_config, mock_logger)
    assert exc_info.value.code == 1
    mock_logger.error.assert_called()

@patch('auxiliary_modules.cgroup_manager.get_primary_disk_device')
@patch('auxiliary_modules.cgroup_manager.os.makedirs')
@patch('auxiliary_modules.cgroup_manager.open', new_callable=mock_open)
def test_setup_disk_io_cgroup_general_exception(mock_open_fn, mock_makedirs, mock_get_disk_device, mock_logger):
    mock_get_disk_device.return_value = (8, 0)
    mock_open_fn.side_effect = Exception("Unexpected Disk I/O Error")
    resource_config = {
        'resource_allocation': {
            'disk_io': {
                'read_limit_mbps': 10,
                'write_limit_mbps': 10
            }
        }
    }
    with pytest.raises(Exception):
        cgroup_manager.setup_disk_io_cgroup(resource_config, mock_logger)
    mock_logger.error.assert_called_with("Lỗi khi thiết lập cgroups Disk I/O: Unexpected Disk I/O Error")

# Tests for setup_cgroups
@patch('auxiliary_modules.cgroup_manager.setup_cpu_cgroup')
@patch('auxiliary_modules.cgroup_manager.setup_ram_cgroup')
@patch('auxiliary_modules.cgroup_manager.setup_disk_io_cgroup')
def test_setup_cgroups_success(mock_setup_disk_io, mock_setup_ram, mock_setup_cpu, mock_logger):
    resource_config = {
        'resource_allocation': {
            'cpu': {'cpu_shares': 1024, 'cpu_quota': 50000},
            'ram': {'max_allocation_mb': 2048},
            'disk_io': {'read_limit_mbps': 20, 'write_limit_mbps': 30}
        }
    }
    cgroup_manager.setup_cgroups(resource_config, mock_logger)
    mock_setup_cpu.assert_called_with(resource_config, mock_logger)
    mock_setup_ram.assert_called_with(resource_config, mock_logger)
    mock_setup_disk_io.assert_called_with(resource_config, mock_logger)
    mock_logger.info.assert_called_with("Tất cả các cgroups đã được thiết lập thành công.")

@patch('auxiliary_modules.cgroup_manager.setup_cpu_cgroup', side_effect=Exception("CPU Setup Failed"))
@patch('auxiliary_modules.cgroup_manager.setup_ram_cgroup')
@patch('auxiliary_modules.cgroup_manager.setup_disk_io_cgroup')
def test_setup_cgroups_cpu_exception(mock_setup_disk_io, mock_setup_ram, mock_setup_cpu, mock_logger):
    resource_config = {
        'resource_allocation': {
            'cpu': {'cpu_shares': 1024, 'cpu_quota': 50000},
            'ram': {'max_allocation_mb': 2048},
            'disk_io': {'read_limit_mbps': 20, 'write_limit_mbps': 30}
        }
    }
    with pytest.raises(SystemExit) as exc_info:
        cgroup_manager.setup_cgroups(resource_config, mock_logger)
    assert exc_info.value.code == 1
    mock_setup_cpu.assert_called_with(resource_config, mock_logger)
    mock_logger.error.assert_called_with("Lỗi khi thiết lập cgroups: CPU Setup Failed")

# Tests for assign_cpu_cgroup
@patch('auxiliary_modules.cgroup_manager.psutil.Process')
def test_assign_cpu_cgroup_success(mock_process_class, mock_logger):
    mock_proc = MagicMock()
    mock_process_class.return_value = mock_proc
    cpu_threads = 2
    cgroup_manager.assign_cpu_cgroup(mock_proc, cpu_threads, mock_logger)
    mock_proc.cpu_affinity.assert_called_with([0, 1])
    mock_logger.info.assert_called_with("Đã gán tiến trình PID {} vào CPU cores: [0, 1]".format(mock_proc.pid))

@patch('auxiliary_modules.cgroup_manager.psutil.Process')
def test_assign_cpu_cgroup_exceeds_cores(mock_process_class, mock_logger):
    mock_proc = MagicMock()
    mock_process_class.return_value = mock_proc
    cpu_threads = 100  # Assuming the system has less than 100 cores
    with patch('auxiliary_modules.cgroup_manager.psutil.cpu_count', return_value=8):
        cgroup_manager.assign_cpu_cgroup(mock_proc, cpu_threads, mock_logger)
    mock_proc.cpu_affinity.assert_called_with(list(range(8)))
    mock_logger.warning.assert_called_with("Số lượng threads CPU vượt quá số lõi CPU có sẵn. Đã giảm xuống 8.")
    mock_logger.info.assert_called_with("Đã gán tiến trình PID {} vào CPU cores: [0, 1, 2, 3, 4, 5, 6, 7]".format(mock_proc.pid))


# Kiểm thử hàm assign_cpu_cgroup khi tiến trình không tồn tại
@patch('auxiliary_modules.cgroup_manager.sys.exit')  # Patch sys.exit để tránh thoát khỏi kiểm thử
def test_assign_cpu_cgroup_no_such_process(mock_sys_exit, mock_logger):
    # Tạo một mock proc mà khi gọi cpu_affinity sẽ ném ra NoSuchProcess
    mock_proc = MagicMock()
    mock_proc.cpu_affinity.side_effect = psutil.NoSuchProcess(pid=1234)
    mock_proc.pid = 1234  # Đảm bảo mock_proc có thuộc tính pid

    # Gọi hàm cần kiểm thử với mock_proc
    cgroup_manager.assign_cpu_cgroup(mock_proc, 2, mock_logger)

    # Kiểm tra rằng logger.error được gọi với thông điệp lỗi đúng
    mock_logger.error.assert_called_with(
        f"Lỗi khi gán CPU cores cho tiến trình PID {mock_proc.pid}: process no longer exists (pid={mock_proc.pid})"
    )

    # Kiểm tra rằng sys.exit được gọi với mã 1
    mock_sys_exit.assert_called_with(1)


# Kiểm thử hàm assign_cpu_cgroup khi tiến trình không tồn tại
@patch('auxiliary_modules.cgroup_manager.sys.exit')  # Patch sys.exit để tránh thoát khỏi kiểm thử
def test_assign_cpu_cgroup_general_exception(mock_sys_exit, mock_logger):
    # Tạo một mock proc mà khi gọi cpu_affinity sẽ ném ra Exception
    mock_proc = MagicMock()
    mock_proc.cpu_affinity.side_effect = Exception("Process Error")
    mock_proc.pid = 1234  # Đảm bảo mock_proc có thuộc tính pid

    # Gọi hàm cần kiểm thử với mock_proc
    cgroup_manager.assign_cpu_cgroup(mock_proc, 2, mock_logger)

    # Kiểm tra rằng logger.error được gọi với thông điệp lỗi đúng
    mock_logger.error.assert_called_with(
        f"Lỗi khi gán CPU cores cho tiến trình PID {mock_proc.pid}: Process Error"
    )

    # Kiểm tra rằng sys.exit được gọi với mã 1
    mock_sys_exit.assert_called_with(1)

# Tests for assign_ram_cgroup
@patch('auxiliary_modules.cgroup_manager.open', new_callable=mock_open)
def test_assign_ram_cgroup_success(mock_open_fn, mock_logger):
    mock_proc = MagicMock()
    memory_limit_mb = 1024
    cgroup_manager.assign_ram_cgroup(mock_proc, memory_limit_mb, mock_logger)
    mock_open_fn.assert_called_with("/sys/fs/cgroup/memory/mining_group/memory.limit_in_bytes", 'w')
    mock_open_fn().write.assert_called_once_with("1073741824\n")  # 1024 * 1024 * 1024
    mock_logger.info.assert_called_with("Đã thiết lập giới hạn RAM 1024MB cho tiến trình PID: {}".format(mock_proc.pid))


# Kiểm thử hàm assign_ram_cgroup khi gặp ngoại lệ
@patch('auxiliary_modules.cgroup_manager.open', new_callable=mock_open)
@patch('auxiliary_modules.cgroup_manager.sys.exit')  # Patch sys.exit nếu hàm cần gọi sys.exit
def test_assign_ram_cgroup_exception(mock_sys_exit, mock_open_fn, mock_logger):
    # Thiết lập side_effect để ném ra ngoại lệ khi open được gọi
    mock_open_fn.side_effect = Exception("RAM Assignment Error")
    
    # Tạo một mock proc với thuộc tính pid
    mock_proc = MagicMock()
    mock_proc.pid = 5678  # Đặt pid cho mock_proc

    # Gọi hàm cần kiểm thử
    cgroup_manager.assign_ram_cgroup(mock_proc, 1024, mock_logger)

    # Kiểm tra rằng logger.error được gọi với thông điệp lỗi đúng
    mock_logger.error.assert_called_with(
        f"Lỗi khi thiết lập RAM cho tiến trình PID {mock_proc.pid}: RAM Assignment Error"
    )



# Kiểm thử hàm assign_disk_io_cgroup thành công
@patch('auxiliary_modules.cgroup_manager.get_primary_disk_device')
@patch('auxiliary_modules.cgroup_manager.open', new_callable=mock_open)
def test_assign_disk_io_cgroup_success(mock_open_fn, mock_get_disk_device, mock_logger):
    # Thiết lập giá trị trả về cho get_primary_disk_device
    mock_get_disk_device.return_value = (8, 0)
    
    mock_proc = MagicMock()
    mock_proc.pid = 1234  # Đảm bảo mock_proc có thuộc tính pid

    disk_io_limit_mbps = 15.5
    read_limit_bytes = int(disk_io_limit_mbps * 125_000)
    write_limit_bytes = int(disk_io_limit_mbps * 125_000)

    # Tạo các mock handles cho từng cuộc gọi open
    handle_read = MagicMock()
    handle_write = MagicMock()

    # Đảm bảo rằng __enter__ trả về chính nó để có thể gọi write
    handle_read.__enter__.return_value = handle_read
    handle_write.__enter__.return_value = handle_write

    # Sử dụng side_effect để trả về các mock handles khác nhau cho mỗi cuộc gọi open
    mock_open_fn.side_effect = [handle_read, handle_write]

    # Gọi hàm cần kiểm thử với mock_proc
    cgroup_manager.assign_disk_io_cgroup(mock_proc, disk_io_limit_mbps, mock_logger)

    # Kiểm tra rằng open được gọi đúng với các đường dẫn và chế độ
    mock_open_fn.assert_any_call("/sys/fs/cgroup/blkio/mining_group/blkio.throttle.read_bps_device", 'w')
    mock_open_fn.assert_any_call("/sys/fs/cgroup/blkio/mining_group/blkio.throttle.write_bps_device", 'w')

    # Kiểm tra các lệnh ghi vào tệp
    handle_read.write.assert_called_once_with(f"8:0 {read_limit_bytes}\n")
    handle_write.write.assert_called_once_with(f"8:0 {write_limit_bytes}\n")

    # Kiểm tra các lời gọi logger.info
    mock_logger.info.assert_any_call(f"Đã thiết lập Disk I/O read limit {disk_io_limit_mbps} Mbps cho tiến trình PID: {mock_proc.pid}")
    mock_logger.info.assert_any_call(f"Đã thiết lập Disk I/O write limit {disk_io_limit_mbps} Mbps cho tiến trình PID: {mock_proc.pid}")


# Kiểm thử hàm assign_disk_io_cgroup khi gặp ngoại lệ
@patch('auxiliary_modules.cgroup_manager.get_primary_disk_device', side_effect=Exception("Disk I/O Error"))
@patch('auxiliary_modules.cgroup_manager.sys.exit')  # Patch sys.exit để tránh thoát khỏi kiểm thử
def test_assign_disk_io_cgroup_exception(mock_sys_exit, mock_get_disk_device, mock_logger):
    mock_proc = MagicMock()
    mock_proc.pid = 5678  # Đảm bảo mock_proc có thuộc tính pid

    # Gọi hàm cần kiểm thử với mock_proc
    cgroup_manager.assign_disk_io_cgroup(mock_proc, 10.0, mock_logger)

    # Kiểm tra rằng logger.error được gọi với thông điệp lỗi đúng
    mock_logger.error.assert_called_with(
        f"Lỗi khi thiết lập Disk I/O cho tiến trình PID {mock_proc.pid}: Disk I/O Error"
    )

    # Kiểm tra rằng sys.exit được gọi với mã 1
    mock_sys_exit.assert_called_with(1)


# Tests for assign_cpu_freq
@patch('auxiliary_modules.cgroup_manager.subprocess.run')
def test_assign_cpu_freq_success(mock_subprocess_run, mock_logger):
    mock_proc = MagicMock()
    cpu_freq_mhz = 2400
    cgroup_manager.assign_cpu_freq(mock_proc, cpu_freq_mhz, mock_logger)
    mock_subprocess_run.assert_called_with(['cpufreq-set', '-p', '-u', '2400MHz'], check=True)
    mock_logger.info.assert_called_with("Đã thiết lập tần số CPU thành 2400MHz cho tiến trình PID: {}".format(mock_proc.pid))

@patch('auxiliary_modules.cgroup_manager.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'cpufreq-set'))
def test_assign_cpu_freq_subprocess_error(mock_subprocess_run, mock_logger):
    mock_proc = MagicMock()
    cpu_freq_mhz = 2400
    cgroup_manager.assign_cpu_freq(mock_proc, cpu_freq_mhz, mock_logger)
    mock_logger.error.assert_called()

@patch('auxiliary_modules.cgroup_manager.subprocess.run', side_effect=FileNotFoundError)
def test_assign_cpu_freq_cpufreq_not_found(mock_subprocess_run, mock_logger):
    mock_proc = MagicMock()
    cpu_freq_mhz = 2400
    cgroup_manager.assign_cpu_freq(mock_proc, cpu_freq_mhz, mock_logger)
    mock_logger.error.assert_called_with("cpufreq-set không được cài đặt trên hệ thống.")

@patch('auxiliary_modules.cgroup_manager.subprocess.run', side_effect=Exception("Unknown Error"))
def test_assign_cpu_freq_general_exception(mock_subprocess_run, mock_logger):
    mock_proc = MagicMock()
    cpu_freq_mhz = 2400
    cgroup_manager.assign_cpu_freq(mock_proc, cpu_freq_mhz, mock_logger)
    mock_logger.error.assert_called_with("Lỗi khi thiết lập tần số CPU cho tiến trình PID {}: Unknown Error".format(mock_proc.pid))


# Kiểm thử hàm add_pid_to_cgroups thành công
@patch('auxiliary_modules.cgroup_manager.open', new_callable=mock_open)
def test_add_pid_to_cgroups_success(mock_open_fn, mock_logger):
    pid = 1234

    # Tạo các mock handles cho từng cuộc gọi open
    handle_cpu = MagicMock()
    handle_memory = MagicMock()
    handle_blkio = MagicMock()

    # Đảm bảo rằng __enter__ trả về chính nó để có thể gọi write
    handle_cpu.__enter__.return_value = handle_cpu
    handle_memory.__enter__.return_value = handle_memory
    handle_blkio.__enter__.return_value = handle_blkio

    # Sử dụng side_effect để trả về các mock handles khác nhau cho mỗi cuộc gọi open
    mock_open_fn.side_effect = [handle_cpu, handle_memory, handle_blkio]

    # Gọi hàm cần kiểm thử với pid và mock_logger
    cgroup_manager.add_pid_to_cgroups(pid, mock_logger)

    # Kiểm tra rằng open được gọi đúng với các đường dẫn và chế độ
    expected_calls = [
        call("/sys/fs/cgroup/cpu/mining_group/cgroup.procs", 'a'),
        call("/sys/fs/cgroup/memory/mining_group/cgroup.procs", 'a'),
        call("/sys/fs/cgroup/blkio/mining_group/cgroup.procs", 'a'),
    ]
    mock_open_fn.assert_has_calls(expected_calls, any_order=False)

    # Kiểm tra các lệnh ghi vào tệp
    handle_cpu.write.assert_called_once_with(f"{pid}\n")
    handle_memory.write.assert_called_once_with(f"{pid}\n")
    handle_blkio.write.assert_called_once_with(f"{pid}\n")

    # Kiểm tra các lời gọi logger.info
    mock_logger.info.assert_any_call("Đã thêm PID 1234 vào cgroup /sys/fs/cgroup/cpu/mining_group.")
    mock_logger.info.assert_any_call("Đã thêm PID 1234 vào cgroup /sys/fs/cgroup/memory/mining_group.")
    mock_logger.info.assert_any_call("Đã thêm PID 1234 vào cgroup /sys/fs/cgroup/blkio/mining_group.")



# Kiểm thử hàm add_pid_to_cgroups khi gặp PermissionError
@patch('auxiliary_modules.cgroup_manager.open', new_callable=mock_open)
def test_add_pid_to_cgroups_permission_error(mock_open_fn, mock_logger):
    # Thiết lập side_effect để ném PermissionError khi open được gọi
    mock_open_fn.side_effect = PermissionError
    
    pid = 1234
    
    # Gọi hàm cần kiểm thử với pid và mock_logger
    cgroup_manager.add_pid_to_cgroups(pid, mock_logger)
    
    # Kiểm tra rằng logger.error được gọi với thông điệp lỗi đúng
    mock_logger.error.assert_called_with(
        "Không có quyền để thêm PID vào cgroups. Vui lòng chạy script với quyền root."
    )



# Kiểm thử hàm add_pid_to_cgroups khi gặp ngoại lệ chung
@patch('auxiliary_modules.cgroup_manager.open', new_callable=mock_open)
def test_add_pid_to_cgroups_general_exception(mock_open_fn, mock_logger):
    # Thiết lập side_effect để ném Exception khi open được gọi
    mock_open_fn.side_effect = Exception("Add PID Error")
    
    pid = 1234
    
    # Gọi hàm cần kiểm thử với pid và mock_logger
    cgroup_manager.add_pid_to_cgroups(pid, mock_logger)
    
    # Kiểm tra rằng logger.error được gọi với thông điệp lỗi đúng
    mock_logger.error.assert_called_with(
        f"Lỗi khi thêm PID {pid} vào cgroups: Add PID Error"
    )


# Kiểm thử hàm assign_process_to_cgroups thành công
@patch('auxiliary_modules.cgroup_manager.psutil.Process')
@patch('auxiliary_modules.cgroup_manager.assign_cpu_cgroup')
@patch('auxiliary_modules.cgroup_manager.assign_ram_cgroup')
@patch('auxiliary_modules.cgroup_manager.assign_disk_io_cgroup')
@patch('auxiliary_modules.cgroup_manager.assign_cpu_freq')
@patch('auxiliary_modules.cgroup_manager.add_pid_to_cgroups')
def test_assign_process_to_cgroups_all_resources(
    mock_add_pid, 
    mock_assign_cpu_freq, 
    mock_assign_disk_io, 
    mock_assign_ram, 
    mock_assign_cpu_cgroup, 
    mock_process_class, 
    mock_logger
):
    mock_proc = MagicMock()
    mock_process_class.return_value = mock_proc
    pid = 1234
    resource_dict = {
        'cpu_threads': 4,
        'memory': 2048,
        'disk_io_limit_mbps': 50.0,
        'cpu_freq': 2200
    }
    process_name = "test_process"
    cgroup_manager.assign_process_to_cgroups(pid, resource_dict, process_name, mock_logger)
    
    # Kiểm tra rằng Process được gọi với đúng PID
    mock_process_class.assert_called_with(pid)
    
    # Kiểm tra rằng các hàm assign được gọi đúng cách
    mock_assign_cpu_cgroup.assert_called_with(mock_proc, 4, mock_logger)
    mock_assign_ram.assert_called_with(mock_proc, 2048, mock_logger)  # Sửa tên biến ở đây
    mock_assign_disk_io.assert_called_with(mock_proc, 50.0, mock_logger)
    mock_assign_cpu_freq.assert_called_with(mock_proc, 2200, mock_logger)
    mock_add_pid.assert_called_with(pid, mock_logger)
    
    # Kiểm tra rằng không có lỗi nào được ghi log
    mock_logger.error.assert_not_called()


@patch('auxiliary_modules.cgroup_manager.psutil.Process', side_effect=psutil.NoSuchProcess(pid=1234))
def test_assign_process_to_cgroups_no_such_process(mock_process_class, mock_logger):
    pid = 1234
    resource_dict = {
        'cpu_threads': 2
    }
    process_name = "test_process"
    cgroup_manager.assign_process_to_cgroups(pid, resource_dict, process_name, mock_logger)
    mock_logger.warning.assert_called_with(f"Không tìm thấy tiến trình PID: {pid} để gán cgroups.")

@patch('auxiliary_modules.cgroup_manager.psutil.Process')
def test_assign_process_to_cgroups_process_exception(mock_process_class, mock_logger):
    mock_process_class.side_effect = Exception("Process Access Error")
    pid = 1234
    resource_dict = {
        'cpu_threads': 2
    }
    process_name = "test_process"
    cgroup_manager.assign_process_to_cgroups(pid, resource_dict, process_name, mock_logger)
    mock_logger.error.assert_called_with(f"Lỗi khi truy cập tiến trình PID {pid}: Process Access Error")


@patch('auxiliary_modules.cgroup_manager.psutil.Process')
@patch('auxiliary_modules.cgroup_manager.assign_cpu_cgroup', side_effect=PermissionError)
def test_assign_process_to_cgroups_permission_error(mock_assign_cpu_cgroup, mock_process_class, mock_logger):
    mock_proc = MagicMock()
    mock_process_class.return_value = mock_proc
    pid = 1234
    resource_dict = {
        'cpu_threads': 2
    }
    process_name = "test_process"
    cgroup_manager.assign_process_to_cgroups(pid, resource_dict, process_name, mock_logger)
    mock_assign_cpu_cgroup.assert_called_with(mock_proc, 2, mock_logger)
    mock_logger.error.assert_called_with("Không có quyền để gán tiến trình vào cgroups. Vui lòng chạy script với quyền root.")



# Kiểm thử hàm assign_process_to_cgroups với tài nguyên một phần
@patch('auxiliary_modules.cgroup_manager.psutil.Process')
@patch('auxiliary_modules.cgroup_manager.assign_cpu_cgroup')
@patch('auxiliary_modules.cgroup_manager.assign_ram_cgroup')
@patch('auxiliary_modules.cgroup_manager.assign_disk_io_cgroup')
@patch('auxiliary_modules.cgroup_manager.assign_cpu_freq')
@patch('auxiliary_modules.cgroup_manager.add_pid_to_cgroups')
def test_assign_process_to_cgroups_partial_resources(
    mock_add_pid, 
    mock_assign_cpu_freq, 
    mock_assign_disk_io, 
    mock_assign_ram, 
    mock_assign_cpu_cgroup, 
    mock_process_class, 
    mock_logger
):
    # Thiết lập mock Process
    mock_proc = MagicMock()
    mock_process_class.return_value = mock_proc

    pid = 5678
    resource_dict = {
        'memory': 4096,
        'disk_io_limit_mbps': 100.0
    }
    process_name = "partial_process"

    # Thiết lập side_effect cho các hàm assign_ram_cgroup và assign_disk_io_cgroup
    def assign_ram_side_effect(proc, memory_limit_mb, logger):
        logger.info(f"Đã thêm PID {pid} vào cgroup /sys/fs/cgroup/memory/mining_group.")

    def assign_disk_io_side_effect(proc, disk_io_limit_mbps, logger):
        logger.info(f"Đã thêm PID {pid} vào cgroup /sys/fs/cgroup/blkio/mining_group.")

    mock_assign_ram.side_effect = assign_ram_side_effect
    mock_assign_disk_io.side_effect = assign_disk_io_side_effect

    # Gọi hàm cần kiểm thử với pid và mock_logger
    cgroup_manager.assign_process_to_cgroups(pid, resource_dict, process_name, mock_logger)

    # Kiểm tra rằng Process được gọi với đúng PID
    mock_process_class.assert_called_with(pid)

    # Kiểm tra rằng các hàm assign được gọi đúng cách
    mock_assign_cpu_cgroup.assert_not_called()
    mock_assign_ram.assert_called_with(mock_proc, 4096, mock_logger)
    mock_assign_disk_io.assert_called_with(mock_proc, 100.0, mock_logger)
    mock_assign_cpu_freq.assert_not_called()
    mock_add_pid.assert_called_with(pid, mock_logger)

    # Kiểm tra rằng không có lỗi nào được ghi log
    mock_logger.error.assert_not_called()

    # Kiểm tra các thông điệp logger.info
    mock_logger.info.assert_any_call(f"Đã thêm PID {pid} vào cgroup /sys/fs/cgroup/memory/mining_group.")
    mock_logger.info.assert_any_call(f"Đã thêm PID {pid} vào cgroup /sys/fs/cgroup/blkio/mining_group.")
