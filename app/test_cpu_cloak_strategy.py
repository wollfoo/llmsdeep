# test_cpu_cloak_strategy.py

import logging
import time
import psutil
import subprocess
import os
import traceback
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

def start_test_process(logger):
    """
    Khởi chạy một tiến trình kiểm thử CPU-intensive (sử dụng lệnh 'yes'),
    chuyển hướng output về DEVNULL để tránh block khi buffer đầy.
    
    Args:
        logger (logging.Logger): Đối tượng logger để ghi log.
    
    Returns:
        subprocess.Popen: Đối tượng tiến trình đã được khởi chạy.
    """
    try:
        # Chạy 'yes' với output và error đều chuyển vào DEVNULL
        test_process = subprocess.Popen(
            ['yes'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info(f"Đã khởi chạy tiến trình kiểm thử 'yes' với PID {test_process.pid}.")
        return test_process
    except Exception as e:
        logger.error(f"Lỗi khi khởi chạy tiến trình kiểm thử: {e}")
        raise

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
    try:
        p = psutil.Process(pid)
        for _ in range(duration):
            cpu = p.cpu_percent(interval=interval)
            usage.append(cpu)
            if logger:
                logger.debug(f"PID {pid} CPU usage: {cpu}%")
    except psutil.NoSuchProcess:
        if logger:
            logger.warning(f"Tiến trình với PID {pid} đã kết thúc.")
    except Exception as e:
        if logger:
            logger.error(f"Lỗi khi giám sát CPU usage cho PID {pid}: {e}")
    return usage

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
        'cache_enabled': True             # Bật tính năng caching
    }
    
    # Khởi tạo CpuCloakStrategy
    cpu_cloak_strategy = CpuCloakStrategy(config, logger)
    
    # Bắt đầu tiến trình kiểm thử (yes)
    test_process = start_test_process(logger)
    
    try:
        # Áp dụng CpuCloakStrategy cho tiến trình kiểm thử
        # (Hàm apply() ở phiên bản mới cho phép chúng ta truyền vào Popen)
        adjustments = cpu_cloak_strategy.apply(test_process)
        logger.info(f"Các điều chỉnh đã áp dụng: {adjustments}")
        
        # Giám sát CPU usage trước khi chạy các nhiệm vụ
        logger.info("Bắt đầu giám sát CPU usage trước khi chạy các nhiệm vụ...")
        pre_throttle_usage = monitor_cpu_usage(test_process.pid, interval=1, duration=5, logger=logger)
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
    
    finally:
        # Kết thúc tiến trình kiểm thử
        try:
            test_process.terminate()
            test_process.wait(timeout=5)
            logger.info(f"Đã kết thúc tiến trình kiểm thử PID {test_process.pid}.")
        except Exception as e:
            logger.error(f"Lỗi khi kết thúc tiến trình kiểm thử: {e}")
        
        # Xóa các cgroup đã tạo để làm sạch môi trường (nếu có)
        try:
            cpu_cgroup = f"cpu_cloak_{test_process.pid}"
            cpuset_cgroup = f"cpuset_cloak_{test_process.pid}"
            
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
