import psutil
import logging

# Cấu hình logging để hiển thị thông tin chi tiết
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_process_and_get_obj(process_pid):
    """
    Kiểm tra xem PID có hợp lệ không và trả về đối tượng Process nếu tồn tại.
    """
    try:
        if psutil.pid_exists(process_pid):  # Kiểm tra PID tồn tại
            p_obj = psutil.Process(process_pid)
            logging.info(f"Process with PID {process_pid} exists: {p_obj.name()}")
            return p_obj  # Trả về đối tượng Process
        else:
            logging.warning(f"Process with PID {process_pid} no longer exists.")
            return None  # PID không tồn tại
    except psutil.NoSuchProcess:
        logging.error(f"Process PID not found (pid={process_pid})")
        return None  # Lỗi: Process không tồn tại
    except Exception as e:
        logging.error(f"Unexpected error with PID {process_pid}: {e}")
        return None  # Lỗi khác

# Thử nghiệm hàm với các PID hợp lệ và không hợp lệ
if __name__ == "__main__":
    # PID hợp lệ (thay bằng PID thực tế trên hệ thống)
    pid_valid = 1  # PID 1 thường là process init/systemd
    # PID không hợp lệ
    pid_invalid = 99999  # PID giả định không tồn tại

    # Kiểm tra PID hợp lệ
    result_valid = check_process_and_get_obj(pid_valid)
    if result_valid:
        logging.info(f"Process name: {result_valid.name()}")  # Lấy tên process nếu tồn tại

    # Kiểm tra PID không hợp lệ
    result_invalid = check_process_and_get_obj(pid_invalid)
    if not result_invalid:
        logging.info(f"No process found for PID {pid_invalid}.")
