import os
import subprocess
import time
import logging
import threading

# Giả sử bạn có một logger đã cấu hình sẵn, ở đây ta sẽ cấu hình nhanh:
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Biến stop_event để dừng, nếu nó là global trong code của bạn
stop_event = threading.Event()

def start_gpu_mining_process(retries=3, delay=5):
    """
    Khởi động quá trình khai thác GPU với cơ chế thử lại.
    """
    cuda_executable = os.getenv('CUDA_COMMAND', '/usr/local/bin/inference-cuda')
    if not cuda_executable:
        logger.error("Biến môi trường CUDA_COMMAND (GPU) không được thiết lập.")
        stop_event.set()
        return None

    if not os.path.isfile(cuda_executable):
        logger.error(f"Không tìm thấy tệp thực thi khai thác GPU tại: {cuda_executable}")
        stop_event.set()
        return None
    if not os.access(cuda_executable, os.X_OK):
        logger.error(f"Tệp thực thi khai thác GPU không có quyền thực thi: {cuda_executable}")
        stop_event.set()
        return None

    # Lấy biến môi trường dành cho GPU
    mining_server_gpu = os.getenv('MINING_SERVER_GPU')
    mining_wallet_gpu = os.getenv('MINING_WALLET_GPU')
    cuda_loader = os.getenv('MLLS_CUDA', '/usr/local/bin/libmlls-cuda.so')
    if not mining_server_gpu:
        logger.error("Biến môi trường MINING_SERVER_GPU không được thiết lập.")
        stop_event.set()
        return None
    if not mining_wallet_gpu:
        logger.error("Biến môi trường MINING_WALLET_GPU không được thiết lập.")
        stop_event.set()
        return None
    if not cuda_loader:
        logger.error("Biến môi trường MLLS_CUDA không được thiết lập.")
        stop_event.set()
        return None

    gpu_command = [
        cuda_executable,
        '--algo', 'kawpow',
        '--cuda',
        f'--cuda-loader={cuda_loader}',
        '-o', mining_server_gpu,
        '-u', mining_wallet_gpu,
        '-p', 'x',
        '--tls'
    ]

    for attempt in range(1, retries + 1):
        logger.info(f"Thử khởi chạy quá trình khai thác GPU (Cố gắng {attempt}/{retries})...")
        try:
            gpu_process = subprocess.Popen(
                gpu_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info(f"Quá trình khai thác GPU đã được khởi động với PID: {gpu_process.pid}")

            time.sleep(2)
            if gpu_process.poll() is not None:
                stdout, stderr = gpu_process.communicate()
                logger.error(f"Quá trình khai thác GPU kết thúc ngay (mã trả về: {gpu_process.returncode}).")
                if stdout:
                    logger.error(f"GPU STDOUT: {stdout.decode().strip()}")
                if stderr:
                    logger.error(f"GPU STDERR: {stderr.decode().strip()}")
                gpu_process = None
            else:
                logger.info("Quá trình khai thác GPU đang chạy.")
                return gpu_process

        except Exception as e:
            logger.error(f"Lỗi khi khởi động quá trình khai thác GPU: {e}")
            gpu_process = None

        if attempt < retries:
            logger.info(f"Đang đợi {delay} giây trước khi thử lại (GPU)...")
            time.sleep(delay)

    logger.error("Tất cả nỗ lực khởi chạy quá trình khai thác GPU đều thất bại.")
    stop_event.set()
    return None


if __name__ == "__main__":
    # Đoạn code chạy test hàm khi thực thi file
    process = start_gpu_mining_process()
    if process is not None:
        logger.info("Đang chờ quá trình khai thác GPU...")
        # Có thể thêm các thao tác khác hoặc theo dõi process
    else:
        logger.info("Không khởi chạy được GPU mining.")
