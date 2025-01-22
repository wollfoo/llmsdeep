import time
import psutil
from multiprocessing import Pool

def cpu_intensive_task(n):
    # Task CPU-bound: Tính toán số Fibonacci
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

if __name__ == "__main__":
    physical_cores = psutil.cpu_count(logical=False)
    print(f"[TEST] Lõi vật lý: {physical_cores}")

    # Test với 4 processes (70% của 6 lõi)
    with Pool(processes=4) as pool:
        pool.map(cpu_intensive_task, [10_000_000] * 100)  # Tạo 100 task nặng