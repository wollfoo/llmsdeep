from gpu_manager import GPUResourceManager

if __name__ == "__main__":
    # Khởi tạo GPUResourceManager
    gpu_manager = GPUResourceManager()

    # GPU Index (thường là 0 trên các node có 1 GPU)
    gpu_index = 0

    # Thông số xung nhịp mong muốn
    sm_clock = 1000  # SM clock (MHz)
    mem_clock = 877  # Memory clock (MHz, không đổi trong trường hợp này)

    # Thử đặt xung nhịp GPU
    success = gpu_manager.set_gpu_clocks(gpu_index, sm_clock, mem_clock)

    if success:
        print(f"Successfully set clocks: SM={sm_clock} MHz, Memory={mem_clock} MHz.")
    else:
        print("Failed to set GPU clocks.")
