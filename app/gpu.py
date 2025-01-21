import pynvml

def get_max_temperature(gpu_index=0):
    """
    Lấy ngưỡng nhiệt độ tối đa cho phép của GPU.

    :param gpu_index: Chỉ số GPU (default = 0).
    :return: Nhiệt độ tối đa cho phép (°C) hoặc None nếu không thành công.
    """
    try:
        # Khởi tạo NVML
        pynvml.nvmlInit()

        # Lấy handle GPU
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

        # Lấy ngưỡng nhiệt độ tối đa
        max_temp = pynvml.nvmlDeviceGetTemperatureThreshold(handle, pynvml.NVML_TEMPERATURE_THRESHOLD_GPU_MAX)
        return max_temp
    except pynvml.NVMLError as error:
        print(f"Lỗi NVML: {error}")
        return None
    finally:
        # Đóng NVML
        pynvml.nvmlShutdown()

# Kiểm tra nhiệt độ tối đa của GPU đầu tiên
gpu_index = 0
max_temp = get_max_temperature(gpu_index)
if max_temp is not None:
    print(f"Nhiệt độ tối đa của GPU {gpu_index}: {max_temp}°C")
else:
    print(f"Không thể lấy nhiệt độ tối đa của GPU {gpu_index}.")
