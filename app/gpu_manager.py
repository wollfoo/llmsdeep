import pynvml
import subprocess
import logging

class GPUResourceManager:
    def __init__(self):
        self.logger = logging.getLogger("GPUResourceManager")
        logging.basicConfig(level=logging.DEBUG)
        self.gpu_initialized = False
        self.initialize_nvml()

    def initialize_nvml(self):
        try:
            pynvml.nvmlInit()
            self.logger.info("NVML initialized successfully.")
            self.gpu_initialized = True
        except Exception as e:
            self.logger.error(f"Failed to initialize NVML: {e}")

    def set_gpu_clocks(self, gpu_index, sm_clock, mem_clock):
        if not self.gpu_initialized:
            self.logger.error("NVML not initialized.")
            return False

        try:
            # Set SM clock
            cmd_sm = [
                'nvidia-smi',
                '-i', str(gpu_index),
                '--lock-gpu-clocks=' + str(sm_clock)
            ]
            subprocess.run(cmd_sm, check=True)
            self.logger.info(f"Set SM clock={sm_clock} MHz for GPU={gpu_index}.")

            # Set MEM clock
            cmd_mem = [
                'nvidia-smi',
                '-i', str(gpu_index),
                '--lock-memory-clocks=' + str(mem_clock)
            ]
            subprocess.run(cmd_mem, check=True)
            self.logger.info(f"Set MEM clock={mem_clock} MHz for GPU={gpu_index}.")

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to set clocks: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error setting clocks: {e}")
            return False
