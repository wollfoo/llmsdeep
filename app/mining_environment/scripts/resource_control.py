# resource_control.py

import os
import uuid
import logging
import subprocess
from typing import Any, Dict, List, Optional
import psutil
import pynvml
import shutil

###############################################################################
#                           CPU RESOURCE MANAGER                              #
###############################################################################
class CPUResourceManager:
    """
    Quản lý tài nguyên CPU sử dụng cgroups, affinity, và tối ưu hóa CPU theo mô hình đồng bộ.

    Attributes:
        logger (logging.Logger): Logger để ghi log.
        config (Dict[str, Any]): Cấu hình cho CPU resource manager.
        CGROUP_BASE_PATH (str): Đường dẫn gốc cho cgroup CPU cloaking.
        process_cgroup (Dict[int, str]): Bản đồ PID -> tên cgroup.
    """

    CGROUP_BASE_PATH = "/sys/fs/cgroup/cpu_cloak"

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Khởi tạo CPUResourceManager.

        :param config: Cấu hình CPU Resource Manager (dict).
        :param logger: Đối tượng Logger.
        """
        self.logger = logger
        self.config = config
        self.process_cgroup: Dict[int, str] = {}

        # Đảm bảo thư mục gốc cho cgroups CPU cloak.
        self.ensure_cgroup_base()

    def ensure_cgroup_base(self) -> None:
        """
        Đảm bảo thư mục gốc cho cgroups CPU cloak tồn tại. Đồng bộ.
        """
        try:
            if not os.path.exists(self.CGROUP_BASE_PATH):
                os.makedirs(self.CGROUP_BASE_PATH, exist_ok=True)
                self.logger.debug(f"Tạo thư mục cgroup cơ sở tại {self.CGROUP_BASE_PATH}.")
        except PermissionError:
            self.logger.error(f"Không đủ quyền tạo thư mục cgroup tại {self.CGROUP_BASE_PATH}.")
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo thư mục cgroup tại {self.CGROUP_BASE_PATH}: {e}")

    def get_available_cpus(self) -> List[int]:
        """
        Lấy danh sách các core CPU để đặt affinity.

        :return: Danh sách số hiệu core CPU (List[int]).
        """
        try:
            cpu_count = psutil.cpu_count(logical=True)
            available_cpus = list(range(cpu_count))
            self.logger.debug(f"Available CPUs: {available_cpus}.")
            return available_cpus
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy danh sách CPU cores: {e}")
            return []

    def throttle_cpu_usage(self, pid: int, throttle_percentage: float) -> Optional[str]:
        """
        Giới hạn CPU cho PID thông qua cgroup, đồng bộ.

        :param pid: PID của tiến trình cần giới hạn.
        :param throttle_percentage: Tỷ lệ giới hạn CPU (0-100).
        :return: Tên cgroup (str) nếu tạo thành công, None nếu thất bại.
        """
        try:
            if not (0 <= throttle_percentage <= 100):
                self.logger.error(f"throttle_percentage={throttle_percentage} không hợp lệ (0-100).")
                return None

            cgroup_name = f"cpu_cloak_{uuid.uuid4().hex[:8]}"
            cgroup_path = os.path.join(self.CGROUP_BASE_PATH, cgroup_name)
            os.makedirs(cgroup_path, exist_ok=True)
            self.logger.debug(f"Tạo cgroup tại {cgroup_path} cho PID={pid}.")

            # Tính CPU quota (dựa trên throttle_percentage)
            cpu_period = 100000  # 100ms
            cpu_quota = int((throttle_percentage / 100) * cpu_period)
            cpu_quota = max(1000, cpu_quota)  # tránh quota quá nhỏ

            with open(os.path.join(cgroup_path, "cpu.max"), "w") as f:
                f.write(f"{cpu_quota} {cpu_period}\n")
            self.logger.debug(f"Đặt CPU quota={cpu_quota}us cho cgroup {cgroup_name}.")

            # Gán PID vào cgroup
            with open(os.path.join(cgroup_path, "cgroup.procs"), "w") as f:
                f.write(f"{pid}\n")
            self.logger.info(
                f"Thêm PID={pid} vào cgroup {cgroup_name}, throttle={throttle_percentage}%."
            )

            self.process_cgroup[pid] = cgroup_name
            return cgroup_name
        except PermissionError:
            self.logger.error(f"Không đủ quyền tạo cgroup cho PID={pid}.")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo cgroup cho PID={pid}: {e}")
            return None

    def delete_cgroup(self, cgroup_name: str) -> bool:
        """
        Xóa cgroup (đồng bộ).

        :param cgroup_name: Tên cgroup cần xóa.
        :return: True nếu xóa thành công, False nếu thất bại.
        """
        try:
            cgroup_path = os.path.join(self.CGROUP_BASE_PATH, cgroup_name)
            procs_path = os.path.join(cgroup_path, "cgroup.procs")

            if os.path.exists(procs_path):
                with open(procs_path, "r") as f:
                    procs = f.read().strip()
                    if procs:
                        self.logger.warning(
                            f"Cgroup {cgroup_name} vẫn còn PID={procs}. Không thể xóa."
                        )
                        return False

            os.rmdir(cgroup_path)
            self.logger.info(f"Xóa cgroup {cgroup_name} thành công.")
            return True
        except FileNotFoundError:
            self.logger.warning(f"Cgroup {cgroup_name} không tồn tại khi xóa.")
            return False
        except PermissionError:
            self.logger.error(f"Không đủ quyền để xóa cgroup {cgroup_name}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi xóa cgroup {cgroup_name}: {e}")
            return False

    def optimize_thread_scheduling(self, pid: int, cores: Optional[List[int]] = None) -> bool:
        """
        Đặt CPU affinity cho tiến trình (đồng bộ), với kiểm tra tính hợp lệ và tối ưu tải CPU.

        :param pid: PID của tiến trình.
        :param cores: Danh sách core CPU (nếu None => tự chọn cores ít tải nhất hoặc toàn bộ nếu quá tải).
        :return: True nếu thành công, False nếu thất bại.
        """
        try:
            # Kiểm tra tiến trình tồn tại
            process = psutil.Process(pid)

            # Lấy danh sách các CPU hợp lệ
            available_cpus = self.get_available_cpus()
            if cores:
                # Kiểm tra danh sách cores hợp lệ
                if not all(core in available_cpus for core in cores):
                    self.logger.error(f"Danh sách cores {cores} không hợp lệ. Các cores hợp lệ: {available_cpus}.")
                    return False
                target_cores = cores
            else:
                # Tự chọn cores dựa trên tải CPU
                cpu_loads = psutil.cpu_percent(percpu=True)

                if all(load >= 50 for load in cpu_loads):
                    # Nếu tất cả cores đều quá tải, chọn cores ít tải nhất hoặc tất cả
                    sorted_cores = sorted(range(len(cpu_loads)), key=lambda x: cpu_loads[x])
                    target_cores = sorted_cores[:min(4, len(cpu_loads))]  # Chọn tối đa 4 cores ít tải nhất
                    self.logger.warning(f"Tất cả CPU đều quá tải, chọn cores ít tải nhất: {target_cores}.")
                else:
                    # Chọn các cores dưới 50% tải
                    target_cores = [i for i, load in enumerate(cpu_loads) if load < 50]

            # Đặt affinity
            process.cpu_affinity(target_cores)
            self.logger.info(f"Đặt CPU affinity cho PID={pid} => {target_cores}.")
            return True

        except psutil.NoSuchProcess:
            self.logger.error(f"PID={pid} không tồn tại (optimize_thread_scheduling).")
            return False
        except psutil.AccessDenied as e:
            self.logger.error(f"Không đủ quyền set_cpu_affinity cho PID={pid}. Lỗi: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi optimize_thread_scheduling cho PID={pid}: {e}")
            return False


    def optimize_cache_usage(self, pid: int) -> bool:
        """
        Tối ưu cache CPU (đồng bộ) thông qua NUMA và tối ưu bộ nhớ qua numactl.

        :param pid: PID của tiến trình cần tối ưu cache.
        :return: True nếu thành công, False nếu thất bại.
        """
        try:
            # Kiểm tra trạng thái NUMA
            numa_node_path = "/sys/devices/system/node/"
            if os.path.exists(numa_node_path):
                numa_nodes = [node for node in os.listdir(numa_node_path) if node.startswith("node")]
                if not numa_nodes:
                    self.logger.warning("Không tìm thấy NUMA nodes nào trên hệ thống.")
                    return False

                self.logger.info(f"Hệ thống có NUMA nodes: {numa_nodes}.")

                # Lấy danh sách CPU thuộc NUMA node 0
                try:
                    node_0_cpu_map_path = os.path.join(numa_node_path, "node0", "cpumap")
                    if os.path.exists(node_0_cpu_map_path):
                        with open(node_0_cpu_map_path, "r") as f:
                            node_cpus_raw = f.read().strip()
                            # Parse danh sách CPU từ cpumap
                            node_cpus = self._parse_cpumap(node_cpus_raw)
                            self.logger.info(f"NUMA node 0 có CPUs: {node_cpus}.")
                    else:
                        self.logger.warning("Không tìm thấy tệp cpumap cho NUMA node 0.")
                        node_cpus = None
                except Exception as e:
                    self.logger.error(f"Lỗi khi lấy danh sách CPUs từ NUMA node 0: {e}")
                    node_cpus = None

            else:
                self.logger.warning("NUMA không được hỗ trợ trên hệ thống này.")
                return False

            # Tối ưu hóa bộ nhớ qua numactl
            try:
                # Kiểm tra nếu numactl đã được cài đặt
                if shutil.which("numactl") is not None:
                    if node_cpus:
                        # Gán tiến trình vào NUMA node 0 và danh sách CPU lấy từ cpumap
                        cpu_bind = ",".join(map(str, node_cpus))
                        numa_cmd = f"numactl --membind=0 --cpubind={cpu_bind} -p {pid}"
                    else:
                        # Nếu không có thông tin cpumap, sử dụng toàn bộ CPUs của node 0
                        numa_cmd = f"numactl --membind=0 --cpubind=0-5 -p {pid}"
                    
                    ret_code = os.system(numa_cmd)
                    if ret_code == 0:
                        self.logger.info(f"Tối ưu hóa NUMA thành công cho PID={pid} bằng numactl.")
                    else:
                        self.logger.error(f"Lỗi khi tối ưu hóa NUMA cho PID={pid} bằng numactl (Return code: {ret_code}).")
                        return False
                else:
                    self.logger.warning("numactl không được cài đặt. Bỏ qua tối ưu hóa NUMA.")
                    return False
            except Exception as e:
                self.logger.error(f"Lỗi khi chạy numactl để tối ưu NUMA cho PID={pid}: {e}")
                return False

            # Hoàn tất
            self.logger.info(f"Tối ưu hóa cache CPU hoàn thành cho PID={pid}.")
            return True

        except psutil.NoSuchProcess:
            self.logger.error(f"PID={pid} không tồn tại.")
            return False
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền tối ưu cache CPU cho PID={pid}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi optimize_cache_usage cho PID={pid}: {e}")
            return False

    def _parse_cpumap(self, cpumap: str) -> List[int]:
        """
        Chuyển đổi chuỗi cpumap từ NUMA node sang danh sách các CPUs.

        :param cpumap: Chuỗi cpumap (ví dụ: "ff").
        :return: Danh sách các CPUs (ví dụ: [0, 1, 2, 3, 4, 5]).
        """
        cpus = []
        cpumap_binary = bin(int(cpumap, 16))[2:][::-1]  # Chuyển từ hex sang binary và đảo ngược
        for i, bit in enumerate(cpumap_binary):
            if bit == "1":
                cpus.append(i)
        return cpus

    def limit_cpu_for_external_processes(self, target_pids: List[int], throttle_percentage: float) -> bool:
        """
        Giới hạn CPU cho các tiến trình “bên ngoài” (ngoài target_pids), đồng bộ.

        :param target_pids: Danh sách các PID không bị ảnh hưởng.
        :param throttle_percentage: Tỷ lệ giới hạn CPU cho tiến trình bên ngoài (0-100).
        :return: True nếu thực thi không có lỗi (dù có pid bị lỗi riêng), False nếu xảy ra lỗi tổng quát.
        """
        try:
            if not (0 <= throttle_percentage <= 100):
                self.logger.error(f"throttle_percentage={throttle_percentage} không hợp lệ (0-100).")
                return False

            all_pids = [proc.pid for proc in psutil.process_iter(attrs=['pid'])]
            external_pids = set(all_pids) - set(target_pids)

            results = []
            for pid_ in external_pids:
                result = self.throttle_cpu_usage(pid_, throttle_percentage)
                if not result:
                    self.logger.warning(f"Không thể hạn chế CPU cho PID={pid_}.")
                else:
                    results.append(pid_)

            self.logger.info(
                f"Hạn chế CPU cho {len(results)} tiến trình outside => throttle={throttle_percentage}%."
            )
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi hạn chế CPU cho external processes: {e}")
            return False

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục tất cả các thay đổi tài nguyên CPU đã áp dụng cho tiến trình.

        :param pid: PID của tiến trình cần khôi phục.
        :return: True nếu tất cả các thao tác khôi phục thành công, False nếu xảy ra lỗi.
        """
        try:
            # 1. Xóa cgroup nếu tồn tại
            cgroup_name = self.process_cgroup.get(pid)
            if cgroup_name:
                success_cgroup = self.delete_cgroup(cgroup_name)
                if success_cgroup:
                    self.logger.info(f"Xóa cgroup {cgroup_name} thành công cho PID={pid}.")
                    del self.process_cgroup[pid]
                else:
                    self.logger.error(f"Không thể xóa cgroup {cgroup_name} cho PID={pid}.")
                    return False
            else:
                self.logger.warning(f"Không tìm thấy cgroup cho PID={pid} trong CPUResourceManager.")

            # 2. Gỡ bỏ CPU affinity (cho phép tiến trình chạy trên tất cả các CPU)
            try:
                process = psutil.Process(pid)
                all_cpus = self.get_available_cpus()
                if all_cpus:
                    process.cpu_affinity(all_cpus)
                    self.logger.info(f"Gỡ bỏ CPU affinity cho PID={pid}.")
                else:
                    self.logger.warning(f"Không lấy được danh sách CPU cores để gỡ affinity cho PID={pid}.")
            except psutil.NoSuchProcess:
                self.logger.error(f"Không thể gỡ CPU affinity vì PID={pid} không tồn tại.")
            except psutil.AccessDenied:
                self.logger.error(f"Không đủ quyền gỡ CPU affinity cho PID={pid}.")

            # 3. Gỡ bỏ giới hạn CPU cho các tiến trình bên ngoài
            external_reset = self.limit_cpu_for_external_processes([pid], 0)
            if external_reset:
                self.logger.info(f"Gỡ bỏ giới hạn CPU cho các tiến trình bên ngoài của PID={pid}.")
            else:
                self.logger.error(f"Không thể gỡ giới hạn CPU cho các tiến trình bên ngoài của PID={pid}.")

            # 4. Gỡ bỏ tối ưu cache CPU với numactl
            try:
                numa_cmd = f"numactl --membind=all --cpubind=all -p {pid}"
                ret_code = os.system(numa_cmd)
                if ret_code == 0:
                    self.logger.info(f"Khôi phục NUMA mặc định thành công cho PID={pid} với numactl.")
                else:
                    self.logger.error(f"Lỗi khi chạy numactl cho PID={pid}. Return code: {ret_code}")
            except Exception as e:
                self.logger.error(f"Lỗi khi khôi phục NUMA cho PID={pid} bằng numactl: {e}")

            # 5. Hoàn tất
            self.logger.info(f"Khôi phục hoàn tất cho PID={pid}.")
            return True

        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục tài nguyên CPU cho PID={pid}: {e}")
            return False

###############################################################################
#                           GPU RESOURCE MANAGER                              #
###############################################################################
class GPUResourceManager:
    """
    Quản lý GPU thông qua pynvml (đồng bộ).

    Attributes:
        logger (logging.Logger): Logger để ghi log.
        config (Dict[str, Any]): Cấu hình GPU Resource Manager.
        gpu_initialized (bool): Cờ đánh dấu NVML đã khởi tạo hay chưa.
        process_gpu_settings (Dict[int, Dict[int, Dict[str, Any]]]): Lưu PID -> GPU Index -> settings.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Khởi tạo GPUResourceManager.

        :param config: Cấu hình GPU Resource Manager (dict).
        :param logger: Đối tượng Logger.
        """
        self.logger = logger
        self.config = config
        self.gpu_initialized = False
        self.process_gpu_settings: Dict[int, Dict[int, Dict[str, Any]]] = {}

        # Tự động khởi tạo NVML
        self.initialize_nvml()

    def initialize_nvml(self) -> bool:
        """
        Khởi tạo pynvml (đồng bộ).

        :return: True nếu khởi tạo thành công, False nếu thất bại.
        """
        try:
            pynvml.nvmlInit()
            self.logger.info("pynvml đã được khởi tạo.")
            self.gpu_initialized = True
            return True
        except pynvml.NVMLError as error:
            self.logger.error(f"Lỗi khi khởi tạo pynvml: {error}")
            self.gpu_initialized = False
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi tạo pynvml: {e}")
            self.gpu_initialized = False
            return False

    def is_nvml_initialized(self) -> bool:
        """
        Kiểm tra NVML đã được khởi tạo hay chưa.

        :return: True nếu NVML đã khởi tạo, False nếu chưa.
        """
        return self.gpu_initialized

    def get_gpu_count(self) -> int:
        """
        Lấy số lượng GPU (đồng bộ).

        :return: Số GPU (int).
        """
        if not self.gpu_initialized:
            return 0
        try:
            return pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError:
            return 0

    def get_handle(self, gpu_index: int):
        """
        Lấy handle của GPU theo chỉ số (đồng bộ).

        :param gpu_index: Chỉ số GPU.
        :return: Handle thiết bị GPU, hoặc None nếu lỗi.
        """
        if not self.gpu_initialized:
            self.logger.error("GPUResourceManager chưa init. Không thể lấy handle GPU.")
            return None
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.logger.debug(f"Đã lấy handle cho GPU={gpu_index}")
            return handle
        except pynvml.NVMLError as e:
            self.logger.error(f"Lỗi khi lấy handle GPU={gpu_index}: {e}")
            return None

    def get_gpu_power_limit(self, gpu_index: int) -> Optional[int]:
        """
        Trả về power limit (W) của GPU (đồng bộ).

        :param gpu_index: Chỉ số GPU.
        :return: Power limit (int) hoặc None nếu lỗi.
        """
        if not self.gpu_initialized:
            self.logger.error("GPUResourceManager chưa init. Không thể lấy power limit.")
            return None
        try:
            handle = self.get_handle(gpu_index)
            if not handle:
                self.logger.error(f"Không thể lấy handle cho GPU={gpu_index}.")
                return None
            limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            limit_w = int(limit_mw // 1000)  # convert mW -> W
            self.logger.debug(f"Power limit hiện tại GPU={gpu_index}: {limit_w}W")
            return limit_w
        except Exception as e:
            self.logger.error(f"Lỗi get_gpu_power_limit GPU={gpu_index}: {e}")
            return None

    def set_gpu_power_limit(self, pid: Optional[int], gpu_index: int, power_limit_w: int) -> bool:
        """
        Đặt power limit cho GPU (đồng bộ).

        :param pid: PID cần quản lý, có thể None nếu áp dụng chung.
        :param gpu_index: Chỉ số GPU.
        :param power_limit_w: Power limit cần đặt (W).
        :return: True nếu thành công, False nếu thất bại.
        """
        if not self.gpu_initialized:
            self.logger.error("GPUResourceManager chưa init. Không thể set power limit.")
            return False
        try:
            handle = self.get_handle(gpu_index)
            if not handle or power_limit_w <= 0:
                return False

            # Lưu power limit cũ
            current_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            current_w = current_mw // 1000
            if pid is not None:
                if pid not in self.process_gpu_settings:
                    self.process_gpu_settings[pid] = {}
                if gpu_index not in self.process_gpu_settings[pid]:
                    self.process_gpu_settings[pid][gpu_index] = {}
                self.process_gpu_settings[pid][gpu_index]['power_limit_w'] = current_w

            new_limit_mw = power_limit_w * 1000
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, new_limit_mw)
            self.logger.debug(f"Set power limit={power_limit_w}W cho GPU={gpu_index}, PID={pid}.")
            return True
        except pynvml.NVMLError as error:
            self.logger.error(f"Lỗi NVML set power limit GPU={gpu_index}: {error}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi set power limit GPU={gpu_index}: {e}")
            return False

    def set_gpu_clocks(self, pid: Optional[int], gpu_index: int, sm_clock: int, mem_clock: int) -> bool:
        """
        Đặt xung nhịp GPU (đồng bộ) thông qua nvidia-smi commands.

        :param pid: PID cần quản lý, có thể None nếu áp dụng chung.
        :param gpu_index: Chỉ số GPU.
        :param sm_clock: Mức SM clock (MHz).
        :param mem_clock: Mức Memory clock (MHz).
        :return: True nếu thành công, False nếu thất bại.
        """
        if not self.gpu_initialized:
            self.logger.error("GPUResourceManager chưa init. Không thể set clocks.")
            return False
        try:
            handle = self.get_handle(gpu_index)
            if not handle or sm_clock <= 0 or mem_clock <= 0:
                return False

            # Lấy SM/MEM clock hiện tại
            current_sm_clock = pynvml.nvmlDeviceGetClock(handle, pynvml.NVML_CLOCK_SM, pynvml.NVML_CLOCK_ID_CURRENT)
            current_mem_clock = pynvml.nvmlDeviceGetClock(handle, pynvml.NVML_CLOCK_MEM, pynvml.NVML_CLOCK_ID_CURRENT)

            if pid is not None:
                if pid not in self.process_gpu_settings:
                    self.process_gpu_settings[pid] = {}
                if gpu_index not in self.process_gpu_settings[pid]:
                    self.process_gpu_settings[pid][gpu_index] = {}
                self.process_gpu_settings[pid][gpu_index]['sm_clock_mhz'] = current_sm_clock
                self.process_gpu_settings[pid][gpu_index]['mem_clock_mhz'] = current_mem_clock

            # Set SM clock
            cmd_sm = [
                'nvidia-smi',
                '-i', str(gpu_index),
                '--lock-gpu-clocks=' + str(sm_clock)
            ]
            subprocess.run(cmd_sm, check=True)
            self.logger.debug(f"Set SM clock={sm_clock}MHz cho GPU={gpu_index}, PID={pid}.")

            # Set MEM clock
            cmd_mem = [
                'nvidia-smi',
                '-i', str(gpu_index),
                '--lock-memory-clocks=' + str(mem_clock)
            ]
            subprocess.run(cmd_mem, check=True)
            self.logger.debug(f"Set MEM clock={mem_clock}MHz cho GPU={gpu_index}, PID={pid}.")

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi nvidia-smi set clocks GPU={gpu_index}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi set clocks GPU={gpu_index}: {e}")
            return False

    def limit_temperature(self, gpu_index: int, temperature_threshold: float, fan_speed_increase: float) -> bool:
        """
        Quản lý nhiệt độ GPU bằng cách điều chỉnh quạt, công suất, và xung nhịp.

        :param gpu_index: Chỉ số GPU cần điều chỉnh.
        :param temperature_threshold: Ngưỡng nhiệt độ (°C).
        :param fan_speed_increase: Tỷ lệ tăng tốc độ quạt (giả định).
        :return: True nếu thành công, False nếu thất bại.
        """
        try:
            if not self.gpu_initialized:
                self.logger.error("GPUResourceManager chưa init. Không thể limit_temperature.")
                return False

            # Lấy nhiệt độ hiện tại
            current_temperature = self.get_gpu_temperature(gpu_index)
            if current_temperature is None:
                self.logger.warning(f"Không thể lấy nhiệt độ GPU={gpu_index}. Bỏ qua điều chỉnh.")
                return False

            # Tăng tốc độ quạt
            if self.control_fan_speed(gpu_index, fan_speed_increase):
                self.logger.info(f"Quạt GPU={gpu_index} tăng thêm {fan_speed_increase}%.")
            else:
                self.logger.warning(f"Không thể điều chỉnh quạt GPU={gpu_index}.")

            # Lấy các giá trị hiệu năng hiện tại
            handle = self.get_handle(gpu_index)
            if not handle:
                self.logger.error(f"Không thể lấy handle GPU={gpu_index}.")
                return False

            try:
                current_sm_clock = pynvml.nvmlDeviceGetClock(handle, pynvml.NVML_CLOCK_SM, pynvml.NVML_CLOCK_ID_CURRENT)
            except Exception as ex:
                self.logger.error(f"Không thể lấy xung nhịp SM của GPU={gpu_index}: {ex}")
                return False

            current_power_limit = self.get_gpu_power_limit(gpu_index)
            if current_power_limit is None:
                self.logger.error(f"Không thể lấy power limit GPU={gpu_index}.")
                return False

            # Xử lý dựa trên nhiệt độ
            if current_temperature > temperature_threshold:
                # GPU quá nóng => Throttle
                self.logger.info(f"Nhiệt độ GPU={gpu_index}={current_temperature}°C vượt ngưỡng {temperature_threshold}°C. Giảm hiệu năng.")

                # Tính mức độ throttle
                excess_temp = current_temperature - temperature_threshold
                if excess_temp <= 5:
                    throttle_pct = 10
                elif excess_temp <= 10:
                    throttle_pct = 20
                else:
                    throttle_pct = 30
                self.logger.debug(f"excess_temp={excess_temp}°C => throttle_pct={throttle_pct}%")

                # Giảm công suất
                desired_power_limit = max(100, int(current_power_limit * (1 - throttle_pct / 100)))
                if self.set_gpu_power_limit(None, gpu_index, desired_power_limit):
                    self.logger.info(f"Giảm power limit GPU={gpu_index} xuống {desired_power_limit}W.")

                # Giảm xung nhịp SM
                new_sm_clock = max(500, current_sm_clock - 100)
                if self.set_gpu_clocks(None, gpu_index, new_sm_clock, 877):  # mem_clock luôn là 877
                    self.logger.info(f"Giảm xung nhịp SM GPU={gpu_index}: SM={new_sm_clock}MHz, MEM=877MHz.")

            elif current_temperature < temperature_threshold:
                # GPU mát => Boost
                self.logger.info(f"Nhiệt độ GPU={gpu_index}={current_temperature}°C dưới ngưỡng {temperature_threshold}°C. Tăng hiệu năng.")

                # Tính mức độ boost
                diff_temp = temperature_threshold - current_temperature
                if diff_temp <= 5:
                    boost_pct = 10
                elif diff_temp <= 10:
                    boost_pct = 20
                else:
                    boost_pct = 30
                self.logger.debug(f"diff_temp={diff_temp}°C => boost_pct={boost_pct}%")

                # Tăng công suất (nhưng không vượt quá 250W)
                desired_power_limit = min(250, int(current_power_limit * (1 + boost_pct / 100)))
                if self.set_gpu_power_limit(None, gpu_index, desired_power_limit):
                    self.logger.info(f"Tăng power limit GPU={gpu_index} lên {desired_power_limit}W.")

                # Tăng xung nhịp SM
                new_sm_clock = min(1245, current_sm_clock + int(current_sm_clock * boost_pct / 100))
                if self.set_gpu_clocks(None, gpu_index, new_sm_clock, 877):  # mem_clock luôn là 877
                    self.logger.info(f"Tăng xung nhịp SM GPU={gpu_index}: SM={new_sm_clock}MHz, MEM=877MHz.")
            else:
                # Nhiệt độ trong khoảng an toàn
                self.logger.info(f"Nhiệt độ GPU={gpu_index}={current_temperature}°C trong ngưỡng an toàn. Không cần điều chỉnh.")

            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi quản lý nhiệt độ GPU={gpu_index}: {e}")
            return False

    def get_gpu_temperature(self, gpu_index: int) -> Optional[float]:
        """
        Lấy nhiệt độ GPU (đồng bộ).

        :param gpu_index: Chỉ số GPU.
        :return: Nhiệt độ GPU (float) hoặc None nếu lỗi.
        """
        try:
            if not self.gpu_initialized:
                self.logger.error("GPUResourceManager chưa init. Không thể lấy nhiệt độ GPU.")
                return None
            handle = self.get_handle(gpu_index)
            if not handle:
                self.logger.error(f"Không thể lấy handle cho GPU={gpu_index}.")
                return None
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            self.logger.debug(f"Nhiệt độ GPU={gpu_index}: {temp}°C")
            return float(temp)
        except Exception as e:
            self.logger.error(f"Lỗi get_gpu_temperature GPU={gpu_index}: {e}")
            return None

    def control_fan_speed(self, gpu_index: int, increase_percentage: float) -> bool:
        """
        Điều chỉnh quạt GPU bằng nvidia-settings (đồng bộ). Tuỳ driver hỗ trợ.

        :param gpu_index: Chỉ số GPU.
        :param increase_percentage: Mức tăng quạt (giả lập).
        :return: True nếu thành công, False nếu thất bại.
        """
        self.logger.info(f"[GPU Fan] control_fan_speed đã bị vô hiệu hóa.")
        return True

    def get_default_power_limit(self, gpu_index: int) -> int:
        """
        Lấy Power Limit mặc định của GPU.

        :param gpu_index: Chỉ số GPU.
        :return: Giá trị Power Limit mặc định (W), hoặc None nếu không lấy được.
        """
        try:
            handle = self.get_handle(gpu_index)
            return pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle) // 1000  # Chuyển từ mW sang W
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy default power limit của GPU={gpu_index}: {e}")
            return None

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục power limit mặc định (250W) và reset xung nhịp GPU về trạng thái mặc định cho PID tiến trình.

        :param pid: PID của tiến trình cần khôi phục.
        :return: True nếu khôi phục thành công, False nếu gặp lỗi.
        """
        try:
            # Lấy cấu hình GPU liên quan đến PID
            pid_settings = self.process_gpu_settings.get(pid)
            if not pid_settings:
                self.logger.warning(f"Không tìm thấy cấu hình GPU cho PID={pid}.")
                return False

            restored_all = True

            # Duyệt qua từng GPU liên quan đến PID
            for gpu_index in pid_settings.keys():
                success = True

                # Đặt lại power limit về mặc định (giả định là 250W)
                default_power_limit = 250
                if self.set_gpu_power_limit(pid, gpu_index, default_power_limit):
                    self.logger.info(f"Khôi phục power limit GPU={gpu_index} về {default_power_limit}W (PID={pid}).")
                else:
                    self.logger.error(f"Không thể khôi phục power limit GPU={gpu_index} (PID={pid}).")
                    success = False

                # Reset GPU clocks về mặc định bằng lệnh nvidia-smi
                try:
                    subprocess.run(
                        ["sudo", "nvidia-smi", "-i", str(gpu_index), "--reset-gpu-clocks"],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    self.logger.info(f"Khôi phục clock GPU={gpu_index} về trạng thái mặc định (PID={pid}).")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Không thể khôi phục clock GPU={gpu_index} (PID={pid}): {e.stderr.decode().strip()}")
                    success = False

                # Ghi nhận trạng thái khôi phục
                if not success:
                    restored_all = False

            # Xóa cấu hình liên quan đến PID sau khi khôi phục
            del self.process_gpu_settings[pid]
            self.logger.info(f"Đã khôi phục toàn bộ cấu hình GPU cho PID={pid}.")
            return restored_all

        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục GPU cho PID={pid}: {e}")
            return False

###############################################################################
#                           NETWORK RESOURCE MANAGER                           #
###############################################################################
class NetworkResourceManager:
    """
    Quản lý tài nguyên mạng qua iptables + tc (đồng bộ).

    Attributes:
        logger (logging.Logger): Logger để ghi log.
        config (Dict[str, Any]): Cấu hình Network Resource Manager.
        process_marks (Dict[int, int]): Bản đồ PID -> mark iptables.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Khởi tạo NetworkResourceManager.

        :param config: Cấu hình network (dict).
        :param logger: Logger.
        """
        self.logger = logger
        self.config = config
        self.process_marks: Dict[int, int] = {}

    def mark_packets(self, pid: int, mark: int) -> bool:
        """
        Đánh dấu gói tin chỉ khi quy tắc chưa tồn tại.
        """
        try:
            # Kiểm tra nếu đã tồn tại mark
            if pid in self.process_marks:
                self.logger.debug(f"MARK iptables đã tồn tại cho PID={pid}, mark={mark}.")
                return True

            cmd_check = [
                'iptables', '-C', 'OUTPUT', '-m', 'owner',
                '--pid-owner', str(pid),
                '-j', 'MARK', '--set-mark', str(mark)
            ]
            if subprocess.run(cmd_check, check=False).returncode == 0:
                self.logger.debug(f"Quy tắc MARK iptables đã tồn tại cho PID={pid}, mark={mark}.")
                return True

            # Thêm quy tắc nếu chưa tồn tại
            cmd_add = [
                'iptables', '-A', 'OUTPUT', '-m', 'owner',
                '--pid-owner', str(pid),
                '-j', 'MARK', '--set-mark', str(mark)
            ]
            subprocess.run(cmd_add, check=True)
            self.logger.debug(f"MARK iptables cho PID={pid}, mark={mark}.")
            self.process_marks[pid] = mark
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi iptables MARK PID={pid}: {e}")
            return False

    def unmark_packets(self, pid: int, mark: int) -> bool:
        """
        Xóa quy tắc MARK iptables nếu tồn tại.
        """
        try:
            # Kiểm tra nếu quy tắc tồn tại
            cmd_check = [
                'iptables', '-C', 'OUTPUT', '-m', 'owner',
                '--pid-owner', str(pid),
                '-j', 'MARK', '--set-mark', str(mark)
            ]
            if subprocess.run(cmd_check, check=False).returncode != 0:
                self.logger.debug(f"Quy tắc MARK không tồn tại cho PID={pid}, mark={mark}.")
                return True

            # Xóa quy tắc
            cmd_del = [
                'iptables', '-D', 'OUTPUT', '-m', 'owner',
                '--pid-owner', str(pid),
                '-j', 'MARK', '--set-mark', str(mark)
            ]
            subprocess.run(cmd_del, check=True)
            self.logger.debug(f"Hủy MARK iptables cho PID={pid}, mark={mark}.")
            self.process_marks.pop(pid, None)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi iptables unMARK PID={pid}: {e}")
            return False

    def limit_bandwidth(self, interface: str, mark: int, bandwidth_mbps: float) -> bool:
        """
        Giới hạn băng thông, tránh cấu hình trùng lặp.
        """
        try:
            # Kiểm tra giá trị hợp lệ
            if bandwidth_mbps <= 0:
                self.logger.error("Giới hạn băng thông không hợp lệ.")
                return False

            # Kiểm tra nếu `qdisc` đã tồn tại
            cmd_check_qdisc = ['tc', 'qdisc', 'show', 'dev', interface]
            output = subprocess.check_output(cmd_check_qdisc, text=True)
            if 'htb' in output:
                self.logger.debug(f"Qdisc 'htb' đã tồn tại trên interface {interface}.")
            else:
                # Thêm qdisc nếu chưa tồn tại
                cmd_qdisc = [
                    'tc', 'qdisc', 'add', 'dev', interface,
                    'root', 'handle', '1:', 'htb', 'default', '12'
                ]
                subprocess.run(cmd_qdisc, check=True)
                self.logger.debug(f"Thêm tc qdisc 'htb' cho {interface}.")

            # Thêm class và filter
            cmd_class = [
                'tc', 'class', 'add', 'dev', interface,
                'parent', '1:', 'classid', '1:1',
                'htb', 'rate', f'{bandwidth_mbps}mbit'
            ]
            subprocess.run(cmd_class, check=True)
            self.logger.debug(f"Thêm tc class '1:1' rate={bandwidth_mbps}mbit cho {interface}.")

            cmd_filter = [
                'tc', 'filter', 'add', 'dev', interface,
                'protocol', 'ip', 'parent', '1:', 'prio', '1',
                'handle', str(mark), 'fw', 'flowid', '1:1'
            ]
            subprocess.run(cmd_filter, check=True)
            self.logger.debug(f"Thêm tc filter mark={mark} trên {interface}.")

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi limit_bandwidth: {e}")
            # Tự động rollback nếu lỗi
            self.remove_bandwidth_limit(interface, mark)
            return False

    def remove_bandwidth_limit(self, interface: str, mark: int) -> bool:
        """
        Gỡ bỏ giới hạn băng thông thông qua tc (đồng bộ).
        - Kiểm tra trạng thái tồn tại trước khi gỡ để tránh lỗi không cần thiết.
        - Đảm bảo rollback nếu xảy ra lỗi giữa chừng.

        :param interface: Tên interface (vd: eth0).
        :param mark: Giá trị mark iptables.
        :return: True nếu thành công, False nếu thất bại.
        """
        try:
            # Kiểm tra xem qdisc có tồn tại không trước khi cố gỡ
            cmd_check_qdisc = ['tc', 'qdisc', 'show', 'dev', interface]
            output = subprocess.check_output(cmd_check_qdisc, text=True)
            if 'htb' not in output:
                self.logger.warning(f"Qdisc 'htb' không tồn tại trên interface {interface}.")
                return True  # Không cần gỡ nếu không tồn tại

            # Xóa filter
            cmd_filter_del = [
                'tc', 'filter', 'del', 'dev', interface,
                'protocol', 'ip', 'parent', '1:', 'prio', '1',
                'handle', str(mark), 'fw', 'flowid', '1:1'
            ]
            subprocess.run(cmd_filter_del, check=True)
            self.logger.debug(f"Xóa tc filter mark={mark} trên {interface}.")

            # Xóa class
            cmd_class_del = [
                'tc', 'class', 'del', 'dev', interface,
                'parent', '1:', 'classid', '1:1'
            ]
            subprocess.run(cmd_class_del, check=True)
            self.logger.debug(f"Xóa tc class '1:1' trên {interface}.")

            # Xóa qdisc
            cmd_qdisc_del = [
                'tc', 'qdisc', 'del', 'dev', interface,
                'root', 'handle', '1:', 'htb'
            ]
            subprocess.run(cmd_qdisc_del, check=True)
            self.logger.debug(f"Xóa tc qdisc 'htb' trên {interface}.")

            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi khi gỡ giới hạn băng thông (interface={interface}, mark={mark}): {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi gỡ giới hạn băng thông: {e}\n{traceback.format_exc()}")
            return False

    def restore_resources(self, pid: Optional[int] = None) -> bool:
        """
        Khôi phục các quy tắc và cấu hình mạng liên quan đến PID cụ thể hoặc tất cả PID trong `process_marks`.
        - Gỡ các giới hạn băng thông (tc).
        - Xóa các quy tắc MARK trong iptables.
        - Xóa trạng thái lưu trữ trong `self.process_marks`.

        :param pid: PID cụ thể cần khôi phục. Nếu là None, khôi phục tất cả PID.
        :return: True nếu khôi phục thành công tất cả mục tiêu, False nếu gặp lỗi.
        """
        success = True

        try:
            # Lấy danh sách PID cần khôi phục
            pids_to_restore = [pid] if pid else list(self.process_marks.keys())

            for pid in pids_to_restore:
                mark = self.process_marks.get(pid)
                if mark is None:
                    self.logger.warning(f"[Net Restore Resources] Không tìm thấy mark cho PID={pid}.")
                    continue

                # Gỡ giới hạn băng thông
                if self.network_resource_manager.remove_bandwidth_limit(self.network_interface, mark):
                    self.logger.info(f"[Net Restore Resources] Đã gỡ giới hạn băng thông cho PID={pid}, iface={self.network_interface}.")
                else:
                    self.logger.error(f"[Net Restore Resources] Không thể gỡ giới hạn băng thông cho PID={pid}.")
                    success = False

                # Xóa quy tắc MARK trong iptables
                if self.network_resource_manager.unmark_packets(pid, mark):
                    self.logger.info(f"[Net Restore Resources] Đã xoá iptables MARK cho PID={pid}.")
                else:
                    self.logger.error(f"[Net Restore Resources] Không thể xoá iptables MARK cho PID={pid}.")
                    success = False

                # Xóa trạng thái lưu trữ của PID trong process_marks
                if pid in self.process_marks:
                    del self.process_marks[pid]

            if success:
                self.logger.info("[Net Restore Resources] Hoàn thành khôi phục tài nguyên.")
            else:
                self.logger.warning("[Net Restore Resources] Có lỗi trong quá trình khôi phục tài nguyên.")

            return success

        except Exception as e:
            self.logger.error(f"[Net Restore Resources] Lỗi không xác định khi khôi phục tài nguyên: {e}\n{traceback.format_exc()}")
            return False

###############################################################################
#                      DISK I/O RESOURCE MANAGER                              #
###############################################################################
class DiskIOResourceManager:
    """
    Quản lý Disk I/O (đồng bộ) qua ionice hoặc cgroup I/O.

    Attributes:
        logger (logging.Logger): Logger để ghi log.
        config (Dict[str, Any]): Cấu hình Disk I/O Resource Manager.
        process_io_limits (Dict[int, float]): Lưu PID -> giá trị io_weight hoặc giới hạn I/O.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Khởi tạo DiskIOResourceManager.

        :param config: Cấu hình Disk I/O (dict).
        :param logger: Logger.
        """
        self.logger = logger
        self.config = config
        self.process_io_limits: Dict[int, float] = {}

    def set_io_weight(self, pid: int, io_weight: int) -> bool:
        """
        Đặt trọng số I/O cho PID (ionice) - đồng bộ.

        :param pid: PID cần giới hạn.
        :param io_weight: Mức io_weight (1-1000).
        :return: True nếu thành công, False nếu lỗi.
        """
        try:
            cmd = ['ionice', '-c', '2', '-n', str(io_weight), '-p', str(pid)]
            subprocess.run(cmd, check=True)
            self.logger.debug(f"Set io_weight={io_weight} cho PID={pid} qua ionice.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi ionice set_io_weight PID={pid}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi set_io_weight PID={pid}: {e}")
            return False

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục Disk I/O => set ionice class=0 (best effort) - đồng bộ.

        :param pid: PID cần khôi phục Disk I/O.
        :return: True nếu thành công, False nếu lỗi.
        """
        try:
            cmd = ['ionice', '-c', '0', '-p', str(pid)]
            subprocess.run(cmd, check=True)
            self.logger.info(f"Khôi phục Disk I/O cho PID={pid} (ionice class=0).")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi ionice restore_resources PID={pid}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi restore_resources Disk cho PID={pid}: {e}")
            return False


###############################################################################
#                       CACHE RESOURCE MANAGER                                #
###############################################################################
class CacheResourceManager:
    """
    Quản lý Cache (đồng bộ).

    Attributes:
        logger (logging.Logger): Logger để ghi log.
        config (Dict[str, Any]): Cấu hình Cache Resource Manager.
        dropped_pids (List[int]): Lưu danh sách PID từng được drop cache.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Khởi tạo CacheResourceManager.

        :param config: Cấu hình Cache (dict).
        :param logger: Logger.
        """
        self.logger = logger
        self.config = config
        self.dropped_pids: List[int] = []

    def drop_caches(self, pid: Optional[int] = None) -> bool:
        """
        Drop caches (đồng bộ).

        :param pid: PID liên quan (nếu muốn lưu thêm vào dropped_pids).
        :return: True nếu thành công, False nếu lỗi.
        """
        try:
            cmd = ['sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches']
            subprocess.run(cmd, check=True)
            self.logger.debug("Đã drop caches.")
            if pid:
                self.dropped_pids.append(pid)
            return True
        except subprocess.CalledProcessError:
            self.logger.error("Không đủ quyền drop_caches hoặc lệnh thất bại.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi drop_caches: {e}")
            return False

    def limit_cache_usage(self, cache_limit_percent: float, pid: Optional[int] = None) -> bool:
        """
        Giới hạn cache => Tối giản: drop caches + log (đồng bộ).
        Chưa có cơ chế kernel-level limit caches.

        :param cache_limit_percent: Tỷ lệ cache limit (0-100).
        :param pid: PID nếu muốn lưu info, mặc định=None.
        :return: True nếu thành công, False nếu lỗi.
        """
        try:
            success = self.drop_caches(pid)
            if not success:
                return False
            self.logger.debug(f"Giới hạn cache => {cache_limit_percent}%. (chưa có logic chi tiết)")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi limit_cache_usage: {e}")
            return False

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục cache => limit_cache_usage(100) (đồng bộ).

        :param pid: PID cần khôi phục cache.
        :return: True nếu thành công, False nếu lỗi.
        """
        try:
            success = self.limit_cache_usage(100.0, pid)
            if success:
                self.logger.info(f"Khôi phục Cache cho PID={pid} => 100%.")
            else:
                self.logger.error(f"Không thể khôi phục Cache cho PID={pid}.")
            return success
        except Exception as e:
            self.logger.error(f"Lỗi restore_resources Cache cho PID={pid}: {e}")
            return False


###############################################################################
#                       MEMORY RESOURCE MANAGER                               #
###############################################################################
class MemoryResourceManager:
    """
    Quản lý Memory qua psutil rlimit (đồng bộ).

    Attributes:
        logger (logging.Logger): Logger để ghi log.
        config (Dict[str, Any]): Cấu hình Memory Resource Manager.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Khởi tạo MemoryResourceManager.

        :param config: Cấu hình Memory (dict).
        :param logger: Logger.
        """
        self.logger = logger
        self.config = config

    def set_memory_limit(self, pid: int, memory_limit_mb: int) -> bool:
        """
        Đặt memory limit (MB) cho tiến trình (đồng bộ).

        :param pid: PID cần giới hạn bộ nhớ.
        :param memory_limit_mb: Giới hạn bộ nhớ (MB).
        :return: True nếu thành công, False nếu lỗi.
        """
        try:
            process = psutil.Process(pid)
            mem_bytes = memory_limit_mb * 1024 * 1024
            process.rlimit(psutil.RLIMIT_AS, (mem_bytes, mem_bytes))
            self.logger.debug(f"Đặt memory_limit={memory_limit_mb}MB cho PID={pid}.")
            return True
        except psutil.NoSuchProcess:
            self.logger.error(f"PID={pid} không tồn tại (set_memory_limit).")
            return False
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền set_memory_limit cho PID={pid}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi set_memory_limit cho PID={pid}: {e}")
            return False

    def get_memory_limit(self, pid: int) -> float:
        """
        Lấy memory limit (bytes) cho tiến trình (đồng bộ).

        :param pid: PID cần kiểm tra limit.
        :return: Giá trị memory limit (bytes), hoặc 0.0 nếu lỗi.
        """
        try:
            process = psutil.Process(pid)
            mem_limit = process.rlimit(psutil.RLIMIT_AS)
            if mem_limit and mem_limit[1] != psutil.RLIM_INFINITY:
                self.logger.debug(f"Memory limit PID={pid}={mem_limit[1]} bytes.")
                return float(mem_limit[1])
            else:
                self.logger.debug(f"PID={pid} không giới hạn bộ nhớ.")
                return float('inf')
        except Exception as e:
            self.logger.error(f"Lỗi get_memory_limit PID={pid}: {e}")
            return 0.0

    def remove_memory_limit(self, pid: int) -> bool:
        """
        Khôi phục memory => không giới hạn (đồng bộ).

        :param pid: PID cần bỏ giới hạn.
        :return: True nếu thành công, False nếu lỗi.
        """
        try:
            process = psutil.Process(pid)
            process.rlimit(psutil.RLIMIT_AS, (psutil.RLIM_INFINITY, psutil.RLIM_INFINITY))
            self.logger.debug(f"Khôi phục memory cho PID={pid} => không giới hạn.")
            return True
        except psutil.NoSuchProcess:
            self.logger.error(f"PID={pid} không tồn tại khi remove_memory_limit.")
            return False
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền remove_memory_limit cho PID={pid}.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi remove_memory_limit cho PID={pid}: {e}")
            return False

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục memory => remove_memory_limit (đồng bộ).

        :param pid: PID cần khôi phục memory.
        :return: True nếu thành công, False nếu lỗi.
        """
        return self.remove_memory_limit(pid)


###############################################################################
#                     RESOURCE CONTROL FACTORY                                #
###############################################################################
class ResourceControlFactory:
    """
    Factory tạo các resource manager (CPU, GPU, Network, Disk I/O, Cache, Memory) theo mô hình đồng bộ.
    """

    @staticmethod
    def create_resource_managers(config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
        """
        Khởi tạo tất cả resource managers theo mô hình đồng bộ.

        :param config: Cấu hình ResourceManager (dict).
        :param logger: Logger dùng để ghi log.
        :return: Dictionary chứa các resource managers.
        """
        resource_managers = {}
        manager_classes = {
            'cpu': CPUResourceManager,
            'gpu': GPUResourceManager,
            'network': NetworkResourceManager,
            'disk_io': DiskIOResourceManager,
            'cache': CacheResourceManager,
            'memory': MemoryResourceManager,
        }

        for name, manager_class in manager_classes.items():
            try:
                logger.info(f"Đang khởi tạo {name} manager...")
                manager_instance = manager_class(config, logger)
                resource_managers[name] = manager_instance
                logger.info(f"{name.capitalize()} manager đã được khởi tạo thành công.")
            except Exception as e:
                logger.error(f"Lỗi khi khởi tạo {name} manager: {e}", exc_info=True)

        if not resource_managers:
            logger.error("Không có resource managers nào được khởi tạo.")
            raise RuntimeError("Tất cả resource managers đều khởi tạo thất bại.")

        logger.info("Tất cả resource managers đã được khởi tạo.")
        return resource_managers
