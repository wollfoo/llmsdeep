# resource_control.py

import os
import uuid
import logging
import subprocess
import re
import psutil
import pynvml
import shutil
from typing import Any, Dict, List, Optional

###############################################################################
#                           CPU RESOURCE MANAGER                              #
###############################################################################

import os
import re
import uuid
import psutil
import shutil
import logging
import subprocess
from typing import Dict, Any, List, Optional

class CPUResourceManager:
    """
    Quản lý tài nguyên CPU sử dụng cgroups (v1), affinity, và tối ưu hóa CPU.

    Attributes:
        logger (logging.Logger): Logger để ghi log.
        config (Dict[str, Any]): Cấu hình cho CPU resource manager.
        CGROUP_CPU_BASE (str): Đường dẫn gốc cgroup CPU (cgroup v1).
        process_cgroup (Dict[int, str]): Bản đồ PID -> tên cgroup.
    """

    # Sử dụng cgroup v1 trên đường dẫn gốc /sys/fs/cgroup/cpu
    CGROUP_CPU_BASE = "/sys/fs/cgroup/cpu"

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Khởi tạo CPUResourceManager.

        :param config: Cấu hình CPU Resource Manager (dict).
        :param logger: Đối tượng Logger.
        """
        self.logger = logger
        self.config = config
        self.process_cgroup: Dict[int, str] = {}

        # Đảm bảo thư mục gốc cho cgroup CPU (v1)
        self.ensure_cgroup_base()

    def ensure_cgroup_base(self) -> None:
        """
        Đảm bảo thư mục gốc cho cgroup CPU (v1) tồn tại.
        Trong trường hợp muốn gom nhóm cgroup riêng, có thể tạo thêm thư mục con.
        """
        try:
            if not os.path.exists(self.CGROUP_CPU_BASE):
                os.makedirs(self.CGROUP_CPU_BASE, exist_ok=True)
                self.logger.debug(f"Tạo thư mục cgroup cơ sở tại {self.CGROUP_CPU_BASE}.")
        except PermissionError:
            self.logger.error(f"Không đủ quyền tạo thư mục cgroup tại {self.CGROUP_CPU_BASE}.")
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo thư mục cgroup tại {self.CGROUP_CPU_BASE}: {e}")

    def throttle_cpu_usage(self, pid: int, throttle_percentage: float, cgroup_name: Optional[str] = None, cores: Optional[List[int]] = None) -> Optional[str]:
        """
        Giới hạn (throttle) CPU cho một tiến trình (PID) thông qua cgroup v1.
        Nếu cgroup đã tồn tại, chỉ cần cập nhật giá trị throttle_percentage.

        :param pid: PID của tiến trình cần giới hạn.
        :param throttle_percentage: Tỷ lệ giới hạn CPU (0-100).
        :param cgroup_name: Tên cgroup, nếu None thì sẽ tạo mới.
        :param cores: Danh sách các core được sử dụng (nếu None, sử dụng toàn bộ CPU logic).
        :return: Tên cgroup (str) nếu thành công, None nếu thất bại.
        """
        try:
            if not (0 <= throttle_percentage <= 100):
                self.logger.error(f"throttle_percentage={throttle_percentage} không hợp lệ (0-100).")
                return None

            # Nếu chưa có cgroup_name, tạo tên cgroup mới
            if not cgroup_name:
                cgroup_name = f"cpu_cloak_{uuid.uuid4().hex[:8]}"
                cgroup_path = os.path.join(self.CGROUP_CPU_BASE, cgroup_name)
                os.makedirs(cgroup_path, exist_ok=True)
                self.logger.debug(f"Tạo cgroup tại {cgroup_path} cho PID={pid}.")

                # Thêm PID vào cgroup
                cgroup_procs_path = os.path.join(cgroup_path, "cgroup.procs")
                with open(cgroup_procs_path, "w") as f:
                    f.write(str(pid))
                self.process_cgroup[pid] = cgroup_name
            else:
                # Nếu cgroup_name đã tồn tại, chỉ cần cập nhật giá trị
                cgroup_path = os.path.join(self.CGROUP_CPU_BASE, cgroup_name)
                if not os.path.exists(cgroup_path):
                    self.logger.error(f"Cgroup {cgroup_name} không tồn tại.")
                    return None

            # Chu kỳ CPU mặc định là 100ms = 100000 microseconds
            cpu_period = 100000

            # Lấy tổng số core từ danh sách cores truyền vào
            if cores is not None:
                n_cores = len(cores)
                if n_cores == 0:
                    self.logger.error("Danh sách cores rỗng, không thể giới hạn CPU.")
                    return None
            else:
                n_cores = psutil.cpu_count(logical=True)  # Mặc định toàn bộ CPU logic

            # Tính toán quota
            if throttle_percentage < 100:
                cpu_quota = int((throttle_percentage / 100.0) * n_cores * cpu_period)
            else:
                cpu_quota = -1  # Không giới hạn

            # Ghi cfs_period_us
            cpu_period_path = os.path.join(cgroup_path, "cpu.cfs_period_us")
            with open(cpu_period_path, "w") as f:
                f.write(str(cpu_period))

            # Ghi cfs_quota_us
            cpu_quota_path = os.path.join(cgroup_path, "cpu.cfs_quota_us")
            with open(cpu_quota_path, "w") as f:
                f.write(str(cpu_quota))

            self.logger.info(
                f"Đặt cpu.cfs_period_us={cpu_period}, cpu.cfs_quota_us={cpu_quota} "
                f"cho cgroup {cgroup_name}, throttle={throttle_percentage}%, cores={cores or 'toàn bộ'}."
            )

            return cgroup_name

        except PermissionError:
            self.logger.error(f"Không đủ quyền tạo hoặc cập nhật cgroup cho PID={pid}.")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo hoặc cập nhật cgroup cho PID={pid}: {e}")
            return None
            
    def delete_cgroup(self, cgroup_name: str) -> bool:
        """
        Xóa một cgroup (v1) đã được tạo.

        :param cgroup_name: Tên cgroup cần xóa.
        :return: True nếu xóa thành công, False nếu thất bại.
        """
        try:
            cgroup_path = os.path.join(self.CGROUP_CPU_BASE, cgroup_name)
            procs_path = os.path.join(cgroup_path, "cgroup.procs")

            # Kiểm tra xem cgroup có còn PID nào không
            if os.path.exists(procs_path):
                with open(procs_path, "r") as f:
                    procs = f.read().strip().splitlines()
                    if procs:
                        self.logger.warning(
                            f"Cgroup {cgroup_name} vẫn chứa các PID: {', '.join(procs)}. Không thể xóa."
                        )
                        return False

            # Kiểm tra và xóa cgroup
            if os.path.exists(cgroup_path):
                try:
                    os.rmdir(cgroup_path)  # Xóa nếu thư mục rỗng
                    self.logger.info(f"Xóa cgroup {cgroup_name} thành công.")
                    return True
                except OSError as e:
                    self.logger.error(f"Lỗi khi xóa thư mục cgroup {cgroup_name}: {e}")
                    return False
            else:
                self.logger.warning(f"Cgroup {cgroup_name} không tồn tại khi xóa.")
                return False

        except PermissionError:
            self.logger.error(f"Không đủ quyền để xóa cgroup {cgroup_name}. Vui lòng kiểm tra quyền truy cập.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi xóa cgroup {cgroup_name}: {e}")
            return False

    def get_available_cpus(self) -> List[int]:
        """
        Lấy danh sách các core CPU (ID) để đặt affinity.

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

    def optimize_thread_scheduling(
        self, pid: int, cores: Optional[List[int]] = None, use_even_odd: Optional[str] = None, cgroup_name: Optional[str] = None) -> bool:
        """
        Đặt CPU affinity và cập nhật CPU mask cho cgroup.

        :param pid: PID của tiến trình.
        :param cores: Danh sách core CPU được chỉ định (ưu tiên nếu có).
        :param use_even_odd: 'even' để chọn cores chẵn, 'odd' để chọn cores lẻ.
        :param cgroup_name: Tên cgroup để cập nhật CPU mask (nếu có).
        :return: True nếu thành công, False nếu thất bại.
        """
        try:
            # Kiểm tra tiến trình tồn tại
            process = psutil.Process(pid)

            # Lấy danh sách các CPU hợp lệ
            available_cpus = self.get_available_cpus()
            if cores:
                # Nếu cores được chỉ định, kiểm tra tính hợp lệ
                if not all(core in available_cpus for core in cores):
                    self.logger.error(
                        f"Danh sách cores {cores} không hợp lệ. Các cores hợp lệ: {available_cpus}."
                    )
                    return False
                target_cores = cores
            else:
                # Xử lý nhóm cores chẵn hoặc lẻ dựa trên use_even_odd
                if use_even_odd == "even":
                    target_cores = [core for core in available_cpus if core % 2 == 0]
                    if not target_cores:
                        self.logger.error("Không tìm thấy cores chẵn.")
                        return False
                    self.logger.info(f"Chọn cores chẵn: {target_cores}")
                elif use_even_odd == "odd":
                    target_cores = [core for core in available_cpus if core % 2 != 0]
                    if not target_cores:
                        self.logger.error("Không tìm thấy cores lẻ.")
                        return False
                    self.logger.info(f"Chọn cores lẻ: {target_cores}")
                else:
                    self.logger.error("Tham số use_even_odd không hợp lệ. Chỉ nhận 'even' hoặc 'odd'.")
                    return False

            # Đặt CPU affinity cho tiến trình
            process.cpu_affinity(target_cores)
            self.logger.info(f"Đặt CPU affinity cho PID={pid} => {target_cores}.")

            # Nếu có cgroup_name, cập nhật cpuset.cpus trong cgroup
            if cgroup_name:
                cgroup_path = os.path.join(self.CGROUP_CPU_BASE, cgroup_name)
                cpuset_cpus_path = os.path.join(cgroup_path, "cpuset.cpus")
                if not os.path.exists(cpuset_cpus_path):
                    self.logger.error(f"cpuset.cpus không tồn tại trong cgroup {cgroup_name}.")
                    return False
                # Ghi danh sách core vào cpuset.cpus
                with open(cpuset_cpus_path, "w") as f:
                    f.write(",".join(map(str, target_cores)))
                self.logger.info(
                    f"Cập nhật cpuset.cpus cho cgroup {cgroup_name} => {target_cores}."
                )

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
        Tối ưu hóa cache CPU (NUMA) cho tiến trình thông qua numactl (nếu hệ thống hỗ trợ).

        :param pid: PID của tiến trình cần tối ưu.
        :return: True nếu thành công, False nếu thất bại.
        """
        try:
            # Kiểm tra numactl đã được cài đặt
            if shutil.which("numactl") is None:
                self.logger.error("numactl không được cài đặt. Bỏ qua tối ưu hóa NUMA.")
                return False

            # Lấy NUMA node tốt nhất
            numa_info = self._get_best_numa_node()
            if not numa_info or "node" not in numa_info or "cpus" not in numa_info:
                self.logger.warning("Không tìm thấy NUMA node phù hợp.")
                return False

            best_numa_node = numa_info["node"]
            numa_node_path = f"/sys/devices/system/node/node{best_numa_node}"

            # Kiểm tra sự tồn tại của NUMA node
            if not os.path.exists(numa_node_path):
                self.logger.error(f"NUMA node {best_numa_node} không tồn tại.")
                return False

            # Đọc danh sách CPUs từ cpulist/cpumap
            try:
                cpulist_path = f"{numa_node_path}/cpulist"
                if os.path.exists(cpulist_path):
                    with open(cpulist_path, "r") as f:
                        cpulist = f.read().strip()
                        node_cpus = self._parse_cpulist(cpulist)
                else:
                    with open(f"{numa_node_path}/cpumap", "r") as f:
                        cpumap = f.read().strip()
                        node_cpus = self._parse_cpumap(cpumap)

                if not node_cpus:
                    self.logger.warning(f"NUMA node {best_numa_node} không có CPUs khả dụng.")
                    return False

                self.logger.debug(f"NUMA node {best_numa_node} có CPUs: {node_cpus}.")

            except FileNotFoundError:
                self.logger.error(f"Tệp cpumap hoặc cpulist không tồn tại trong NUMA node {best_numa_node}.")
                return False
            except Exception as e:
                self.logger.error(f"Lỗi khi đọc CPU từ NUMA node {best_numa_node}: {e}")
                return False

            # Xây dựng lệnh numactl
            command = [
                "numactl",
                "--membind", str(best_numa_node),
                "--cpunodebind", str(best_numa_node),
                "--", "taskset", "-p", str(pid)
            ]

            self.logger.info(f"Áp dụng NUMA policy cho PID={pid}: node={best_numa_node}, CPUs={node_cpus}.")

            # Thực thi lệnh numactl
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                self.logger.info(f"Tối ưu NUMA thành công cho PID={pid}.")
                return True
            else:
                error_msg = result.stderr.strip() or "Không có thông báo lỗi."
                self.logger.error(f"Lỗi numactl (code={result.returncode}): {error_msg}")
                return False

        except psutil.NoSuchProcess:
            self.logger.error(f"PID={pid} không tồn tại.")
            return False
        except psutil.AccessDenied:
            self.logger.error(f"Không đủ quyền truy cập PID={pid}.")
            return False
        except subprocess.TimeoutExpired:
            self.logger.error("numactl timeout sau 10 giây.")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi tối ưu NUMA cho PID={pid}: {e}")
            return False

    def _parse_cpulist(self, cpulist: str) -> List[int]:
        """
        Chuyển đổi chuỗi cpulist (ví dụ: "0-5") sang danh sách CPUs.
        """
        cpus = []
        ranges = cpulist.split(",")
        for r in ranges:
            if "-" in r:
                start, end = map(int, r.split("-"))
                cpus.extend(range(start, end + 1))
            else:
                cpus.append(int(r))
        return cpus

    def _parse_cpumap(self, cpumap: str) -> List[int]:
        """
        Ví dụ chuyển đổi từ cpumap hex (không chi tiết ở đây), tùy dự án mà parse.
        Tạm thời trả về danh sách rỗng hoặc cần custom parse.
        """
        self.logger.warning("Hàm _parse_cpumap chưa được triển khai chi tiết. Trả về rỗng.")
        return []

    def _get_best_numa_node(self) -> Optional[Dict[str, Any]]:
        """
        Lựa chọn NUMA node tốt nhất dựa trên 'numactl --hardware' (nếu hệ thống hỗ trợ).
        Trả về None nếu không khả dụng.
        """
        try:
            result = subprocess.run(
                ["numactl", "--hardware"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode != 0:
                self.logger.error(f"Lỗi khi chạy `numactl --hardware`: {result.stderr.strip()}")
                return None

            output = result.stdout.strip()
            self.logger.debug(f"Đầu ra của `numactl --hardware`:\n{output}")

            numa_nodes = {}
            current_node = None

            for line in output.splitlines():
                line = line.strip()
                if line.startswith("available:"):
                    match = re.search(r"available:\s+(\d+)\s+nodes", line)
                    if match and int(match.group(1)) == 0:
                        self.logger.warning("Hệ thống không hỗ trợ NUMA.")
                        return None

                elif line.startswith("node") and "cpus:" in line:
                    match = re.match(r"node (\d+) cpus:\s*(.*)", line)
                    if match:
                        current_node = int(match.group(1))
                        cpu_list = [int(cpu) for cpu in match.group(2).split()]
                        numa_nodes[current_node] = {"cpus": cpu_list, "size": 0}

                elif current_node is not None and "size:" in line:
                    match = re.match(r"node\s+\d+\s+size:\s*(\d+)\s*MB", line)
                    if match:
                        numa_nodes[current_node]["size"] = int(match.group(1))
                        self.logger.debug(
                            f"NUMA node {current_node} size cập nhật: {numa_nodes[current_node]['size']} MB"
                        )

            self.logger.info(f"NUMA nodes phát hiện: {numa_nodes}")

            if not numa_nodes:
                self.logger.warning("Không tìm thấy NUMA nodes nào.")
                return None

            # Chọn node có bộ nhớ lớn nhất
            best_node = None
            max_memory = 0
            for node, info in numa_nodes.items():
                if info["size"] > max_memory:
                    best_node = node
                    max_memory = info["size"]

            if best_node is not None:
                self.logger.info(f"NUMA node tốt nhất: Node {best_node} (bộ nhớ {max_memory} MB).")
                return {"node": best_node, "cpus": numa_nodes[best_node]["cpus"]}
            else:
                self.logger.warning("Không tìm thấy NUMA node nào phù hợp.")
                return None

        except FileNotFoundError:
            self.logger.error("Lệnh `numactl` không tồn tại. Vui lòng cài đặt numactl.")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi khi chọn NUMA node tốt nhất: {e}")
            return None

    def limit_cpu_for_external_processes(self, target_pids: List[int], throttle_percentage: float) -> bool:
        """
        Giới hạn CPU cho tất cả tiến trình bên ngoài, ngoại trừ danh sách target_pids.

        :param target_pids: Danh sách các PID không bị ảnh hưởng.
        :param throttle_percentage: Tỷ lệ giới hạn CPU (0-100) cho tiến trình bên ngoài.
        :return: True nếu hạn chế thành công, False nếu xảy ra lỗi.
        """
        try:
            if not (0 <= throttle_percentage <= 100):
                self.logger.error(f"throttle_percentage={throttle_percentage} không hợp lệ (0-100).")
                return False

            # Lấy danh sách tất cả PID
            all_pids = [proc.pid for proc in psutil.process_iter(attrs=['pid'])]
            external_pids = set(all_pids) - set(target_pids)

            results = []
            for pid_ in external_pids:
                res = self.throttle_cpu_usage(pid_, throttle_percentage)
                if not res:
                    self.logger.warning(f"Không thể hạn chế CPU cho PID={pid_}.")
                else:
                    results.append(pid_)

            self.logger.info(
                f"Hạn chế CPU cho {len(results)} tiến trình bên ngoài => throttle={throttle_percentage}%."
            )
            return True

        except Exception as e:
            self.logger.error(f"Lỗi khi hạn chế CPU cho external processes: {e}")
            return False

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục tất cả thay đổi tài nguyên CPU đã áp dụng cho tiến trình.

        :param pid: PID của tiến trình cần khôi phục.
        :return: True nếu khôi phục thành công, False nếu lỗi.
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
                self.logger.warning(f"Không tìm thấy cgroup cho PID={pid}.")

            # 2. Gỡ bỏ CPU affinity (mở lại toàn bộ CPU)
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

            # 4. Gỡ bỏ tối ưu NUMA (nếu có)
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
        process_marks (Dict[int, int]): Bản đồ UID -> mark iptables.
    """

    def __init__(self, config: Dict[str, any], logger: logging.Logger):
        """
        Khởi tạo NetworkResourceManager.

        :param config: Cấu hình network (dict).
        :param logger: Logger.
        """
        self.logger = logger
        self.config = config
        self.process_marks: Dict[int, int] = {}

    # ======================
    #  ĐÁNH DẤU GÓI TIN (iptables)
    # ======================

    def mark_packets(self, uid: int, mark: int) -> bool:
        """
        Đánh dấu gói tin chỉ khi quy tắc chưa tồn tại, sử dụng UID.

        :param uid: UID của tiến trình cần đánh dấu gói tin.
        :param mark: Giá trị MARK iptables.
        :return: True nếu thành công, False nếu thất bại.
        """
        try:
            # Kiểm tra nếu đã tồn tại quy tắc
            if self._check_iptables_rule(uid, mark):
                self.logger.debug(f"MARK iptables đã tồn tại cho UID={uid}, mark={mark}.")
                return True

            # Thêm quy tắc iptables
            cmd_add = [
                'iptables', '-A', 'OUTPUT', '-m', 'owner',
                '--uid-owner', str(uid),
                '-j', 'MARK', '--set-mark', str(mark)
            ]
            subprocess.run(cmd_add, check=True)
            self.logger.info(f"Đánh dấu MARK iptables thành công: UID={uid}, mark={mark}.")
            self.process_marks[uid] = mark
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi iptables MARK UID={uid}: {e}")
            return False

    def unmark_packets(self, uid: int, mark: int) -> bool:
        """
        Xóa quy tắc MARK iptables nếu tồn tại.

        :param uid: UID của tiến trình cần xóa quy tắc.
        :param mark: Giá trị MARK iptables.
        :return: True nếu thành công, False nếu thất bại.
        """
        try:
            if not self._check_iptables_rule(uid, mark):
                self.logger.debug(f"Quy tắc MARK không tồn tại cho UID={uid}, mark={mark}.")
                return True

            # Xóa quy tắc iptables
            cmd_del = [
                'iptables', '-D', 'OUTPUT', '-m', 'owner',
                '--uid-owner', str(uid),
                '-j', 'MARK', '--set-mark', str(mark)
            ]
            subprocess.run(cmd_del, check=True)
            self.logger.info(f"Đã xóa MARK iptables: UID={uid}, mark={mark}.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi iptables unMARK UID={uid}: {e}")
            return False

    def _check_iptables_rule(self, uid: int, mark: int) -> bool:
        """
        Kiểm tra xem quy tắc MARK iptables đã tồn tại hay chưa.

        :param uid: UID cần kiểm tra.
        :param mark: Giá trị MARK cần kiểm tra.
        :return: True nếu tồn tại, False nếu không tồn tại.
        """
        cmd_check = [
            'iptables', '-C', 'OUTPUT', '-m', 'owner',
            '--uid-owner', str(uid),
            '-j', 'MARK', '--set-mark', str(mark)
        ]
        return subprocess.run(cmd_check, check=False).returncode == 0

    # ======================
    #  GIỚI HẠN BĂNG THÔNG (tc)
    # ======================

    def limit_bandwidth(self, interface: str, mark: int, bandwidth_mbps: float) -> bool:
        """
        Giới hạn băng thông cho các gói tin được đánh dấu.

        :param interface: Giao diện mạng (vd: eth0).
        :param mark: Giá trị MARK iptables.
        :param bandwidth_mbps: Băng thông tối đa (mbps).
        :return: True nếu thành công, False nếu thất bại.
        """
        try:
            if bandwidth_mbps <= 0:
                self.logger.error("Giới hạn băng thông không hợp lệ.")
                return False

            # Kiểm tra nếu `qdisc` đã tồn tại
            if not self._check_tc_qdisc(interface):
                cmd_qdisc = [
                    'tc', 'qdisc', 'add', 'dev', interface,
                    'root', 'handle', '1:', 'htb', 'default', '12'
                ]
                subprocess.run(cmd_qdisc, check=True)
                self.logger.info(f"Thêm qdisc 'htb' cho {interface}.")

            # Kiểm tra và thêm class
            if not self._check_tc_class(interface, '1:1'):
                cmd_class = [
                    'tc', 'class', 'add', 'dev', interface,
                    'parent', '1:', 'classid', '1:1',
                    'htb', 'rate', f'{bandwidth_mbps}mbit'
                ]
                subprocess.run(cmd_class, check=True)
                self.logger.info(f"Thêm class '1:1' rate={bandwidth_mbps}mbit cho {interface}.")

            # Kiểm tra và thêm filter
            if not self._check_tc_filter(interface, mark):
                cmd_filter = [
                    'tc', 'filter', 'add', 'dev', interface,
                    'protocol', 'ip', 'parent', '1:', 'prio', '1',
                    'handle', str(mark), 'fw', 'flowid', '1:1'
                ]
                subprocess.run(cmd_filter, check=True)
                self.logger.info(f"Thêm filter mark={mark} trên {interface}.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi limit_bandwidth: {e}")
            self.remove_bandwidth_limit(interface, mark)
            return False

    def remove_bandwidth_limit(self, interface: str, mark: int) -> bool:
        """
        Gỡ bỏ giới hạn băng thông trên giao diện.

        :param interface: Giao diện mạng (vd: eth0).
        :param mark: Giá trị MARK iptables.
        :return: True nếu thành công, False nếu thất bại.
        """
        try:
            # Xóa filter
            if self._check_tc_filter(interface, mark):
                cmd_filter = [
                    'tc', 'filter', 'del', 'dev', interface,
                    'protocol', 'ip', 'parent', '1:', 'prio', '1',
                    'handle', str(mark), 'fw', 'flowid', '1:1'
                ]
                subprocess.run(cmd_filter, check=True)
                self.logger.info(f"Xóa filter mark={mark} trên {interface}.")

            # Xóa class
            if self._check_tc_class(interface, '1:1'):
                cmd_class = [
                    'tc', 'class', 'del', 'dev', interface,
                    'parent', '1:', 'classid', '1:1'
                ]
                subprocess.run(cmd_class, check=True)
                self.logger.info(f"Xóa class '1:1' trên {interface}.")

            # Xóa qdisc
            if self._check_tc_qdisc(interface):
                cmd_qdisc = [
                    'tc', 'qdisc', 'del', 'dev', interface,
                    'root', 'handle', '1:', 'htb'
                ]
                subprocess.run(cmd_qdisc, check=True)
                self.logger.info(f"Xóa qdisc 'htb' trên {interface}.")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi remove_bandwidth_limit: {e}")
            return False

    def _check_tc_qdisc(self, interface: str) -> bool:
        cmd_check = ['tc', 'qdisc', 'show', 'dev', interface]
        output = subprocess.check_output(cmd_check, text=True)
        return 'htb' in output

    def _check_tc_class(self, interface: str, classid: str) -> bool:
        cmd_check = ['tc', 'class', 'show', 'dev', interface]
        output = subprocess.check_output(cmd_check, text=True)
        return classid in output

    def _check_tc_filter(self, interface: str, mark: int) -> bool:
        cmd_check = ['tc', 'filter', 'show', 'dev', interface]
        output = subprocess.check_output(cmd_check, text=True)
        return str(mark) in output

    def restore_resources(self, uid: Optional[int] = None) -> bool:
        """
        Khôi phục các tài nguyên mạng liên quan đến UID hoặc tất cả UIDs.
        """
        success = True
        uids_to_restore = [uid] if uid else list(self.process_marks.keys())
        for uid in uids_to_restore:
            mark = self.process_marks.get(uid)
            if mark:
                self.remove_bandwidth_limit(self.config.get("network_interface", "eth0"), mark)
                self.unmark_packets(uid, mark)
                self.process_marks.pop(uid, None)
        return success

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
        :param io_weight: Mức io_weight (0-7 cho Best Effort class).
        :return: True nếu thành công, False nếu lỗi.
        """
        try:
            # Kiểm tra giá trị io_weight hợp lệ
            if not (0 <= io_weight <= 7):
                self.logger.error(f"Giá trị io_weight không hợp lệ: {io_weight}. Hợp lệ: 0-7.")
                return False

            # Kiểm tra tiến trình tồn tại
            if not psutil.pid_exists(pid):
                self.logger.error(f"PID={pid} không tồn tại.")
                return False

            # Lấy thông tin tiến trình để log thêm
            process = psutil.Process(pid)
            process_name = process.name()

            # Xây dựng lệnh
            cmd = ['ionice', '-c', '2', '-n', str(io_weight), '-p', str(pid)]

            # Thực thi lệnh
            subprocess.run(cmd, check=True)
            self.logger.info(f"Set io_weight={io_weight} cho PID={pid} ({process_name}) thành công.")
            self.process_io_limits[pid] = io_weight
            return True

        except psutil.NoSuchProcess:
            self.logger.error(f"Lỗi: PID={pid} không tồn tại.")
            return False
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi ionice set_io_weight PID={pid}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi không xác định trong set_io_weight PID={pid}: {e}\n{traceback.format_exc()}")
            return False

    def restore_resources(self, pid: int) -> bool:
        """
        Khôi phục Disk I/O => set ionice class=0 (best effort) - đồng bộ.

        :param pid: PID cần khôi phục Disk I/O.
        :return: True nếu thành công, False nếu lỗi.
        """
        try:
            # Kiểm tra tiến trình tồn tại
            if not psutil.pid_exists(pid):
                self.logger.error(f"PID={pid} không tồn tại.")
                return False

            # Lấy thông tin tiến trình để log thêm
            process = psutil.Process(pid)
            process_name = process.name()

            # Xây dựng lệnh khôi phục
            cmd = ['ionice', '-c', '0', '-p', str(pid)]

            # Thực thi lệnh
            subprocess.run(cmd, check=True)
            self.logger.info(f"Khôi phục Disk I/O cho PID={pid} ({process_name}) thành công.")
            if pid in self.process_io_limits:
                del self.process_io_limits[pid]
            return True

        except psutil.NoSuchProcess:
            self.logger.error(f"Lỗi: PID={pid} không tồn tại.")
            return False
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Lỗi ionice restore_resources PID={pid}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Lỗi không xác định trong restore_resources PID={pid}: {e}\n{traceback.format_exc()}")
            return False

    def list_io_limits(self) -> Dict[int, float]:
        """
        Liệt kê tất cả các tiến trình và giới hạn I/O hiện tại.

        :return: Bản đồ PID -> io_weight.
        """
        return self.process_io_limits


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
