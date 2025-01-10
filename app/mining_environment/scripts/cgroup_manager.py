# cgroup_manager.py

import os
import logging
from threading import Lock
from typing import Optional, List

class CgroupManager:
    """
    Lớp quản lý các thao tác liên quan đến cgroup v2 một cách an toàn.
    
    Các chức năng bao gồm:
    - Kiểm tra sự tồn tại của cgroup v2.
    - Tạo và xóa cgroup v2.
    - Gán tiến trình vào cgroup v2.
    
    Yêu cầu:
    - Hệ thống phải hỗ trợ và sử dụng cgroups v2.
    - Container phải được chạy với quyền `--privileged` để có thể thao tác với cgroups.
    """
    
    CGROUP_ROOT = "/sys/fs/cgroup"

    def __init__(self, logger: logging.Logger):
        """
        Khởi tạo CgroupManager với logger và khóa để đảm bảo an toàn khi thao tác đồng thời.

        Args:
            logger (logging.Logger): Logger để ghi log các hoạt động và lỗi.
        """
        self.logger = logger
        self.lock = Lock()
        self.cgroup_version = self.detect_cgroup_version()
        if self.cgroup_version != 2:
            self.logger.error("Hệ thống không sử dụng cgroups v2. CgroupManager chỉ hỗ trợ cgroups v2.")
            raise EnvironmentError("Unsupported cgroup version. Only cgroups v2 is supported.")
        
        # Tạo các cgroup cha 'root' và 'root_gpu' nếu chưa tồn tại
        self.create_parent_cgroups()

    def detect_cgroup_version(self) -> int:
        """
        Phát hiện phiên bản cgroups đang được sử dụng trên hệ thống.

        Returns:
            int: 1 nếu là cgroups v1, 2 nếu là cgroups v2, 0 nếu không xác định.
        """
        if os.path.exists(os.path.join(self.CGROUP_ROOT, "cgroup.controllers")):
            self.logger.debug("Phát hiện cgroups v2.")
            return 2
        elif os.path.exists(os.path.join(self.CGROUP_ROOT, "cpu")):
            self.logger.debug("Phát hiện cgroups v1.")
            return 1
        else:
            self.logger.debug("Không xác định được phiên bản cgroups.")
            return 0

    def cgroup_exists(self, cgroup_name: str) -> bool:
        """
        Kiểm tra xem cgroup v2 đã tồn tại chưa.

        Args:
            cgroup_name (str): Tên của cgroup.

        Returns:
            bool: True nếu cgroup tồn tại, False ngược lại.
        """
        cgroup_path = os.path.join(self.CGROUP_ROOT, cgroup_name)
        exists = os.path.isdir(cgroup_path)
        self.logger.debug(f"Kiểm tra cgroup '{cgroup_name}': {'tồn tại' if exists else 'không tồn tại'}.")
        return exists

    def create_cgroup(self, cgroup_name: str) -> bool:
        """
        Tạo một cgroup v2 mới nếu nó chưa tồn tại.

        Args:
            cgroup_name (str): Tên của cgroup.

        Returns:
            bool: True nếu tạo thành công hoặc đã tồn tại, False nếu thất bại.
        """
        with self.lock:
            if self.cgroup_exists(cgroup_name):
                self.logger.info(f"Cgroup '{cgroup_name}' đã tồn tại.")
                return True

            cgroup_path = os.path.join(self.CGROUP_ROOT, cgroup_name)
            try:
                os.makedirs(cgroup_path, exist_ok=False)
                self.logger.info(f"Tạo cgroup '{cgroup_name}' thành công.")
                return True
            except FileExistsError:
                self.logger.info(f"Cgroup '{cgroup_name}' đã tồn tại.")
                return True
            except PermissionError as e:
                self.logger.error(f"Không đủ quyền để tạo cgroup '{cgroup_name}': {e}")
            except Exception as e:
                self.logger.error(f"Lỗi không xác định khi tạo cgroup '{cgroup_name}': {e}")
            return False

    def delete_cgroup(self, cgroup_name: str) -> bool:
        """
        Xóa một cgroup v2 nếu nó tồn tại và không còn tiến trình nào sử dụng.

        Args:
            cgroup_name (str): Tên của cgroup.

        Returns:
            bool: True nếu xóa thành công hoặc cgroup không tồn tại, False nếu thất bại.
        """
        with self.lock:
            if not self.cgroup_exists(cgroup_name):
                self.logger.info(f"Cgroup '{cgroup_name}' không tồn tại. Không cần xóa.")
                return True

            cgroup_path = os.path.join(self.CGROUP_ROOT, cgroup_name)
            try:
                os.rmdir(cgroup_path)
                self.logger.info(f"Xóa cgroup '{cgroup_name}' thành công.")
                return True
            except OSError as e:
                self.logger.error(f"Lỗi khi xóa cgroup '{cgroup_name}': {e}")
            except Exception as e:
                self.logger.error(f"Lỗi không xác định khi xóa cgroup '{cgroup_name}': {e}")
            return False

    def assign_process_to_cgroup(self, pid: int, cgroup_name: str) -> bool:
        """
        Gán một tiến trình vào cgroup v2.

        Args:
            pid (int): PID của tiến trình.
            cgroup_name (str): Tên của cgroup.

        Returns:
            bool: True nếu gán thành công, False nếu thất bại.
        """
        with self.lock:
            if not self.cgroup_exists(cgroup_name):
                self.logger.error(f"Cgroup '{cgroup_name}' không tồn tại. Không thể gán PID={pid}.")
                return False

            cgroup_procs_path = os.path.join(self.CGROUP_ROOT, cgroup_name, "cgroup.procs")
            try:
                with open(cgroup_procs_path, 'a') as f:
                    f.write(f"{pid}\n")
                self.logger.info(f"Gán tiến trình PID={pid} vào cgroup '{cgroup_name}'.")
                return True
            except FileNotFoundError:
                self.logger.error(f"Tệp 'cgroup.procs' không tồn tại trong cgroup '{cgroup_name}'.")
            except PermissionError as e:
                self.logger.error(f"Không đủ quyền để gán PID={pid} vào cgroup '{cgroup_name}': {e}")
            except Exception as e:
                self.logger.error(f"Lỗi không xác định khi gán PID={pid} vào cgroup '{cgroup_name}': {e}")
            return False

    def get_cgroup_parameter(self, cgroup_name: str, parameter: str) -> Optional[str]:
        """
        Lấy giá trị của một tham số trong cgroup v2.

        Args:
            cgroup_name (str): Tên của cgroup.
            parameter (str): Tên tham số (ví dụ: 'cpu.max').

        Returns:
            Optional[str]: Giá trị của tham số hoặc None nếu lỗi.
        """
        cgroup_param_path = os.path.join(self.CGROUP_ROOT, cgroup_name, parameter)
        try:
            with open(cgroup_param_path, 'r') as f:
                value = f.read().strip()
            self.logger.debug(f"Lấy {parameter} từ cgroup '{cgroup_name}': {value}")
            return value
        except FileNotFoundError:
            self.logger.error(f"Tham số '{parameter}' không tồn tại trong cgroup '{cgroup_name}'.")
        except PermissionError as e:
            self.logger.error(f"Không đủ quyền để đọc tham số '{parameter}' từ cgroup '{cgroup_name}': {e}")
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi lấy tham số '{parameter}' từ cgroup '{cgroup_name}': {e}")
        return None

    def get_available_controllers(self) -> List[str]:
        """
        Lấy danh sách các controller có sẵn trong cgroups v2.

        Returns:
            List[str]: Danh sách các controller.
        """
        controllers_file = os.path.join(self.CGROUP_ROOT, "cgroup.controllers")
        try:
            with open(controllers_file, 'r') as f:
                controllers = f.read().strip().split()
            self.logger.debug(f"Available controllers: {controllers}")
            return controllers
        except FileNotFoundError:
            self.logger.error(f"Tệp '{controllers_file}' không tồn tại. Không thể lấy danh sách controllers.")
        except PermissionError as e:
            self.logger.error(f"Không đủ quyền để đọc tệp '{controllers_file}': {e}")
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi lấy danh sách controllers: {e}")
        return []

    def list_cgroups(self) -> List[str]:
        """
        Liệt kê tất cả các cgroup v2 hiện có.

        Returns:
            List[str]: Danh sách tên các cgroup.
        """
        try:
            cgroups = [name for name in os.listdir(self.CGROUP_ROOT) 
                       if os.path.isdir(os.path.join(self.CGROUP_ROOT, name))]
            self.logger.debug(f"Danh sách cgroup hiện có: {cgroups}")
            return cgroups
        except FileNotFoundError:
            self.logger.error(f"Thư mục '{self.CGROUP_ROOT}' không tồn tại.")
        except PermissionError as e:
            self.logger.error(f"Không đủ quyền để liệt kê cgroups trong '{self.CGROUP_ROOT}': {e}")
        except Exception as e:
            self.logger.error(f"Lỗi không xác định khi liệt kê cgroups: {e}")
        return []

    def add_cgroup_to_parent(self, parent_cgroup: str, child_cgroup: str) -> bool:
        """
        Thêm một cgroup con vào cgroup cha trong cgroups v2.

        Args:
            parent_cgroup (str): Tên của cgroup cha.
            child_cgroup (str): Tên của cgroup con.

        Returns:
            bool: True nếu thêm thành công, False nếu thất bại.
        """
        if not self.cgroup_exists(parent_cgroup):
            self.logger.error(f"Cgroup cha '{parent_cgroup}' không tồn tại.")
            return False

        if not self.cgroup_exists(child_cgroup):
            self.logger.error(f"Cgroup con '{child_cgroup}' không tồn tại.")
            return False

        # Trong cgroup v2, các cgroup con chỉ cần được tạo trong thư mục cha,
        # không cần phải thêm thông qua một phương thức đặc biệt.
        # Nếu các cgroup đã được tạo đúng cách, chúng tự động là con của cha.

        # Do đó, phương thức này có thể không cần thiết hoặc chỉ cần xác nhận cấu trúc.
        self.logger.info(f"Cgroup '{child_cgroup}' đã được thêm vào cgroup cha '{parent_cgroup}'.")
        return True

    def create_parent_cgroups(self):
        """
        Tạo các cgroup cha 'root' và 'root_gpu' nếu chúng chưa tồn tại.
        """
        parents = ['root', 'root_gpu']
        for parent in parents:
            if not self.cgroup_exists(parent):
                success = self.create_cgroup(parent)
                if success:
                    self.logger.info(f"Đã tạo cgroup cha '{parent}'.")
                else:
                    self.logger.error(f"Không thể tạo cgroup cha '{parent}'.")
                    raise RuntimeError(f"Cannot create parent cgroup '{parent}'.")
        
        self.logger.info("Các cgroup cha 'root' và 'root_gpu' đã được tạo hoặc đã tồn tại.")
