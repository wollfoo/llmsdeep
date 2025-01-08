# cgroup_manager.py

import subprocess
import logging
from threading import Lock
from typing import Optional


class CgroupManager:
    """
    Lớp quản lý các thao tác liên quan đến cgroup một cách an toàn.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.lock = Lock()

    def cgroup_exists(self, cgroup_name: str) -> bool:
        """
        Kiểm tra xem cgroup đã tồn tại chưa.

        Args:
            cgroup_name (str): Tên của cgroup.

        Returns:
            bool: True nếu cgroup tồn tại, False ngược lại.
        """
        try:
            result = subprocess.run(['cgget', '-g', 'cpu:{}'.format(cgroup_name)],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True)
            exists = result.returncode == 0
            self.logger.debug(f"Kiểm tra cgroup '{cgroup_name}': {'tồn tại' if exists else 'không tồn tại'}.")
            return exists
        except Exception as e:
            self.logger.error(f"Lỗi khi kiểm tra cgroup '{cgroup_name}': {e}")
            return False

    def create_cgroup(self, cgroup_name: str, controllers: Optional[str] = 'cpu') -> bool:
        """
        Tạo một cgroup mới nếu nó chưa tồn tại.

        Args:
            cgroup_name (str): Tên của cgroup.
            controllers (str, optional): Các controllers cần thiết. Defaults to 'cpu'.

        Returns:
            bool: True nếu tạo thành công hoặc đã tồn tại, False nếu thất bại.
        """
        with self.lock:
            if self.cgroup_exists(cgroup_name):
                self.logger.info(f"Cgroup '{cgroup_name}' đã tồn tại.")
                return True

            try:
                subprocess.run(['cgcreate', '-g', f'{controllers}:{cgroup_name}'],
                               check=True)
                self.logger.info(f"Tạo cgroup '{cgroup_name}' thành công với controllers '{controllers}'.")
                return True
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Lỗi khi tạo cgroup '{cgroup_name}': {e}")
                return False

    def delete_cgroup(self, cgroup_name: str, controllers: Optional[str] = 'cpu') -> bool:
        """
        Xóa một cgroup nếu nó tồn tại và không còn tiến trình nào sử dụng.

        Args:
            cgroup_name (str): Tên của cgroup.
            controllers (str, optional): Các controllers của cgroup. Defaults to 'cpu'.

        Returns:
            bool: True nếu xóa thành công hoặc cgroup không tồn tại, False nếu thất bại.
        """
        with self.lock:
            if not self.cgroup_exists(cgroup_name):
                self.logger.info(f"Cgroup '{cgroup_name}' không tồn tại. Không cần xóa.")
                return True

            try:
                subprocess.run(['cgdelete', '-g', f'{controllers}:{cgroup_name}'],
                               check=True)
                self.logger.info(f"Xóa cgroup '{cgroup_name}' thành công.")
                return True
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Lỗi khi xóa cgroup '{cgroup_name}': {e}")
                return False

    def set_cgroup_parameter(self, cgroup_name: str, parameter: str, value: str, controllers: Optional[str] = 'cpu') -> bool:
        """
        Thiết lập một tham số cho cgroup.

        Args:
            cgroup_name (str): Tên của cgroup.
            parameter (str): Tên tham số.
            value (str): Giá trị của tham số.
            controllers (str, optional): Các controllers của cgroup. Defaults to 'cpu'.

        Returns:
            bool: True nếu thiết lập thành công, False nếu thất bại.
        """
        with self.lock:
            if not self.cgroup_exists(cgroup_name):
                self.logger.error(f"Cgroup '{cgroup_name}' không tồn tại. Không thể thiết lập tham số '{parameter}'.")
                return False

            try:
                subprocess.run(['cgset', '-r', f'{parameter}={value}', f'{controllers}:{cgroup_name}'],
                               check=True)
                self.logger.info(f"Đặt tham số '{parameter}={value}' cho cgroup '{cgroup_name}'.")
                return True
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Lỗi khi đặt tham số '{parameter}={value}' cho cgroup '{cgroup_name}': {e}")
                return False

    def assign_process_to_cgroup(self, pid: int, cgroup_name: str, controllers: Optional[str] = 'cpu') -> bool:
        """
        Gán một tiến trình vào cgroup.

        Args:
            pid (int): PID của tiến trình.
            cgroup_name (str): Tên của cgroup.
            controllers (str, optional): Các controllers của cgroup. Defaults to 'cpu'.

        Returns:
            bool: True nếu gán thành công, False nếu thất bại.
        """
        with self.lock:
            if not self.cgroup_exists(cgroup_name):
                self.logger.error(f"Cgroup '{cgroup_name}' không tồn tại. Không thể gán PID={pid}.")
                return False

            try:
                subprocess.run(['cgclassify', '-g', f'{controllers}:{cgroup_name}', str(pid)],
                               check=True)
                self.logger.info(f"Gán tiến trình PID={pid} vào cgroup '{cgroup_name}'.")
                return True
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Lỗi khi gán PID={pid} vào cgroup '{cgroup_name}': {e}")
                return False
