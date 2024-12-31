import os
import logging
from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient
from azure.core.exceptions import HttpResponseError
from datetime import datetime, timedelta, timezone

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PreRunCheck")

class AzureLogAnalyticsClient:
    """
    Lớp để lấy danh sách Workspace GUIDs và thực hiện kiểm tra kết nối.
    """
    def __init__(self, credential, logger):
        self.credential = credential
        self.logger = logger
        self.logs_client = LogsQueryClient(credential)

    def discover_resources(self, resource_type: str):
        """
        Hàm lấy danh sách tài nguyên từ Azure.
        Args:
            resource_type (str): Loại tài nguyên cần lấy.
        Returns:
            List[Dict[str, Any]]: Danh sách tài nguyên.
        """
        from azure.mgmt.resource import ResourceManagementClient
        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        if not subscription_id:
            raise ValueError("Biến môi trường AZURE_SUBSCRIPTION_ID không được định nghĩa.")

        resource_client = ResourceManagementClient(self.credential, subscription_id)
        resources = []
        for resource in resource_client.resources.list():
            if resource.type == resource_type:
                resources.append({"id": resource.id, "name": resource.name})
        return resources

    def get_workspace_ids(self) -> list:
        """
        Lấy danh sách Workspace GUIDs từ tài nguyên Log Analytics.
        """
        try:
            resources = self.discover_resources('Microsoft.OperationalInsights/workspaces')
            # Trích xuất GUID từ Resource ID
            workspace_ids = [res['id'].split('/')[-1] for res in resources if 'id' in res]
            if workspace_ids:
                self.logger.info(f"Đã tìm thấy {len(workspace_ids)} Workspace GUIDs.")
            else:
                self.logger.warning("Không tìm thấy Log Analytics Workspace nào.")
            return workspace_ids
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy danh sách Workspace GUIDs: {e}")
            return []

    def check_workspace_has_data(self, workspace_id: str) -> bool:
        """
        Kiểm tra xem Workspace có chứa dữ liệu Heartbeat hay không.

        Args:
            workspace_id (str): Workspace GUID.

        Returns:
            bool: True nếu Workspace chứa dữ liệu, False nếu không.
        """
        try:
            query = """
            Heartbeat
            | take 1
            """
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=7)

            response = self.logs_client.query_workspace(
                workspace_id=workspace_id,
                query=query,
                timespan=(start_time, end_time),
            )

            if response.tables and response.tables[0].rows:
                self.logger.info(f"Workspace GUID {workspace_id} chứa dữ liệu Heartbeat.")
                return True
            else:
                self.logger.warning(
                    f"Workspace GUID {workspace_id} không chứa dữ liệu Heartbeat. "
                    "Vui lòng kiểm tra cấu hình gửi log từ nguồn hoặc đảm bảo log đã được kích hoạt."
                )
                return False
        except HttpResponseError as e:
            self.logger.error(
                f"Lỗi khi kiểm tra dữ liệu trong Workspace GUID {workspace_id}: {e}\n"
                "Kiểm tra lại quyền truy cập hoặc cấu hình gửi log."
            )
            return False


    def test_logs_query(self, workspace_id: str) -> bool:
        """
        Thực hiện truy vấn mẫu trên Workspace để kiểm tra kết nối.

        Args:
            workspace_id (str): Workspace GUID cần kiểm tra.

        Returns:
            bool: True nếu kết nối thành công, False nếu không.
        """
        try:
            query = """
            Heartbeat
            | summarize count() by bin(TimeGenerated, 1h)
            """
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=1)

            response = self.logs_client.query_workspace(
                workspace_id=workspace_id,  # GUID thay vì Resource ID
                query=query,
                timespan=(start_time, end_time),
            )

            if response.tables and response.tables[0].rows:
                self.logger.info(f"Kết nối thành công đến Workspace GUID: {workspace_id}.")
                return True
            else:
                self.logger.warning(f"Không có dữ liệu trả về từ Workspace GUID: {workspace_id}.")
                return False
        except HttpResponseError as e:
            if e.status_code == 404:
                self.logger.error(
                    f"Lỗi `PathNotFoundError` trên Workspace GUID: {workspace_id}. "
                    "Workspace có thể không tồn tại hoặc chưa chứa dữ liệu log."
                )
            else:
                self.logger.error(f"Lỗi khi truy vấn logs trên Workspace GUID {workspace_id}: {e}")
            return False

def main():
    """
    Kiểm tra cấu hình và kết nối trước khi chạy ứng dụng.
    """
    try:
        logger.info("Bắt đầu kiểm tra trước khi chạy ứng dụng...")

        # Lấy credential từ DefaultAzureCredential
        credential = DefaultAzureCredential()

        # Khởi tạo AzureLogAnalyticsClient
        client = AzureLogAnalyticsClient(credential, logger)

        # Lấy danh sách Workspace IDs
        workspace_ids = client.get_workspace_ids()
        if not workspace_ids:
            logger.error("Không tìm thấy bất kỳ Workspace GUID nào. Kiểm tra lại cấu hình Azure.")
            return

        # Kiểm tra kết nối đến từng Workspace
        for workspace_id in workspace_ids:
            logger.info(f"Kiểm tra dữ liệu trên Workspace GUID: {workspace_id}")
            if not client.check_workspace_has_data(workspace_id):
                logger.warning(f"Bỏ qua Workspace GUID {workspace_id} vì không có dữ liệu Heartbeat.")
                continue
            logger.info(f"Kiểm tra kết nối đến Workspace GUID: {workspace_id}")
            if not client.test_logs_query(workspace_id):
                logger.warning(f"Kết nối thất bại đến Workspace GUID: {workspace_id}. Kiểm tra lại quyền truy cập và cấu hình.")

        logger.info("Kiểm tra hoàn tất.")

    except Exception as e:
        logger.critical(f"Lỗi nghiêm trọng trong kiểm tra: {e}")

if __name__ == "__main__":
    main()
