# azure_log_analytics_client.py

import os
import re
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

from azure.core.exceptions import HttpResponseError
from azure.monitor.query import LogsQueryClient
from azure.identity import ClientSecretCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.resourcegraph import ResourceGraphClient
from azure.mgmt.resourcegraph.models import QueryRequest

# Thêm import cho Log Analytics Management
from azure.mgmt.loganalytics import LogAnalyticsManagementClient
from azure.mgmt.loganalytics.models import Workspace


###############################################################################
# LỚP CƠ SỞ (AzureBaseClient)
###############################################################################
class AzureBaseClient:
    """
    Lớp cơ sở để xử lý xác thực và cấu hình chung cho các client Azure.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
        if not self.subscription_id:
            self.logger.error("AZURE_SUBSCRIPTION_ID không được thiết lập.")
            raise ValueError("AZURE_SUBSCRIPTION_ID không được thiết lập.")

        self.credential = self.authenticate()
        self.resource_graph_client = ResourceGraphClient(self.credential)
        self.resource_management_client = ResourceManagementClient(
            self.credential, 
            self.subscription_id
        )

    def authenticate(self) -> ClientSecretCredential:
        """
        Xác thực với Azure AD bằng ClientSecretCredential (Azure.Identity).
        """
        client_id = os.getenv('AZURE_CLIENT_ID')
        client_secret = os.getenv('AZURE_CLIENT_SECRET')
        tenant_id = os.getenv('AZURE_TENANT_ID')

        if not all([client_id, client_secret, tenant_id]):
            self.logger.error("AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, và AZURE_TENANT_ID phải được thiết lập.")
            raise ValueError("Thiếu thông tin xác thực Azure.")

        try:
            credential = ClientSecretCredential(
                client_id=client_id,
                client_secret=client_secret,
                tenant_id=tenant_id
            )
            self.logger.info("Đã xác thực thành công với Azure AD (ClientSecretCredential).")
            return credential
        except Exception as e:
            self.logger.error(f"Lỗi khi xác thực với Azure AD: {e}")
            raise e

    def discover_resources(self, resource_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Sử dụng Azure Resource Graph để khám phá tài nguyên theo loại.
        VD: 'Microsoft.OperationalInsights/workspaces' để tìm Log Analytics Workspace.
        """
        try:
            query = "Resources"
            if resource_type:
                query += f" | where type =~ '{resource_type}'"
            query += " | project name, type, resourceGroup, id"

            request = QueryRequest(
                subscriptions=[self.subscription_id],
                query=query
            )

            response = self.resource_graph_client.resources(request)
            if not isinstance(response.data, list):
                self.logger.warning("Dữ liệu trả về không phải danh sách.")
                return []

            resources = []
            for res in response.data:
                resources.append({
                    'id': res.get('id', 'N/A'),
                    'name': res.get('name', 'N/A'),
                    'type': res.get('type', 'N/A'),
                    'resourceGroup': res.get('resourceGroup', 'N/A'),
                })

            self.logger.info(f"Đã khám phá {len(resources)} tài nguyên.")
            return resources
        except Exception as e:
            self.logger.error(f"Lỗi khi khám phá tài nguyên: {e}", exc_info=True)
            return []


###############################################################################
# LỚP CHÍNH (AzureLogAnalyticsClient)
###############################################################################
class AzureLogAnalyticsClient(AzureBaseClient):
    """
    Lớp tương tác với Azure Log Analytics:
    - Khám phá workspace bằng Resource Graph
    - Query logs bằng LogsQueryClient
    - Quản trị workspace (lấy info chi tiết) bằng LogAnalyticsManagementClient
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        if not self.credential:
            raise AttributeError("Credential không được định nghĩa trong AzureBaseClient.")

        # Client để query logs
        self.logs_client = LogsQueryClient(self.credential)

        # Client để quản trị (tạo/xoá/lấy thông tin) Log Analytics workspace
        self.log_analytics_mgmt_client = LogAnalyticsManagementClient(
            self.credential, 
            self.subscription_id
        )

        # Danh sách workspace_ids (theo Resource Graph)
        self.workspace_ids = self.get_workspace_ids()

    def get_workspace_ids(self) -> List[str]:
        """
        Lấy danh sách Workspace Resource ID từ tài nguyên Log Analytics 
        (thông qua discover_resources).
        """
        try:
            resources = self.discover_resources('Microsoft.OperationalInsights/workspaces')
            if not resources:
                self.logger.warning("Không tìm thấy Log Analytics Workspace nào.")
                return []

            workspace_ids = [res['id'] for res in resources if 'id' in res]
            self.logger.info(f"Đã tìm thấy {len(workspace_ids)} Log Analytics Workspaces.")
            return workspace_ids
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy Workspace IDs: {e}")
            return []

    def parse_log_analytics_id(self, resource_id: str) -> Dict[str, str]:
        """
        Tách subscription_id, resource_group, workspace_name từ Resource ID.
        Ví dụ:
          /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.OperationalInsights/workspaces/<ws>
        """
        pattern = (
            r"^/subscriptions/(?P<sub>[^/]+)/resourceGroups/(?P<rg>[^/]+)/providers/Microsoft\.OperationalInsights/workspaces/(?P<ws>[^/]+)$"
        )
        match = re.match(pattern, resource_id.strip())
        if not match:
            raise ValueError(f"Resource ID không đúng format: {resource_id}")
        return {
            "subscription_id": match.group("sub"),
            "resource_group": match.group("rg"),
            "workspace_name": match.group("ws")
        }

    def get_workspace_details(self, resource_id: str) -> Optional[Workspace]:
        """
        Từ 1 resource_id, parse ra resource_group & workspace_name,
        rồi gọi LogAnalyticsManagementClient để lấy thông tin chi tiết.
        """
        try:
            parsed = self.parse_log_analytics_id(resource_id)
            resource_group = parsed["resource_group"]
            workspace_name = parsed["workspace_name"]

            ws = self.log_analytics_mgmt_client.workspaces.get(
                resource_group_name=resource_group,
                workspace_name=workspace_name
            )
            self.logger.info(
                f"Đã lấy thông tin workspace '{ws.name}' tại group '{resource_group}' (location={ws.location})."
            )
            return ws
        except Exception as e:
            self.logger.error(f"Không thể get workspace details cho resource_id={resource_id}: {e}")
            return None

    def query_logs(self, query: str, days: int = 7) -> List[Dict[str, Any]]:
        results = []
        if days < 0:
            self.logger.error("days phải >= 0.")
            return results
        if not self.workspace_ids:
            self.logger.error("Không có Workspace nào để truy vấn logs.")
            return results
        if not query:
            self.logger.error("Query không được để trống.")
            return results

        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days)
            timespan = (start_time, end_time)

            for resource_id in self.workspace_ids:
                # 1) Lấy thông tin workspace chi tiết
                ws_details = self.get_workspace_details(resource_id)
                if not ws_details:
                    # get_workspace_details() bị lỗi => skip
                    continue

                # 2) Thay vì dùng resource_id, ta dùng ws_details.customer_id (GUID)
                customer_id = ws_details.customer_id  # dạng "b3ffe677-a68f-4eac-98e6-3d65c71865cc"
                if not customer_id:
                    self.logger.warning(f"Workspace {ws_details.name} không có customer_id.")
                    continue

                # 3) Gọi query_logs dùng GUID
                try:
                    response = self.logs_client.query_workspace(
                        workspace_id=customer_id,
                        query=query,
                        timespan=timespan
                    )
                    if not isinstance(response.tables, list):
                        self.logger.warning(f"Kết quả không hợp lệ từ Workspace GUID {customer_id}.")
                        continue

                    workspace_results = []
                    for table in response.tables:
                        if table.rows:
                            for row in table.rows:
                                row_dict = dict(zip(table.columns, row))
                                workspace_results.append(row_dict)

                    if workspace_results:
                        results.extend(workspace_results)
                        self.logger.info(
                            f"Query thành công trên workspace '{ws_details.name}' (GUID={customer_id})."
                        )
                    else:
                        self.logger.warning(f"Không có dữ liệu trả về từ workspace '{ws_details.name}'.")
                except HttpResponseError as http_error:
                    self.logger.error(
                        f"Lỗi HTTP khi query workspace GUID={customer_id}: {http_error}"
                    )
                except Exception as workspace_error:
                    self.logger.error(
                        f"Lỗi khác trên workspace GUID={customer_id}: {workspace_error}"
                    )

            self.logger.info(f"Tổng cộng lấy được {len(results)} dòng dữ liệu từ tất cả các workspace.")
        except Exception as e:
            self.logger.critical(f"Lỗi nghiêm trọng khi truy vấn logs: {e}", exc_info=True)

        return results

    def query_logs_with_time_range(
        self, query: str, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Truy vấn logs với khoảng thời gian cụ thể trên Azure Log Analytics.
        """
        results = []
        if not self.workspace_ids:
            self.logger.error("Không có Workspace ID để truy vấn logs.")
            return results

        if start_time > end_time:
            self.logger.error("start_time phải <= end_time.")
            return results

        try:
            for resource_id in self.workspace_ids:
                ws_details = self.get_workspace_details(resource_id)
                if not ws_details:
                    continue
                customer_id = ws_details.customer_id
                if not customer_id:
                    self.logger.warning(f"Workspace {ws_details.name} không có customer_id.")
                    continue

                try:
                    response = self.logs_client.query_workspace(
                        workspace_id=customer_id,
                        query=query,
                        timespan=(start_time, end_time)
                    )

                    for table in response.tables:
                        for row in table.rows:
                            row_dict = dict(zip(table.columns, row))
                            results.append(row_dict)

                    self.logger.info(
                        f"Đã truy vấn logs thành công trên Workspace GUID={customer_id}"
                    )
                except Exception as workspace_error:
                    self.logger.error(f"Lỗi trên Workspace GUID={customer_id}: {workspace_error}")

        except Exception as e:
            self.logger.critical(f"Lỗi nghiêm trọng khi truy vấn logs: {e}")
        return results

    def query_aml_logs(self, days: int = 2) -> List[Dict[str, Any]]:
        """
        Truy vấn tất cả Category (thay vì chỉ AML) trong AzureDiagnostics,
        thời gian mặc định = 2 ngày.
        """
        kql = f"""
        AzureDiagnostics
        | where TimeGenerated > ago({days}d)
        // Không lọc Category => lấy tất cả
        | project TimeGenerated, ResourceId, Category, OperationName
        | limit 50
        """
        return self.query_logs(kql, days=days)


###############################################################################
# HƯỚNG DẪN KIỂM THỬ NHANH (runtime)
###############################################################################
if __name__ == "__main__":
    # Thiết lập logger
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(ch)

    try:
        # 1. Khởi tạo client
        azure_log_client = AzureLogAnalyticsClient(logger=logger)

        # 2. In ra danh sách workspace_id (Resource ID đầy đủ) đã khám phá
        if not azure_log_client.workspace_ids:
            logger.warning("Không có workspace nào để kiểm thử.")
        else:
            for idx, ws_id in enumerate(azure_log_client.workspace_ids, start=1):
                logger.info(f"[{idx}] Workspace Resource ID: {ws_id}")

        # 3. Lấy info chi tiết của workspace đầu tiên (nếu có)
        if azure_log_client.workspace_ids:
            first_ws_id = azure_log_client.workspace_ids[0]
            ws_info = azure_log_client.get_workspace_details(first_ws_id)
            if ws_info:
                logger.info(f"Workspace {ws_info.name} có CustomerId (Workspace GUID): {ws_info.customer_id}")

        # 4. Thử query AML logs trong 2 ngày (tất cả Category)
        aml_logs = azure_log_client.query_aml_logs(days=2)
        logger.info(f"Kết quả AML logs (tất cả Category) trong 2 ngày:\n{aml_logs}")

    except Exception as e:
        logger.error(f"Runtime test gặp lỗi: {e}", exc_info=True)
