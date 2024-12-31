# azure_clients.py

import os
import logging
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Union, Tuple

from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
from azure.monitor.query import MetricsQueryClient, MetricAggregationType, LogsQueryClient
from azure.mgmt.security import SecurityCenter
from azure.loganalytics import LogAnalyticsDataClient, models as logmodels
from azure.loganalytics.models import QueryBody
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.resource import ResourceManagementClient
from azure.identity import ClientSecretCredential
from azure.mgmt.resourcegraph import ResourceGraphClient
from azure.mgmt.resourcegraph.models import QueryRequest
from azure.mgmt.machinelearningservices import AzureMachineLearningWorkspaces
from azure.mgmt.loganalytics import LogAnalyticsManagementClient

from azure.ai.anomalydetector import AnomalyDetectorClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.anomalydetector.models import (
    TimeSeriesPoint,
    UnivariateDetectionOptions,
    UnivariateEntireDetectionResult
)

from openai import AzureOpenAI

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

        # Dùng ClientSecretCredential (SDK Azure mới)
        self.credential = self.authenticate()

        # Khởi tạo ResourceGraphClient, ResourceManagementClient
        self.resource_graph_client = ResourceGraphClient(self.credential)
        self.resource_management_client = ResourceManagementClient(self.credential, self.subscription_id)

    def authenticate(self) -> ClientSecretCredential:
        """
        Xác thực với Azure AD bằng ClientSecretCredential (Azure.Identity).
        Không sử dụng hàm cũ signed_session().
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
        Sử dụng Azure Resource Graph để tự động khám phá tài nguyên theo loại.
        Nếu không truyền `resource_type`, hàm sẽ trả về tất cả tài nguyên.

        Args:
            resource_type (Optional[str]): Loại tài nguyên cần tìm kiếm, ví dụ: 
                'Microsoft.Compute/virtualMachines'.
                Nếu None, tìm kiếm tất cả tài nguyên.

        Returns:
            List[Dict[str, Any]]: Danh sách tài nguyên được tìm thấy.
        """
        try:
            # Xây dựng truy vấn
            query = "Resources"
            if resource_type:
                query += f" | where type =~ '{resource_type}'"
            query += " | project name, type, resourceGroup, id"

            # Tạo yêu cầu truy vấn
            request = QueryRequest(
                subscriptions=[self.subscription_id],
                query=query
            )

            # Gửi truy vấn
            response = self.resource_graph_client.resources(request)
            self.logger.debug(f"Response Data: {response.data}")

            # Xử lý kết quả trả về
            resources = []
            for res in response.data or []:  # Xử lý trường hợp response.data là None
                resource_dict = {
                    'id': res.get('id', 'N/A'),
                    'name': res.get('name', 'N/A'),
                    'type': res.get('type', 'N/A'),
                    'resourceGroup': res.get('resourceGroup', 'N/A')
                }
                resources.append(resource_dict)

            self.logger.info(f"Đã khám phá {len(resources)} tài nguyên.")
            return resources
        except Exception as e:
            self.logger.error(f"Lỗi khi khám phá tài nguyên: {e}", exc_info=True)
            return []

class AzureMonitorClient(AzureBaseClient):
    """
    Lớp để tương tác với Azure Monitor.
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.client = MetricsQueryClient(self.credential)


    def get_metrics(
        self,
        resource_id: str,
        metric_names: List[str],
        timespan: Optional[Union[str, Tuple[datetime, datetime]]] = None,
        interval: Optional[str] = "PT1M",
        aggregations: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """
        Lấy metrics từ Azure Monitor cho một tài nguyên cụ thể.

        Args:
            resource_id (str): ID của tài nguyên cần lấy metrics.
            metric_names (List[str]): Danh sách tên metrics cần lấy.
            timespan (Optional[Union[str, Tuple[datetime, datetime]]]): Khoảng thời gian.
            interval (Optional[str]): Khoảng thời gian giữa các điểm dữ liệu (ISO 8601).
            aggregations (Optional[List[str]]): Loại tổng hợp (ví dụ: ["Average", "Maximum"]).

        Returns:
            Dict[str, List[float]]: Metrics theo thời gian.
        """
        if not resource_id or not metric_names:
            self.logger.error("resource_id và metric_names không được để trống.")
            return {}

        # Đảm bảo `timespan` có định dạng chính xác
        try:
            if timespan is None:
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=1)  # Mặc định là 1 giờ qua
                timespan = (start_time, end_time)
            elif isinstance(timespan, str):
                # Chuyển đổi ISO8601 sang tuple nếu cần
                start_time, end_time = map(datetime.fromisoformat, timespan.split("/"))
                timespan = (start_time, end_time)
            elif isinstance(timespan, tuple):
                if not (isinstance(timespan[0], datetime) and isinstance(timespan[1], datetime)):
                    raise ValueError("timespan tuple không hợp lệ. Phải chứa hai đối tượng datetime.")
            else:
                raise ValueError(f"timespan không hợp lệ: {timespan}")

        except Exception as e:
            self.logger.error(f"Lỗi khi xử lý timespan: {e}\nTimespan đầu vào: {timespan}")
            return {}

        # Đảm bảo `aggregations` có giá trị mặc định
        if aggregations is None:
            aggregations = ["Average"]

        try:
            self.logger.info(
                f"Truy vấn metrics cho tài nguyên {resource_id} với các metrics {metric_names}, "
                f"timespan: ({timespan[0]}, {timespan[1]}), interval: {interval}, aggregations: {aggregations}."
            )

            # Gọi API để lấy metrics
            response = self.client.query_resource(
                resource_uri=resource_id,
                metric_names=metric_names,
                timespan=timespan,
                granularity=interval,
                aggregations=aggregations
            )

            # Xử lý kết quả trả về
            metrics = {}
            for metric in response.metrics:
                metrics[metric.name] = [
                    dp.average or dp.total or dp.minimum or dp.maximum or dp.count or 0
                    for ts in metric.timeseries for dp in ts.data
                ]

            self.logger.info(f"Đã lấy thành công metrics cho tài nguyên {resource_id}: {metrics}")
            return metrics

        except Exception as e:
            self.logger.error(
                f"Lỗi khi lấy metrics từ Azure Monitor cho tài nguyên {resource_id}: {e}\n"
                f"Timespan: ({timespan[0]}, {timespan[1]}), Metrics: {metric_names}, Interval: {interval}, Aggregations: {aggregations}\n"
                f"Stack Trace:\n{traceback.format_exc()}"
            )
            return {}



class AzureSentinelClient(AzureBaseClient):
    """
    Lớp để tương tác với Azure Sentinel (thông qua SecurityCenter).
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.security_client = SecurityCenter(self.credential, self.subscription_id)

    def get_recent_alerts(self, days: int = 1) -> List[Any]:
        """
        Lấy các cảnh báo (alerts) gần đây từ Azure Security Center.
        """
        try:
            alerts = self.security_client.alerts.list()
            recent_alerts = []
            cutoff_time = datetime.utcnow() - timedelta(days=days)

            if not isinstance(cutoff_time, datetime):
                raise ValueError(f"cutoff_time không hợp lệ: {cutoff_time}. Phải là kiểu datetime.")

            for alert in alerts:
                if (hasattr(alert, 'properties') 
                    and hasattr(alert.properties, 'created_time') 
                    and alert.properties.created_time >= cutoff_time):
                    recent_alerts.append(alert)

            self.logger.info(f"Đã lấy {len(recent_alerts)} alerts từ Azure Sentinel trong {days} ngày gần đây.")
            return recent_alerts
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy alerts từ Azure Sentinel: {e}")
            return []

class AzureLogAnalyticsClient(AzureBaseClient):
    """
    Lớp để tương tác với Azure Log Analytics bằng LogsQueryClient.
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.logs_client = LogsQueryClient(self.credential)
        self.workspace_ids = self.get_workspace_ids()

    def get_workspace_ids(self) -> List[str]:
        """
        Lấy danh sách Workspace IDs từ tài nguyên Log Analytics.

        Returns:
            List[str]: Danh sách Workspace IDs.
        """
        try:
            resources = self.discover_resources('Microsoft.OperationalInsights/workspaces')
            workspace_ids = [res['id'] for res in resources if 'id' in res]
            if workspace_ids:
                self.logger.info(f"Đã tìm thấy {len(workspace_ids)} Log Analytics Workspaces.")
            else:
                self.logger.warning("Không tìm thấy Log Analytics Workspace nào.")
            return workspace_ids
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy Workspace IDs: {e}")
            return []


    def query_logs(self, query: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Thực hiện truy vấn log trên Azure Log Analytics.

        Args:
            query (str): Câu lệnh KQL để truy vấn logs.
            days (int): Khoảng thời gian truy vấn (mặc định là 7 ngày).

        Returns:
            List[Dict[str, Any]]: Danh sách logs kết quả.
        """
        results = []

        # Kiểm tra workspace IDs
        if not self.workspace_ids:
            self.logger.error("Không có Workspace ID để truy vấn logs.")
            return results

        # Kiểm tra query
        if not query:
            self.logger.error("Query không được để trống.")
            return results

        try:
            # Tính toán thời gian truy vấn
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days)

            # Xác thực timespan
            timespan = (start_time, end_time)
            if not (isinstance(start_time, datetime) and isinstance(end_time, datetime)):
                raise ValueError(
                    f"timespan không hợp lệ: start_time={start_time}, end_time={end_time}. "
                    "Cả hai phải là kiểu datetime."
                )

            # Duyệt qua các workspace
            for workspace_id in self.workspace_ids:
                try:
                    self.logger.debug(
                        f"Truy vấn logs trên Workspace ID {workspace_id} với timespan: {timespan}."
                    )

                    # Gọi API để lấy logs
                    response = self.logs_client.query_workspace(
                        workspace_id=workspace_id,
                        query=query,
                        timespan=timespan
                    )

                    # Xử lý kết quả trả về
                    workspace_results = []
                    for table in response.tables:
                        if table.rows:
                            workspace_results.extend(
                                [dict(zip(table.columns, row)) for row in table.rows]
                            )

                    if workspace_results:
                        results.extend(workspace_results)
                        self.logger.info(
                            f"Đã truy vấn thành công trên Workspace ID {workspace_id}. "
                            f"Lấy được {len(workspace_results)} dòng dữ liệu."
                        )
                    else:
                        self.logger.warning(
                            f"Không có dữ liệu trả về từ Workspace ID {workspace_id}."
                        )

                except ResourceNotFoundError as not_found_error:
                    self.logger.error(
                        f"Workspace không tồn tại hoặc không có dữ liệu trên Workspace ID {workspace_id}: {not_found_error}\n"
                        f"Query: {query}, Timespan: {timespan}\n"
                    )
                except HttpResponseError as http_error:
                    self.logger.error(
                        f"Lỗi HTTP trên Workspace ID {workspace_id}: {http_error}\n"
                        f"Query: {query}, Timespan: {timespan}\n"
                    )
                except Exception as workspace_error:
                    self.logger.error(
                        f"Lỗi không xác định trên Workspace ID {workspace_id}: {workspace_error}\n"
                        f"Query: {query}, Timespan: {timespan}\n"
                        f"Stack Trace:\n{traceback.format_exc()}"
                    )

            self.logger.info(f"Tổng cộng lấy được {len(results)} dòng dữ liệu từ tất cả các workspace.")
            return results

        except ValueError as ve:
            self.logger.error(
                f"Lỗi xác thực thời gian trong query_logs: {ve}\n"
                f"Query: {query}, Days: {days}"
            )
        except Exception as e:
            self.logger.critical(
                f"Lỗi nghiêm trọng khi thực hiện truy vấn logs: {e}\n"
                f"Query: {query}, Days: {days}, Timespan: ({start_time}, {end_time})\n"
                f"Stack Trace:\n{traceback.format_exc()}"
            )

        return []

    def query_logs_with_time_range(
        self, query: str, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Truy vấn logs với khoảng thời gian cụ thể trên Azure Log Analytics.

        Args:
            query (str): Câu lệnh KQL để truy vấn logs.
            start_time (datetime): Thời gian bắt đầu.
            end_time (datetime): Thời gian kết thúc.

        Returns:
            List[Dict[str, Any]]: Danh sách logs kết quả.
        """
        results = []
        if not self.workspace_ids:
            self.logger.error("Không có Workspace ID để truy vấn logs.")
            return results

        try:
            # Kiểm tra định dạng của start_time và end_time
            if not isinstance(start_time, datetime) or not isinstance(end_time, datetime):
                raise ValueError(
                    f"start_time hoặc end_time không hợp lệ: start_time={start_time}, end_time={end_time}. "
                    "Cả hai phải là kiểu datetime."
                )

            for workspace_id in self.workspace_ids:
                try:
                    # Thực hiện truy vấn
                    response = self.logs_client.query_workspace(
                        workspace_id=workspace_id,
                        query=query,
                        timespan=(start_time, end_time)  # Tuple định dạng đúng
                    )

                    # Xử lý kết quả trả về
                    for table in response.tables:
                        for row in table.rows:
                            results.append(dict(zip(table.columns, row)))

                    self.logger.info(
                        f"Đã truy vấn logs thành công trên Workspace ID: {workspace_id} "
                        f"với timespan từ {start_time.isoformat()} đến {end_time.isoformat()}."
                    )

                except Exception as workspace_error:
                    self.logger.error(
                        f"Lỗi khi truy vấn logs trên Workspace ID {workspace_id}: {workspace_error}\n"
                        f"Query: {query}, Timespan: ({start_time}, {end_time})\n"
                        f"Stack Trace:\n{traceback.format_exc()}"
                    )

            return results

        except ValueError as ve:
            self.logger.error(
                f"Lỗi xác thực thời gian trong query_logs_with_time_range: {ve}\n"
                f"start_time={start_time}, end_time={end_time}"
            )
        except Exception as e:
            self.logger.critical(
                f"Lỗi nghiêm trọng khi truy vấn logs với khoảng thời gian cụ thể: {e}\n"
                f"Query: {query}, Timespan: ({start_time}, {end_time})\n"
                f"Stack Trace:\n{traceback.format_exc()}"
            )
        return []

class AzureSecurityCenterClient(AzureBaseClient):
    """
    Lớp để tương tác với Azure Security Center.
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.security_client = SecurityCenter(self.credential, self.subscription_id)

    def get_security_assessments(self) -> List[Any]:
        """
        Lấy các đánh giá bảo mật từ Azure Security Center.
        """
        try:
            scope = f"/subscriptions/{self.subscription_id}"  # Phạm vi subscription
            self.logger.info(f"Lấy đánh giá bảo mật cho scope: {scope}")
            assessments = self.security_client.assessments.list(scope=scope)
            assessments_list = list(assessments)
            self.logger.info(f"Đã lấy {len(assessments_list)} đánh giá bảo mật từ Azure Security Center.")
            return assessments_list
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy đánh giá bảo mật từ Azure Security Center: {e}")
            return []

    def get_secure_scores(self) -> List[Any]:
        """
        Lấy điểm bảo mật từ Azure Security Center.
        """
        try:
            self.logger.info("Lấy điểm bảo mật...")
            secure_scores = self.security_client.secure_scores.list()
            secure_scores_list = list(secure_scores)
            self.logger.info(f"Đã lấy {len(secure_scores_list)} điểm bảo mật từ Azure Security Center.")
            return secure_scores_list
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy điểm bảo mật từ Azure Security Center: {e}")
            return []

class AzureNetworkWatcherClient(AzureBaseClient):
    """
    Lớp để tương tác với Azure Network Watcher.
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.network_client = NetworkManagementClient(self.credential, self.subscription_id)

    def get_network_watcher_name(self, resource_group: str) -> Optional[str]:
        """
        Lấy tên của Network Watcher dựa trên Resource Group.

        Args:
            resource_group (str): Tên Resource Group.

        Returns:
            Optional[str]: Tên Network Watcher nếu tìm thấy, None nếu không.
        """
        try:
            region = self.resource_management_client.resource_groups.get(resource_group).location
            watchers = self.network_client.network_watchers.list_all()
            for watcher in watchers:
                if watcher.location.lower() == region.lower():
                    self.logger.info(f"Đã tìm thấy Network Watcher '{watcher.name}' cho vùng '{region}'.")
                    return watcher.name
            self.logger.warning(f"Không tìm thấy Network Watcher cho vùng '{region}'.")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi khi tìm Network Watcher cho Resource Group '{resource_group}': {e}")
            return None

    def get_flow_logs(self, network_watcher_resource_group: str, network_watcher_name: str, nsg_name: str) -> List[Any]:
        """
        Lấy các flow logs từ Network Security Groups (NSGs) thông qua Azure Network Watcher.

        Args:
            network_watcher_resource_group (str): Tên Resource Group chứa Network Watcher.
            network_watcher_name (str): Tên Network Watcher.
            nsg_name (str): Tên Network Security Group.

        Returns:
            List[Any]: Danh sách Flow Logs.
        """
        try:
            flow_log_configurations = self.network_client.flow_logs.list(
                resource_group_name=network_watcher_resource_group,
                network_watcher_name=network_watcher_name
            )
            flow_logs = [
                log for log in flow_log_configurations 
                if hasattr(log, 'target_resource_id') and log.target_resource_id.endswith(nsg_name)
            ]
            self.logger.info(
                f"Đã lấy {len(flow_logs)} flow logs từ NSG {nsg_name} trong Resource Group {network_watcher_resource_group}."
            )
            return flow_logs
        except Exception as e:
            self.logger.error(
                f"Lỗi khi lấy flow logs từ NSG {nsg_name} trong Resource Group {network_watcher_resource_group}: {e}"
            )
            return []

    def create_flow_log(
        self, resource_group: str, network_watcher_name: str, nsg_name: str, flow_log_name: str,
        params: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Tạo một flow log mới cho một NSG.

        Args:
            resource_group (str): Resource Group của NSG.
            network_watcher_name (str): Tên Network Watcher.
            nsg_name (str): Tên NSG.
            flow_log_name (str): Tên Flow Log.
            params (Dict[str, Any]): Cấu hình Flow Log.

        Returns:
            Optional[Any]: Thông tin Flow Log vừa tạo.
        """
        try:
            flow_log = self.network_client.flow_logs.begin_create_or_update(
                resource_group_name=resource_group,
                network_security_group_name=nsg_name,
                flow_log_name=flow_log_name,
                parameters=params
            ).result()
            self.logger.info(
                f"Đã tạo flow log {flow_log_name} cho NSG {nsg_name} trong Resource Group {resource_group}."
            )
            return flow_log
        except Exception as e:
            self.logger.error(
                f"Lỗi khi tạo flow log {flow_log_name} cho NSG {nsg_name} trong Resource Group {resource_group}: {e}"
            )
            return None

    def delete_flow_log(
        self, resource_group: str, network_watcher_name: str, nsg_name: str, flow_log_name: str
    ) -> bool:
        """
        Xóa một flow log từ NSG.

        Args:
            resource_group (str): Resource Group của NSG.
            network_watcher_name (str): Tên Network Watcher.
            nsg_name (str): Tên NSG.
            flow_log_name (str): Tên Flow Log.

        Returns:
            bool: True nếu xóa thành công, False nếu không.
        """
        try:
            self.network_client.flow_logs.begin_delete(
                resource_group_name=resource_group,
                network_security_group_name=nsg_name,
                flow_log_name=flow_log_name
            ).result()
            self.logger.info(
                f"Đã xóa flow log {flow_log_name} từ NSG {nsg_name} trong Resource Group {resource_group}."
            )
            return True
        except Exception as e:
            self.logger.error(
                f"Lỗi khi xóa flow log {flow_log_name} từ NSG {nsg_name} trong Resource Group {resource_group}: {e}"
            )
            return False

    def check_flow_log_status(self, network_watcher_resource_group: str, network_watcher_name: str, flow_log_name: str) -> Optional[Dict[str, Any]]:
        """
        Kiểm tra trạng thái của Flow Log.

        Args:
            network_watcher_resource_group (str): Resource Group của Network Watcher.
            network_watcher_name (str): Tên Network Watcher.
            flow_log_name (str): Tên Flow Log.

        Returns:
            Optional[Dict[str, Any]]: Thông tin trạng thái Flow Log, None nếu không tìm thấy.
        """
        try:
            flow_log = self.network_client.flow_logs.get(
                resource_group_name=network_watcher_resource_group,
                network_watcher_name=network_watcher_name,
                flow_log_name=flow_log_name
            )
            self.logger.info(f"Trạng thái Flow Log {flow_log_name}: {flow_log.provisioning_state}")
            return {
                "id": flow_log.id,
                "state": flow_log.provisioning_state,
                "storageId": flow_log.storage_id,
                "targetResourceId": flow_log.target_resource_id
            }
        except Exception as e:
            self.logger.error(f"Lỗi khi kiểm tra trạng thái Flow Log {flow_log_name}: {e}")
            return None

class AzureTrafficAnalyticsClient(AzureBaseClient):
    """
    Lớp để tương tác với Azure Traffic Analytics sử dụng azure-loganalytics và azure-mgmt-network.
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.log_analytics_client = LogAnalyticsDataClient(self.credential)
        self.network_client = NetworkManagementClient(self.credential, self.subscription_id)
        self.workspace_ids = self.get_traffic_workspace_ids()

    def get_traffic_workspace_ids(self) -> List[str]:
        try:
            resources = self.discover_resources('Microsoft.OperationalInsights/workspaces')
            traffic_workspace_ids = [
                res['id'] for res in resources
                if 'id' in res and 'trafficanalytics' in res.get('name', '').lower()
            ]
            if traffic_workspace_ids:
                self.logger.info(f"Đã tìm thấy {len(traffic_workspace_ids)} Traffic Analytics Workspaces.")
            else:
                self.logger.warning("Không tìm thấy Traffic Analytics Workspace nào.")
            return traffic_workspace_ids
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy Workspace IDs của Traffic Analytics: {e}")
            return []

    def get_traffic_data(
        self, query: Optional[str] = None, timespan: Optional[str] = "P1D"
    ) -> List[Any]:
        """
        Lấy dữ liệu Traffic Analytics từ Azure Log Analytics.
        """
        results = []
        if not self.workspace_ids:
            self.logger.error("Không có Workspace ID của Traffic Analytics để lấy dữ liệu.")
            return results
        try:
            if not query:
                query = """
                AzureNetworkAnalytics_CL
                | where TimeGenerated > ago(1d)
                | summarize Count = count() by bin(TimeGenerated, 1h), SourceIP_s, DestinationIP_s
                """
            for workspace_id in self.workspace_ids:
                if isinstance(timespan, str):
                    # Kiểm tra định dạng timespan nếu sử dụng chuỗi
                    if not timespan.startswith("P") and not timespan.endswith("D"):
                        raise ValueError(f"timespan không hợp lệ: {timespan}. Phải tuân theo ISO8601.")
                
                body = logmodels.QueryBody(query=query, timespan=timespan)
                response = self.log_analytics_client.query(workspace_id=workspace_id, body=body)
                
                if response.tables:
                    results.extend(response.tables)
                    self.logger.info(
                        f"Đã lấy dữ liệu Traffic Analytics thành công từ Workspace ID: {workspace_id}."
                    )
                else:
                    self.logger.info(f"Không có dữ liệu trả về từ Workspace ID: {workspace_id}.")
            return results
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy dữ liệu Traffic Analytics: {e}")
            return []

    def enable_traffic_analytics(
        self,
        resource_group: str,
        network_watcher_name: str,
        nsg_name: str,
        workspace_resource_id: str,
        storage_account_id: str,
        retention_days: int = 7
    ) -> bool:
        """
        Bật Traffic Analytics cho một NSG bằng cách tạo hoặc cập nhật Flow Logs.
        """
        try:
            parameters = {
                "location": self.get_nsg_location(resource_group, nsg_name),
                "enabled": True,
                "storageId": storage_account_id,
                "retentionPolicy": {
                    "days": retention_days,
                    "enabled": True
                },
                "format": "JSON",
                "flowAnalyticsConfiguration": {
                    "networkWatcherFlowAnalyticsConfiguration": {
                        "enabled": True,
                        "workspaceId": workspace_resource_id,
                        "workspaceRegion": self.get_workspace_region(workspace_resource_id),
                        "trafficAnalyticsInterval": 10
                    }
                }
            }
            self.network_client.flow_logs.begin_create_or_update(
                resource_group_name=resource_group,
                network_security_group_name=nsg_name,
                flow_log_name=f"{nsg_name}-flowlog",
                parameters=parameters
            ).result()
            self.logger.info(f"Đã bật Traffic Analytics cho NSG {nsg_name} trong Resource Group {resource_group}.")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi bật Traffic Analytics cho NSG {nsg_name}: {e}")
            return False

    def disable_traffic_analytics(
        self, resource_group: str, network_watcher_name: str, nsg_name: str
    ) -> bool:
        """
        Tắt Traffic Analytics cho một NSG bằng cách xóa Flow Logs.
        """
        try:
            self.network_client.flow_logs.begin_delete(
                resource_group_name=resource_group,
                network_security_group_name=nsg_name,
                flow_log_name=f"{nsg_name}-flowlog"
            ).result()
            self.logger.info(f"Đã tắt Traffic Analytics cho NSG {nsg_name} trong Resource Group {resource_group}.")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi tắt Traffic Analytics cho NSG {nsg_name}: {e}")
            return False

    def get_nsg_location(self, resource_group: str, nsg_name: str) -> str:
        """
        Lấy vị trí của Network Security Group.
        """
        try:
            nsg = self.network_client.network_security_groups.get(resource_group, nsg_name)
            return nsg.location
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy vị trí của NSG {nsg_name}: {e}")
            return "eastus"

    def get_workspace_region(self, workspace_resource_id: str) -> str:
        """
        Lấy khu vực của Log Analytics Workspace.
        """
        try:
            workspace = self.resource_management_client.resources.get_by_id(
                workspace_resource_id,
                '2015-11-01-preview'
            )
            return workspace.location
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy khu vực của Workspace {workspace_resource_id}: {e}")
            return "eastus"

class AzureMLClient(AzureBaseClient):
    """
    Lớp để tương tác với Azure Machine Learning Clusters.
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.ml_client = AzureMachineLearningWorkspaces(self.credential, self.subscription_id)
        self.compute_resource_type = 'Microsoft.MachineLearningServices/workspaces/computes'

    def discover_ml_clusters(self) -> List[Dict[str, Any]]:
        try:
            resources = self.discover_resources(self.compute_resource_type)
            self.logger.info(f"Đã khám phá {len(resources)} Azure ML Clusters.")
            return resources
        except Exception as e:
            self.logger.error(f"Lỗi khi khám phá Azure ML Clusters: {e}")
            return []

    def get_ml_cluster_metrics(
        self, 
        compute_id: str, 
        metric_names: List[str],
        timespan: Optional[str] = 'PT1H', 
        interval: Optional[str] = 'PT1M'
    ) -> Dict[str, List[float]]:
        if not compute_id or not metric_names:
            self.logger.error("compute_id và metric_names không được để trống.")
            return {}
        try:
            monitor_client = MetricsQueryClient(self.credential)
            metrics_data = monitor_client.list(
                resource_id=compute_id,
                metric_names=metric_names,
                timespan=timespan,
                interval=interval,
                aggregations=[MetricAggregationType.AVERAGE]
            )
            metrics = {}
            for metric in metrics_data.metrics:
                metrics[metric.name] = []
                for ts in metric.timeseries:
                    for dp in ts.data:
                        value = dp.average or dp.total or dp.minimum or dp.maximum or dp.count or 0
                        metrics[metric.name].append(value)
            self.logger.info(f"Đã lấy metrics cho ML Cluster {compute_id}: {metric_names}")
            return metrics
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy metrics từ ML Cluster {compute_id}: {e}")
            return {}

class AzureAnomalyDetectorClient:
    """
    Lớp để tương tác với Azure Anomaly Detector.
    """
    def __init__(self, logger: logging.Logger, config: Dict[str, Any]):
        self.logger = logger
        self.endpoint = config.get("azure_anomaly_detector", {}).get("api_base")
        self.api_key = os.getenv("ANOMALY_DETECTOR_API_KEY")

        self.logger.debug(f"Azure Anomaly Detector Endpoint: {self.endpoint}")
        self.logger.debug(f"Azure Anomaly Detector API Key: {'***' if self.api_key else None}")

        if not self.endpoint or not self.api_key:
            self.logger.error("Thông tin endpoint hoặc api_key cho Azure Anomaly Detector không được thiết lập.")
            raise ValueError("Thiếu thông tin cấu hình cho Azure Anomaly Detector.")

        try:
            self.client = AnomalyDetectorClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key)
            )
            self.logger.info("Đã kết nối thành công với Azure Anomaly Detector.")
        except Exception as e:
            self.logger.error(f"Lỗi khi kết nối với Azure Anomaly Detector: {e}")
            raise e

    def detect_anomalies(self, metric_data: Dict[str, Any]) -> bool:
        """
        Phát hiện bất thường dựa trên dữ liệu metrics gửi đến.
        """
        try:
            for pid, metrics in metric_data.items():
                if not isinstance(metrics, dict):
                    self.logger.warning(f"Metric data cho PID {pid} không phải dict, bỏ qua.")
                    continue

                cpu_usage = metrics.get('cpu_usage', [])
                if not isinstance(cpu_usage, list):
                    self.logger.warning(f"cpu_usage cho PID {pid} không phải list, bỏ qua.")
                    continue
                if len(cpu_usage) < 12:
                    self.logger.warning(f"Tiến trình PID {pid} có ít hơn 12 điểm dữ liệu, bỏ qua.")
                    continue

                series = []
                for i, usage in enumerate(cpu_usage):
                    if not isinstance(usage, (int, float)):
                        self.logger.warning(f"usage={usage} cho PID {pid} không phải kiểu số, bỏ qua điểm dữ liệu.")
                        continue
                    timestamp = datetime.utcnow() - timedelta(minutes=len(cpu_usage) - i)
                    if not isinstance(timestamp, datetime):
                        raise ValueError(f"timestamp không hợp lệ: {timestamp}. Phải là kiểu datetime.")
                    series.append(TimeSeriesPoint(timestamp=timestamp.isoformat(), value=usage))

                options = UnivariateDetectionOptions(
                    series=series,
                    granularity="minutely",
                    sensitivity=95
                )
                response: UnivariateEntireDetectionResult = self.client.detect_univariate_entire_series(options=options)

                if any(response.is_anomaly):
                    self.logger.warning(f"Phát hiện bất thường trong tiến trình PID {pid}.")
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi phát hiện bất thường với Azure Anomaly Detector: {e}")
            return False

    def validate_configuration(self):
        """
        Xác thực cấu hình trước khi sử dụng client.
        """
        if not self.endpoint or not self.api_key:
            self.logger.error("Endpoint hoặc API Key chưa được cấu hình đúng cách.")
            raise ValueError("Cấu hình Azure Anomaly Detector không đầy đủ.")
        self.logger.info("Cấu hình Azure Anomaly Detector đã được xác thực thành công.")

    def log_configuration(self):
        """
        Ghi log thông tin cấu hình để hỗ trợ gỡ lỗi.
        """
        self.logger.debug(f"Endpoint: {self.endpoint}")
        self.logger.debug(f"API Key: {'***' if self.api_key else None}")

class AzureOpenAIClient(AzureBaseClient):
    """
    Lớp để tương tác với Azure OpenAI Service.
    """
    def __init__(self, logger: logging.Logger, config: Dict[str, Any]):
        super().__init__(logger)
        self.endpoint = config.get("azure_openai", {}).get("api_base")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.deployment_name = config.get("azure_openai", {}).get("deployment_name")
        self.api_version = config.get("azure_openai", {}).get("api_version", "2023-03-15-preview")

        if not self.api_key:
            self.logger.error("API key cho Azure OpenAI không được thiết lập trong biến môi trường.")
            raise ValueError("Thiếu thông tin API key trong biến môi trường OPENAI_API_KEY.")

        if not self.endpoint:
            self.logger.error("Endpoint cho Azure OpenAI không được thiết lập trong config.")
            raise ValueError("Thiếu thông tin endpoint trong cấu hình.")

        self.initialize_openai()

    def initialize_openai(self):
        if not self.endpoint or not self.api_key or not self.deployment_name:
            self.logger.error("Thiếu thông tin cấu hình cho Azure OpenAI.")
            raise ValueError("Thiếu thông tin endpoint, api_key hoặc deployment_name.")

        try:
            self.client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_version=self.api_version,
                api_key=self.api_key
            )
            self.logger.info("Đã cấu hình thành công Azure OpenAI Service.")
        except Exception as e:
            self.logger.error(f"Lỗi khi cấu hình Azure OpenAI Service: {e}")
            raise e

    def get_optimization_suggestions(self, state_data: Dict[str, Any]) -> List[float]:
        """
        Gửi dữ liệu trạng thái hệ thống đến Azure OpenAI và nhận gợi ý tối ưu hóa.
        """
        try:
            prompt = self.construct_prompt(state_data)
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "Bạn là một chuyên gia tối ưu hóa tài nguyên hệ thống."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.5
            )
            suggestion_text = response.choices[0].message.content.strip()
            suggestions = []
            if isinstance(suggestion_text, str) and suggestion_text:
                for x in suggestion_text.split(','):
                    try:
                        suggestions.append(float(x.strip()))
                    except ValueError:
                        self.logger.warning(f"Không parse được '{x.strip()}' thành float, bỏ qua.")
            self.logger.info(f"Nhận được gợi ý tối ưu hóa từ Azure OpenAI: {suggestions}")
            return suggestions
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy gợi ý từ Azure OpenAI Service: {e}")
            return []

    def construct_prompt(self, state_data: Dict[str, Any]) -> str:
        """
        Xây dựng prompt gửi đến OpenAI dựa trên dữ liệu trạng thái hệ thống.
        """
        prompt = "Dựa trên các thông số hệ thống sau đây, đề xuất các điều chỉnh tối ưu hóa tài nguyên:\n"
        for pid, metrics in state_data.items():
            if not isinstance(metrics, dict):
                self.logger.warning(f"metrics cho PID {pid} không phải dict, bỏ qua.")
                continue

            prompt += (
                f"Tiến trình PID {pid}: CPU Usage: {metrics.get('cpu_usage_percent', 0)}%, "
                f"RAM Usage: {metrics.get('memory_usage_mb', 0)}MB, "
                f"GPU Usage: {metrics.get('gpu_usage_percent', 0)}%, "
                f"Disk I/O: {metrics.get('disk_io_mbps', 0)}Mbps, "
                f"Network Bandwidth: {metrics.get('network_bandwidth_mbps', 0)}Mbps, "
                f"Cache Limit: {metrics.get('cache_limit_percent', 0)}%.\n"
            )
        prompt += (
            "Hãy trả về các hành động tối ưu hóa dưới dạng danh sách số, ví dụ: "
            "[cpu_threads, frequency, ram_allocation_mb, gpu_usage_percent, disk_io_limit_mbps, "
            "network_bandwidth_limit_mbps, cache_limit_percent]."
        )
        return prompt
