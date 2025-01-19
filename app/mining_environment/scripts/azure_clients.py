import os
import logging
import asyncio
import re
import functools
import psutil
import pynvml
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

from .auxiliary_modules.models import ConfigModel

# Các import từ SDK Azure
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
from azure.monitor.query import (
    MetricsQueryClient,
    MetricAggregationType,
    LogsQueryClient,
    LogsQueryResult,
)
from azure.mgmt.security import SecurityCenter
from azure.loganalytics import LogAnalyticsDataClient
from azure.loganalytics.models import QueryBody
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.resource import ResourceManagementClient
from azure.identity import ClientSecretCredential, DefaultAzureCredential
from azure.mgmt.resourcegraph import ResourceGraphClient
from azure.mgmt.resourcegraph.models import QueryRequest
from azure.mgmt.loganalytics import LogAnalyticsManagementClient
from azure.mgmt.loganalytics.models import Workspace
from azure.ai.anomalydetector import AnomalyDetectorClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.anomalydetector.models import (
    TimeSeriesPoint,
    UnivariateDetectionOptions,
    UnivariateEntireDetectionResult,
)

###############################################################################
#              DECORATOR ĐỂ RETRY BẤT ĐỒNG BỘ (EVENT-DRIVEN-STYLE)           #
###############################################################################

def async_retry(exception_to_check: Any, tries: int = 4, delay: float = 3.0, backoff: float = 2.0):
    """
    Decorator bất đồng bộ để retry một coroutine khi gặp exception cụ thể.
    """
    def decorator_retry(func: Any):
        @functools.wraps(func)
        async def wrapper_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return await func(*args, **kwargs)
                except exception_to_check as e:
                    logging.getLogger(__name__).warning(
                        f"Lỗi '{e}' xảy ra trong '{func.__name__}'. Thử lại sau {mdelay} giây..."
                    )
                    await asyncio.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return await func(*args, **kwargs)
        return wrapper_retry
    return decorator_retry

###############################################################################
#         LỚP CƠ SỞ: AzureBaseClient (EVENT-DRIVEN FRIENDLY)                  #
###############################################################################

class AzureBaseClient:
    """
    Lớp cơ sở cho các client Azure khác. Cung cấp logic authenticate() và discover_resources().
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
        if not self.subscription_id:
            self.logger.error("AZURE_SUBSCRIPTION_ID không được thiết lập.")
            raise ValueError("AZURE_SUBSCRIPTION_ID không được thiết lập.")

        self.credential = self.authenticate()
        self.resource_graph_client = ResourceGraphClient(self.credential)
        self.resource_management_client = ResourceManagementClient(self.credential, self.subscription_id)

    def authenticate(self) -> ClientSecretCredential:
        """
        Thực hiện authenticate, lưu credential.
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
            self.logger.info("Đã xác thực Azure AD (ClientSecretCredential) thành công.")
            return credential
        except Exception as e:
            self.logger.error(f"Lỗi khi xác thực với Azure AD: {e}")
            raise e

    def discover_resources(self, resource_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Khám phá resource thông qua Resource Graph (đồng bộ). 
        Event-driven => Gọi khi “cần”.
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
                self.logger.warning("Dữ liệu trả về không phải là list.")
                return []

            resources = []
            for res in response.data:
                resources.append({
                    'id': res.get('id', 'N/A'),
                    'name': res.get('name', 'N/A'),
                    'type': res.get('type', 'N/A'),
                    'resourceGroup': res.get('resourceGroup', 'N/A')
                })
            self.logger.info(f"Đã khám phá {len(resources)} tài nguyên.")
            return resources
        except Exception as e:
            self.logger.error(f"Lỗi khi khám phá tài nguyên: {e}", exc_info=True)
            return []

###############################################################################
#           AZURE SENTINEL CLIENT (EVENT-DRIVEN)                              #
###############################################################################

class AzureSentinelClient(AzureBaseClient):
    """
    Client Azure Sentinel.  
    Dùng SecurityCenter để list() alert => filter theo ngày.
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.security_client = SecurityCenter(self.credential, self.subscription_id)

    async def get_recent_alerts(self, days: int = 1) -> List[Any]:
        """
        Lấy alert gần đây từ Azure Sentinel. 
        """
        try:
            # Chuyển qua to_thread => non-block
            alerts = await asyncio.to_thread(lambda: list(self.security_client.alerts.list()))
            recent_alerts = []
            cutoff_time = datetime.utcnow() - timedelta(days=days)

            for alert in alerts:
                created_time = None
                if hasattr(alert, 'properties') and hasattr(alert.properties, 'created_time'):
                    created_time = alert.properties.created_time
                if created_time and created_time >= cutoff_time:
                    recent_alerts.append(alert)

            self.logger.info(f"Đã lấy {len(recent_alerts)} alerts từ Azure Sentinel trong {days} ngày gần đây.")
            return recent_alerts
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy alerts từ Azure Sentinel: {e}")
            return []

###############################################################################
#    AZURE LOG ANALYTICS CLIENT (EVENT-DRIVEN, ASYNC FACTORY METHOD)          #
###############################################################################

class AzureLogAnalyticsClient(AzureBaseClient):
    """
    Client tương tác Log Analytics (LogsQueryClient).
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.logs_client = LogsQueryClient(self.credential)
        self.log_analytics_mgmt_client = LogAnalyticsManagementClient(self.credential, self.subscription_id)
        self.workspace_ids: List[str] = []

    @classmethod
    async def create(cls, logger: logging.Logger) -> 'AzureLogAnalyticsClient':
        instance = cls(logger)
        await instance.initialize()
        return instance

    async def initialize(self):
        self.workspace_ids = await self.get_workspace_ids()

    async def get_workspace_ids(self) -> List[str]:
        """
        Lấy danh sách workspace IDs => return [resource_id,...].
        """
        try:
            # discover_resources() là hàm đồng bộ => dùng to_thread
            resources = await asyncio.to_thread(self.discover_resources, 'Microsoft.OperationalInsights/workspaces')
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
        pattern = (
            r"^/subscriptions/(?P<sub>[^/]+)/resourceGroups/(?P<rg>[^/]+)/providers/"
            r"Microsoft\.OperationalInsights/workspaces/(?P<ws>[^/]+)$"
        )
        match = re.match(pattern, resource_id.strip())
        if not match:
            raise ValueError(f"Resource ID không đúng format: {resource_id}")
        return {
            "subscription_id": match.group("sub"),
            "resource_group": match.group("rg"),
            "workspace_name": match.group("ws")
        }

    async def get_workspace_details(self, resource_id: str) -> Optional[Workspace]:
        """
        Lấy chi tiết 1 workspace => workspace object.
        """
        try:
            parsed = self.parse_log_analytics_id(resource_id)
            resource_group = parsed["resource_group"]
            workspace_name = parsed["workspace_name"]

            # Dùng partial + to_thread
            ws = await asyncio.to_thread(
                functools.partial(
                    self.log_analytics_mgmt_client.workspaces.get,
                    resource_group_name=resource_group,
                    workspace_name=workspace_name
                )
            )
            self.logger.info(
                f"Đã lấy thông tin workspace '{ws.name}' (RG='{resource_group}')."
            )
            return ws
        except Exception as e:
            self.logger.error(f"Không thể get workspace details cho {resource_id}: {e}")
            return None

    async def query_logs(self, query: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Query logs => days => timespan.
        """
        results = []
        if days < 0:
            self.logger.error("days phải >= 0.")
            return results
        if not self.workspace_ids:
            self.logger.error("Không có Workspace ID để truy vấn logs.")
            return results
        if not query:
            self.logger.error("Query không được để trống.")
            return results

        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days)
            timespan = (start_time, end_time)

            for resource_id in self.workspace_ids:
                ws_details = await self.get_workspace_details(resource_id)
                if not ws_details:
                    continue
                customer_id = ws_details.customer_id
                if not customer_id:
                    self.logger.warning(f"Workspace {ws_details.name} không có customer_id (GUID).")
                    continue

                try:
                    response = await asyncio.to_thread(
                        functools.partial(
                            self.logs_client.query_workspace,
                            workspace_id=customer_id,
                            query=query,
                            timespan=timespan
                        )
                    )
                    if not isinstance(response.tables, list):
                        self.logger.warning(f"KQ không hợp lệ (Workspace GUID={customer_id}).")
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
                            f"Truy vấn thành công workspace '{ws_details.name}' (GUID={customer_id})."
                        )
                    else:
                        self.logger.warning(f"Không có dữ liệu trả về từ workspace '{ws_details.name}'.")
                except HttpResponseError as http_error:
                    self.logger.error(f"Lỗi HTTP Workspace GUID={customer_id}: {http_error}")
                except Exception as workspace_error:
                    self.logger.error(f"Lỗi Workspace GUID={customer_id}: {workspace_error}")

            self.logger.info(f"Tổng cộng lấy {len(results)} dòng log.")
        except Exception as e:
            self.logger.critical(f"Lỗi nghiêm trọng khi truy vấn logs: {e}", exc_info=True)
        return results

    async def query_logs_with_time_range(
        self, query: str, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Query logs 1 khoảng thời gian tùy chọn.
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
                ws_details = await self.get_workspace_details(resource_id)
                if not ws_details:
                    continue
                customer_id = ws_details.customer_id
                if not customer_id:
                    self.logger.warning(f"Workspace {ws_details.name} không có customer_id.")
                    continue

                try:
                    response = await asyncio.to_thread(
                        functools.partial(
                            self.logs_client.query_workspace,
                            workspace_id=customer_id,
                            query=query,
                            timespan=(start_time, end_time)
                        )
                    )
                    for table in response.tables:
                        for row in table.rows:
                            row_dict = dict(zip(table.columns, row))
                            results.append(row_dict)
                    self.logger.info(f"Query logs OK Workspace GUID={customer_id} (time range).")
                except Exception as workspace_error:
                    self.logger.error(f"Lỗi Workspace GUID={customer_id}: {workspace_error}")

        except Exception as e:
            self.logger.critical(f"Lỗi nghiêm trọng query logs: {e}")
        return results

    async def query_aml_logs(self, days: int = 1) -> List[Dict[str, Any]]:
        """
        Query AML logs => KQL => AzureDiagnostics
        """
        kql = f"""
        AzureDiagnostics
        | where TimeGenerated > ago({days}d)
        | project TimeGenerated, ResourceId, Category, OperationName
        | limit 50
        """
        return await self.query_logs(kql, days=days)

###############################################################################
#        AZURE NETWORK WATCHER CLIENT (EVENT-DRIVEN)                          #
###############################################################################

class AzureNetworkWatcherClient(AzureBaseClient):
    """
    Client quản lý Network Watcher: get_flow_logs, create_flow_log, etc.
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.network_client = NetworkManagementClient(self.credential, self.subscription_id)

    async def get_network_watcher_name(self, resource_group: str) -> Optional[str]:
        """
        Lấy tên network watcher cho 1 resource group => location => watchers
        """
        try:
            # Lấy location resource group => to_thread
            rg_obj = await asyncio.to_thread(
                functools.partial(
                    self.resource_management_client.resource_groups.get,
                    resource_group_name=resource_group
                )
            )
            region = rg_obj.location

            watchers = await asyncio.to_thread(lambda: list(self.network_client.network_watchers.list_all()))
            for watcher in watchers:
                if watcher.location.lower() == region.lower():
                    self.logger.info(f"Network Watcher '{watcher.name}' cho vùng '{region}'.")
                    return watcher.name
            self.logger.warning(f"Không tìm thấy Network Watcher cho vùng '{region}'.")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi khi tìm Network Watcher cho RG '{resource_group}': {e}")
            return None

    async def get_flow_logs(self, nw_rg: str, nw_name: str, nsg_name: str) -> List[Any]:
        """
        Lấy flow logs từ NSG
        """
        try:
            flow_log_configurations = await asyncio.to_thread(
                functools.partial(
                    self.network_client.flow_logs.list,
                    resource_group_name=nw_rg,
                    network_watcher_name=nw_name
                )
            )
            # Filter log
            flow_logs = []
            for log in flow_log_configurations:
                if hasattr(log, 'target_resource_id') and log.target_resource_id.endswith(nsg_name):
                    flow_logs.append(log)
            self.logger.info(f"Lấy {len(flow_logs)} flow logs từ NSG={nsg_name}, RG={nw_rg}.")
            return flow_logs
        except Exception as e:
            self.logger.error(
                f"Lỗi khi lấy flow logs từ NSG {nsg_name}, RG={nw_rg}: {e}"
            )
            return []

    async def create_flow_log(self, resource_group: str, network_watcher_name: str, 
                              nsg_name: str, flow_log_name: str, params: Dict[str, Any]) -> Optional[Any]:
        """
        Tạo flow log => to_thread
        """
        try:
            flow_log_op = await asyncio.to_thread(
                functools.partial(
                    self.network_client.flow_logs.begin_create_or_update,
                    resource_group_name=resource_group,
                    network_security_group_name=nsg_name,
                    flow_log_name=flow_log_name,
                    parameters=params
                )
            )
            flow_log = flow_log_op.result()
            self.logger.info(
                f"Đã tạo flow log {flow_log_name} cho NSG {nsg_name}, RG {resource_group}."
            )
            return flow_log
        except Exception as e:
            self.logger.error(
                f"Lỗi khi tạo flow log {flow_log_name}, NSG {nsg_name}, RG {resource_group}: {e}"
            )
            return None

    async def delete_flow_log(self, resource_group: str, network_watcher_name: str,
                              nsg_name: str, flow_log_name: str) -> bool:
        """
        Xóa flow log => to_thread
        """
        try:
            delete_op = await asyncio.to_thread(
                functools.partial(
                    self.network_client.flow_logs.begin_delete,
                    resource_group_name=resource_group,
                    network_security_group_name=nsg_name,
                    flow_log_name=flow_log_name
                )
            )
            delete_op.result()
            self.logger.info(
                f"Đã xóa flow log {flow_log_name} (NSG={nsg_name}, RG={resource_group})."
            )
            return True
        except Exception as e:
            self.logger.error(
                f"Lỗi khi xóa flow log {flow_log_name}, NSG={nsg_name}, RG={resource_group}: {e}"
            )
            return False

    async def check_flow_log_status(self, nw_rg: str, nw_name: str, flow_log_name: str) -> Optional[Dict[str, Any]]:
        """
        Kiểm tra trạng thái Flow Log => to_thread
        """
        try:
            flow_log = await asyncio.to_thread(
                functools.partial(
                    self.network_client.flow_logs.get,
                    resource_group_name=nw_rg,
                    network_watcher_name=nw_name,
                    flow_log_name=flow_log_name
                )
            )
            self.logger.info(
                f"Flow Log {flow_log_name}, provisioning_state={flow_log.provisioning_state}."
            )
            return {
                "id": flow_log.id,
                "state": flow_log.provisioning_state,
                "storageId": flow_log.storage_id,
                "targetResourceId": flow_log.target_resource_id
            }
        except Exception as e:
            self.logger.error(f"Lỗi check Flow Log {flow_log_name}: {e}")
            return None

###############################################################################
#         AZURE ANOMALY DETECTOR CLIENT (EVENT-DRIVEN)                        #
###############################################################################

class AzureAnomalyDetectorClient(AzureBaseClient):
    """
    Client Azure Anomaly Detector (event-driven).
    """
    def __init__(self, logger: logging.Logger, config: Dict[str, Any]):
        super().__init__(logger)
        self.logger = logger

        if isinstance(config, ConfigModel):
            config_dict = config.to_dict()
        else:
            config_dict = config
        self.config = config_dict

        self.endpoint = self.config.get("azure_anomaly_detector", {}).get("api_base")
        self.api_key = os.getenv("ANOMALY_DETECTOR_API_KEY")

        self.logger.debug(f"Endpoint Anomaly Detector: {self.endpoint}")
        self.logger.debug(f"API Key: {'***' if self.api_key else None}")

        if not self.endpoint or not self.api_key:
            self.logger.error("endpoint hoặc api_key cho Azure Anomaly Detector chưa thiết lập.")
            raise ValueError("Thiếu cấu hình Azure Anomaly Detector.")

        try:
            self.client = AnomalyDetectorClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key)
            )
            self.logger.info("Kết nối Azure Anomaly Detector thành công.")
        except Exception as e:
            self.logger.error(f"Lỗi kết nối Azure Anomaly Detector: {e}")
            raise e

    async def detect_anomalies(self, metric_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Phát hiện bất thường cho các PID (nhiều metric).
        """
        anomalies = {}
        metrics_to_analyze = ['cpu_usage', 'gpu_usage', 'cache_usage', 'network_usage']
        min_data_points = 12
        granularity = self.config.get("granularity", "minutely")

        async def analyze_pid_metric(pid: str, metric_name: str, values: List[float]) -> Optional[str]:
            """
            Phân tích 1 metric => trả về metric_name nếu có bất thường, None nếu ko.
            """
            try:
                if len(values) < min_data_points:
                    self.logger.warning(
                        f"PID={pid} {metric_name}: không đủ dữ liệu (cần >= {min_data_points})."
                    )
                    return None

                from datetime import datetime, timedelta
                current_time = datetime.utcnow()
                series = []
                for i, val in enumerate(values):
                    if not isinstance(val, (int, float)):
                        self.logger.warning(
                            f"PID={pid} {metric_name}: giá trị không hợp lệ ({val})."
                        )
                        continue
                    timestamp = current_time - timedelta(minutes=len(values) - i)
                    series.append(TimeSeriesPoint(timestamp=timestamp.isoformat(), value=val))

                options = UnivariateDetectionOptions(
                    series=series,
                    granularity=granularity,
                    sensitivity=95
                )

                # Gọi API detect => blocking => to_thread
                response = await asyncio.to_thread(
                    self.client.detect_univariate_entire_series,
                    options=options
                )
                if any(response.is_anomaly):
                    self.logger.warning(f"PID={pid} bất thường metric={metric_name}.")
                    return metric_name
            except Exception as e:
                self.logger.error(f"Lỗi phân tích PID={pid}, metric={metric_name}: {e}")
            return None

        tasks = []
        for pid, metrics in metric_data.items():
            if not isinstance(metrics, dict):
                self.logger.warning(f"PID={pid}: metric_data không hợp lệ => bỏ qua.")
                continue
            for metric_name in metrics_to_analyze:
                metric_values = metrics.get(metric_name, [])
                task = asyncio.create_task(analyze_pid_metric(pid, metric_name, metric_values))
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        idx = 0
        pids_list = list(metric_data.keys())
        for pid in pids_list:
            metrics = metric_data[pid]
            if not isinstance(metrics, dict):
                continue
            for metric_name in metrics_to_analyze:
                result = results[idx]
                idx += 1
                if isinstance(result, Exception):
                    self.logger.error(f"Lỗi phân tích PID={pid}, metric={metric_name}: {result}")
                    continue
                if result:  # => metric_name
                    anomalies.setdefault(pid, []).append(metric_name)

        return anomalies

    def validate_configuration(self):
        """
        Kiểm tra cấu hình Anomaly Detector.
        """
        if not self.endpoint or not self.api_key:
            self.logger.error("Endpoint/API Key chưa cấu hình đúng.")
            raise ValueError("Cấu hình Azure Anomaly Detector thiếu.")
        self.logger.info("Cấu hình Azure Anomaly Detector OK.")

    def log_configuration(self):
        """
        Log cấu hình (debug).
        """
        self.logger.debug(f"Endpoint: {self.endpoint}")
        self.logger.debug(f"API Key: {'***' if self.api_key else None}")



# ====================================
# AzureOpenAIClient (Không thay đổi)
# ====================================

# class AzureOpenAIClient(AzureBaseClient):
#     """
#     Lớp tích hợp với Azure OpenAI, cho phép lấy gợi ý tối ưu hoá tài nguyên
#     dựa trên dữ liệu trạng thái hệ thống. 
#     Kiểm tra độ dài prompt/response để giám sát token usage.
#     """
#     def __init__(self, logger: logging.Logger, config: Dict[str, Any]):
#         super().__init__(logger)
        
#         self.endpoint = config.get("azure_openai", {}).get("api_base")
#         self.api_key = os.getenv("OPENAI_API_KEY")
#         self.deployment_name = config.get("azure_openai", {}).get("deployment_name")
#         self.api_version = config.get("azure_openai", {}).get("api_version", "2023-03-15-preview")

#         # Kiểm tra thiết yếu
#         if not self.api_key:
#             self.logger.error("API key cho Azure OpenAI chưa được thiết lập (OPENAI_API_KEY).")
#             raise ValueError("Thiếu thông tin API key trong biến môi trường OPENAI_API_KEY.")
#         if not self.endpoint:
#             self.logger.error("Endpoint cho Azure OpenAI chưa được thiết lập trong config.")
#             raise ValueError("Thiếu thông tin azure_openai.api_base.")
#         if not self.deployment_name:
#             self.logger.error("Deployment name cho Azure OpenAI chưa được thiết lập trong config.")
#             raise ValueError("Thiếu thông tin azure_openai.deployment_name.")

#         self.initialize_openai()

#     def initialize_openai(self):
#         if not self.endpoint or not self.api_key or not self.deployment_name:
#             self.logger.error("Thiếu thông tin endpoint, api_key, deployment_name.")
#             raise ValueError("Cần có đủ endpoint, api_key, deployment_name.")
#         try:
#             self.client = AzureOpenAI(
#                 azure_endpoint=self.endpoint,
#                 api_version=self.api_version,
#                 api_key=self.api_key
#             )
#             self.logger.info("Đã cấu hình Azure OpenAI Service thành công.")
#         except Exception as e:
#             self.logger.error(f"Lỗi khi cấu hình AzureOpenAI: {e}")
#             raise e

#     async def get_optimization_suggestions(
#         self,
#         server_config: Dict[str, Any],
#         optimization_goals: Dict[str, str],
#         state_data: Dict[str, Any]
#     ) -> List[float]:
#         """
#         Gửi prompt tới Azure OpenAI và nhận các gợi ý tối ưu hóa.

#         :param server_config: Thông tin cấu hình máy chủ (tĩnh).
#         :param optimization_goals: Mục tiêu tối ưu hóa cho từng tài nguyên.
#         :param state_data: Dữ liệu trạng thái hệ thống hiện tại (động).
#         :return: Danh sách 7 giá trị float đại diện cho cấu hình tối ưu.
#         """
#         try:
#             # Xác thực định dạng của state_data
#             if not isinstance(state_data, dict):
#                 self.logger.error(
#                     f"state_data không phải là dict. Dữ liệu nhận được: {state_data}"
#                 )
#                 # Gán các giá trị mặc định hoặc bỏ qua PID này
#                 cpu = 0
#                 ram = 0
#                 gpu = 0
#                 net_bw = 0
#                 cache = 0
#             else:
#                 # Construct prompt
#                 prompt = self.construct_prompt(server_config, optimization_goals, state_data)

#             # Log độ dài prompt
#             prompt_len = len(prompt)
#             self.logger.debug(f"Prompt length = {prompt_len} characters.")

#             # Định nghĩa các tin nhắn gửi tới mô hình
#             messages = [
#                 {
#                     "role": "system",
#                     "content": (
#                         "You are an expert system resource optimizer. Based on the provided server configuration and optimization goals, "
#                         "please suggest an optimization plan to enhance system performance and resource utilization efficiently. "
#                         "Provide exactly one single line with 7 numerical values in CSV format representing the optimized configuration. "
#                         "Do not add any additional explanations."
#                     )
#                 },
#                 {
#                     "role": "user",
#                     "content": (
#                         f"{prompt}\n\n"
#                         "Please return EXACTLY 7 comma-separated floats in the following order:\n"
#                         "[cpu_threads, frequency_mhz, ram_allocation_mb, gpu_usage_percent, disk_io_limit_mbps, network_bandwidth_limit_mbps, cache_limit_percent].\n"
#                         "Do not add any additional text or new lines."
#                     )
#                 }
#             ]

#             # Log độ dài của từng tin nhắn
#             user_msg_len = len(messages[1]["content"])
#             system_msg_len = len(messages[0]["content"])
#             self.logger.debug(f"System message length = {system_msg_len}, User message length = {user_msg_len}.")

#             # Gửi yêu cầu tới Azure OpenAI
#             response = await asyncio.get_event_loop().run_in_executor(
#                 None,
#                 functools.partial(
#                     self.client.chat.completions.create,
#                     model=self.deployment_name,  # Sử dụng deployment_name làm model
#                     messages=messages,
#                     max_tokens=70,  # Tăng max_tokens để đảm bảo nhận đủ 7 giá trị
#                     temperature=0.0
#                 )
#             )

#             # Kiểm tra phản hồi có chứa choices hay không
#             if not response.choices:
#                 self.logger.error("Phản hồi từ Azure OpenAI không chứa lựa chọn nào.")
#                 return [0.0] * 7  # Trả về giá trị mặc định

#             # Lấy và xử lý phản hồi
#             suggestion_text = response.choices[0].message.content.strip()

#             # Log độ dài phản hồi
#             response_len = len(suggestion_text)
#             self.logger.debug(f"Raw response length = {response_len} characters.")

#             # Xóa xuống dòng nếu có
#             suggestion_text = suggestion_text.replace('\n', ' ').replace('\r', ' ')

#             # Chuyển đổi phản hồi thành danh sách float
#             suggestions_raw = []
#             for x in suggestion_text.split(','):
#                 try:
#                     suggestions_raw.append(float(x.strip()))
#                 except ValueError:
#                     self.logger.warning(f"Không thể parse '{x.strip()}' thành float, gán giá trị 0.0.")
#                     suggestions_raw.append(0.0)  # Gán giá trị mặc định nếu không thể parse

#             # Giới hạn số lượng giá trị ở 7
#             if len(suggestions_raw) > 7:
#                 self.logger.warning(f"Nhận được nhiều hơn 7 giá trị: {suggestions_raw}. Giới hạn chỉ lấy 7 giá trị đầu.")
#                 suggestions_raw = suggestions_raw[:7]
#             elif len(suggestions_raw) < 7:
#                 self.logger.warning(f"Số lượng gợi ý nhận được ít hơn 7: {suggestions_raw}. Bổ sung giá trị 0.0.")
#                 suggestions_raw += [0.0] * (7 - len(suggestions_raw))

#             # Xử lý tần số (frequency_mhz)
#             if len(suggestions_raw) >= 2:
#                 suggestions_raw[1] = self._parse_frequency(suggestions_raw[1])
#             else:
#                 self.logger.warning("Không đủ dữ liệu để xử lý tần số. Gán giá trị 0.0.")
#                 suggestions_raw.append(0.0)  # Bổ sung giá trị mặc định

#             # Đảm bảo danh sách luôn có 7 giá trị
#             suggestions_raw = suggestions_raw[:7]
#             if len(suggestions_raw) < 7:
#                 suggestions_raw += [0.0] * (7 - len(suggestions_raw))

#             self.logger.info(f"Nhận được gợi ý tối ưu hóa từ Azure OpenAI (đã parse freq): {suggestions_raw}")
#             return suggestions_raw

#         def construct_prompt(
#             self,
#             server_config: Dict[str, Any],
#             optimization_goals: Dict[str, str],
#             state_data: Dict[str, Any]
#         ) -> str:
#             """
#             Xây dựng prompt dựa trên cấu hình máy chủ, mục tiêu tối ưu hóa và dữ liệu trạng thái hệ thống.

#             :param server_config: Thông tin cấu hình máy chủ (tĩnh).
#             :param optimization_goals: Mục tiêu tối ưu hóa cho từng tài nguyên.
#             :param state_data: Dữ liệu trạng thái hệ thống hiện tại (động).
#             :return: Chuỗi prompt đầy đủ.
#             """
#             prompt = f"Current Server: {server_config.get('server_type', 'Standard_NC12s_v3')} on Azure Cloud\n"
#             prompt += "Resource Limits:\n"
#             resource_limits = server_config.get('resource_limits', {})
#             prompt += f"- CPU Usage Limit: {resource_limits.get('cpu_usage_percent', 'N/A')}%\n"
#             prompt += f"- RAM Usage Limit: {resource_limits.get('ram_usage_percent', 'N/A')}%\n"
#             prompt += f"- GPU Usage Limit: {resource_limits.get('gpu_usage_percent', 'N/A')}%\n"
#             prompt += f"- Network Bandwidth Limit: {resource_limits.get('network_bandwidth_mbps', 'N/A')} Mbps\n"
#             prompt += f"- Storage Usage Limit: {resource_limits.get('storage_usage_percent', 'N/A')}%\n\n"

#             prompt += "Current System Parameters:\n"
#             for pid, metrics_info in state_data.items():
#                 if not isinstance(metrics_info, dict):
#                     self.logger.error(
#                         f"Metrics_info cho PID {pid} không phải là dict. Dữ liệu nhận được: {metrics_info}"
#                     )
#                     # Gán các giá trị mặc định hoặc bỏ qua PID này
#                     cpu = 0
#                     ram = 0
#                     gpu = 0
#                     net_bw = 0
#                     cache = 0
#                 else:
#                     cpu = metrics_info.get('cpu_usage_percent', 0)
#                     ram = metrics_info.get('memory_usage_mb', 0)
#                     gpu = metrics_info.get('gpu_usage_percent', 0)
#                     net_bw = metrics_info.get('network_bandwidth_mbps', 0)
#                     cache = metrics_info.get('cache_limit_percent', 0)

#                 prompt += (
#                     f"PID {pid}: CPU={cpu}%, RAM={ram}MB, GPU={gpu}%, Net={net_bw}Mbps, Cache={cache}%.\n"
#                 )
#             prompt += "\n"

#             prompt += "Optimization Goals:\n"
#             for key, description in optimization_goals.items():
#                 prompt += f"- **{key}**: {description}\n"

#             return prompt

#         def _parse_frequency(self, freq_val: float) -> float:
#             """
#             Chuyển đổi tần số từ GHz sang MHz nếu cần thiết.

#             :param freq_val: Giá trị tần số.
#             :return: Giá trị tần số đã chuyển đổi.
#             """
#             if freq_val < 100:
#                 mhz_val = freq_val * 1000.0
#                 self.logger.debug(f"Converting {freq_val} GHz to {mhz_val} MHz")
#                 return mhz_val
#             else:
#                 return freq_val
