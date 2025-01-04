# azure_clients.py

import os
import logging
import time
import traceback
import re  
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Union, Tuple
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
# Vẫn giữ import MetricsQueryClient, MetricAggregationType, LogsQueryClient (nếu sau này muốn sử dụng)
from azure.monitor.query import MetricsQueryClient, MetricAggregationType, LogsQueryClient, LogsQueryResult
from azure.mgmt.security import SecurityCenter
from azure.loganalytics import LogAnalyticsDataClient, models as logmodels
from azure.loganalytics.models import QueryBody
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.resource import ResourceManagementClient
from azure.identity import ClientSecretCredential
from azure.mgmt.resourcegraph import ResourceGraphClient
from azure.mgmt.resourcegraph.models import QueryRequest


from azure.identity import DefaultAzureCredential

# Import để quản trị Log Analytics (mới thêm)
from azure.mgmt.loganalytics import LogAnalyticsManagementClient
from azure.mgmt.loganalytics.models import Workspace

from azure.ai.anomalydetector import AnomalyDetectorClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.anomalydetector.models import (
    TimeSeriesPoint,
    UnivariateDetectionOptions,
    UnivariateEntireDetectionResult
    # Đã lược bỏ MultivariateDetectionOptions, MultivariateEntireDetectionResult
)

from openai import AzureOpenAI


class AzureBaseClient:
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
                self.logger.warning("Dữ liệu trả về không phải là danh sách.")
                return []

            resources = [
                {
                    'id': res.get('id', 'N/A'),
                    'name': res.get('name', 'N/A'),
                    'type': res.get('type', 'N/A'),
                    'resourceGroup': res.get('resourceGroup', 'N/A'),
                }
                for res in response.data
            ]

            self.logger.info(f"Đã khám phá {len(resources)} tài nguyên.")
            return resources
        except Exception as e:
            self.logger.error(f"Lỗi khi khám phá tài nguyên: {e}", exc_info=True)
            return []


class AzureSentinelClient(AzureBaseClient):
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.security_client = SecurityCenter(self.credential, self.subscription_id)

    def get_recent_alerts(self, days: int = 1) -> List[Any]:
        try:
            alerts = self.security_client.alerts.list()
            recent_alerts = []
            cutoff_time = datetime.utcnow() - timedelta(days=days)

            if not isinstance(cutoff_time, datetime):
                raise ValueError(f"cutoff_time không hợp lệ: {cutoff_time}.")

            for alert in alerts:
                if (
                    hasattr(alert, 'properties')
                    and hasattr(alert.properties, 'created_time')
                    and alert.properties.created_time >= cutoff_time
                ):
                    recent_alerts.append(alert)

            self.logger.info(f"Đã lấy {len(recent_alerts)} alerts từ Azure Sentinel trong {days} ngày gần đây.")
            return recent_alerts
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy alerts từ Azure Sentinel: {e}")
            return []


class AzureLogAnalyticsClient(AzureBaseClient):
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        if not hasattr(self, "credential") or not self.credential:
            raise AttributeError("Credential không được định nghĩa trong AzureBaseClient.")

        self.logs_client = LogsQueryClient(self.credential)
        self.log_analytics_mgmt_client = LogAnalyticsManagementClient(self.credential, self.subscription_id)
        self.workspace_ids = self.get_workspace_ids()

    def get_workspace_ids(self) -> List[str]:
        try:
            resources = self.discover_resources('Microsoft.OperationalInsights/workspaces')
            if not resources:
                self.logger.warning("Không tìm thấy tài nguyên Log Analytics Workspace nào.")
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

    def get_workspace_details(self, resource_id: str) -> Optional[Workspace]:
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
            self.logger.error("Giá trị days phải >= 0.")
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
                ws_details = self.get_workspace_details(resource_id)
                if not ws_details:
                    continue
                customer_id = ws_details.customer_id
                if not customer_id:
                    self.logger.warning(f"Workspace {ws_details.name} không có customer_id (GUID).")
                    continue

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
                            f"Đã truy vấn thành công trên workspace '{ws_details.name}' (GUID={customer_id})."
                        )
                    else:
                        self.logger.warning(
                            f"Không có dữ liệu trả về từ workspace '{ws_details.name}'."
                        )

                except HttpResponseError as http_error:
                    self.logger.error(f"Lỗi HTTP trên Workspace GUID={customer_id}: {http_error}")
                except Exception as workspace_error:
                    self.logger.error(f"Lỗi trên Workspace GUID={customer_id}: {workspace_error}")

            self.logger.info(f"Tổng cộng lấy được {len(results)} dòng dữ liệu từ tất cả các workspace.")
        except Exception as e:
            self.logger.critical(f"Lỗi nghiêm trọng khi truy vấn logs: {e}", exc_info=True)
        return results

    def query_logs_with_time_range(
        self, query: str, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
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

    def query_aml_logs(self, days: int = 1) -> List[Dict[str, Any]]:
        kql = f"""
        AzureDiagnostics
        | where TimeGenerated > ago({days}d)
        // BỎ PHẦN LỌC Category:
        // | where Category in ("AmlComputeClusterEvent", "AmlComputeJobEvent")
        | project TimeGenerated, ResourceId, Category, OperationName
        | limit 50
        """
        return self.query_logs(kql, days=days)


class AzureNetworkWatcherClient(AzureBaseClient):
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.network_client = NetworkManagementClient(self.credential, self.subscription_id)

    def get_network_watcher_name(self, resource_group: str) -> Optional[str]:
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


#
# ===========================
# AzureAnomalyDetectorClient
# ===========================
#

class AzureAnomalyDetectorClient(AzureBaseClient):
    def __init__(self, logger: logging.Logger, config: Dict[str, Any]):
        """
        Khởi tạo AzureAnomalyDetectorClient với các tối ưu và cải thiện hiệu suất.

        Args:
            logger (logging.Logger): Đối tượng logger để ghi log.
            config (Dict[str, Any]): Cấu hình chứa endpoint của Azure Anomaly Detector và các thiết lập khác.
        """
        super().__init__(logger)
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

    def detect_anomalies(self, metric_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Phát hiện bất thường (đơn biến) trong dữ liệu metrics và trả về
        các PID cùng với các metrics có bất thường.

        Args:
            metric_data (Dict[str, Any]): Dữ liệu metrics cho từng tiến trình,
                ví dụ:
                {
                    "1234": {
                        "cpu_usage": [...],
                        "gpu_usage": [...],
                        "cache_usage": [...],
                        "network_usage": [...]
                    },
                    "2345": {
                        ...
                    }
                }

        Returns:
            Dict[str, List[str]]:
                {
                    PID: [danh_sach_metric_bat_thuong],
                    ...
                }
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        anomalies = {}
        metrics_to_analyze = ['cpu_usage', 'gpu_usage', 'cache_usage', 'network_usage']
        min_data_points = 12
        max_workers = 10  # Số luồng tối đa cho ThreadPoolExecutor

        def analyze_pid_metric(pid: str, metric_name: str, metric_values: List[float]) -> Optional[str]:
            """
            Phân tích một metric của một PID để phát hiện bất thường.

            Args:
                pid (str): Mã tiến trình (PID).
                metric_name (str): Tên metric.
                metric_values (List[float]): Danh sách giá trị metric (theo thời gian).

            Returns:
                Optional[str]: Tên metric nếu phát hiện bất thường, None nếu không.
            """
            try:
                if not isinstance(metric_values, list):
                    self.logger.warning(f"{metric_name} cho PID {pid} không phải list, bỏ qua.")
                    return None
                if len(metric_values) < min_data_points:
                    self.logger.warning(f"Tiến trình PID {pid} có ít hơn {min_data_points} điểm dữ liệu cho {metric_name}, bỏ qua.")
                    return None

                # Tạo chuỗi dữ liệu (time series) dạng TimeSeriesPoint
                series = []
                current_time = datetime.utcnow()
                total_data = len(metric_values)

                for i, value in enumerate(metric_values):
                    if not isinstance(value, (int, float)):
                        self.logger.warning(f"{metric_name}={value} cho PID {pid} không phải kiểu số, bỏ qua.")
                        continue
                    # Tính timestamp lùi dần theo chiều dài dữ liệu
                    timestamp = current_time - timedelta(minutes=total_data - i)
                    series.append(TimeSeriesPoint(timestamp=timestamp.isoformat(), value=value))

                # Tạo tùy chọn univariate detection
                options = UnivariateDetectionOptions(
                    series=series,
                    granularity="minutely",
                    sensitivity=95
                )

                # Gọi API detect_univariate_entire_series
                response: UnivariateEntireDetectionResult = self.client.detect_univariate_entire_series(options=options)

                # Nếu có bất kỳ điểm nào is_anomaly=True
                if any(response.is_anomaly):
                    self.logger.warning(f"Phát hiện bất thường trong PID {pid} cho {metric_name}.")
                    return metric_name

            except Exception as e:
                self.logger.error(f"Lỗi khi phân tích PID {pid} cho {metric_name}: {e}")
            return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pid_metric = {}

            for pid, metrics in metric_data.items():
                if not isinstance(metrics, dict):
                    self.logger.warning(f"Metric data cho PID {pid} không phải dict, bỏ qua.")
                    continue

                for metric_name in metrics_to_analyze:
                    metric_values = metrics.get(metric_name, [])
                    future = executor.submit(analyze_pid_metric, pid, metric_name, metric_values)
                    future_to_pid_metric[future] = (pid, metric_name)

            for future in as_completed(future_to_pid_metric):
                pid, metric_name = future_to_pid_metric[future]
                try:
                    result = future.result()
                    if result:
                        # Nếu phát hiện bất thường, lưu lại
                        if pid not in anomalies:
                            anomalies[pid] = []
                        anomalies[pid].append(result)
                except Exception as e:
                    self.logger.error(f"Lỗi khi xử lý PID {pid} cho {metric_name}: {e}")

        return anomalies

    def validate_configuration(self):
        """
        Xác thực cấu hình kết nối với Azure Anomaly Detector.
        """
        if not self.endpoint or not self.api_key:
            self.logger.error("Endpoint hoặc API Key chưa được cấu hình đúng cách.")
            raise ValueError("Cấu hình Azure Anomaly Detector không đầy đủ.")
        self.logger.info("Cấu hình Azure Anomaly Detector đã được xác thực thành công.")

    def log_configuration(self):
        """
        Ghi lại cấu hình hiện tại cho mục đích debug.
        """
        self.logger.debug(f"Endpoint: {self.endpoint}")
        self.logger.debug(f"API Key: {'***' if self.api_key else None}")

#
# ===========================
# AzureOpenAIClient
# ===========================
#

import os
import logging
from typing import Dict, Any, List

from openai import AzureOpenAI

class AzureBaseClient:
    def __init__(self, logger: logging.Logger):
        self.logger = logger


class AzureOpenAIClient(AzureBaseClient):
    """
    Lớp tích hợp với Azure OpenAI, cho phép lấy gợi ý tối ưu hoá tài nguyên
    dựa trên dữ liệu trạng thái hệ thống. 
    Kiểm tra độ dài prompt/response để giám sát token usage.
    """
    def __init__(self, logger: logging.Logger, config: Dict[str, Any]):
        super().__init__(logger)
        self.endpoint = config.get("azure_openai", {}).get("api_base")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.deployment_name = config.get("azure_openai", {}).get("deployment_name")
        self.api_version = config.get("azure_openai", {}).get("api_version", "2023-03-15-preview")

        # Kiểm tra thiết yếu
        if not self.api_key:
            self.logger.error("API key cho Azure OpenAI chưa được thiết lập (OPENAI_API_KEY).")
            raise ValueError("Thiếu thông tin API key trong biến môi trường OPENAI_API_KEY.")
        if not self.endpoint:
            self.logger.error("Endpoint cho Azure OpenAI chưa được thiết lập trong config.")
            raise ValueError("Thiếu thông tin azure_openai.api_base.")

        self.initialize_openai()

    def initialize_openai(self):
        if not self.endpoint or not self.api_key or not self.deployment_name:
            self.logger.error("Thiếu thông tin endpoint, api_key, deployment_name.")
            raise ValueError("Cần có đủ endpoint, api_key, deployment_name.")
        try:
            self.client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_version=self.api_version,
                api_key=self.api_key
            )
            self.logger.info("Đã cấu hình Azure OpenAI Service thành công.")
        except Exception as e:
            self.logger.error(f"Lỗi khi cấu hình AzureOpenAI: {e}")
            raise e

    def get_optimization_suggestions(self, state_data: Dict[str, Any]) -> List[float]:
        """
        Gửi prompt lên Azure OpenAI để lấy gợi ý tối ưu tài nguyên.
        Format output: [cpu_threads, frequency, ram_allocation_mb, gpu_usage_percent, net_bw_limit, cache_limit].
        => frequency: Nếu OpenAI trả về GHz => convert sang MHz, nếu MHz => truyền thẳng.
        """
        try:
            prompt = self.construct_prompt(state_data)

            # Log độ dài prompt
            prompt_len = len(prompt)
            self.logger.debug(f"Prompt length = {prompt_len} characters.")

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a system resource optimization expert. "
                        "Output exactly one single line with 6 numeric values in CSV format. "
                        "No explanation. If frequency is in GHz, just provide the numeric value (e.g. 2.5)."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\n"
                        "Please return EXACTLY 6 comma-separated floats:\n"
                        "[cpu_threads, frequency (GHz or MHz?), ram_allocation_mb, gpu_usage_percent, "
                        "network_bandwidth_limit_mbps, cache_limit_percent].\n"
                        "No additional text, no new lines."
                    )
                }
            ]

            # Tính độ dài tạm cho messages
            user_msg_len = len(messages[1]["content"])
            system_msg_len = len(messages[0]["content"])
            self.logger.debug(f"System message length = {system_msg_len}, User message length = {user_msg_len}.")

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=50,
                temperature=0.0
            )

            suggestion_text = response.choices[0].message.content.strip()

            # Log độ dài response
            response_len = len(suggestion_text)
            self.logger.debug(f"Raw response length = {response_len} characters.")

            # Xóa xuống dòng (nếu có)
            suggestion_text = suggestion_text.replace('\n', ' ')

            suggestions_raw = []
            for x in suggestion_text.split(','):
                try:
                    suggestions_raw.append(float(x.strip()))
                except ValueError:
                    self.logger.warning(f"Không parse được '{x.strip()}' thành float, bỏ qua.")

            # Dừng ở 6 giá trị
            suggestions_raw = suggestions_raw[:6]

            # Giả định format: [threads, freq, ram, gpu, net_bw, cache]
            if len(suggestions_raw) < 6:
                self.logger.warning(f"Số lượng gợi ý nhận được ít hơn 6: {suggestions_raw}")
                suggestions_raw += [0.0] * (6 - len(suggestions_raw))

            # Bước parse frequency
            freq = self._parse_frequency(suggestions_raw[1])
            suggestions_raw[1] = freq

            self.logger.info(f"Nhận được gợi ý tối ưu hóa từ Azure OpenAI (đã parse freq): {suggestions_raw}")
            return suggestions_raw

        except Exception as e:
            self.logger.error(f"Lỗi khi lấy gợi ý từ AzureOpenAI: {e}")
            return []

    def _parse_frequency(self, freq_val: float) -> float:
        """
        Giả định:
         - Nếu freq_val < 100 => coi như GHZ => convert sang MHz (freq_val*1000)
         - Nếu freq_val >= 100 => coi như MHz => không đổi
        """
        if freq_val < 100:
            mhz_val = freq_val * 1000.0
            self.logger.debug(f"Convert {freq_val} GHz => {mhz_val} MHz")
            return mhz_val
        else:
            return freq_val

    def construct_prompt(self, state_data: Dict[str, Any]) -> str:
        prompt = "Based on these system parameters, propose an optimization plan:\n"
        for pid, metrics_info in state_data.items():
            m = self._normalize_metrics(pid, metrics_info)
            cpu_usage = m.get('cpu_usage_percent', 0)
            mem_usage = m.get('memory_usage_mb', 0)
            gpu_usage = m.get('gpu_usage_percent', 0)
            net_bw    = m.get('network_bandwidth_mbps', 0)
            cache_lim = m.get('cache_limit_percent', 0)

            prompt += (
                f"PID {pid}: CPU={cpu_usage}%, RAM={mem_usage}MB, "
                f"GPU={gpu_usage}%, Net={net_bw}Mbps, Cache={cache_lim}%.\n"
            )
        return prompt

    def _normalize_metrics(self, pid: str, raw_data: Any) -> Dict[str, Any]:
        if isinstance(raw_data, dict):
            return raw_data
        elif isinstance(raw_data, list):
            self.logger.debug(f"PID {pid} => metrics là list => bọc vào dict: {raw_data}")
            return {"list_values": raw_data}
        elif isinstance(raw_data, (float, int)):
            self.logger.debug(f"PID {pid} => metrics là số => bọc vào dict: {raw_data}")
            return {"value": raw_data}
        else:
            self.logger.error(f"PID {pid} => metrics không hợp lệ: {type(raw_data)} | {raw_data}")
            return {}
