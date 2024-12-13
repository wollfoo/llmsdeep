# azure_clients.py

import os
import logging
from typing import List, Dict, Any, Optional
import datetime
import time

from azure.monitor.query import MetricsQueryClient, MetricAggregationType
from azure.mgmt.security import SecurityCenter
from azure.loganalytics import LogAnalyticsDataClient, models as logmodels
from azure.loganalytics.models import QueryBody
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.resource import ResourceManagementClient
from azure.identity import ClientSecretCredential
from azure.mgmt.resourcegraph import ResourceGraphClient
from azure.mgmt.machinelearningservices import AzureMachineLearningWorkspaces


from azure.ai.anomalydetector import AnomalyDetectorClient
from azure.ai.anomalydetector.models import UnivariateDetectionOptions, UnivariateEntireDetectionResult

import openai

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
            self.logger.info("Đã xác thực thành công với Azure AD.")
            return credential
        except Exception as e:
            self.logger.error(f"Lỗi khi xác thực với Azure AD: {e}")
            raise e

    def discover_resources(self, resource_type: str) -> List[Dict[str, Any]]:
        """
        Sử dụng Azure Resource Graph để tự động khám phá tài nguyên theo loại.
        """
        try:
            query = f"Resources | where type =~ '{resource_type}' | project name, resourceGroup, id"
            response = self.resource_graph_client.resources(query, subscriptions=[self.subscription_id])
            resources = [res for res in response]
            self.logger.info(f"Đã khám phá {len(resources)} tài nguyên loại {resource_type}.")
            return resources
        except Exception as e:
            self.logger.error(f"Lỗi khi khám phá tài nguyên loại {resource_type}: {e}")
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
        timespan: Optional[str] = 'PT1H', 
        interval: Optional[str] = 'PT1M',
        aggregations: Optional[List[MetricAggregationType]] = None
    ) -> Dict[str, List[float]]:
        if not resource_id or not metric_names:
            self.logger.error("resource_id và metric_names không được để trống.")
            return {}
        if aggregations is None:
            aggregations = [MetricAggregationType.AVERAGE]
        
        try:
            metrics_data = self.client.list(
                resource_id=resource_id,
                metric_names=metric_names,
                timespan=timespan,
                interval=interval,
                aggregations=aggregations
            )
            metrics = {}
            for metric in metrics_data.metrics:
                metrics[metric.name] = []
                for ts in metric.timeseries:
                    for dp in ts.data:
                        # Sử dụng giá trị tương ứng trong các aggregation
                        value = dp.average or dp.total or dp.minimum or dp.maximum or dp.count or 0
                        metrics[metric.name].append(value)
            self.logger.info(f"Đã lấy metrics cho tài nguyên {resource_id}: {metric_names}")
            return metrics
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy metrics từ Azure Monitor cho tài nguyên {resource_id}: {e}")
            return {}

class AzureSentinelClient(AzureBaseClient):
    """
    Lớp để tương tác với Azure Sentinel.
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.security_client = SecurityCenter(self.credential, self.subscription_id)

    def get_recent_alerts(self, days: int = 1) -> List[Any]:
        """
        Lấy các cảnh báo (alerts) gần đây từ Azure Security Center.

        Args:
            days (int): Số ngày để lấy các cảnh báo gần đây.

        Returns:
            List[Any]: Danh sách các alerts.
        """
        try:
            alerts = self.security_client.alerts.list()
            recent_alerts = []
            cutoff_time = datetime.datetime.utcnow() - datetime.timedelta(days=days)
            for alert in alerts:
                # Kiểm tra sự tồn tại của thuộc tính 'properties' và 'created_time'
                if hasattr(alert, 'properties') and hasattr(alert.properties, 'created_time') and \
                   alert.properties.created_time >= cutoff_time:
                    recent_alerts.append(alert)
            self.logger.info(f"Đã lấy {len(recent_alerts)} alerts từ Azure Sentinel trong {days} ngày gần đây.")
            return recent_alerts
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy alerts từ Azure Sentinel: {e}")
            return []

class AzureLogAnalyticsClient(AzureBaseClient):
    """
    Lớp để tương tác với Azure Log Analytics.
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.client = LogAnalyticsDataClient(self.credential)
        self.workspace_ids = self.get_workspace_ids()

    def get_workspace_ids(self) -> List[str]:
        """
        Tự động khám phá và lấy tất cả Workspace IDs của Log Analytics.

        Returns:
            List[str]: Danh sách các Workspace ID.
        """
        try:
            resources = self.discover_resources('Microsoft.OperationalInsights/workspaces')
            workspace_ids = [res['id'] for res in resources]
            if workspace_ids:
                self.logger.info(f"Đã tìm thấy {len(workspace_ids)} Log Analytics Workspaces.")
            else:
                self.logger.warning("Không tìm thấy Log Analytics Workspace nào.")
            return workspace_ids
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy Workspace IDs: {e}")
            return []

    def query_logs(self, query: str, timespan: Optional[str] = "P1D") -> List[Any]:
        """
        Thực hiện truy vấn log trên Azure Log Analytics.

        Args:
            query (str): Truy vấn Kusto (KQL) để thực hiện.
            timespan (Optional[str]): Khoảng thời gian để thực hiện truy vấn (ISO 8601 duration hoặc start/end).

        Returns:
            List[Any]: Danh sách các bảng kết quả từ các Workspace.
        """
        results = []
        if not self.workspace_ids:
            self.logger.error("Không có Workspace ID để truy vấn logs.")
            return results
        try:
            for workspace_id in self.workspace_ids:
                body = logmodels.QueryBody(query=query, timespan=timespan)
                response = self.client.query(workspace_id=workspace_id, body=body)
                if response.tables:
                    results.extend(response.tables)
                    self.logger.info(f"Đã thực hiện truy vấn logs thành công trên Workspace ID: {workspace_id}.")
                else:
                    self.logger.info(f"Không có dữ liệu trả về từ Workspace ID: {workspace_id}.")
            return results
        except Exception as e:
            self.logger.error(f"Lỗi khi truy vấn logs từ Azure Log Analytics: {e}")
            return []

    def query_logs_with_time_range(self, query: str, start_time: datetime.datetime, end_time: datetime.datetime) -> List[Any]:
        """
        Thực hiện truy vấn log với khoảng thời gian cụ thể trên Azure Log Analytics.

        Args:
            query (str): Truy vấn Kusto (KQL) để thực hiện.
            start_time (datetime.datetime): Thời gian bắt đầu.
            end_time (datetime.datetime): Thời gian kết thúc.

        Returns:
            List[Any]: Danh sách các bảng kết quả từ các Workspace.
        """
        results = []
        if not self.workspace_ids:
            self.logger.error("Không có Workspace ID để truy vấn logs.")
            return results
        try:
            timespan = f"{start_time.isoformat()}/{end_time.isoformat()}"
            for workspace_id in self.workspace_ids:
                body = logmodels.QueryBody(query=query, timespan=timespan)
                response = self.client.query(workspace_id=workspace_id, body=body)
                if response.tables:
                    results.extend(response.tables)
                    self.logger.info(f"Đã thực hiện truy vấn logs thành công trên Workspace ID: {workspace_id} với timespan: {timespan}.")
                else:
                    self.logger.info(f"Không có dữ liệu trả về từ Workspace ID: {workspace_id} với timespan: {timespan}.")
            return results
        except Exception as e:
            self.logger.error(f"Lỗi khi truy vấn logs với khoảng thời gian cụ thể từ Azure Log Analytics: {e}")
            return []

class AzureSecurityCenterClient(AzureBaseClient):
    """
    Lớp để tương tác với Azure Security Center.
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.security_client = SecurityCenter(self.credential, self.subscription_id)


    def get_security_recommendations(self) -> List[Any]:
        """
        Lấy các khuyến nghị bảo mật từ Azure Security Center.

        Returns:
            List[Any]: Danh sách các khuyến nghị bảo mật.
        """
        try:
            # Sử dụng API security_recommendations.list()
            recommendations = self.security_client.security_recommendations.list()
            recommendations_list = list(recommendations)
            self.logger.info(f"Đã lấy {len(recommendations_list)} security recommendations từ Azure Security Center.")
            return recommendations_list
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy security recommendations từ Azure Security Center: {e}")
            return []

class AzureNetworkWatcherClient(AzureBaseClient):
    """
    Lớp để tương tác với Azure Network Watcher.
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.network_client = NetworkManagementClient(self.credential, self.subscription_id)

    def get_flow_logs(self, resource_group: str, network_watcher_name: str, nsg_name: str) -> List[Any]:
        """
        Lấy các flow logs từ Network Security Groups (NSGs) thông qua Azure Network Watcher.

        Args:
            resource_group (str): Tên Resource Group chứa NSG.
            network_watcher_name (str): Tên Network Watcher.
            nsg_name (str): Tên Network Security Group.

        Returns:
            List[Any]: Danh sách các flow log configurations.
        """
        try:
            flow_log_configurations = self.network_client.flow_logs.list(
                resource_group_name=resource_group,
                network_security_group_name=nsg_name
            )
            flow_logs = list(flow_log_configurations)
            self.logger.info(f"Đã lấy {len(flow_logs)} flow logs từ NSG {nsg_name} trong Resource Group {resource_group}.")
            return flow_logs
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy flow logs từ NSG {nsg_name} trong Resource Group {resource_group}: {e}")
            return []

    def create_flow_log(self, resource_group: str, network_watcher_name: str, nsg_name: str, flow_log_name: str, params: Dict[str, Any]) -> Optional[Any]:
        """
        Tạo một flow log mới cho một NSG.

        Args:
            resource_group (str): Tên Resource Group chứa NSG.
            network_watcher_name (str): Tên Network Watcher.
            nsg_name (str): Tên Network Security Group.
            flow_log_name (str): Tên của flow log mới.
            params (Dict[str, Any]): Thông số cấu hình cho flow log.

        Returns:
            Optional[Any]: Thông tin về flow log đã tạo hoặc None nếu lỗi.
        """
        try:
            flow_log = self.network_client.flow_logs.begin_create_or_update(
                resource_group_name=resource_group,
                network_security_group_name=nsg_name,
                flow_log_name=flow_log_name,
                parameters=params
            ).result()
            self.logger.info(f"Đã tạo flow log {flow_log_name} cho NSG {nsg_name} trong Resource Group {resource_group}.")
            return flow_log
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo flow log {flow_log_name} cho NSG {nsg_name} trong Resource Group {resource_group}: {e}")
            return None

    def delete_flow_log(self, resource_group: str, network_watcher_name: str, nsg_name: str, flow_log_name: str) -> bool:
        """
        Xóa một flow log từ NSG.

        Args:
            resource_group (str): Tên Resource Group chứa NSG.
            network_watcher_name (str): Tên Network Watcher.
            nsg_name (str): Tên Network Security Group.
            flow_log_name (str): Tên của flow log cần xóa.

        Returns:
            bool: True nếu xóa thành công, False nếu có lỗi.
        """
        try:
            self.network_client.flow_logs.begin_delete(
                resource_group_name=resource_group,
                network_security_group_name=nsg_name,
                flow_log_name=flow_log_name
            ).result()
            self.logger.info(f"Đã xóa flow log {flow_log_name} từ NSG {nsg_name} trong Resource Group {resource_group}.")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi xóa flow log {flow_log_name} từ NSG {nsg_name} trong Resource Group {resource_group}: {e}")
            return False

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
        """
        Tự động khám phá và lấy Workspace IDs của Traffic Analytics.

        Returns:
            List[str]: Danh sách các Workspace ID.
        """
        try:
            resources = self.discover_resources('Microsoft.OperationalInsights/workspaces')
            # Lọc các workspace liên quan đến Traffic Analytics
            traffic_workspace_ids = [res['id'] for res in resources if 'trafficanalytics' in res['name'].lower()]
            if traffic_workspace_ids:
                self.logger.info(f"Đã tìm thấy {len(traffic_workspace_ids)} Traffic Analytics Workspaces.")
            else:
                self.logger.warning("Không tìm thấy Traffic Analytics Workspace nào.")
            return traffic_workspace_ids
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy Workspace IDs của Traffic Analytics: {e}")
            return []

    def get_traffic_data(self, query: Optional[str] = None, timespan: Optional[str] = "P1D") -> List[Any]:
        """
        Lấy dữ liệu Traffic Analytics từ Azure Log Analytics.

        Args:
            query (Optional[str]): Truy vấn Kusto (KQL) để thực hiện. Nếu không cung cấp, sử dụng truy vấn mặc định.
            timespan (Optional[str]): Khoảng thời gian để thực hiện truy vấn (ISO 8601 duration hoặc start/end).

        Returns:
            List[Any]: Danh sách các bảng kết quả từ các Workspace.
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
                body = logmodels.QueryBody(query=query, timespan=timespan)
                response = self.log_analytics_client.query(workspace_id=workspace_id, body=body)
                if response.tables:
                    results.extend(response.tables)
                    self.logger.info(f"Đã lấy dữ liệu Traffic Analytics thành công từ Workspace ID: {workspace_id}.")
                else:
                    self.logger.info(f"Không có dữ liệu trả về từ Workspace ID: {workspace_id}.")
            return results
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy dữ liệu Traffic Analytics: {e}")
            return []

    def enable_traffic_analytics(self, resource_group: str, network_watcher_name: str, nsg_name: str, workspace_resource_id: str, storage_account_id: str, retention_days: int = 7) -> bool:
        """
        Bật Traffic Analytics cho một NSG bằng cách tạo hoặc cập nhật Flow Logs.

        Args:
            resource_group (str): Tên Resource Group chứa NSG.
            network_watcher_name (str): Tên Network Watcher.
            nsg_name (str): Tên Network Security Group.
            workspace_resource_id (str): Resource ID của Log Analytics Workspace.
            storage_account_id (str): Resource ID của Storage Account để lưu trữ Flow Logs.
            retention_days (int): Số ngày lưu trữ Flow Logs.

        Returns:
            bool: True nếu bật thành công, False nếu có lỗi.
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

    def disable_traffic_analytics(self, resource_group: str, network_watcher_name: str, nsg_name: str) -> bool:
        """
        Tắt Traffic Analytics cho một NSG bằng cách xóa Flow Logs.

        Args:
            resource_group (str): Tên Resource Group chứa NSG.
            network_watcher_name (str): Tên Network Watcher.
            nsg_name (str): Tên Network Security Group.

        Returns:
            bool: True nếu tắt thành công, False nếu có lỗi.
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

        Args:
            resource_group (str): Tên Resource Group chứa NSG.
            nsg_name (str): Tên Network Security Group.

        Returns:
            str: Vị trí của NSG.
        """
        try:
            nsg = self.network_client.network_security_groups.get(resource_group, nsg_name)
            return nsg.location
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy vị trí của NSG {nsg_name}: {e}")
            return "eastus"  # Mặc định là eastus nếu không lấy được vị trí

    def get_workspace_region(self, workspace_resource_id: str) -> str:
        """
        Lấy khu vực của Log Analytics Workspace.

        Args:
            workspace_resource_id (str): Resource ID của Log Analytics Workspace.

        Returns:
            str: Khu vực của Workspace.
        """
        try:
            workspace = self.resource_management_client.resources.get_by_id(workspace_resource_id, '2015-11-01-preview')
            return workspace.location
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy khu vực của Workspace {workspace_resource_id}: {e}")
            return "eastus"  # Mặc định là eastus nếu không lấy được khu vực

class AzureMLClient(AzureBaseClient):
    """
    Lớp để tương tác với Azure Machine Learning Clusters.
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.ml_client = AzureMachineLearningWorkspaces(self.credential, self.subscription_id)
        self.compute_resource_type = 'Microsoft.MachineLearningServices/workspaces/computes'

    def discover_ml_clusters(self) -> List[Dict[str, Any]]:
        """
        Khám phá các Azure ML Clusters.

        Returns:
            List[Dict[str, Any]]: Danh sách các ML Clusters.
        """
        try:
            resources = self.discover_resources(self.compute_resource_type)
            self.logger.info(f"Đã khám phá {len(resources)} Azure ML Clusters.")
            return resources
        except Exception as e:
            self.logger.error(f"Lỗi khi khám phá Azure ML Clusters: {e}")
            return []

    def get_ml_cluster_metrics(self, compute_id: str, metric_names: List[str], timespan: Optional[str] = 'PT1H', interval: Optional[str] = 'PT1M') -> Dict[str, List[float]]:
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
                        # Sử dụng giá trị tương ứng trong các aggregation
                        value = dp.average or dp.total or dp.minimum or dp.maximum or dp.count or 0
                        metrics[metric.name].append(value)
            self.logger.info(f"Đã lấy metrics cho ML Cluster {compute_id}: {metric_names}")
            return metrics
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy metrics từ ML Cluster {compute_id}: {e}")
            return {}
    
class AzureAnomalyDetectorClient(AzureBaseClient):
    """
    Lớp để tương tác với Azure Anomaly Detector.
    """
    def __init__(self, logger: logging.Logger, config: Dict[str, Any]):
        super().__init__(logger)
        self.endpoint = config.get("azure_anomaly_detector", {}).get("api_base")
        self.api_key = config.get("azure_anomaly_detector", {}).get("api_key")
        self.client = self.authenticate()
    
    def authenticate(self) -> AnomalyDetectorClient:
        """
        Khởi tạo AnomalyDetectorClient với endpoint và API key.
        """
        if not self.endpoint or not self.api_key:
            self.logger.error("Thông tin endpoint hoặc api_key cho Azure Anomaly Detector không được thiết lập.")
            raise ValueError("Thiếu thông tin cấu hình cho Azure Anomaly Detector.")
        
        try:
            client = AnomalyDetectorClient(endpoint=self.endpoint, credential=self.api_key)
            self.logger.info("Đã kết nối thành công với Azure Anomaly Detector.")
            return client
        except Exception as e:
            self.logger.error(f"Lỗi khi kết nối với Azure Anomaly Detector: {e}")
            raise e
    
    def detect_anomalies(self, metric_data: Dict[str, Any]) -> bool:
        """
        Phát hiện bất thường dựa trên dữ liệu metrics gửi đến.

        Args:
            metric_data (Dict[str, Any]): Dữ liệu metrics của các tiến trình.

        Returns:
            bool: True nếu phát hiện bất thường, False ngược lại.
        """
        try:
            # Giả định metric_data là dict với keys là PID và values là dict chứa các metrics
            for pid, metrics in metric_data.items():
                # Chuẩn bị dữ liệu time series
                series = []
                cpu_usage = metrics.get('cpu_usage_percent', [])
                if not cpu_usage:
                    continue
                for i, usage in enumerate(cpu_usage):
                    timestamp = datetime.datetime.utcnow() - datetime.timedelta(minutes=len(cpu_usage)-i)
                    series.append({"timestamp": timestamp.isoformat(), "value": usage})
                
                # Sử dụng UnivariateDetectionOptions
                options = UnivariateDetectionOptions(
                    series=series,
                    granularity="minutely",
                    sensitivity=95  # Điều chỉnh sensitivity theo yêu cầu
                )
                
                # Gửi yêu cầu phát hiện bất thường
                response: UnivariateEntireDetectionResult = self.client.detect_univariate_entire_series(options)
                
                # Kiểm tra kết quả
                if any(response.is_anomaly):
                    self.logger.warning(f"Phát hiện bất thường trong tiến trình PID {pid}.")
                    return True
            
            return False
        except Exception as e:
            self.logger.error(f"Lỗi khi phát hiện bất thường với Azure Anomaly Detector: {e}")
            return False

class AzureOpenAIClient(AzureBaseClient):
    """
    Lớp để tương tác với Azure OpenAI Service.
    """
    def __init__(self, logger: logging.Logger, config: Dict[str, Any]):
        super().__init__(logger)
        self.endpoint = config.get("azure_openai", {}).get("api_base")
        self.api_key = config.get("azure_openai", {}).get("api_key")
        self.deployment_name = config.get("azure_openai", {}).get("deployment_name")
        self.api_version = config.get("azure_openai", {}).get("api_version", "2023-03-15-preview")
        self.initialize_openai()
    
    def initialize_openai(self):
        """
        Cấu hình OpenAI với endpoint và API key.
        """
        if not self.endpoint or not self.api_key or not self.deployment_name:
            self.logger.error("Thông tin endpoint, api_key hoặc deployment_name cho Azure OpenAI không được thiết lập.")
            raise ValueError("Thiếu thông tin cấu hình cho Azure OpenAI Service.")
        
        try:
            openai.api_type = "azure"
            openai.api_base = self.endpoint
            openai.api_version = self.api_version
            openai.api_key = self.api_key
            self.logger.info("Đã cấu hình thành công Azure OpenAI Service.")
        except Exception as e:
            self.logger.error(f"Lỗi khi cấu hình Azure OpenAI Service: {e}")
            raise e
    
    def get_optimization_suggestions(self, state_data: Dict[str, Any]) -> List[float]:
        """
        Gửi dữ liệu trạng thái hệ thống đến Azure OpenAI và nhận gợi ý tối ưu hóa.

        Args:
            state_data (Dict[str, Any]): Dữ liệu trạng thái của hệ thống.

        Returns:
            List[float]: Danh sách các hành động tối ưu hóa đề xuất.
        """
        try:
            prompt = self.construct_prompt(state_data)
            response = openai.Completion.create(
                engine=self.deployment_name,
                prompt=prompt,
                max_tokens=150,
                temperature=0.5,
                n=1,
                stop=None,
            )
            suggestion_text = response.choices[0].text.strip()
            # Giả định rằng gợi ý được trả về dưới dạng danh sách số, ngăn cách bằng dấu phẩy
            suggestions = [float(x.strip()) for x in suggestion_text.split(',') if x.strip()]
            self.logger.info(f"Nhận được gợi ý tối ưu hóa từ Azure OpenAI: {suggestions}")
            return suggestions
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy gợi ý từ Azure OpenAI Service: {e}")
            return []
    
    def construct_prompt(self, state_data: Dict[str, Any]) -> str:
        """
        Xây dựng prompt gửi đến OpenAI dựa trên dữ liệu trạng thái hệ thống.

        Args:
            state_data (Dict[str, Any]): Dữ liệu trạng thái của hệ thống.

        Returns:
            str: Prompt hoàn chỉnh.
        """
        prompt = "Dựa trên các thông số hệ thống sau đây, đề xuất các điều chỉnh tối ưu hóa tài nguyên dưới dạng danh sách số:\n"
        for pid, metrics in state_data.items():
            prompt += f"Tiến trình PID {pid}: CPU Usage: {metrics.get('cpu_usage_percent', 0)}%, RAM Usage: {metrics.get('memory_usage_mb', 0)}MB, "
            prompt += f"GPU Usage: {metrics.get('gpu_usage_percent', 0)}%, Disk I/O: {metrics.get('disk_io_mbps', 0)}Mbps, "
            prompt += f"Network Bandwidth: {metrics.get('network_bandwidth_mbps', 0)}Mbps, Cache Limit: {metrics.get('cache_limit_percent', 0)}%.\n"
        prompt += "Hãy trả về các hành động tối ưu hóa dưới dạng danh sách số, ví dụ: [cpu_threads, frequency, ram_allocation_mb, gpu_usage_percent, disk_io_limit_mbps, network_bandwidth_limit_mbps, cache_limit_percent]."
        return prompt
