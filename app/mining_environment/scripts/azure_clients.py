# azure_clients.py

import os
import logging
from typing import List, Dict, Any
import datetime  # Đảm bảo import datetime nếu sử dụng trong các lớp

from azure.monitor.query import MonitorClient, MetricAggregationType
from azure.mgmt.security import SecurityCenterClient
from azure.loganalytics import LogAnalyticsDataClient, models
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.resource import ResourceManagementClient
from azure.identity import ClientSecretCredential
from azure.resourcegraph import ResourceGraphClient



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

    def authenticate(self):
        client_id = os.getenv('AZURE_CLIENT_ID')
        client_secret = os.getenv('AZURE_CLIENT_SECRET')
        tenant_id = os.getenv('AZURE_TENANT_ID')

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
            query = f"resources | where type =~ '{resource_type}' | project name, resourceGroup, id"
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
        self.client = MonitorClient(self.credential, self.subscription_id)

    def get_metrics(self, resource_id: str, metric_names: List[str], timespan: str = 'PT1H', interval: str = 'PT1M') -> Dict[str, List[float]]:
        """
        Lấy các chỉ số (metrics) của tài nguyên từ Azure Monitor.
        """
        try:
            metrics_data = self.client.metrics.list(
                resource_id,
                timespan=timespan,
                interval=interval,
                metricnames=','.join(metric_names),
                aggregation=MetricAggregationType.AVERAGE
            )
            metrics = {}
            for metric in metrics_data.value:
                metrics[metric.name.value] = [
                    datapoint.average for datapoint in metric.timeseries[0].data
                    if datapoint.average is not None
                ]
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
        self.client = SecurityCenterClient(self.credential, self.subscription_id)

    def get_recent_alerts(self, days: int = 1) -> List[Any]:
        """
        Lấy các cảnh báo (alerts) gần đây từ Azure Sentinel.
        """
        try:
            alerts = self.client.alerts.list()
            recent_alerts = []
            cutoff_time = datetime.datetime.utcnow() - datetime.timedelta(days=days)
            for alert in alerts:
                if alert.created_time and alert.created_time >= cutoff_time:
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
        self.workspace_id = self.get_workspace_id()

    def get_workspace_id(self) -> str:
        """
        Tự động khám phá và lấy Workspace ID của Log Analytics.
        """
        try:
            resources = self.discover_resources('Microsoft.OperationalInsights/workspaces')
            if resources:
                workspace = resources[0]
                self.logger.info(f"Đã tìm thấy Log Analytics Workspace: {workspace['name']}")
                return workspace['id'].split('/')[-1]
            else:
                self.logger.warning("Không tìm thấy Log Analytics Workspace nào.")
                return ""
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy Workspace ID: {e}")
            return ""

    def query_logs(self, query: str, timespan: str = "P1D") -> List[Any]:
        """
        Thực hiện truy vấn log trên Azure Log Analytics.
        """
        if not self.workspace_id:
            self.logger.error("Không có Workspace ID để truy vấn logs.")
            return []
        try:
            body = models.QueryBody(query=query, timespan=timespan)
            response = self.client.query(workspace_id=self.workspace_id, body=body)
            self.logger.info(f"Đã thực hiện truy vấn logs: {query}")
            return response.tables
        except Exception as e:
            logger.error(f"Lỗi khi truy vấn logs từ Azure Log Analytics: {e}")
            return []

class AzureSecurityCenterClient(AzureBaseClient):
    """
    Lớp để tương tác với Azure Security Center.
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.client = SecurityCenterClient(self.credential, self.subscription_id)

    def get_security_recommendations(self) -> List[Any]:
        """
        Lấy các khuyến nghị bảo mật từ Azure Security Center.
        """
        try:
            recommendations = self.client.security_recommendations.list()
            recommendations_list = list(recommendations)
            self.logger.info(f"Đã lấy {len(recommendations_list)} security recommendations từ Azure Security Center.")
            return recommendations_list
        except Exception as e:
            logger.error(f"Lỗi khi lấy security recommendations từ Azure Security Center: {e}")
            return []

class AzureNetworkWatcherClient(AzureBaseClient):
    """
    Lớp để tương tác với Azure Network Watcher.
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.client = NetworkManagementClient(self.credential, self.subscription_id)

    def get_flow_logs(self, resource_group: str, network_watcher_name: str, nsg_name: str) -> List[Any]:
        """
        Lấy các flow logs từ Network Security Groups (NSGs) thông qua Azure Network Watcher.
        """
        try:
            flow_log = self.client.network_security_groups.list_flow_logs(
                resource_group_name=resource_group,
                network_security_group_name=nsg_name
            )
            flow_logs = list(flow_log)
            self.logger.info(f"Đã lấy {len(flow_logs)} flow logs từ NSG {nsg_name} trong Resource Group {resource_group}.")
            return flow_logs
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy flow logs từ NSG {nsg_name} trong Resource Group {resource_group}: {e}")
            return []

class AzureTrafficAnalyticsClient(AzureBaseClient):
    """
    Lớp để tương tác với Azure Traffic Analytics.
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.client = LogAnalyticsDataClient(self.credential)
        self.workspace_id = self.get_traffic_workspace_id()

    def get_traffic_workspace_id(self) -> str:
        """
        Tự động khám phá và lấy Workspace ID của Traffic Analytics.
        """
        try:
            resources = self.discover_resources('Microsoft.OperationalInsights/workspaces')
            if resources:
                workspace = resources[0]
                self.logger.info(f"Đã tìm thấy Traffic Analytics Workspace: {workspace['name']}")
                return workspace['id'].split('/')[-1]
            else:
                self.logger.warning("Không tìm thấy Traffic Analytics Workspace nào.")
                return ""
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy Traffic Analytics Workspace ID: {e}")
            return ""

    def get_traffic_data(self) -> List[Any]:
        """
        Lấy dữ liệu Traffic Analytics từ Azure Log Analytics.
        """
        if not self.workspace_id:
            self.logger.error("Không có Traffic Analytics Workspace ID để lấy dữ liệu.")
            return []
        try:
            query = """
            AzureDiagnostics
            | where Category == "NetworkSecurityGroupFlowEvent"
            | summarize Count = count() by bin(TimeGenerated, 1h), SourceIP, DestinationIP
            """
            body = models.QueryBody(query=query, timespan="P1D")
            response = self.client.query(workspace_id=self.workspace_id, body=body)
            self.logger.info("Đã lấy dữ liệu Traffic Analytics thành công.")
            return response.tables
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy dữ liệu Traffic Analytics: {e}")
            return []

class AzureMLClient(AzureBaseClient):
    """
    Lớp để tương tác với Azure Machine Learning Clusters.
    """
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        # Không cần client cụ thể vì sử dụng ResourceGraphClient để khám phá
        pass

    def discover_ml_clusters(self) -> List[Dict[str, Any]]:
        """
        Khám phá các Azure ML Clusters.
        """
        resource_type = 'Microsoft.MachineLearningServices/workspaces/computes'
        clusters = self.discover_resources(resource_type)
        self.logger.info(f"Đã khám phá {len(clusters)} Azure ML Clusters.")
        return clusters
