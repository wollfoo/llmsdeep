import os
import logging
from datetime import datetime, timedelta
from azure.monitor.query import MetricsQueryClient
from azure.identity import DefaultAzureCredential
from azure.mgmt.resourcegraph import ResourceGraphClient
from azure.mgmt.resourcegraph.models import QueryRequest

def discover_resources(credential, subscription_id, resource_type):
    """
    Tìm kiếm tài nguyên Azure dựa trên loại tài nguyên.
    """
    try:
        resource_graph_client = ResourceGraphClient(credential)
        query = f"Resources | where type =~ '{resource_type}' | project name, resourceGroup, id"
        request = QueryRequest(
            subscriptions=[subscription_id],
            query=query
        )
        response = resource_graph_client.resources(request)
        resources = [{"name": res["name"], "resourceGroup": res["resourceGroup"], "id": res["id"]} for res in response.data]
        return resources
    except Exception as e:
        raise RuntimeError(f"Lỗi khi khám phá tài nguyên: {e}")

def test_azure_monitor_query_with_discovery():
    """
    Kiểm thử SDK Azure Monitor Query với khám phá tài nguyên động.
    """
    # Cấu hình Logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("AzureMonitorQueryTest")

    try:
        # Lấy giá trị từ biến môi trường
        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        if not subscription_id:
            logger.error("AZURE_SUBSCRIPTION_ID không được thiết lập.")
            raise ValueError("Thiếu giá trị AZURE_SUBSCRIPTION_ID trong biến môi trường.")

        # Xác thực bằng DefaultAzureCredential
        credential = DefaultAzureCredential()
        logger.info("Đã xác thực thành công bằng DefaultAzureCredential.")

        # Khám phá tài nguyên Microsoft.Compute/virtualMachines
        resource_type = "Microsoft.Compute/virtualMachines"
        logger.info(f"Khám phá tài nguyên loại {resource_type} trong subscription {subscription_id}.")
        resources = discover_resources(credential, subscription_id, resource_type)

        if not resources:
            logger.error(f"Không tìm thấy tài nguyên loại {resource_type} trong subscription {subscription_id}.")
            return

        # Sử dụng tài nguyên đầu tiên cho kiểm thử
        resource = resources[0]
        resource_id = resource["id"]
        resource_group = resource["resourceGroup"]
        vm_name = resource["name"]

        logger.info(f"Đã tìm thấy tài nguyên: {vm_name} trong nhóm {resource_group}.")

        # Khởi tạo MetricsQueryClient
        metrics_client = MetricsQueryClient(credential)

        # Truy vấn metrics khả dụng
        logger.info(f"Truy vấn metrics khả dụng cho tài nguyên {resource_id}.")
        response = metrics_client.query_resource(
            resource_uri=resource_id,
            metric_names=["Percentage CPU"],  # Metric cần kiểm tra
            timespan=(datetime.utcnow() - timedelta(hours=1), datetime.utcnow()),  # 1 giờ qua
            granularity="PT1M",
        )

        # In kết quả metrics
        logger.info("Kết quả metrics truy vấn:")
        for metric in response.metrics:
            logger.info(f"Metric Name: {metric.name}")
            for timeseries in metric.timeseries:
                for data in timeseries.data:
                    logger.info(f"Timestamp: {data.timestamp}, Value: {data.average or data.total or 0}")

    except Exception as e:
        logger.error(f"Lỗi xảy ra trong quá trình kiểm thử Azure Monitor Query: {e}")


if __name__ == "__main__":
    test_azure_monitor_query_with_discovery()
