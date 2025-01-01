import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from azure.identity import ClientSecretCredential, DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.machinelearningservices import AzureMachineLearningWorkspaces
from azure.ai.ml import MLClient
from azure.monitor.query import MetricsQueryClient, MetricAggregationType
from azure.core.exceptions import HttpResponseError

class AzureBaseClient:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
        if not self.subscription_id:
            self.logger.error("AZURE_SUBSCRIPTION_ID không được thiết lập.")
            raise ValueError("AZURE_SUBSCRIPTION_ID không được thiết lập.")

        self.credential = self.authenticate()
        self.resource_management_client = ResourceManagementClient(
            self.credential,
            self.subscription_id
        )

    def authenticate(self) -> ClientSecretCredential:
        client_id = os.getenv('AZURE_CLIENT_ID')
        client_secret = os.getenv('AZURE_CLIENT_SECRET')
        tenant_id = os.getenv('AZURE_TENANT_ID')

        if not all([client_id, client_secret, tenant_id]):
            self.logger.error("Cần AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID trong môi trường.")
            raise ValueError("Thiếu thông tin xác thực Azure.")

        try:
            cred = ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret
            )
            self.logger.info("Đã xác thực thành công với Azure AD (ClientSecretCredential).")
            return cred
        except Exception as e:
            self.logger.error(f"Lỗi khi xác thực với Azure AD: {e}")
            raise e


class AzureMLClient(AzureBaseClient):
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self.aml_mgmt_client = AzureMachineLearningWorkspaces(self.credential, self.subscription_id)

    def discover_all_aml_compute(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Liệt kê toàn bộ AML Workspaces và các Compute trong subscription.
        Trả về dict: { "RG/WorkspaceName": [ { id, name, type, provisioning_state, scale_set_id }, ... ] }
        """
        results = {}
        rg_list = list(self.resource_management_client.resource_groups.list())
        self.logger.info(f"Phát hiện {len(rg_list)} Resource Groups trong subscription {self.subscription_id}.")

        for rg in rg_list:
            rg_name = rg.name
            try:
                workspaces = list(self.aml_mgmt_client.workspaces.list_by_resource_group(rg_name))
                if not workspaces:
                    continue

                self.logger.info(f"  RG='{rg_name}' có {len(workspaces)} AML Workspace.")
                for ws in workspaces:
                    ws_name = ws.name
                    self.logger.info(f"    => Workspace '{ws_name}': liệt kê compute AML...")

                    # Sử dụng DefaultAzureCredential để tương thích
                    ml_credential = DefaultAzureCredential()
                    ml_client = MLClient(
                        credential=ml_credential,
                        subscription_id=self.subscription_id,
                        resource_group_name=rg_name,
                        workspace_name=ws_name
                    )

                    try:
                        compute_list = ml_client.compute.list()
                        comp_results = []
                        for comp in compute_list:
                            # Tạo item cơ bản
                            item = {
                                "id": comp.id,
                                "name": comp.name,
                                "type": comp.type,
                                "provisioning_state": comp.provisioning_state,
                                "scale_set_id": None  # Mặc định là None
                            }

                            # Nếu là amlcompute, ta cố gắng truy xuất Scale Set ID
                            if comp.type.lower() == "amlcompute":
                                # Trước hết, thử property resource_id (nếu SDK có)
                                scale_set_id = getattr(comp, "resource_id", None)
                                # Nếu vẫn None, in __dict__ để kiểm tra
                                if not scale_set_id:
                                    self.logger.info(f"DEBUG comp.__dict__ = {comp.__dict__}")
                                    # Từ log, bạn có thể tìm key "resourceId" hoặc "virtualMachineScaleSetId", v.v.
                                    # Ví dụ:
                                    # scale_set_id = comp.__dict__.get("_other_properties", {}).get("resourceId")

                                item["scale_set_id"] = scale_set_id or None

                            comp_results.append(item)
                            self.logger.info(f"      - compute: {item}")
                        if comp_results:
                            key = f"{rg_name}/{ws_name}"
                            results.setdefault(key, []).extend(comp_results)
                    except Exception as excomp:
                        self.logger.error(f"      Lỗi liệt kê compute workspace='{ws_name}': {excomp}")
            except Exception as exws:
                self.logger.error(f"Lỗi liệt kê workspace trong RG='{rg_name}': {exws}")

        return results

    def get_vm_scale_set_metrics(
        self,
        vmss_id: str,
        metric_names: List[str],
        timespan: str = "PT48H",
        interval: str = "PT1H"
    ) -> Dict[str, List[float]]:
        """
        Lấy metrics hệ thống (CPU, RAM, Disk, Network,...) cho VM Scale Set (bao gồm LowPriority).
        vmss_id ví dụ: /subscriptions/<subID>/resourceGroups/<rg>/providers/Microsoft.Compute/virtualMachineScaleSets/<tên VMSS>
        metric_names ví dụ: ["Percentage CPU", "Network In Total", "Network Out Total"]
        """
        if not vmss_id:
            self.logger.warning("vmss_id rỗng, bỏ qua.")
            return {}

        try:
            monitor_client = MetricsQueryClient(self.credential)

            end_time = datetime.utcnow()
            # timespan dạng PT48H => 48 giờ, parse thô
            hours = int(timespan.replace("PT", "").replace("H", "")) if "H" in timespan else 48
            start_time = end_time - timedelta(hours=hours)

            response = monitor_client.query_resource(
                resource_uri=vmss_id,
                metric_names=metric_names,
                timespan=(start_time, end_time),
                granularity=interval,
                aggregations=[MetricAggregationType.AVERAGE]
            )

            metrics = {}
            for metric in response.metrics:
                metric_values = []
                for ts in metric.timeseries:
                    for dp in ts.data:
                        val = dp.average or 0
                        metric_values.append(val)
                metrics[metric.name] = metric_values

            self.logger.info(f"Đã lấy metrics cho VM Scale Set: {vmss_id}")
            return metrics

        except HttpResponseError as e:
            self.logger.error(f"HttpResponseError: {e.message}")
        except Exception as e:
            self.logger.error(f"Lỗi không xác định: {e}")
        return {}


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("azure_mlclient_combined_main")

    client = AzureMLClient(logger)
    all_compute_map = client.discover_all_aml_compute()

    logger.info("== Kết quả cuối cùng (giám sát CPU, RAM, Disk, Network cho Low-Priority) ==")
    # Ví dụ các metric chính, tuỳ theo doc:
    # https://learn.microsoft.com/en-us/azure/azure-monitor/essentials/metrics-supported#microsoftcomputevirtualmachinescalesets
    wanted_metrics = [
        "Percentage CPU",      # CPU
        "Network In Total",    # Network In
        "Network Out Total",   # Network Out
        "Disk Read Bytes/Sec", # Disk
        # "Available Memory Bytes", ...
    ]

    for ws_key, compute_list in all_compute_map.items():
        logger.info(f"  Workspace: {ws_key}")

        for c in compute_list:
            logger.info(f"    Compute: {c}")

            if c["type"].lower() == "amlcompute":
                vmss_id = c["scale_set_id"]
                if vmss_id:
                    # Lấy metrics VM Scale Set
                    sys_metrics = client.get_vm_scale_set_metrics(
                        vmss_id=vmss_id,
                        metric_names=wanted_metrics,
                        timespan="PT48H",
                        interval="PT1H"
                    )
                    logger.info(f"    => VMSS metrics cho '{c['name']}': {sys_metrics}")
                else:
                    logger.info(f"    => Chưa xác định được Scale Set ID cho '{c['name']}'.")
            else:
                logger.info(f"    => '{c['name']}' không phải amlcompute, bỏ qua.")


if __name__ == "__main__":
    main()
