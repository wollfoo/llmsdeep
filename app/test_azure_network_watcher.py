from azureml.core import Workspace
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.core.exceptions import ResourceExistsError

# Cấu hình thông tin
subscription_id = "74007d19-e81a-44d6-81f3-b00269f8e6a5"
resource_group = "ncslearngning"
workspace_name = "ncslearngning"
location = "eastus"
storage_account_name = "ncslearngning".lower()
nsg_name = "ncslearngning"

# Xác thực và khởi tạo các client
credential = DefaultAzureCredential()
resource_client = ResourceManagementClient(credential, subscription_id)
network_client = NetworkManagementClient(credential, subscription_id)
storage_client = StorageManagementClient(credential, subscription_id)

# 4. Lấy Network Watcher
print("Đang lấy danh sách Network Watcher...")
network_watchers = network_client.network_watchers.list_all()
network_watcher_name = None

for nw in network_watchers:
    if nw.location == location:
        network_watcher_name = nw.name
        break

if not network_watcher_name:
    raise Exception(f"Không tìm thấy Network Watcher trong khu vực '{location}'.")

print(f"Network Watcher '{network_watcher_name}' đã được tìm thấy trong khu vực '{location}'.")

# 5. Lấy Resource ID của NSG
print("Đang lấy thông tin NSG...")
nsg = network_client.network_security_groups.get(
    resource_group_name=resource_group,
    network_security_group_name=nsg_name
)

# 6. Bật Flow Logs cho NSG
flow_log_params = {
    "location": location,
    "enabled": True,
    "storageId": storage_account.id,  # ID của Storage Account
    "targetResourceId": nsg.id,      # Resource ID của NSG
    "retentionPolicy": {"days": 7, "enabled": True},  # Giữ logs 7 ngày
}

print("Đang bật Flow Logs cho NSG...")
network_client.flow_logs.begin_create_or_update(
    resource_group_name=resource_group,
    network_watcher_name=network_watcher_name,
    flow_log_name=f"{nsg_name}-flowlog",
    parameters=flow_log_params
).result()
print(f"Flow Logs đã được bật cho NSG '{nsg_name}'.")
