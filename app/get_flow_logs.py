import logging
from mining_environment.scripts.azure_clients import AzureNetworkWatcherClient

# Khởi tạo logger
logger = logging.getLogger("AzureNetworkWatcherClient")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# Định nghĩa Resource Group và NSG
resource_group_nsg = "amtechllp"  # Resource Group chứa NSG
nsg_name = "amtechllp-nsg"  # Tên NSG
resource_group_watcher = "NetworkWatcherRG"  # Resource Group chứa Network Watcher
network_watcher_name = "NetworkWatcher_southeastasia"  # Tên Network Watcher

try:
    # Khởi tạo client AzureNetworkWatcherClient
    logger.info("Khởi tạo AzureNetworkWatcherClient...")
    network_watcher_client = AzureNetworkWatcherClient(logger)

    # Lấy Flow Logs
    logger.info(f"Lấy Flow Logs cho NSG {nsg_name} từ Resource Group {resource_group_nsg}...")
    flow_logs = network_watcher_client.get_flow_logs(
        resource_group=resource_group_watcher,
        nsg_name=nsg_name,
        network_watcher_name=network_watcher_name
    )

    # In kết quả
    if flow_logs:
        logger.info(f"Flow Logs đã lấy thành công: {flow_logs}")
    else:
        logger.warning("Không có Flow Logs nào được trả về.")
except Exception as e:
    logger.error(f"Lỗi khi lấy Flow Logs: {e}")
