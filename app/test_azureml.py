import logging

# Import lớp AzureMLClient theo đường dẫn gói Python
# Chú ý: Thư mục 'mining_environment' và 'scripts' phải có __init__.py 
from mining_environment.scripts.azure_clients import AzureMLClient

def main():
    logger = logging.getLogger("test_azure_mlclient")
    logger.setLevel(logging.INFO)

    ml_client = AzureMLClient(logger)
    clusters = ml_client.discover_ml_clusters()
    logger.info(f"Đã tìm thấy {len(clusters)} ML clusters: {clusters}")

if __name__ == "__main__":
    main()
