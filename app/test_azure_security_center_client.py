import logging

# Giả sử azure_clients.py nằm cùng thư mục, có AzureSecurityCenterClient
from mining_environment.scripts.azure_clients import AzureSecurityCenterClient


def main():
    """
    Hàm chính để kiểm thử AzureSecurityCenterClient.
    Các biến môi trường cần được set từ trước:
      AZURE_SUBSCRIPTION_ID,
      AZURE_CLIENT_ID,
      AZURE_CLIENT_SECRET,
      AZURE_TENANT_ID
    """

    # 1) Tạo logger đơn giản
    logger = logging.getLogger("test_azure_sc")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # 2) Khởi tạo AzureSecurityCenterClient
    sc_client = AzureSecurityCenterClient(logger=logger)

    # 3) Gọi một số hàm để kiểm tra
    logger.info("=== Lấy Assessments ===")
    assessments = sc_client.get_security_assessments()
    logger.info(f"Số lượng Assessments: {len(assessments)}")

    logger.info("=== Lấy Secure Scores ===")
    scores = sc_client.get_secure_scores()
    logger.info(f"Số lượng Secure Scores: {len(scores)}")

    logger.info("=== Lấy Recommendations ===")
    recommendations = sc_client.get_recommendations()
    logger.info(f"Số lượng Recommendations: {len(recommendations)}")

if __name__ == "__main__":
    main()
